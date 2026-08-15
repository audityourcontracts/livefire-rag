//! Portable fast-index writer and exact dense/lexical/fused reader.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::{
        ArrowWriter,
        arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    },
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use rag_embedding::{EmbeddingProfile, validate_embedding_profile};
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, TransactionBehavior, params, params_from_iter,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const VECTOR_MAGIC: [u8; 8] = *b"LFRAGV1\0";
pub const EMBEDDING_VECTOR_MAGIC: [u8; 8] = *b"LFREMB01";
pub const VECTOR_HEADER_BYTES: u32 = 64;
pub const MAX_RETURNED_OCCURRENCES_PER_HIT: usize = 50;
const PARQUET_WRITE_BATCH_ROWS: usize = 8_192;
const OCCURRENCE_LOOKUP_SCHEMA: &str = "sqlite-occurrence-lookup-v1";
const SQLITE_LEXICAL_SCHEMA: &str = "sqlite-inverted-bm25-v1";
const SQLITE_LEXICAL_APPLICATION_ID: i64 = 0x4c_46_52_33;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("index I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("index JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("index Parquet failed: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("index Arrow failed: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("index occurrence lookup failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("index invariant failed: {0}")]
    Invalid(&'static str),
    #[error("index artifact is corrupt: {0}")]
    Corrupt(&'static str),
}

pub type Result<T> = std::result::Result<T, IndexError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastDocument {
    pub document_id: String,
    pub document_sha256: String,
    pub document_kind: String,
    pub semantic_text: String,
    pub facets_json: String,
    pub relations_json: String,
    pub occurrence_count: u64,
    pub vector_ordinal: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastOccurrence {
    pub occurrence_id: String,
    pub document_id: String,
    pub event_time_ms: Option<u64>,
    pub relation: String,
    pub exact_attributes_json: String,
    pub snapshot_sha256: String,
    pub mapping_sha256: String,
    pub event_id: String,
    pub support_ref: String,
}

/// One vector bound to its canonical document ordinal.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedVector {
    pub vector_ordinal: u64,
    pub values: Vec<f32>,
}

/// A planned, consecutive `LFREMB01` vector result part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedVectorShard {
    pub path: PathBuf,
    pub first_vector_ordinal: u64,
    pub vector_count: u64,
    pub dimensions: u32,
    pub order_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectSummary {
    pub path: String,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: String,
    pub order_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSummary {
    pub path: String,
    pub count: u64,
    pub bytes: u64,
    pub sha256: String,
    pub dimensions: u32,
    pub dtype: String,
    pub header_bytes: u32,
    pub document_order_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalSummary {
    pub path: String,
    pub document_count: u64,
    pub bytes: u64,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub tokenizer: String,
    pub k1: f64,
    pub b: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LexicalFormat {
    JsonV2,
    SqliteV3,
}

struct IndexAssemblyOptions {
    source: SourceBinding,
    build_scope: BuildScope,
    embedding_profile: EmbeddingProfile,
    lexical_format: LexicalFormat,
    pipeline_provenance: Option<PipelineProvenance>,
    test_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccurrenceLookupSummary {
    pub path: String,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: String,
    pub schema: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBinding {
    pub snapshot_sha256: String,
    pub mapping_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildScope {
    Full,
    Sample,
}

/// Immutable upstream identities used to assemble a dataset index.
///
/// General-purpose and legacy index writers may omit this value. Dataset
/// catalogues require it so an otherwise valid index cannot be paired with the
/// wrong prepared corpus or embedding results merely because their row counts
/// and embedding profiles happen to match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineProvenance {
    pub dataset_sha256: String,
    pub prepared_corpus_sha256: String,
    pub embedding_plan_sha256: String,
    pub embedding_result_set_sha256: String,
}

/// Assembly settings for an index that will be admitted into a dataset
/// catalogue.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineIndexOptions {
    pub source: SourceBinding,
    pub build_scope: BuildScope,
    pub embedding_profile: EmbeddingProfile,
    pub provenance: PipelineProvenance,
    pub test_only: bool,
}

impl PipelineProvenance {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.dataset_sha256,
            &self.prepared_corpus_sha256,
            &self.embedding_plan_sha256,
            &self.embedding_result_set_sha256,
        ] {
            decode_sha256(digest)?;
            if digest.bytes().any(|byte| byte.is_ascii_uppercase()) {
                return Err(IndexError::Invalid("pipeline provenance digest"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastIndexManifest {
    pub schema_version: String,
    pub component_sha256: String,
    pub source: SourceBinding,
    pub build_scope: BuildScope,
    pub complete: bool,
    pub documents: ObjectSummary,
    pub occurrences: ObjectSummary,
    pub vectors: VectorSummary,
    pub lexical: LexicalSummary,
    pub occurrence_lookup: OccurrenceLookupSummary,
    pub embedding_profile: EmbeddingProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_provenance: Option<PipelineProvenance>,
    /// True only for deterministic diagnostic vectors that did not come from
    /// the embedding model named by the profile.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub test_only: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildReport {
    pub schema_version: String,
    pub source: SourceBinding,
    pub build_scope: BuildScope,
    pub complete: bool,
    pub document_count: u64,
    pub occurrence_count: u64,
    pub vector_count: u64,
    pub embedding_profile_sha256: String,
    pub accounting: serde_json::Value,
    pub cache_hits: u64,
    pub embedded: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub test_only: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub relations: BTreeSet<String>,
    pub time_start_ms: Option<u64>,
    pub time_end_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Dense,
    Lexical,
    Fused,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchHit {
    pub rank: usize,
    pub document_id: String,
    pub semantic_text: String,
    pub score: f64,
    pub dense_score: Option<f64>,
    pub lexical_score: Option<f64>,
    pub eligible_occurrence_count: u64,
    pub occurrences_exhausted: bool,
    pub occurrences: Vec<EvidenceOccurrence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceOccurrence {
    pub event_time_ms: Option<u64>,
    pub relation: String,
    pub snapshot_sha256: String,
    pub mapping_sha256: String,
    pub event_id: String,
    pub support_ref: String,
}

/// A validated dense query vector bound to one embedding profile.
///
/// Create this once with [`FastIndex::validate_query_vector`] before starting
/// an index-only timing measurement. The values are immutable and shared, so
/// the same query can safely be used by concurrent searches without copying or
/// validating the complete vector again.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileBoundQueryVector {
    values: Arc<[f32]>,
    embedding_profile: EmbeddingProfile,
}

impl ProfileBoundQueryVector {
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[must_use]
    pub fn embedding_profile_sha256(&self) -> &str {
        &self.embedding_profile.sha256
    }
}

pub fn write_fast_index(
    output: &Path,
    source: SourceBinding,
    build_scope: BuildScope,
    documents: &[FastDocument],
    occurrences: &[FastOccurrence],
    vectors: &[Vec<f32>],
    embedding_profile: EmbeddingProfile,
) -> Result<FastIndexManifest> {
    write_fast_index_streaming(
        output,
        source,
        build_scope,
        documents,
        occurrences.iter().cloned().map(Ok),
        vectors.iter().cloned().map(Ok),
        embedding_profile,
    )
}

/// Write the canonical index from one-pass occurrence and vector streams.
///
/// Documents remain ordered metadata because vector ordinals and the lexical
/// corpus are document-scoped. Occurrences and vectors, which dominate a
/// representative build, are validated and written incrementally without a
/// corpus-sized Arrow batch or `Vec<Vec<f32>>` copy.
pub fn write_fast_index_streaming<O, V>(
    output: &Path,
    source: SourceBinding,
    build_scope: BuildScope,
    documents: &[FastDocument],
    occurrences: O,
    vectors: V,
    embedding_profile: EmbeddingProfile,
) -> Result<FastIndexManifest>
where
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<Vec<f32>>>,
{
    let vectors = vectors.into_iter().enumerate().map(|(ordinal, vector)| {
        Ok(OrderedVector {
            vector_ordinal: u64::try_from(ordinal)
                .map_err(|_| IndexError::Invalid("vector ordinal"))?,
            values: vector?,
        })
    });
    write_fast_index_from_streams(
        output,
        source,
        build_scope,
        documents.iter().cloned().map(Ok),
        occurrences,
        vectors,
        embedding_profile,
    )
}

/// Assemble a v2 fast index from bounded, one-pass row streams.
///
/// Documents must be in strictly ascending `document_id` order with contiguous
/// vector ordinals. Occurrences may arrive from any number of inputs. Vectors
/// must be in ordinal order; only one vector row and one Parquet write batch are
/// retained at a time.
pub fn write_fast_index_from_streams<D, O, V>(
    output: &Path,
    source: SourceBinding,
    build_scope: BuildScope,
    documents: D,
    occurrences: O,
    vectors: V,
    embedding_profile: EmbeddingProfile,
) -> Result<FastIndexManifest>
where
    D: IntoIterator<Item = Result<FastDocument>>,
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<OrderedVector>>,
{
    write_fast_index_from_streams_with_format(
        output,
        documents,
        occurrences,
        vectors,
        IndexAssemblyOptions {
            source,
            build_scope,
            embedding_profile,
            lexical_format: LexicalFormat::JsonV2,
            pipeline_provenance: None,
            test_only: false,
        },
    )
}

/// Assemble a v2 dataset index and bind it to the exact validated pipeline
/// components that supplied its documents, occurrences, and vectors.
pub fn write_bound_fast_index_from_streams<D, O, V>(
    output: &Path,
    documents: D,
    occurrences: O,
    vectors: V,
    options: PipelineIndexOptions,
) -> Result<FastIndexManifest>
where
    D: IntoIterator<Item = Result<FastDocument>>,
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<OrderedVector>>,
{
    options.provenance.validate()?;
    write_fast_index_from_streams_with_format(
        output,
        documents,
        occurrences,
        vectors,
        IndexAssemblyOptions {
            source: options.source,
            build_scope: options.build_scope,
            embedding_profile: options.embedding_profile,
            lexical_format: LexicalFormat::JsonV2,
            pipeline_provenance: Some(options.provenance),
            test_only: options.test_only,
        },
    )
}

/// Assemble a scalable v3 index whose lexical postings remain on disk.
///
/// This preserves the v2 tokenizer, BM25 formula, score ordering, document
/// IDs, and occurrence-first filters. The v2 writer remains available for
/// existing readers; callers must opt into this versioned format explicitly.
pub fn write_scalable_fast_index_from_streams<D, O, V>(
    output: &Path,
    source: SourceBinding,
    build_scope: BuildScope,
    documents: D,
    occurrences: O,
    vectors: V,
    embedding_profile: EmbeddingProfile,
) -> Result<FastIndexManifest>
where
    D: IntoIterator<Item = Result<FastDocument>>,
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<OrderedVector>>,
{
    write_fast_index_from_streams_with_format(
        output,
        documents,
        occurrences,
        vectors,
        IndexAssemblyOptions {
            source,
            build_scope,
            embedding_profile,
            lexical_format: LexicalFormat::SqliteV3,
            pipeline_provenance: None,
            test_only: false,
        },
    )
}

/// Assemble a scalable v3 dataset index and bind it to the exact validated
/// pipeline components that supplied its documents, occurrences, and vectors.
pub fn write_bound_scalable_fast_index_from_streams<D, O, V>(
    output: &Path,
    documents: D,
    occurrences: O,
    vectors: V,
    options: PipelineIndexOptions,
) -> Result<FastIndexManifest>
where
    D: IntoIterator<Item = Result<FastDocument>>,
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<OrderedVector>>,
{
    options.provenance.validate()?;
    write_fast_index_from_streams_with_format(
        output,
        documents,
        occurrences,
        vectors,
        IndexAssemblyOptions {
            source: options.source,
            build_scope: options.build_scope,
            embedding_profile: options.embedding_profile,
            lexical_format: LexicalFormat::SqliteV3,
            pipeline_provenance: Some(options.provenance),
            test_only: options.test_only,
        },
    )
}

fn write_fast_index_from_streams_with_format<D, O, V>(
    output: &Path,
    documents: D,
    occurrences: O,
    vectors: V,
    options: IndexAssemblyOptions,
) -> Result<FastIndexManifest>
where
    D: IntoIterator<Item = Result<FastDocument>>,
    O: IntoIterator<Item = Result<FastOccurrence>>,
    V: IntoIterator<Item = Result<OrderedVector>>,
{
    let IndexAssemblyOptions {
        source,
        build_scope,
        embedding_profile,
        lexical_format,
        pipeline_provenance,
        test_only,
    } = options;
    if output.exists() {
        return Err(IndexError::Invalid("output/documents"));
    }
    validate_assembly_identity(&source, &embedding_profile)?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(IndexError::Invalid("output name"))?;
    let staging = parent.join(format!(".{name}.staging-{}", std::process::id()));
    if staging.exists() {
        return Err(IndexError::Invalid("staging exists"));
    }
    fs::create_dir(&staging)?;
    let result = (|| {
        let mut lookup = create_occurrence_lookup(&staging.join("occurrence-index.sqlite3"))?;
        let document_summary = write_documents_streaming(
            &staging.join("documents.parquet"),
            documents.into_iter(),
            &mut lookup,
            lexical_format == LexicalFormat::JsonV2,
        )?;
        let occurrence_rows = write_occurrences_streaming(
            &staging.join("occurrences.parquet"),
            occurrences.into_iter(),
            &source,
            &mut lookup,
        )?;
        finalize_occurrence_lookup(&mut lookup, occurrence_rows, &source)?;
        drop(lookup);
        let vector_count = write_vectors_streaming(
            &staging.join("vectors.f32"),
            vectors.into_iter(),
            embedding_profile.dimensions,
            &document_summary.order_sha256,
            &embedding_profile.normalization,
        )?;
        if vector_count != document_summary.rows {
            return Err(IndexError::Invalid("document/vector count"));
        }
        fs::create_dir(staging.join("lexical"))?;
        let (storage_schema_version, lexical_relative) = match lexical_format {
            LexicalFormat::JsonV2 => {
                write_lexical_from_documents(
                    &staging.join("lexical/index.json"),
                    &staging.join("documents.parquet"),
                    document_summary.rows,
                    document_summary
                        .total_token_count
                        .ok_or(IndexError::Invalid("legacy lexical token count"))?,
                    document_summary
                        .document_frequency
                        .as_ref()
                        .ok_or(IndexError::Invalid("legacy lexical document frequency"))?,
                )?;
                ("livefire.rag.fast-index/2", "lexical/index.json")
            }
            LexicalFormat::SqliteV3 => {
                write_sqlite_lexical_from_documents(
                    &staging.join("lexical/index.sqlite3"),
                    &staging.join("documents.parquet"),
                    document_summary.rows,
                    &document_summary.order_sha256,
                )?;
                ("livefire.rag.fast-index/3", "lexical/index.sqlite3")
            }
        };
        let schema_version = if test_only {
            "livefire.rag.fast-index/4"
        } else {
            storage_schema_version
        };
        let documents_artifact = artifact_summary(
            &staging.join("documents.parquet"),
            "documents.parquet",
            document_summary.rows,
            Some(document_summary.order_sha256.clone()),
        )?;
        let occurrences_artifact = artifact_summary(
            &staging.join("occurrences.parquet"),
            "occurrences.parquet",
            occurrence_rows,
            None,
        )?;
        let vectors_path = staging.join("vectors.f32");
        let lexical_path = staging.join(lexical_relative);
        let occurrence_lookup_path = staging.join("occurrence-index.sqlite3");
        let mut manifest = FastIndexManifest {
            schema_version: schema_version.into(),
            component_sha256: String::new(),
            source,
            build_scope: build_scope.clone(),
            complete: matches!(build_scope, BuildScope::Full),
            documents: documents_artifact,
            occurrences: occurrences_artifact,
            vectors: VectorSummary {
                path: "vectors.f32".into(),
                count: vector_count,
                bytes: fs::metadata(&vectors_path)?.len(),
                sha256: file_sha256(&vectors_path)?,
                dimensions: embedding_profile.dimensions,
                dtype: "f32le".into(),
                header_bytes: VECTOR_HEADER_BYTES,
                document_order_sha256: document_summary.order_sha256,
            },
            lexical: LexicalSummary {
                path: lexical_relative.into(),
                document_count: document_summary.rows,
                bytes: fs::metadata(&lexical_path)?.len(),
                sha256: file_sha256(&lexical_path)?,
                schema: (lexical_format == LexicalFormat::SqliteV3)
                    .then(|| SQLITE_LEXICAL_SCHEMA.into()),
                tokenizer: "ascii_camel_lower_v1".into(),
                k1: 1.2,
                b: 0.75,
            },
            occurrence_lookup: OccurrenceLookupSummary {
                path: "occurrence-index.sqlite3".into(),
                rows: occurrence_rows,
                bytes: fs::metadata(&occurrence_lookup_path)?.len(),
                sha256: file_sha256(&occurrence_lookup_path)?,
                schema: OCCURRENCE_LOOKUP_SCHEMA.into(),
            },
            embedding_profile,
            pipeline_provenance,
            test_only,
        };
        manifest.component_sha256 = manifest_component_sha256(&manifest)?;
        let mut bytes = serde_json::to_vec_pretty(&manifest)?;
        bytes.push(b'\n');
        fs::write(staging.join("index.json"), bytes)?;
        let report = BuildReport {
            schema_version: if test_only {
                "livefire.rag.fast-build-report/2".into()
            } else {
                "livefire.rag.fast-build-report/1".into()
            },
            source: manifest.source.clone(),
            build_scope: manifest.build_scope.clone(),
            complete: manifest.complete,
            document_count: document_summary.rows,
            occurrence_count: occurrence_rows,
            vector_count,
            embedding_profile_sha256: manifest.embedding_profile.sha256.clone(),
            accounting: serde_json::json!({
                "coverage_semantics": "assembled_input_stream_only_not_source_row_coverage",
                "semantic_source_coverage_complete": false,
                "input_documents": document_summary.rows,
                "input_occurrences": occurrence_rows,
                "input_vectors": vector_count,
            }),
            // Embedding is a separate stage for streaming callers. These
            // counters describe work performed by this assembly operation.
            cache_hits: 0,
            embedded: 0,
            test_only,
        };
        let mut report_bytes = serde_json::to_vec_pretty(&report)?;
        report_bytes.push(b'\n');
        fs::write(staging.join("build-report.json"), report_bytes)?;
        fs::rename(&staging, output)?;
        Ok(manifest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

pub struct FastIndex {
    root: PathBuf,
    pub manifest: FastIndexManifest,
    documents_path: PathBuf,
    occurrence_lookup_path: PathBuf,
    vectors_file: File,
    documents: OnceLock<Vec<FastDocument>>,
    lexical: LexicalBackend,
}

enum LexicalBackend {
    JsonV2 {
        path: PathBuf,
        index: OnceLock<LexicalIndex>,
    },
    SqliteV3(SqliteLexicalIndex),
}

struct SqliteLexicalIndex {
    path: PathBuf,
    document_count: u64,
    total_length: u64,
}

type EligibleDocuments = Option<HashSet<String>>;

impl FastIndex {
    /// Fast open validates metadata and vector/document pairing without replaying the parent.
    pub fn open(root: &Path) -> Result<Self> {
        Self::open_internal(root, false)
    }

    /// Open an index for an explicitly test-only catalogue or diagnostic.
    /// Normal query and provider paths deliberately use [`Self::open`].
    pub fn open_allow_test_only(root: &Path) -> Result<Self> {
        Self::open_internal(root, true)
    }

    fn open_internal(root: &Path, allow_test_only: bool) -> Result<Self> {
        let root = fs::canonicalize(root)?;
        let manifest: FastIndexManifest =
            serde_json::from_slice(&fs::read(root.join("index.json"))?)?;
        if manifest.test_only && !allow_test_only {
            return Err(IndexError::Invalid(
                "test-only index requires explicit diagnostic mode",
            ));
        }
        if !matches!(
            manifest.schema_version.as_str(),
            "livefire.rag.fast-index/2" | "livefire.rag.fast-index/3" | "livefire.rag.fast-index/4"
        ) {
            return Err(IndexError::Invalid("manifest version"));
        }
        if (manifest.schema_version == "livefire.rag.fast-index/4") != manifest.test_only {
            return Err(IndexError::Invalid("test-only manifest version binding"));
        }
        if manifest.complete != matches!(manifest.build_scope, BuildScope::Full)
            || manifest.vectors.count != manifest.documents.rows
            || manifest.vectors.dimensions != manifest.embedding_profile.dimensions
            || manifest.vectors.dtype != "f32le"
            || manifest.vectors.header_bytes != VECTOR_HEADER_BYTES
            || manifest.lexical.document_count != manifest.documents.rows
            || manifest.occurrence_lookup.rows != manifest.occurrences.rows
            || manifest.occurrence_lookup.schema != OCCURRENCE_LOOKUP_SCHEMA
            || manifest.lexical.tokenizer != "ascii_camel_lower_v1"
            || !manifest.lexical.k1.is_finite()
            || manifest.lexical.k1 <= 0.0
            || !manifest.lexical.b.is_finite()
            || !(0.0..=1.0).contains(&manifest.lexical.b)
            || validate_embedding_profile(&manifest.embedding_profile).is_err()
            || manifest
                .pipeline_provenance
                .as_ref()
                .is_some_and(|provenance| provenance.validate().is_err())
        {
            return Err(IndexError::Invalid("manifest bindings"));
        }
        match manifest.schema_version.as_str() {
            "livefire.rag.fast-index/2" if manifest.lexical.schema.is_none() => {}
            "livefire.rag.fast-index/3"
                if manifest.lexical.schema.as_deref() == Some(SQLITE_LEXICAL_SCHEMA) => {}
            "livefire.rag.fast-index/4"
                if manifest.lexical.schema.is_none()
                    || manifest.lexical.schema.as_deref() == Some(SQLITE_LEXICAL_SCHEMA) => {}
            _ => return Err(IndexError::Invalid("lexical format binding")),
        }
        if manifest.component_sha256 != manifest_component_sha256(&manifest)? {
            return Err(IndexError::Corrupt("index component identity"));
        }
        let documents_path = safe_artifact(&root, &manifest.documents.path)?;
        let occurrences_path = safe_artifact(&root, &manifest.occurrences.path)?;
        if parquet_row_count(&documents_path)? != manifest.documents.rows
            || parquet_row_count(&occurrences_path)? != manifest.occurrences.rows
        {
            return Err(IndexError::Invalid("manifest row binding"));
        }
        validate_artifact(
            &documents_path,
            manifest.documents.bytes,
            &manifest.documents.sha256,
        )?;
        validate_artifact(
            &occurrences_path,
            manifest.occurrences.bytes,
            &manifest.occurrences.sha256,
        )?;
        let vectors_path = safe_artifact(&root, &manifest.vectors.path)?;
        validate_vector_file(&vectors_path, &manifest.vectors)?;
        let lexical_path = safe_artifact(&root, &manifest.lexical.path)?;
        validate_artifact(
            &vectors_path,
            manifest.vectors.bytes,
            &manifest.vectors.sha256,
        )?;
        validate_artifact(
            &lexical_path,
            manifest.lexical.bytes,
            &manifest.lexical.sha256,
        )?;
        let occurrence_lookup_path = safe_artifact(&root, &manifest.occurrence_lookup.path)?;
        validate_artifact(
            &occurrence_lookup_path,
            manifest.occurrence_lookup.bytes,
            &manifest.occurrence_lookup.sha256,
        )?;
        validate_occurrence_lookup_metadata(&occurrence_lookup_path, &manifest)?;
        let lexical = if manifest.lexical.schema.is_none() {
            LexicalBackend::JsonV2 {
                path: lexical_path,
                index: OnceLock::new(),
            }
        } else {
            let (document_count, total_length) = validate_sqlite_lexical_metadata(
                &lexical_path,
                manifest.documents.rows,
                &manifest.vectors.document_order_sha256,
            )?;
            LexicalBackend::SqliteV3(SqliteLexicalIndex {
                path: lexical_path,
                document_count,
                total_length,
            })
        };
        let vectors_file = File::open(&vectors_path)?;
        Ok(Self {
            root,
            manifest,
            documents_path,
            occurrence_lookup_path,
            vectors_file,
            documents: OnceLock::new(),
            lexical,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Validate and bind a caller-supplied query vector to this index's
    /// embedding profile.
    ///
    /// Callers that need to measure index-only latency should do this before
    /// starting their timer, after obtaining the vector from the embedding
    /// service. The returned value owns an immutable copy and may be reused
    /// across threads.
    pub fn validate_query_vector(&self, query: &[f32]) -> Result<ProfileBoundQueryVector> {
        validate_query_vector_values(
            query,
            self.manifest.embedding_profile.dimensions as usize,
            &self.manifest.embedding_profile.normalization,
        )?;
        Ok(ProfileBoundQueryVector {
            values: Arc::from(query),
            embedding_profile: self.manifest.embedding_profile.clone(),
        })
    }

    /// Search the dense index with a previously validated query vector.
    pub fn search_dense_with_vector(
        &self,
        query: &ProfileBoundQueryVector,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_with_bound_vector(SearchMode::Dense, "", query, filters, top_n)
    }

    /// Search the fused dense and lexical index with a previously validated
    /// query vector.
    pub fn search_fused_with_vector(
        &self,
        query_text: &str,
        query: &ProfileBoundQueryVector,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_with_bound_vector(SearchMode::Fused, query_text, query, filters, top_n)
    }

    pub fn search(
        &self,
        mode: SearchMode,
        query_text: &str,
        query_vector: Option<&[f32]>,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_internal(mode, query_text, query_vector, false, filters, top_n)
    }

    fn search_with_bound_vector(
        &self,
        mode: SearchMode,
        query_text: &str,
        query: &ProfileBoundQueryVector,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        if !matches!(mode, SearchMode::Dense | SearchMode::Fused) {
            return Err(IndexError::Invalid("bound query search mode"));
        }
        self.validate_query_binding(query)?;
        self.search_internal(mode, query_text, Some(query.values()), true, filters, top_n)
    }

    fn search_internal(
        &self,
        mode: SearchMode,
        query_text: &str,
        query_vector: Option<&[f32]>,
        query_vector_is_validated: bool,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        if top_n == 0 || top_n > 100 {
            return Err(IndexError::Invalid("top_n"));
        }
        if filters.relations.len() > 256
            || filters.relations.iter().any(|relation| {
                relation.is_empty() || relation.len() > 128 || relation.contains('\0')
            })
            || matches!((filters.time_start_ms, filters.time_end_ms), (Some(start), Some(end)) if start >= end)
        {
            return Err(IndexError::Invalid("search filters"));
        }
        let eligible = self.eligible(filters)?;
        if matches!(mode, SearchMode::Dense | SearchMode::Fused) && !query_vector_is_validated {
            validate_query_vector_values(
                query_vector.ok_or(IndexError::Invalid("query vector"))?,
                self.manifest.embedding_profile.dimensions as usize,
                &self.manifest.embedding_profile.normalization,
            )?;
        }
        let dense = if matches!(mode, SearchMode::Dense | SearchMode::Fused) {
            Some(self.dense_scores(
                self.documents()?,
                query_vector.ok_or(IndexError::Invalid("query vector"))?,
                &eligible,
            )?)
        } else {
            None
        };
        let lexical = if matches!(mode, SearchMode::Lexical | SearchMode::Fused) {
            Some(self.lexical_scores(query_text, &eligible)?)
        } else {
            None
        };
        let scores = match mode {
            SearchMode::Dense => dense.clone().expect("dense mode"),
            SearchMode::Lexical => lexical.clone().expect("lexical mode"),
            SearchMode::Fused => reciprocal_rank_fusion(
                dense.as_ref().expect("dense fused"),
                lexical.as_ref().expect("lexical fused"),
                60.0,
            ),
        };
        let dense_map = dense
            .unwrap_or_default()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let lexical_map = lexical
            .unwrap_or_default()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let selected = scores.into_iter().take(top_n).collect::<Vec<_>>();
        let selected_ids = selected
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<BTreeSet<_>>();
        let semantic_by_id = self.semantic_texts(&selected_ids)?;
        let occurrences_by_document =
            read_matching_occurrences(&self.occurrence_lookup_path, &selected_ids, filters)?;
        Ok(selected
            .into_iter()
            .enumerate()
            .map(|(index, (id, score))| SearchHit {
                rank: index + 1,
                semantic_text: semantic_by_id
                    .get(&id)
                    .expect("ranked document exists")
                    .clone(),
                dense_score: dense_map.get(&id).copied(),
                lexical_score: lexical_map.get(&id).copied(),
                eligible_occurrence_count: occurrences_by_document
                    .get(&id)
                    .map_or(0, |matched| matched.eligible_count),
                occurrences_exhausted: occurrences_by_document
                    .get(&id)
                    .is_some_and(|matched| matched.rows.len() as u64 == matched.eligible_count),
                occurrences: occurrences_by_document
                    .get(&id)
                    .map(|matched| matched.rows.clone())
                    .unwrap_or_default(),
                document_id: id,
                score,
            })
            .collect())
    }

    fn validate_query_binding(&self, query: &ProfileBoundQueryVector) -> Result<()> {
        if query.embedding_profile != self.manifest.embedding_profile {
            return Err(IndexError::Invalid("query vector profile binding"));
        }
        Ok(())
    }

    fn documents(&self) -> Result<&Vec<FastDocument>> {
        if let Some(documents) = self.documents.get() {
            return Ok(documents);
        }
        let documents = read_documents(File::open(&self.documents_path)?)?;
        if documents.len() as u64 != self.manifest.documents.rows
            || document_order_sha256(&documents) != self.manifest.vectors.document_order_sha256
            || self.manifest.documents.order_sha256.as_deref()
                != Some(self.manifest.vectors.document_order_sha256.as_str())
            || documents
                .iter()
                .enumerate()
                .any(|(ordinal, document)| document.vector_ordinal != ordinal as u64)
        {
            return Err(IndexError::Corrupt("manifest row/order binding"));
        }
        let _ = self.documents.set(documents);
        self.documents
            .get()
            .ok_or(IndexError::Invalid("document initialization"))
    }

    fn lexical_scores(
        &self,
        query: &str,
        eligible: &EligibleDocuments,
    ) -> Result<Vec<(String, f64)>> {
        match &self.lexical {
            LexicalBackend::JsonV2 { path, index } => {
                if index.get().is_none() {
                    let lexical: LexicalIndex = serde_json::from_slice(&fs::read(path)?)?;
                    let documents = self.documents()?;
                    let document_ids = documents
                        .iter()
                        .map(|document| document.document_id.as_str())
                        .collect::<BTreeSet<_>>();
                    let lexical_ids = lexical
                        .documents
                        .iter()
                        .map(|document| document.document_id.as_str())
                        .collect::<BTreeSet<_>>();
                    if lexical.document_count != documents.len() as u64
                        || lexical.documents.len() != documents.len()
                        || lexical_ids.len() != lexical.documents.len()
                        || lexical_ids != document_ids
                    {
                        return Err(IndexError::Corrupt("lexical document association"));
                    }
                    let _ = index.set(lexical);
                }
                Ok(index
                    .get()
                    .ok_or(IndexError::Invalid("lexical index initialization"))?
                    .scores(query, eligible))
            }
            LexicalBackend::SqliteV3(index) => index.scores(query, eligible),
        }
    }

    fn semantic_texts(&self, selected_ids: &BTreeSet<String>) -> Result<HashMap<String, String>> {
        match &self.lexical {
            LexicalBackend::SqliteV3(index) => index.semantic_texts(selected_ids),
            LexicalBackend::JsonV2 { .. } => Ok(self
                .documents()?
                .iter()
                .filter(|document| selected_ids.contains(&document.document_id))
                .map(|document| (document.document_id.clone(), document.semantic_text.clone()))
                .collect()),
        }
    }

    fn eligible(&self, filters: &SearchFilters) -> Result<EligibleDocuments> {
        if filters.relations.is_empty()
            && filters.time_start_ms.is_none()
            && filters.time_end_ms.is_none()
        {
            return Ok(None);
        }
        Ok(Some(
            read_eligible_document_ids(&self.occurrence_lookup_path, filters)?
                .into_iter()
                .collect(),
        ))
    }

    fn dense_scores(
        &self,
        documents: &[FastDocument],
        query: &[f32],
        eligible: &EligibleDocuments,
    ) -> Result<Vec<(String, f64)>> {
        let dimensions = self.manifest.vectors.dimensions as usize;
        let row_bytes = dimensions
            .checked_mul(4)
            .ok_or(IndexError::Invalid("vector row bytes"))?;
        let mut row = vec![0_u8; row_bytes];
        let mut scores =
            Vec::with_capacity(eligible.as_ref().map_or(documents.len(), HashSet::len));
        for document in documents {
            if eligible
                .as_ref()
                .is_some_and(|ids| !ids.contains(&document.document_id))
            {
                continue;
            }
            let offset = u64::from(VECTOR_HEADER_BYTES)
                .checked_add(
                    document
                        .vector_ordinal
                        .checked_mul(row_bytes as u64)
                        .ok_or(IndexError::Invalid("vector offset"))?,
                )
                .ok_or(IndexError::Invalid("vector offset"))?;
            read_exact_at(&self.vectors_file, &mut row, offset)?;
            let mut score = 0.0_f64;
            let mut norm_squared = 0.0_f64;
            for (bytes, query_value) in row.chunks_exact(4).zip(query) {
                let value = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
                if !value.is_finite() {
                    return Err(IndexError::Corrupt("non-finite vector value"));
                }
                let value = f64::from(value);
                score += value * f64::from(*query_value);
                norm_squared += value * value;
            }
            if !score.is_finite()
                || (self.manifest.embedding_profile.normalization == "l2"
                    && (norm_squared.sqrt() - 1.0).abs() > 1.0e-4)
            {
                return Err(IndexError::Corrupt("vector normalization"));
            }
            scores.push((document.document_id.clone(), score));
        }
        sort_scores(&mut scores);
        Ok(scores)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)?;
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        let read = file.seek_read(buffer, offset)?;
        if read == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        offset += read as u64;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut independent = file.try_clone()?;
    independent.seek(SeekFrom::Start(offset))?;
    independent.read_exact(buffer)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LexicalIndex {
    document_count: u64,
    average_length: f64,
    document_frequency: BTreeMap<String, u64>,
    documents: Vec<LexicalDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LexicalDocument {
    document_id: String,
    length: u64,
    terms: BTreeMap<String, u64>,
}

impl LexicalIndex {
    fn scores(&self, query: &str, eligible: &EligibleDocuments) -> Vec<(String, f64)> {
        let terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        let mut scores = self
            .documents
            .iter()
            .filter(|doc| {
                eligible
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&doc.document_id))
            })
            .filter_map(|doc| {
                let score = terms
                    .iter()
                    .map(|term| {
                        let tf = *doc.terms.get(term).unwrap_or(&0) as f64;
                        if tf == 0.0 {
                            return 0.0;
                        }
                        let df = *self.document_frequency.get(term).unwrap_or(&0) as f64;
                        let idf = ((self.document_count as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
                        idf * (tf * 2.2)
                            / (tf
                                + 1.2
                                    * (1.0 - 0.75
                                        + 0.75 * doc.length as f64 / self.average_length.max(1.0)))
                    })
                    .sum::<f64>();
                (score > 0.0).then(|| (doc.document_id.clone(), score))
            })
            .collect::<Vec<_>>();
        sort_scores(&mut scores);
        scores
    }
}

impl SqliteLexicalIndex {
    fn scores(&self, query: &str, eligible: &EligibleDocuments) -> Result<Vec<(String, f64)>> {
        let terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let connection = open_sqlite_lexical(&self.path)?;
        let average_length = self.total_length as f64 / self.document_count.max(1) as f64;
        let mut scores = HashMap::<String, f64>::new();
        let mut frequency =
            connection.prepare("SELECT document_frequency FROM term_stats WHERE term = ?1")?;
        let mut postings = connection.prepare(
            "SELECT document.document_id, document.length, posting.term_frequency
               FROM postings AS posting
               JOIN documents AS document USING (document_ordinal)
              WHERE posting.term = ?1
              ORDER BY posting.document_ordinal",
        )?;
        for term in terms {
            let Some(document_frequency) = frequency
                .query_row([&term], |row| row.get::<_, i64>(0))
                .optional()?
            else {
                continue;
            };
            let document_frequency = u64::try_from(document_frequency)
                .map_err(|_| IndexError::Corrupt("lexical document frequency"))?;
            if document_frequency == 0 || document_frequency > self.document_count {
                return Err(IndexError::Corrupt("lexical document frequency"));
            }
            let idf = ((self.document_count as f64 - document_frequency as f64 + 0.5)
                / (document_frequency as f64 + 0.5)
                + 1.0)
                .ln();
            let rows = postings.query_map([&term], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            for row in rows {
                let (document_id, length, term_frequency) = row?;
                if eligible
                    .as_ref()
                    .is_some_and(|ids| !ids.contains(&document_id))
                {
                    continue;
                }
                let length = u64::try_from(length)
                    .map_err(|_| IndexError::Corrupt("lexical document length"))?;
                let term_frequency = u64::try_from(term_frequency)
                    .map_err(|_| IndexError::Corrupt("lexical term frequency"))?;
                if term_frequency == 0 {
                    return Err(IndexError::Corrupt("lexical term frequency"));
                }
                let tf = term_frequency as f64;
                let score = idf * (tf * 2.2)
                    / (tf + 1.2 * (1.0 - 0.75 + 0.75 * length as f64 / average_length.max(1.0)));
                *scores.entry(document_id).or_default() += score;
            }
        }
        let mut scores = scores.into_iter().collect::<Vec<_>>();
        sort_scores(&mut scores);
        Ok(scores)
    }

    fn semantic_texts(&self, selected_ids: &BTreeSet<String>) -> Result<HashMap<String, String>> {
        let connection = open_sqlite_lexical(&self.path)?;
        let mut statement =
            connection.prepare("SELECT semantic_text FROM documents WHERE document_id = ?1")?;
        let mut texts = HashMap::with_capacity(selected_ids.len());
        for document_id in selected_ids {
            let semantic_text = statement
                .query_row([document_id], |row| row.get::<_, String>(0))
                .optional()?
                .ok_or(IndexError::Corrupt("lexical document association"))?;
            texts.insert(document_id.clone(), semantic_text);
        }
        Ok(texts)
    }
}

fn write_lexical_from_documents(
    path: &Path,
    documents_path: &Path,
    document_count: u64,
    total_length: u64,
    document_frequency: &BTreeMap<String, u64>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"{\"document_count\":")?;
    serde_json::to_writer(&mut writer, &document_count)?;
    writer.write_all(b",\"average_length\":")?;
    serde_json::to_writer(
        &mut writer,
        &(total_length as f64 / document_count.max(1) as f64),
    )?;
    writer.write_all(b",\"document_frequency\":")?;
    serde_json::to_writer(&mut writer, document_frequency)?;
    writer.write_all(b",\"documents\":[")?;
    let mut first = true;
    for document in documents_from_parquet_shards([documents_path]) {
        let document = document?;
        let mut terms = BTreeMap::<String, u64>::new();
        for token in tokenize(&document.semantic_text) {
            *terms.entry(token).or_default() += 1;
        }
        if !first {
            writer.write_all(b",")?;
        }
        first = false;
        serde_json::to_writer(
            &mut writer,
            &LexicalDocument {
                document_id: document.document_id,
                length: terms.values().sum(),
                terms,
            },
        )?;
    }
    writer.write_all(b"]}")?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_sqlite_lexical_from_documents(
    path: &Path,
    documents_path: &Path,
    expected_document_count: u64,
    document_order_sha256: &str,
) -> Result<()> {
    let mut connection = Connection::open(path)?;
    connection.execute_batch(&format!(
        "PRAGMA page_size=4096;
         PRAGMA auto_vacuum=NONE;
         PRAGMA encoding='UTF-8';
         PRAGMA journal_mode=OFF;
         PRAGMA synchronous=OFF;
         PRAGMA temp_store=FILE;
         PRAGMA cache_size=-32768;
         PRAGMA mmap_size=0;
         PRAGMA foreign_keys=ON;
         PRAGMA application_id={SQLITE_LEXICAL_APPLICATION_ID};
         PRAGMA user_version=1;
         CREATE TABLE documents (
           document_ordinal INTEGER PRIMARY KEY NOT NULL,
           document_id TEXT NOT NULL UNIQUE,
           semantic_text TEXT NOT NULL,
           length INTEGER NOT NULL
         ) WITHOUT ROWID;
         CREATE TABLE postings (
           term TEXT NOT NULL,
           document_ordinal INTEGER NOT NULL REFERENCES documents(document_ordinal),
           term_frequency INTEGER NOT NULL,
           PRIMARY KEY(term, document_ordinal)
         ) WITHOUT ROWID;
         CREATE TABLE term_stats (
           term TEXT PRIMARY KEY NOT NULL,
           document_frequency INTEGER NOT NULL
         ) WITHOUT ROWID;
         CREATE TABLE metadata (
           key TEXT PRIMARY KEY NOT NULL,
           value TEXT NOT NULL
         ) WITHOUT ROWID;"
    ))?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Exclusive)?;
    let mut document_count = 0_u64;
    let mut total_length = 0_u64;
    let mut order_hasher = Sha256::new();
    {
        let mut insert_document = transaction.prepare(
            "INSERT INTO documents(document_ordinal, document_id, semantic_text, length)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut insert_posting = transaction.prepare(
            "INSERT INTO postings(term, document_ordinal, term_frequency) VALUES (?1, ?2, ?3)",
        )?;
        let mut update_frequency = transaction.prepare(
            "INSERT INTO term_stats(term, document_frequency) VALUES (?1, 1)
             ON CONFLICT(term) DO UPDATE
             SET document_frequency = document_frequency + 1",
        )?;
        for document in documents_from_parquet_shards([documents_path]) {
            let document = document?;
            if document.vector_ordinal != document_count {
                return Err(IndexError::Invalid("lexical document ordinal"));
            }
            order_hasher.update(document.document_id.as_bytes());
            order_hasher.update([0]);
            let mut terms = BTreeMap::<String, u64>::new();
            for term in tokenize(&document.semantic_text) {
                *terms.entry(term).or_default() += 1;
            }
            let length = terms.values().try_fold(0_u64, |total, count| {
                total
                    .checked_add(*count)
                    .ok_or(IndexError::Invalid("lexical token count"))
            })?;
            insert_document.execute(params![
                i64::try_from(document.vector_ordinal)
                    .map_err(|_| IndexError::Invalid("lexical document ordinal"))?,
                document.document_id,
                document.semantic_text,
                i64::try_from(length).map_err(|_| IndexError::Invalid("lexical token count"))?
            ])?;
            for (term, term_frequency) in terms {
                insert_posting.execute(params![
                    term,
                    i64::try_from(document.vector_ordinal)
                        .map_err(|_| IndexError::Invalid("lexical document ordinal"))?,
                    i64::try_from(term_frequency)
                        .map_err(|_| IndexError::Invalid("lexical term frequency"))?
                ])?;
                update_frequency.execute([term])?;
            }
            document_count = document_count
                .checked_add(1)
                .ok_or(IndexError::Invalid("lexical document count"))?;
            total_length = total_length
                .checked_add(length)
                .ok_or(IndexError::Invalid("lexical token count"))?;
        }
    }
    if document_count != expected_document_count {
        return Err(IndexError::Invalid("lexical document count"));
    }
    let observed_document_order_sha256 = format!("{:x}", order_hasher.finalize());
    if observed_document_order_sha256 != document_order_sha256 {
        return Err(IndexError::Invalid("lexical document order"));
    }
    for (key, value) in [
        ("schema", SQLITE_LEXICAL_SCHEMA.to_owned()),
        ("document_count", document_count.to_string()),
        ("total_length", total_length.to_string()),
        ("document_order_sha256", observed_document_order_sha256),
        ("tokenizer", "ascii_camel_lower_v1".to_owned()),
        ("k1", "1.2".to_owned()),
        ("b", "0.75".to_owned()),
    ] {
        transaction.execute(
            "INSERT INTO metadata(key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
    }
    transaction.commit()?;
    connection.execute_batch("VACUUM;")?;
    drop(connection);
    Ok(())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len() + 16);
    let mut previous_lower = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            if character.is_ascii_uppercase() && previous_lower {
                normalized.push(' ');
            }
            normalized.push(character.to_ascii_lowercase());
            previous_lower = character.is_ascii_lowercase();
        } else {
            normalized.push(' ');
            previous_lower = false;
        }
    }
    normalized.split_whitespace().map(str::to_owned).collect()
}

fn reciprocal_rank_fusion(
    dense: &[(String, f64)],
    lexical: &[(String, f64)],
    k: f64,
) -> Vec<(String, f64)> {
    let mut scores = BTreeMap::<String, f64>::new();
    for ranking in [dense, lexical] {
        for (rank, (id, _)) in ranking.iter().enumerate() {
            *scores.entry(id.clone()).or_default() += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    let mut result = scores.into_iter().collect::<Vec<_>>();
    sort_scores(&mut result);
    result
}

fn sort_scores(scores: &mut [(String, f64)]) {
    scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
}

fn validate_assembly_identity(source: &SourceBinding, profile: &EmbeddingProfile) -> Result<()> {
    if decode_sha256(&source.snapshot_sha256).is_err()
        || decode_sha256(&source.mapping_sha256).is_err()
        || validate_embedding_profile(profile).is_err()
    {
        return Err(IndexError::Invalid("assembly identity"));
    }
    Ok(())
}

#[must_use]
pub fn document_order_sha256(documents: &[FastDocument]) -> String {
    let mut hasher = Sha256::new();
    for document in documents {
        hasher.update(document.document_id.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn write_vectors_streaming<I>(
    path: &Path,
    vectors: I,
    dimensions: u32,
    order_sha: &str,
    normalization: &str,
) -> Result<u64>
where
    I: Iterator<Item = Result<OrderedVector>>,
{
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&VECTOR_MAGIC)?;
    writer.write_all(&VECTOR_HEADER_BYTES.to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&[1_u8, 0_u8])?;
    // The final count is patched after the one-pass vector stream is sealed.
    writer.write_all(&0_u64.to_le_bytes())?;
    writer.write_all(&dimensions.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&decode_sha256(order_sha)?)?;
    let mut count = 0_u64;
    for vector in vectors {
        let vector = vector?;
        if vector.vector_ordinal != count {
            return Err(IndexError::Invalid("vector ordinal"));
        }
        validate_vector_values(&vector.values, dimensions as usize, normalization)?;
        for value in &vector.values {
            writer.write_all(&value.to_le_bytes())?;
        }
        count = count
            .checked_add(1)
            .ok_or(IndexError::Invalid("vector count"))?;
    }
    writer.flush()?;
    drop(writer);
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(16))?;
    file.write_all(&count.to_le_bytes())?;
    file.sync_all()?;
    Ok(count)
}

fn validate_vector_values(vector: &[f32], dimensions: usize, normalization: &str) -> Result<()> {
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(IndexError::Invalid("vector shape or finiteness"));
    }
    if normalization == "l2" {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-4 {
            return Err(IndexError::Invalid("vector normalization"));
        }
    }
    Ok(())
}

fn validate_query_vector_values(
    vector: &[f32],
    dimensions: usize,
    normalization: &str,
) -> Result<()> {
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(IndexError::Invalid("query vector"));
    }
    if normalization == "l2" {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-4 {
            return Err(IndexError::Invalid("query vector normalization"));
        }
    }
    Ok(())
}

fn validate_vector_file(path: &Path, expected: &VectorSummary) -> Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0_u8; 64];
    reader.read_exact(&mut header)?;
    if header[..8] != VECTOR_MAGIC
        || u32::from_le_bytes(header[8..12].try_into().expect("header slice")) != 64
        || u16::from_le_bytes(header[12..14].try_into().expect("header slice")) != 1
        || header[14] != 1
        || header[15] != 0
        || header[28..32] != [0_u8; 4]
        || u64::from_le_bytes(header[16..24].try_into().expect("header slice")) != expected.count
        || u32::from_le_bytes(header[24..28].try_into().expect("header slice"))
            != expected.dimensions
        || header[32..64] != decode_sha256(&expected.document_order_sha256)?
    {
        return Err(IndexError::Invalid("vector header"));
    }
    let expected_bytes = u64::from(VECTOR_HEADER_BYTES)
        .checked_add(
            expected
                .count
                .checked_mul(u64::from(expected.dimensions))
                .and_then(|values| values.checked_mul(4))
                .ok_or(IndexError::Invalid("vector bytes"))?,
        )
        .ok_or(IndexError::Invalid("vector bytes"))?;
    if reader.get_ref().metadata()?.len() != expected_bytes {
        return Err(IndexError::Invalid("vector file length"));
    }
    Ok(())
}

fn decode_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(IndexError::Invalid("sha256"));
    }
    let mut output = [0_u8; 32];
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| IndexError::Invalid("sha256"))?;
    }
    Ok(output)
}

fn parquet_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("valid zstd"),
        ))
        .build()
}

#[cfg(test)]
fn write_documents(path: &Path, rows: &[FastDocument]) -> Result<()> {
    let mut writer = ArrowWriter::try_new(
        File::create(path)?,
        document_schema(),
        Some(parquet_properties()),
    )?;
    for rows in rows.chunks(PARQUET_WRITE_BATCH_ROWS) {
        write_document_batch(&mut writer, rows)?;
    }
    writer.close()?;
    Ok(())
}

fn document_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("document_id", DataType::Utf8, false),
        Field::new("document_sha256", DataType::Utf8, false),
        Field::new("document_kind", DataType::Utf8, false),
        Field::new("semantic_text", DataType::Utf8, false),
        Field::new("facets_json", DataType::Utf8, false),
        Field::new("relations_json", DataType::Utf8, false),
        Field::new("occurrence_count", DataType::UInt64, false),
        Field::new("vector_ordinal", DataType::UInt64, false),
    ]))
}

fn write_document_batch(writer: &mut ArrowWriter<File>, rows: &[FastDocument]) -> Result<()> {
    if !rows.is_empty() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.document_id.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.document_sha256.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.document_kind.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.semantic_text.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.facets_json.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.relations_json.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.occurrence_count),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.vector_ordinal),
            )),
        ];
        writer.write(&RecordBatch::try_new(document_schema(), columns)?)?;
    }
    Ok(())
}

struct DocumentWriteSummary {
    rows: u64,
    order_sha256: String,
    total_token_count: Option<u64>,
    document_frequency: Option<BTreeMap<String, u64>>,
}

fn write_documents_streaming<I>(
    path: &Path,
    documents: I,
    lookup: &mut Connection,
    collect_legacy_lexical_stats: bool,
) -> Result<DocumentWriteSummary>
where
    I: Iterator<Item = Result<FastDocument>>,
{
    let mut writer = ArrowWriter::try_new(
        File::create(path)?,
        document_schema(),
        Some(parquet_properties()),
    )?;
    let mut batch = Vec::with_capacity(PARQUET_WRITE_BATCH_ROWS);
    let mut order_hasher = Sha256::new();
    let mut previous_id: Option<String> = None;
    let mut rows = 0_u64;
    let mut total_token_count = collect_legacy_lexical_stats.then_some(0_u64);
    let mut document_frequency = collect_legacy_lexical_stats.then(BTreeMap::new);
    let transaction = lookup.transaction()?;
    {
        let mut insert = transaction.prepare(
            "INSERT INTO assembly_documents(document_id, expected_occurrences) VALUES (?1, ?2)",
        )?;
        for document in documents {
            let document = document?;
            if document.document_id.is_empty()
                || document.document_id.contains('\0')
                || document.occurrence_count == 0
                || document.vector_ordinal != rows
                || previous_id
                    .as_ref()
                    .is_some_and(|previous| previous >= &document.document_id)
            {
                return Err(IndexError::Invalid("document association"));
            }
            insert.execute(params![
                document.document_id,
                i64::try_from(document.occurrence_count)
                    .map_err(|_| IndexError::Invalid("document occurrence count"))?
            ])?;
            order_hasher.update(document.document_id.as_bytes());
            order_hasher.update([0]);
            if let (Some(total_token_count), Some(document_frequency)) =
                (&mut total_token_count, &mut document_frequency)
            {
                let tokens = tokenize(&document.semantic_text);
                *total_token_count = total_token_count
                    .checked_add(tokens.len() as u64)
                    .ok_or(IndexError::Invalid("lexical token count"))?;
                for term in tokens.into_iter().collect::<BTreeSet<_>>() {
                    *document_frequency.entry(term).or_default() += 1;
                }
            }
            previous_id = Some(document.document_id.clone());
            batch.push(document);
            rows = rows
                .checked_add(1)
                .ok_or(IndexError::Invalid("document count"))?;
            if batch.len() == PARQUET_WRITE_BATCH_ROWS {
                write_document_batch(&mut writer, &batch)?;
                batch.clear();
            }
        }
    }
    transaction.commit()?;
    if rows == 0 {
        return Err(IndexError::Invalid("output/documents"));
    }
    write_document_batch(&mut writer, &batch)?;
    writer.close()?;
    Ok(DocumentWriteSummary {
        rows,
        order_sha256: format!("{:x}", order_hasher.finalize()),
        total_token_count,
        document_frequency,
    })
}

fn occurrence_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("occurrence_id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("event_time_ms", DataType::UInt64, true),
        Field::new("relation", DataType::Utf8, false),
        Field::new("exact_attributes_json", DataType::Utf8, false),
        Field::new("snapshot_sha256", DataType::Utf8, false),
        Field::new("mapping_sha256", DataType::Utf8, false),
        Field::new("event_id", DataType::Utf8, false),
        Field::new("support_ref", DataType::Utf8, false),
    ]))
}

fn write_occurrence_batch(
    writer: &mut ArrowWriter<File>,
    schema: Arc<Schema>,
    rows: &[FastOccurrence],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.occurrence_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.document_id.as_str()),
        )),
        Arc::new(UInt64Array::from(
            rows.iter().map(|r| r.event_time_ms).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.relation.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.exact_attributes_json.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.snapshot_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.mapping_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.event_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.support_ref.as_str()),
        )),
    ];
    writer.write(&RecordBatch::try_new(schema, columns)?)?;
    Ok(())
}

fn write_occurrences_streaming<I>(
    path: &Path,
    rows: I,
    source: &SourceBinding,
    lookup: &mut Connection,
) -> Result<u64>
where
    I: Iterator<Item = Result<FastOccurrence>>,
{
    let schema = occurrence_schema();
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(parquet_properties()))?;
    let mut batch = Vec::with_capacity(PARQUET_WRITE_BATCH_ROWS);
    let mut total = 0_u64;
    let transaction = lookup.transaction_with_behavior(TransactionBehavior::Exclusive)?;
    {
        let mut insert = transaction
            .prepare("INSERT INTO occurrences VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)")?;
        for row in rows {
            let row = row?;
            if row.occurrence_id.is_empty()
                || row.event_id.is_empty()
                || row.support_ref.is_empty()
                || row.snapshot_sha256 != source.snapshot_sha256
                || row.mapping_sha256 != source.mapping_sha256
                || i64::try_from(row.event_time_ms.unwrap_or(0)).is_err()
            {
                return Err(IndexError::Invalid("occurrence source closure"));
            }
            insert.execute(params![
                row.occurrence_id,
                row.document_id,
                row.event_time_ms
                    .and_then(|value| i64::try_from(value).ok()),
                row.relation,
                row.snapshot_sha256,
                row.mapping_sha256,
                row.event_id,
                row.support_ref,
            ])?;
            batch.push(row);
            total = total
                .checked_add(1)
                .ok_or(IndexError::Invalid("occurrence count"))?;
            if batch.len() == PARQUET_WRITE_BATCH_ROWS {
                write_occurrence_batch(&mut writer, occurrence_schema(), &batch)?;
                batch.clear();
            }
        }
    }
    write_occurrence_batch(&mut writer, occurrence_schema(), &batch)?;
    transaction.commit()?;
    writer.close()?;
    Ok(total)
}

fn create_occurrence_lookup(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)?;
    connection.execute_batch(
        "PRAGMA journal_mode=OFF;
         PRAGMA synchronous=OFF;
         PRAGMA temp_store=MEMORY;
         CREATE TABLE assembly_documents (
           document_id TEXT PRIMARY KEY NOT NULL,
           expected_occurrences INTEGER NOT NULL
         ) WITHOUT ROWID;
         CREATE TABLE occurrences (
           occurrence_id TEXT NOT NULL,
           document_id TEXT NOT NULL,
           event_time_ms INTEGER,
           relation TEXT NOT NULL,
           snapshot_sha256 TEXT NOT NULL,
           mapping_sha256 TEXT NOT NULL,
           event_id TEXT NOT NULL,
           support_ref TEXT NOT NULL
         );",
    )?;
    Ok(connection)
}

fn finalize_occurrence_lookup(
    lookup: &mut Connection,
    total: u64,
    source: &SourceBinding,
) -> Result<()> {
    lookup.execute_batch(
        "CREATE UNIQUE INDEX occurrence_id_unique ON occurrences(occurrence_id);
         CREATE UNIQUE INDEX occurrence_event_id_unique ON occurrences(event_id);
         CREATE INDEX occurrence_document_time_relation
           ON occurrences(document_id, event_time_ms, relation);
         CREATE INDEX occurrence_relation_time_document
           ON occurrences(relation, event_time_ms, document_id);",
    )?;
    let count_mismatch = lookup.query_row(
        "SELECT EXISTS(
           SELECT 1
             FROM assembly_documents AS expected
             LEFT JOIN (
               SELECT document_id, count(*) AS actual_occurrences
                 FROM occurrences GROUP BY document_id
             ) AS actual USING (document_id)
            WHERE expected.expected_occurrences != coalesce(actual.actual_occurrences, 0)
           UNION ALL
           SELECT 1
             FROM occurrences AS occurrence
             LEFT JOIN assembly_documents AS expected USING (document_id)
            WHERE expected.document_id IS NULL
         )",
        [],
        |row| row.get::<_, bool>(0),
    )?;
    if count_mismatch {
        return Err(IndexError::Invalid("occurrence count closure"));
    }
    lookup.execute_batch(
        "DROP TABLE assembly_documents;
         CREATE TABLE metadata(key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL) WITHOUT ROWID;",
    )?;
    for (key, value) in [
        ("schema", OCCURRENCE_LOOKUP_SCHEMA.to_owned()),
        ("rows", total.to_string()),
        ("snapshot_sha256", source.snapshot_sha256.clone()),
        ("mapping_sha256", source.mapping_sha256.clone()),
    ] {
        lookup.execute("INSERT INTO metadata VALUES (?1, ?2)", params![key, value])?;
    }
    lookup.execute_batch("ANALYZE; PRAGMA optimize;")?;
    Ok(())
}

struct ParquetShardRows<T> {
    paths: VecDeque<PathBuf>,
    reader: Option<ParquetRecordBatchReader>,
    batch: Option<RecordBatch>,
    batch_row: usize,
    decode: fn(&RecordBatch, usize) -> Result<T>,
}

impl<T> ParquetShardRows<T> {
    fn new<I, P>(paths: I, decode: fn(&RecordBatch, usize) -> Result<T>) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self {
            paths: paths
                .into_iter()
                .map(|path| path.as_ref().to_path_buf())
                .collect(),
            reader: None,
            batch: None,
            batch_row: 0,
            decode,
        }
    }
}

impl<T> Iterator for ParquetShardRows<T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = &self.batch
                && self.batch_row < batch.num_rows()
            {
                let row = self.batch_row;
                self.batch_row += 1;
                return Some((self.decode)(batch, row));
            }
            self.batch = None;
            if let Some(reader) = &mut self.reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        self.batch = Some(batch);
                        self.batch_row = 0;
                        continue;
                    }
                    Some(Err(error)) => return Some(Err(error.into())),
                    None => self.reader = None,
                }
            }
            let path = self.paths.pop_front()?;
            let file = match File::open(path) {
                Ok(file) => file,
                Err(error) => return Some(Err(error.into())),
            };
            match ParquetRecordBatchReaderBuilder::try_new(file)
                .and_then(|builder| builder.with_batch_size(PARQUET_WRITE_BATCH_ROWS).build())
            {
                Ok(reader) => self.reader = Some(reader),
                Err(error) => return Some(Err(error.into())),
            }
        }
    }
}

/// Lazy row iterator over ordered prepared-document Parquet shards.
pub struct DocumentParquetShards(ParquetShardRows<FastDocument>);

impl Iterator for DocumentParquetShards {
    type Item = Result<FastDocument>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub fn documents_from_parquet_shards<I, P>(paths: I) -> DocumentParquetShards
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    DocumentParquetShards(ParquetShardRows::new(paths, decode_fast_document))
}

/// Lazy row iterator over any number of occurrence Parquet shards.
pub struct OccurrenceParquetShards(ParquetShardRows<FastOccurrence>);

impl Iterator for OccurrenceParquetShards {
    type Item = Result<FastOccurrence>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub fn occurrences_from_parquet_shards<I, P>(paths: I) -> OccurrenceParquetShards
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    OccurrenceParquetShards(ParquetShardRows::new(paths, decode_fast_occurrence))
}

struct OpenVectorShard {
    reader: BufReader<File>,
    remaining: u64,
    dimensions: u32,
}

/// Lazy vectors from consecutive `LFREMB01` result parts.
pub struct EmbeddingVectorShards {
    shards: VecDeque<OrderedVectorShard>,
    current: Option<OpenVectorShard>,
    next_ordinal: u64,
}

pub fn vectors_from_embedding_shards<I>(shards: I) -> Result<EmbeddingVectorShards>
where
    I: IntoIterator<Item = OrderedVectorShard>,
{
    let shards = shards.into_iter().collect::<VecDeque<_>>();
    let mut expected_ordinal = 0_u64;
    let mut dimensions = None;
    for shard in &shards {
        if shard.vector_count == 0
            || shard.first_vector_ordinal != expected_ordinal
            || decode_sha256(&shard.order_sha256).is_err()
            || dimensions.is_some_and(|value| value != shard.dimensions)
        {
            return Err(IndexError::Invalid("embedding vector shard order"));
        }
        dimensions = Some(shard.dimensions);
        expected_ordinal = expected_ordinal
            .checked_add(shard.vector_count)
            .ok_or(IndexError::Invalid("embedding vector shard order"))?;
    }
    Ok(EmbeddingVectorShards {
        shards,
        current: None,
        next_ordinal: 0,
    })
}

impl Iterator for EmbeddingVectorShards {
    type Item = Result<OrderedVector>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = &mut self.current
                && current.remaining > 0
            {
                let ordinal = self.next_ordinal;
                let result = read_embedding_vector_row(current, ordinal);
                if result.is_ok() {
                    self.next_ordinal += 1;
                }
                return Some(result);
            }
            self.current = None;
            let shard = self.shards.pop_front()?;
            match open_embedding_vector_shard(&shard) {
                Ok(current) => self.current = Some(current),
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

fn open_embedding_vector_shard(shard: &OrderedVectorShard) -> Result<OpenVectorShard> {
    let mut reader = BufReader::new(File::open(&shard.path)?);
    let mut header = [0_u8; 64];
    reader.read_exact(&mut header)?;
    if header[..8] != EMBEDDING_VECTOR_MAGIC
        || u32::from_le_bytes(header[8..12].try_into().expect("header slice")) != 64
        || u16::from_le_bytes(header[12..14].try_into().expect("header slice")) != 1
        || header[14] != 1
        || header[15] != 0
        || u64::from_le_bytes(header[16..24].try_into().expect("header slice"))
            != shard.vector_count
        || u32::from_le_bytes(header[24..28].try_into().expect("header slice")) != shard.dimensions
        || header[28..32] != [0_u8; 4]
        || header[32..64] != decode_sha256(&shard.order_sha256)?
    {
        return Err(IndexError::Invalid("embedding vector shard header"));
    }
    let expected_bytes = 64_u64
        .checked_add(
            shard
                .vector_count
                .checked_mul(u64::from(shard.dimensions))
                .and_then(|values| values.checked_mul(4))
                .ok_or(IndexError::Invalid("embedding vector shard bytes"))?,
        )
        .ok_or(IndexError::Invalid("embedding vector shard bytes"))?;
    if reader.get_ref().metadata()?.len() != expected_bytes {
        return Err(IndexError::Invalid("embedding vector shard bytes"));
    }
    Ok(OpenVectorShard {
        reader,
        remaining: shard.vector_count,
        dimensions: shard.dimensions,
    })
}

fn read_embedding_vector_row(
    shard: &mut OpenVectorShard,
    vector_ordinal: u64,
) -> Result<OrderedVector> {
    let mut values = Vec::with_capacity(shard.dimensions as usize);
    for _ in 0..shard.dimensions {
        let mut bytes = [0_u8; 4];
        shard.reader.read_exact(&mut bytes)?;
        values.push(f32::from_le_bytes(bytes));
    }
    shard.remaining -= 1;
    Ok(OrderedVector {
        vector_ordinal,
        values,
    })
}

fn read_documents(file: File) -> Result<Vec<FastDocument>> {
    read_parquet(file, decode_fast_document)
}

fn decode_fast_document(batch: &RecordBatch, row: usize) -> Result<FastDocument> {
    let vector_ordinal = if batch.schema().index_of("vector_ordinal").is_ok() {
        number(batch, "vector_ordinal", row)?
    } else {
        number(batch, "document_ordinal", row)?
    };
    Ok(FastDocument {
        document_id: text(batch, "document_id", row)?,
        document_sha256: text(batch, "document_sha256", row)?,
        document_kind: text(batch, "document_kind", row)?,
        semantic_text: text(batch, "semantic_text", row)?,
        facets_json: text(batch, "facets_json", row)?,
        relations_json: text(batch, "relations_json", row)?,
        occurrence_count: number(batch, "occurrence_count", row)?,
        vector_ordinal,
    })
}

fn decode_fast_occurrence(batch: &RecordBatch, row: usize) -> Result<FastOccurrence> {
    Ok(FastOccurrence {
        occurrence_id: text(batch, "occurrence_id", row)?,
        document_id: text(batch, "document_id", row)?,
        event_time_ms: optional_number(batch, "event_time_ms", row)?,
        relation: text(batch, "relation", row)?,
        exact_attributes_json: text(batch, "exact_attributes_json", row)?,
        snapshot_sha256: text(batch, "snapshot_sha256", row)?,
        mapping_sha256: text(batch, "mapping_sha256", row)?,
        event_id: text(batch, "event_id", row)?,
        support_ref: text(batch, "support_ref", row)?,
    })
}
fn read_matching_occurrences(
    path: &Path,
    document_ids: &BTreeSet<String>,
    filters: &SearchFilters,
) -> Result<BTreeMap<String, MatchedOccurrences>> {
    let mut matched = BTreeMap::<String, MatchedOccurrences>::new();
    if document_ids.is_empty() {
        return Ok(matched);
    }
    let connection = open_occurrence_lookup(path)?;
    let placeholders = std::iter::repeat_n("?", document_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let mut sql = format!(
        "SELECT document_id, event_time_ms, relation, snapshot_sha256,
                mapping_sha256, event_id, support_ref
           FROM occurrences WHERE document_id IN ({placeholders})"
    );
    let mut parameters = document_ids
        .iter()
        .cloned()
        .map(ValueParam::Text)
        .collect::<Vec<_>>();
    append_filter_sql(&mut sql, &mut parameters, filters);
    sql.push_str(" ORDER BY document_id, occurrence_id");
    let mut statement = connection.prepare(&sql)?;
    let mut rows = statement.query(params_from_iter(parameters.iter().map(ValueParam::as_sql)))?;
    while let Some(row) = rows.next()? {
        let document_id: String = row.get(0)?;
        let event_time_ms = row
            .get::<_, Option<i64>>(1)?
            .map(u64::try_from)
            .transpose()
            .map_err(|_| IndexError::Corrupt("negative occurrence timestamp"))?;
        let entry = matched.entry(document_id).or_default();
        entry.eligible_count += 1;
        if entry.rows.len() < MAX_RETURNED_OCCURRENCES_PER_HIT {
            entry.rows.push(EvidenceOccurrence {
                event_time_ms,
                relation: row.get(2)?,
                snapshot_sha256: row.get(3)?,
                mapping_sha256: row.get(4)?,
                event_id: row.get(5)?,
                support_ref: row.get(6)?,
            });
        }
    }
    Ok(matched)
}

fn read_eligible_document_ids(path: &Path, filters: &SearchFilters) -> Result<Vec<String>> {
    let connection = open_occurrence_lookup(path)?;
    let mut sql = "SELECT DISTINCT document_id FROM occurrences WHERE 1=1".to_owned();
    let mut parameters = Vec::new();
    append_filter_sql(&mut sql, &mut parameters, filters);
    sql.push_str(" ORDER BY document_id");
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(
        params_from_iter(parameters.iter().map(ValueParam::as_sql)),
        |row| row.get::<_, String>(0),
    )?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

enum ValueParam {
    Text(String),
    Integer(i64),
}

impl ValueParam {
    fn as_sql(&self) -> &dyn rusqlite::ToSql {
        match self {
            Self::Text(value) => value,
            Self::Integer(value) => value,
        }
    }
}

fn append_filter_sql(sql: &mut String, parameters: &mut Vec<ValueParam>, filters: &SearchFilters) {
    if !filters.relations.is_empty() {
        let placeholders = std::iter::repeat_n("?", filters.relations.len())
            .collect::<Vec<_>>()
            .join(",");
        sql.push_str(&format!(" AND relation IN ({placeholders})"));
        parameters.extend(filters.relations.iter().cloned().map(ValueParam::Text));
    }
    if let Some(start) = filters.time_start_ms {
        sql.push_str(" AND event_time_ms >= ?");
        parameters.push(ValueParam::Integer(
            i64::try_from(start).unwrap_or(i64::MAX),
        ));
    }
    if let Some(end) = filters.time_end_ms {
        sql.push_str(" AND event_time_ms < ?");
        parameters.push(ValueParam::Integer(i64::try_from(end).unwrap_or(i64::MAX)));
    }
}

fn open_occurrence_lookup(path: &Path) -> Result<Connection> {
    Ok(Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?)
}

#[derive(Default)]
struct MatchedOccurrences {
    eligible_count: u64,
    rows: Vec<EvidenceOccurrence>,
}
fn read_parquet<T, F: Fn(&RecordBatch, usize) -> Result<T>>(
    file: File,
    decode: F,
) -> Result<Vec<T>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            out.push(decode(&batch, row)?);
        }
    }
    Ok(out)
}

fn parquet_row_count(path: &Path) -> Result<u64> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    u64::try_from(builder.metadata().file_metadata().num_rows())
        .map_err(|_| IndexError::Invalid("Parquet row count"))
}

fn artifact_summary(
    path: &Path,
    relative: &str,
    rows: u64,
    order_sha256: Option<String>,
) -> Result<ObjectSummary> {
    Ok(ObjectSummary {
        path: relative.into(),
        rows,
        bytes: fs::metadata(path)?.len(),
        sha256: file_sha256(path)?,
        order_sha256,
    })
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_artifact(path: &Path, bytes: u64, sha256: &str) -> Result<()> {
    if fs::metadata(path)?.len() != bytes || file_sha256(path)? != sha256 {
        return Err(IndexError::Corrupt("artifact content digest"));
    }
    Ok(())
}

fn manifest_component_sha256(manifest: &FastIndexManifest) -> Result<String> {
    let mut material = serde_json::json!({
        "schema_version": manifest.schema_version,
        "source": manifest.source,
        "build_scope": manifest.build_scope,
        "complete": manifest.complete,
        "documents": manifest.documents,
        "occurrences": manifest.occurrences,
        "vectors": manifest.vectors,
        "lexical": manifest.lexical,
        "occurrence_lookup": manifest.occurrence_lookup,
        "embedding_profile": manifest.embedding_profile,
    });
    if let Some(provenance) = &manifest.pipeline_provenance {
        material
            .as_object_mut()
            .expect("manifest identity material is an object")
            .insert(
                "pipeline_provenance".into(),
                serde_json::to_value(provenance)?,
            );
    }
    if manifest.test_only {
        material
            .as_object_mut()
            .expect("manifest identity material is an object")
            .insert("test_only".into(), serde_json::Value::Bool(true));
    }
    // SDK component identities use RFC 8785 JSON Canonicalization Scheme.
    let bytes = serde_json_canonicalizer::to_vec(&material)
        .map_err(|_| IndexError::Invalid("component identity canonicalization"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_occurrence_lookup_metadata(path: &Path, manifest: &FastIndexManifest) -> Result<()> {
    let connection = open_occurrence_lookup(path)?;
    let read = |key: &str| -> Result<String> {
        Ok(
            connection.query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get(0)
            })?,
        )
    };
    if read("schema")? != OCCURRENCE_LOOKUP_SCHEMA
        || read("rows")? != manifest.occurrences.rows.to_string()
        || read("snapshot_sha256")? != manifest.source.snapshot_sha256
        || read("mapping_sha256")? != manifest.source.mapping_sha256
    {
        return Err(IndexError::Corrupt("occurrence lookup metadata"));
    }
    let rows = u64::try_from(connection.query_row(
        "SELECT count(*) FROM occurrences",
        [],
        |row| row.get::<_, i64>(0),
    )?)
    .map_err(|_| IndexError::Corrupt("negative occurrence lookup row count"))?;
    if rows != manifest.occurrences.rows {
        return Err(IndexError::Corrupt("occurrence lookup row count"));
    }
    Ok(())
}

fn open_sqlite_lexical(path: &Path) -> Result<Connection> {
    Ok(Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?)
}

fn validate_sqlite_lexical_metadata(
    path: &Path,
    expected_document_count: u64,
    expected_document_order_sha256: &str,
) -> Result<(u64, u64)> {
    let connection = open_sqlite_lexical(path)?;
    let quick_check =
        connection.query_row("PRAGMA quick_check(1)", [], |row| row.get::<_, String>(0))?;
    let application_id =
        connection.query_row("PRAGMA application_id", [], |row| row.get::<_, i64>(0))?;
    let user_version =
        connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
    if quick_check != "ok" || application_id != SQLITE_LEXICAL_APPLICATION_ID || user_version != 1 {
        return Err(IndexError::Corrupt("lexical SQLite header"));
    }
    let read = |key: &str| -> Result<String> {
        Ok(
            connection.query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get(0)
            })?,
        )
    };
    let document_count = read("document_count")?
        .parse::<u64>()
        .map_err(|_| IndexError::Corrupt("lexical document count"))?;
    let total_length = read("total_length")?
        .parse::<u64>()
        .map_err(|_| IndexError::Corrupt("lexical token count"))?;
    if read("schema")? != SQLITE_LEXICAL_SCHEMA
        || read("document_order_sha256")? != expected_document_order_sha256
        || read("tokenizer")? != "ascii_camel_lower_v1"
        || read("k1")? != "1.2"
        || read("b")? != "0.75"
        || document_count != expected_document_count
    {
        return Err(IndexError::Corrupt("lexical metadata binding"));
    }
    let (rows, minimum, maximum) = connection.query_row(
        "SELECT count(*), min(document_ordinal), max(document_ordinal) FROM documents",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<i64>>(2)?,
            ))
        },
    )?;
    let rows = u64::try_from(rows).map_err(|_| IndexError::Corrupt("lexical document count"))?;
    let expected_maximum = i64::try_from(document_count.saturating_sub(1))
        .map_err(|_| IndexError::Corrupt("lexical document ordinal"))?;
    if rows != document_count
        || minimum != Some(0)
        || maximum != Some(expected_maximum)
        || connection.query_row(
            "SELECT EXISTS(
               SELECT 1 FROM documents WHERE length < 0
               UNION ALL
               SELECT 1 FROM postings WHERE term_frequency <= 0
               UNION ALL
               SELECT 1 FROM term_stats
                WHERE document_frequency <= 0 OR document_frequency > ?1
               UNION ALL
               SELECT 1 FROM postings AS posting
               LEFT JOIN documents AS document USING (document_ordinal)
               WHERE document.document_ordinal IS NULL
             )",
            [i64::try_from(document_count)
                .map_err(|_| IndexError::Corrupt("lexical document count"))?],
            |row| row.get::<_, bool>(0),
        )?
    {
        return Err(IndexError::Corrupt("lexical row binding"));
    }
    Ok((document_count, total_length))
}

fn safe_artifact(root: &Path, value: &str) -> Result<PathBuf> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(IndexError::Invalid("artifact path"));
    }
    let path = fs::canonicalize(root.join(relative))?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(IndexError::Invalid("artifact containment"));
    }
    Ok(path)
}
fn text(batch: &RecordBatch, name: &'static str, row: usize) -> Result<String> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| IndexError::Invalid(name))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StringArray>()
        .filter(|a| !a.is_null(row))
        .map(|a| a.value(row).to_owned())
        .ok_or(IndexError::Invalid(name))
}
fn number(batch: &RecordBatch, name: &'static str, row: usize) -> Result<u64> {
    optional_number(batch, name, row)?.ok_or(IndexError::Invalid(name))
}
fn optional_number(batch: &RecordBatch, name: &'static str, row: usize) -> Result<Option<u64>> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| IndexError::Invalid(name))?;
    let array = batch
        .column(index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(IndexError::Invalid(name))?;
    Ok((!array.is_null(row)).then(|| array.value(row)))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use super::*;
    use tempfile::tempdir;
    fn profile() -> EmbeddingProfile {
        EmbeddingProfile {
            id: "p".into(),
            version: "1".into(),
            sha256: "a".repeat(64),
            model: "fake".into(),
            dimensions: 2,
            normalization: "l2".into(),
            vector_derivation: None,
            query_instruction: None,
            query_composition: None,
        }
    }
    fn fixture() -> (Vec<FastDocument>, Vec<FastOccurrence>, Vec<Vec<f32>>) {
        let docs = vec![
            FastDocument {
                document_id: "a".into(),
                document_sha256: "1".into(),
                document_kind: "activity".into(),
                semantic_text: "encoded PowerShell logging bypass".into(),
                facets_json: "[]".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: 0,
            },
            FastDocument {
                document_id: "b".into(),
                document_sha256: "2".into(),
                document_kind: "activity".into(),
                semantic_text: "normal browser launch".into(),
                facets_json: "[]".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: 1,
            },
        ];
        let occ = docs
            .iter()
            .map(|d| FastOccurrence {
                occurrence_id: format!("o-{}", d.document_id),
                document_id: d.document_id.clone(),
                event_time_ms: Some(10),
                relation: "ocsf_process_activity".into(),
                exact_attributes_json: "{}".into(),
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
                event_id: format!("e-{}", d.document_id),
                support_ref: "sup".into(),
            })
            .collect();
        (docs, occ, vec![vec![1.0, 0.0], vec![0.0, 1.0]])
    }

    fn provenance(digit: char) -> PipelineProvenance {
        PipelineProvenance {
            dataset_sha256: digit.to_string().repeat(64),
            prepared_corpus_sha256: "4".repeat(64),
            embedding_plan_sha256: "5".repeat(64),
            embedding_result_set_sha256: "6".repeat(64),
        }
    }

    #[test]
    fn bound_index_seals_and_reopens_exact_pipeline_provenance() {
        let root = tempdir().unwrap();
        let build = |out: &Path, pipeline_provenance| {
            let (documents, occurrences, vectors) = fixture();
            write_bound_scalable_fast_index_from_streams(
                out,
                documents.into_iter().map(Ok),
                occurrences.into_iter().map(Ok),
                vectors.into_iter().enumerate().map(|(ordinal, values)| {
                    Ok(OrderedVector {
                        vector_ordinal: ordinal as u64,
                        values,
                    })
                }),
                PipelineIndexOptions {
                    source: SourceBinding {
                        snapshot_sha256: "a".repeat(64),
                        mapping_sha256: "b".repeat(64),
                    },
                    build_scope: BuildScope::Sample,
                    embedding_profile: profile(),
                    provenance: pipeline_provenance,
                    test_only: false,
                },
            )
            .unwrap()
        };
        let first_provenance = provenance('3');
        let first = build(&root.path().join("first"), first_provenance.clone());
        let second = build(&root.path().join("second"), provenance('7'));
        assert_eq!(first.pipeline_provenance, Some(first_provenance));
        assert_ne!(first.component_sha256, second.component_sha256);
        assert_eq!(
            FastIndex::open(&root.path().join("first"))
                .unwrap()
                .manifest,
            first
        );
    }

    #[test]
    fn normal_open_refuses_test_only_index_and_diagnostic_open_accepts_it() {
        let root = tempdir().unwrap();
        let out = root.path().join("test-index");
        let (documents, occurrences, vectors) = fixture();
        let manifest = write_bound_scalable_fast_index_from_streams(
            &out,
            documents.into_iter().map(Ok),
            occurrences.into_iter().map(Ok),
            vectors.into_iter().enumerate().map(|(ordinal, values)| {
                Ok(OrderedVector {
                    vector_ordinal: ordinal as u64,
                    values,
                })
            }),
            PipelineIndexOptions {
                source: SourceBinding {
                    snapshot_sha256: "a".repeat(64),
                    mapping_sha256: "b".repeat(64),
                },
                build_scope: BuildScope::Sample,
                embedding_profile: profile(),
                provenance: provenance('3'),
                test_only: true,
            },
        )
        .unwrap();
        assert!(manifest.test_only);
        assert_eq!(manifest.schema_version, "livefire.rag.fast-index/4");
        assert!(FastIndex::open(&out).is_err());
        assert!(
            FastIndex::open_allow_test_only(&out)
                .unwrap()
                .manifest
                .test_only
        );
        let report: BuildReport =
            serde_json::from_slice(&fs::read(out.join("build-report.json")).unwrap()).unwrap();
        assert!(report.test_only);
        assert_eq!(report.schema_version, "livefire.rag.fast-build-report/2");
    }

    #[test]
    fn direct_index_roundtrip_and_search() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (d, o, v) = fixture();
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &d,
            &o,
            &v,
            profile(),
        )
        .unwrap();
        let report: BuildReport =
            serde_json::from_slice(&fs::read(out.join("build-report.json")).unwrap()).unwrap();
        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.embedded, 0);
        assert_eq!(
            report.accounting["coverage_semantics"],
            "assembled_input_stream_only_not_source_row_coverage"
        );
        let index = FastIndex::open(&out).unwrap();
        assert!(!index.manifest.complete);
        let dense_hits = index
            .search(
                SearchMode::Dense,
                "",
                Some(&[1.0, 0.0]),
                &SearchFilters::default(),
                2,
            )
            .unwrap();
        assert_eq!(dense_hits[0].document_id, "a");
        assert!(dense_hits[0].occurrences_exhausted);
        assert_eq!(
            index
                .search(
                    SearchMode::Lexical,
                    "logging bypass",
                    None,
                    &SearchFilters::default(),
                    2
                )
                .unwrap()[0]
                .document_id,
            "a"
        );
        assert_eq!(
            index
                .search(
                    SearchMode::Fused,
                    "logging bypass",
                    Some(&[1.0, 0.0]),
                    &SearchFilters::default(),
                    2
                )
                .unwrap()[0]
                .document_id,
            "a"
        );
    }

    #[test]
    fn bound_query_search_matches_existing_dense_and_fused_results() {
        let (_root, out) = built_index();
        let index = FastIndex::open(&out).unwrap();
        let filters = SearchFilters::default();
        let values = [1.0, 0.0];
        let query = index.validate_query_vector(&values).unwrap();

        let existing_dense = index
            .search(SearchMode::Dense, "", Some(&values), &filters, 2)
            .unwrap();
        let bound_dense = index.search_dense_with_vector(&query, &filters, 2).unwrap();
        assert_eq!(bound_dense, existing_dense);

        let existing_fused = index
            .search(
                SearchMode::Fused,
                "logging bypass",
                Some(&values),
                &filters,
                2,
            )
            .unwrap();
        let bound_fused = index
            .search_fused_with_vector("logging bypass", &query, &filters, 2)
            .unwrap();
        assert_eq!(bound_fused, existing_fused);
        assert_eq!(query.values(), values);
        assert_eq!(
            query.embedding_profile_sha256(),
            index.manifest.embedding_profile.sha256
        );
    }

    #[test]
    fn bound_query_validates_values_and_rejects_a_different_profile() {
        let (_root, out) = built_index();
        let index = FastIndex::open(&out).unwrap();
        assert!(matches!(
            index.validate_query_vector(&[1.0]),
            Err(IndexError::Invalid("query vector"))
        ));
        assert!(matches!(
            index.validate_query_vector(&[f32::NAN, 0.0]),
            Err(IndexError::Invalid("query vector"))
        ));
        assert!(matches!(
            index.validate_query_vector(&[0.5, 0.5]),
            Err(IndexError::Invalid("query vector normalization"))
        ));

        let query = index.validate_query_vector(&[1.0, 0.0]).unwrap();
        let other_root = tempdir().unwrap();
        let other_out = other_root.path().join("index");
        let (documents, occurrences, vectors) = fixture();
        let mut other_profile = profile();
        other_profile.sha256 = "c".repeat(64);
        write_fast_index(
            &other_out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &vectors,
            other_profile,
        )
        .unwrap();
        let other_index = FastIndex::open(&other_out).unwrap();
        assert!(matches!(
            other_index.search_dense_with_vector(&query, &SearchFilters::default(), 1),
            Err(IndexError::Invalid("query vector profile binding"))
        ));
    }

    #[test]
    fn open_verifies_content_bound_artifacts_before_use() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (documents, occurrences, vectors) = fixture();
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &vectors,
            profile(),
        )
        .unwrap();
        fs::write(out.join("lexical/index.json"), b"not-json\n").unwrap();
        assert!(matches!(
            FastIndex::open(&out),
            Err(IndexError::Corrupt("artifact content digest"))
        ));
    }
    #[test]
    fn relation_filter_is_occurrence_first() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (d, mut o, v) = fixture();
        o[1].relation = "other".into();
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Full,
            &d,
            &o,
            &v,
            profile(),
        )
        .unwrap();
        let index = FastIndex::open(&out).unwrap();
        let filters = SearchFilters {
            relations: BTreeSet::from(["other".into()]),
            ..Default::default()
        };
        let hits = index
            .search(SearchMode::Dense, "", Some(&[1.0, 0.0]), &filters, 2)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id, "b");
    }

    #[test]
    fn query_uses_bound_occurrence_lookup_without_replaying_parquet() {
        let (_root, out) = built_index();
        let index = FastIndex::open(&out).unwrap();
        fs::remove_file(out.join("occurrences.parquet")).unwrap();
        let filters = SearchFilters {
            relations: BTreeSet::from(["ocsf_process_activity".into()]),
            time_start_ms: Some(0),
            time_end_ms: Some(20),
        };
        let hits = index
            .search(SearchMode::Lexical, "logging bypass", None, &filters, 1)
            .unwrap();
        assert_eq!(hits[0].document_id, "a");
        assert_eq!(hits[0].occurrences[0].event_id, "e-a");
    }

    #[test]
    fn high_fanout_hit_bounds_returned_references_and_reports_exhaustion() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (mut documents, mut occurrences, vectors) = fixture();
        for sequence in 1..=50 {
            let mut occurrence = occurrences[0].clone();
            occurrence.occurrence_id = format!("o-a-{sequence:02}");
            occurrence.event_id = format!("e-a-{sequence:02}");
            occurrences.push(occurrence);
        }
        documents[0].occurrence_count = 51;
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &vectors,
            profile(),
        )
        .unwrap();
        let index = FastIndex::open(&out).unwrap();
        let hit = index
            .search(
                SearchMode::Lexical,
                "logging bypass",
                None,
                &SearchFilters::default(),
                1,
            )
            .unwrap()
            .remove(0);
        assert_eq!(hit.eligible_occurrence_count, 51);
        assert_eq!(hit.occurrences.len(), MAX_RETURNED_OCCURRENCES_PER_HIT);
        assert!(!hit.occurrences_exhausted);
    }

    #[test]
    fn writer_rejects_duplicate_snapshot_event_pointers() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (mut documents, mut occurrences, vectors) = fixture();
        let mut duplicate = occurrences[0].clone();
        duplicate.occurrence_id = "different-occurrence-id".into();
        occurrences.push(duplicate);
        documents[0].occurrence_count = 2;
        assert!(matches!(
            write_fast_index(
                &out,
                SourceBinding {
                    snapshot_sha256: "a".repeat(64),
                    mapping_sha256: "b".repeat(64),
                },
                BuildScope::Full,
                &documents,
                &occurrences,
                &vectors,
                profile(),
            ),
            Err(IndexError::Sqlite(_))
        ));
        assert!(!out.exists());
    }

    fn built_index() -> (tempfile::TempDir, PathBuf) {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (documents, occurrences, vectors) = fixture();
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &vectors,
            profile(),
        )
        .unwrap();
        (root, out)
    }

    fn write_scalable_fixture(
        output: &Path,
        documents: &[FastDocument],
        occurrences: &[FastOccurrence],
        vectors: &[Vec<f32>],
    ) -> FastIndexManifest {
        write_scalable_fast_index_from_streams(
            output,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            documents.iter().cloned().map(Ok),
            occurrences.iter().cloned().map(Ok),
            vectors
                .iter()
                .cloned()
                .enumerate()
                .map(|(ordinal, values)| {
                    Ok(OrderedVector {
                        vector_ordinal: ordinal as u64,
                        values,
                    })
                }),
            profile(),
        )
        .unwrap()
    }

    #[test]
    fn scalable_lexical_v3_matches_v2_scores_order_and_occurrence_filters() {
        let root = tempdir().unwrap();
        let v2_path = root.path().join("v2");
        let v3_path = root.path().join("v3");
        let (documents, mut occurrences, vectors) = fixture();
        occurrences[1].relation = "other".into();
        write_fast_index(
            &v2_path,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            &occurrences,
            &vectors,
            profile(),
        )
        .unwrap();
        let v3_manifest = write_scalable_fixture(&v3_path, &documents, &occurrences, &vectors);
        assert_eq!(v3_manifest.schema_version, "livefire.rag.fast-index/3");
        assert_eq!(
            v3_manifest.lexical.schema.as_deref(),
            Some(SQLITE_LEXICAL_SCHEMA)
        );
        assert_eq!(v3_manifest.lexical.path, "lexical/index.sqlite3");
        let v2 = FastIndex::open(&v2_path).unwrap();
        let v3 = FastIndex::open(&v3_path).unwrap();
        for query in [
            "logging bypass",
            "normal launch",
            "PowerShell normal",
            "absent-term",
        ] {
            let v2_hits = v2
                .search(
                    SearchMode::Lexical,
                    query,
                    None,
                    &SearchFilters::default(),
                    2,
                )
                .unwrap();
            let v3_hits = v3
                .search(
                    SearchMode::Lexical,
                    query,
                    None,
                    &SearchFilters::default(),
                    2,
                )
                .unwrap();
            assert_eq!(v3_hits, v2_hits, "{query}");
        }
        let filters = SearchFilters {
            relations: BTreeSet::from(["ocsf_process_activity".into()]),
            ..Default::default()
        };
        let v2_filtered = v2
            .search(SearchMode::Lexical, "normal launch", None, &filters, 2)
            .unwrap();
        let v3_filtered = v3
            .search(SearchMode::Lexical, "normal launch", None, &filters, 2)
            .unwrap();
        assert_eq!(v3_filtered, v2_filtered);
        assert!(v3_filtered.is_empty());
    }

    #[test]
    fn scalable_lexical_v3_uses_document_id_for_exact_ties_without_loading_parquet() {
        let root = tempdir().unwrap();
        let out = root.path().join("v3");
        let (mut documents, occurrences, vectors) = fixture();
        for document in &mut documents {
            document.semantic_text = "same exact terms".into();
        }
        write_scalable_fixture(&out, &documents, &occurrences, &vectors);
        let index = FastIndex::open(&out).unwrap();
        fs::remove_file(out.join("documents.parquet")).unwrap();
        let hits = index
            .search(
                SearchMode::Lexical,
                "same",
                None,
                &SearchFilters::default(),
                2,
            )
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|hit| hit.document_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(hits[0].lexical_score, hits[1].lexical_score);
        assert_eq!(hits[0].semantic_text, "same exact terms");
    }

    #[test]
    fn scalable_lexical_v3_is_byte_deterministic_and_rejects_resealed_metadata_drift() {
        let root = tempdir().unwrap();
        let first = root.path().join("first");
        let second = root.path().join("second");
        let (documents, occurrences, vectors) = fixture();
        let first_manifest = write_scalable_fixture(&first, &documents, &occurrences, &vectors);
        let second_manifest = write_scalable_fixture(&second, &documents, &occurrences, &vectors);
        let tracked_fixture: FastIndexManifest = serde_json::from_str(include_str!(
            "../../../rust-fixtures/index/fast-index-manifest.v3.json"
        ))
        .unwrap();
        assert_eq!(first_manifest, tracked_fixture);
        assert_eq!(
            tracked_fixture.component_sha256,
            manifest_component_sha256(&tracked_fixture).unwrap()
        );
        assert_eq!(first_manifest.lexical, second_manifest.lexical);
        assert_eq!(
            fs::read(first.join("lexical/index.sqlite3")).unwrap(),
            fs::read(second.join("lexical/index.sqlite3")).unwrap()
        );

        let lexical_path = first.join("lexical/index.sqlite3");
        let connection = Connection::open(&lexical_path).unwrap();
        connection
            .execute(
                "UPDATE metadata SET value = 'wrong-schema' WHERE key = 'schema'",
                [],
            )
            .unwrap();
        drop(connection);
        let mut manifest: FastIndexManifest =
            serde_json::from_slice(&fs::read(first.join("index.json")).unwrap()).unwrap();
        manifest.lexical.bytes = fs::metadata(&lexical_path).unwrap().len();
        manifest.lexical.sha256 = file_sha256(&lexical_path).unwrap();
        manifest.component_sha256 = manifest_component_sha256(&manifest).unwrap();
        let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        bytes.push(b'\n');
        fs::write(first.join("index.json"), bytes).unwrap();
        assert!(matches!(
            FastIndex::open(&first),
            Err(IndexError::Corrupt("lexical metadata binding"))
        ));
    }

    #[test]
    fn scalable_lexical_v3_supports_concurrent_read_only_queries() {
        const THREADS: usize = 8;
        let root = tempdir().unwrap();
        let out = root.path().join("v3");
        let (documents, occurrences, vectors) = fixture();
        write_scalable_fixture(&out, &documents, &occurrences, &vectors);
        let index = Arc::new(FastIndex::open(&out).unwrap());
        let barrier = Arc::new(Barrier::new(THREADS));
        let joins = (0..THREADS)
            .map(|_| {
                let index = Arc::clone(&index);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..20 {
                        let hit = index
                            .search(
                                SearchMode::Lexical,
                                "logging bypass",
                                None,
                                &SearchFilters::default(),
                                1,
                            )
                            .unwrap()
                            .remove(0);
                        assert_eq!(hit.document_id, "a");
                    }
                })
            })
            .collect::<Vec<_>>();
        for join in joins {
            join.join().unwrap();
        }
    }

    #[test]
    fn rejects_tampered_occurrence_and_document_artifacts() {
        let (_root, out) = built_index();
        let path = out.join("occurrences.parquet");
        let mut bytes = fs::read(&path).unwrap();
        bytes[32] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            FastIndex::open(&out),
            Err(IndexError::Corrupt("artifact content digest"))
        ));

        let (_root, out) = built_index();
        let path = out.join("documents.parquet");
        let mut documents = read_documents(File::open(&path).unwrap()).unwrap();
        documents[0].vector_ordinal = 10;
        documents[1].vector_ordinal = 11;
        fs::remove_file(&path).unwrap();
        write_documents(&path, &documents).unwrap();
        assert!(matches!(
            FastIndex::open(&out),
            Err(IndexError::Corrupt("artifact content digest"))
        ));
    }

    #[test]
    fn rejects_corrupt_vectors_and_lexical_associations() {
        let (_root, out) = built_index();
        let path = out.join("vectors.f32");
        let mut bytes = fs::read(&path).unwrap();
        bytes[VECTOR_HEADER_BYTES as usize..VECTOR_HEADER_BYTES as usize + 4]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(path, bytes).unwrap();
        assert!(matches!(
            FastIndex::open(&out),
            Err(IndexError::Corrupt("artifact content digest"))
        ));

        let (_root, out) = built_index();
        let path = out.join("lexical/index.json");
        let mut lexical: LexicalIndex = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        lexical.documents[0].document_id = "foreign".into();
        fs::write(&path, serde_json::to_vec(&lexical).unwrap()).unwrap();
        assert!(matches!(
            FastIndex::open(&out),
            Err(IndexError::Corrupt("artifact content digest"))
        ));
    }

    fn write_occurrence_shard(path: &Path, rows: &[FastOccurrence]) {
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            occurrence_schema(),
            Some(parquet_properties()),
        )
        .unwrap();
        write_occurrence_batch(&mut writer, occurrence_schema(), rows).unwrap();
        writer.close().unwrap();
    }

    fn write_embedding_shard(path: &Path, order_sha256: &str, rows: &[Vec<f32>]) {
        let dimensions = rows.first().unwrap().len() as u32;
        let mut writer = BufWriter::new(File::create(path).unwrap());
        writer.write_all(&EMBEDDING_VECTOR_MAGIC).unwrap();
        writer.write_all(&64_u32.to_le_bytes()).unwrap();
        writer.write_all(&1_u16.to_le_bytes()).unwrap();
        writer.write_all(&[1, 0]).unwrap();
        writer
            .write_all(&(rows.len() as u64).to_le_bytes())
            .unwrap();
        writer.write_all(&dimensions.to_le_bytes()).unwrap();
        writer.write_all(&0_u32.to_le_bytes()).unwrap();
        writer
            .write_all(&decode_sha256(order_sha256).unwrap())
            .unwrap();
        for row in rows {
            for value in row {
                writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        writer.flush().unwrap();
    }

    #[test]
    fn prepared_document_ordinal_is_used_as_vector_ordinal() {
        let root = tempdir().unwrap();
        let path = root.path().join("prepared.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("document_ordinal", DataType::UInt64, false),
            Field::new("document_id", DataType::Utf8, false),
            Field::new("document_sha256", DataType::Utf8, false),
            Field::new("semantic_text", DataType::Utf8, false),
            Field::new("document_kind", DataType::Utf8, false),
            Field::new("facets_json", DataType::Utf8, false),
            Field::new("relations_json", DataType::Utf8, false),
            Field::new("occurrence_count", DataType::UInt64, false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["doc-a"])),
            Arc::new(StringArray::from(vec!["hash-a"])),
            Arc::new(StringArray::from(vec!["semantic body"])),
            Arc::new(StringArray::from(vec!["activity"])),
            Arc::new(StringArray::from(vec!["{}"])),
            Arc::new(StringArray::from(vec!["[]"])),
            Arc::new(UInt64Array::from(vec![1])),
        ];
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema.clone(), None).unwrap();
        writer
            .write(&RecordBatch::try_new(schema, columns).unwrap())
            .unwrap();
        writer.close().unwrap();
        let document = documents_from_parquet_shards([path])
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(document.document_id, "doc-a");
        assert_eq!(document.vector_ordinal, 0);
    }

    #[test]
    fn shard_adapters_assemble_multiple_occurrence_and_vector_inputs() {
        let root = tempdir().unwrap();
        let (documents, occurrences, vectors) = fixture();
        let documents_path = root.path().join("prepared-documents.parquet");
        let occurrence_a = root.path().join("occurrences-a.parquet");
        let occurrence_b = root.path().join("occurrences-b.parquet");
        let vector_a = root.path().join("vectors-a.f32");
        let vector_b = root.path().join("vectors-b.f32");
        write_documents(&documents_path, &documents).unwrap();
        write_occurrence_shard(&occurrence_a, &occurrences[..1]);
        write_occurrence_shard(&occurrence_b, &occurrences[1..]);
        let order_a = "1".repeat(64);
        let order_b = "2".repeat(64);
        write_embedding_shard(&vector_a, &order_a, &vectors[..1]);
        write_embedding_shard(&vector_b, &order_b, &vectors[1..]);
        let vector_rows = vectors_from_embedding_shards([
            OrderedVectorShard {
                path: vector_a,
                first_vector_ordinal: 0,
                vector_count: 1,
                dimensions: 2,
                order_sha256: order_a,
            },
            OrderedVectorShard {
                path: vector_b,
                first_vector_ordinal: 1,
                vector_count: 1,
                dimensions: 2,
                order_sha256: order_b,
            },
        ])
        .unwrap();
        let output = root.path().join("index");
        let manifest = write_fast_index_from_streams(
            &output,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Full,
            documents_from_parquet_shards([documents_path]),
            occurrences_from_parquet_shards([occurrence_a, occurrence_b]),
            vector_rows,
            profile(),
        )
        .unwrap();
        assert_eq!(manifest.documents.rows, 2);
        assert_eq!(manifest.occurrences.rows, 2);
        assert_eq!(manifest.vectors.count, 2);
        let index = FastIndex::open(&output).unwrap();
        assert_eq!(
            index
                .search(
                    SearchMode::Dense,
                    "",
                    Some(&[0.0, 1.0]),
                    &SearchFilters::default(),
                    1,
                )
                .unwrap()[0]
                .document_id,
            "b"
        );
    }

    #[test]
    fn embedding_shards_open_lazily_and_enforce_consecutive_association() {
        let root = tempdir().unwrap();
        let first_path = root.path().join("first.f32");
        let missing_path = root.path().join("missing.f32");
        let order = "3".repeat(64);
        write_embedding_shard(&first_path, &order, &[vec![1.0, 0.0]]);
        let mut rows = vectors_from_embedding_shards([
            OrderedVectorShard {
                path: first_path,
                first_vector_ordinal: 0,
                vector_count: 1,
                dimensions: 2,
                order_sha256: order.clone(),
            },
            OrderedVectorShard {
                path: missing_path,
                first_vector_ordinal: 1,
                vector_count: 1,
                dimensions: 2,
                order_sha256: order.clone(),
            },
        ])
        .unwrap();
        assert_eq!(rows.next().unwrap().unwrap().vector_ordinal, 0);
        assert!(matches!(rows.next().unwrap(), Err(IndexError::Io(_))));

        assert!(matches!(
            vectors_from_embedding_shards([
                OrderedVectorShard {
                    path: PathBuf::from("unused-a"),
                    first_vector_ordinal: 0,
                    vector_count: 1,
                    dimensions: 2,
                    order_sha256: order.clone(),
                },
                OrderedVectorShard {
                    path: PathBuf::from("unused-b"),
                    first_vector_ordinal: 2,
                    vector_count: 1,
                    dimensions: 2,
                    order_sha256: order,
                },
            ]),
            Err(IndexError::Invalid("embedding vector shard order"))
        ));
    }

    #[test]
    fn streaming_writer_chunks_high_fanout_occurrences_and_closes_lookup() {
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let (mut documents, mut occurrences, vectors) = fixture();
        let template = occurrences[0].clone();
        occurrences.retain(|row| row.document_id != "a");
        for sequence in 0..(PARQUET_WRITE_BATCH_ROWS + 17) {
            let mut occurrence = template.clone();
            occurrence.occurrence_id = format!("o-a-{sequence:08}");
            occurrence.event_id = format!("e-a-{sequence:08}");
            occurrences.push(occurrence);
        }
        documents[0].occurrence_count = (PARQUET_WRITE_BATCH_ROWS + 17) as u64;
        let manifest = write_fast_index_streaming(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Sample,
            &documents,
            occurrences.into_iter().map(Ok),
            vectors.into_iter().map(Ok),
            profile(),
        )
        .unwrap();
        assert_eq!(
            manifest.occurrences.rows,
            (PARQUET_WRITE_BATCH_ROWS + 18) as u64
        );
        assert_eq!(manifest.occurrence_lookup.rows, manifest.occurrences.rows);
        let index = FastIndex::open(&out).unwrap();
        let hit = index
            .search(
                SearchMode::Lexical,
                "logging bypass",
                None,
                &SearchFilters::default(),
                1,
            )
            .unwrap()
            .remove(0);
        assert_eq!(
            hit.eligible_occurrence_count,
            (PARQUET_WRITE_BATCH_ROWS + 17) as u64
        );
        assert_eq!(hit.occurrences.len(), MAX_RETURNED_OCCURRENCES_PER_HIT);
    }

    #[test]
    fn concurrent_dense_searches_use_positional_vector_reads() {
        const DOCUMENTS: usize = 4_096;
        const DIMENSIONS: usize = 128;
        const THREADS: usize = 8;
        let root = tempdir().unwrap();
        let out = root.path().join("index");
        let mut documents = Vec::with_capacity(DOCUMENTS);
        let mut occurrences = Vec::with_capacity(DOCUMENTS);
        let mut vectors = Vec::with_capacity(DOCUMENTS);
        for ordinal in 0..DOCUMENTS {
            let document_id = format!("doc-{ordinal:08}");
            documents.push(FastDocument {
                document_id: document_id.clone(),
                document_sha256: "1".into(),
                document_kind: "activity".into(),
                semantic_text: format!("document {ordinal}"),
                facets_json: "{}".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: ordinal as u64,
            });
            occurrences.push(FastOccurrence {
                occurrence_id: format!("occ-{ordinal:08}"),
                document_id,
                event_time_ms: Some(ordinal as u64),
                relation: "ocsf_process_activity".into(),
                exact_attributes_json: "[]".into(),
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
                event_id: format!("evt-{ordinal:08}"),
                support_ref: format!("sup-{ordinal:08}"),
            });
            let mut vector = vec![0.0; DIMENSIONS];
            vector[ordinal % DIMENSIONS] = 1.0;
            vectors.push(vector);
        }
        let mut embedding_profile = profile();
        embedding_profile.dimensions = DIMENSIONS as u32;
        write_fast_index(
            &out,
            SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            BuildScope::Full,
            &documents,
            &occurrences,
            &vectors,
            embedding_profile,
        )
        .unwrap();
        let index = Arc::new(FastIndex::open(&out).unwrap());
        let mut warmup = vec![0.0; DIMENSIONS];
        warmup[0] = 1.0;
        index
            .search(
                SearchMode::Dense,
                "",
                Some(&warmup),
                &SearchFilters::default(),
                1,
            )
            .unwrap();
        let barrier = Arc::new(Barrier::new(THREADS));
        let joins = (0..THREADS)
            .map(|thread_id| {
                let index = Arc::clone(&index);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let mut query = vec![0.0; DIMENSIONS];
                    query[thread_id] = 1.0;
                    let query = index.validate_query_vector(&query).unwrap();
                    barrier.wait();
                    for _ in 0..10 {
                        let hit = index
                            .search_dense_with_vector(&query, &SearchFilters::default(), 1)
                            .unwrap()
                            .remove(0);
                        assert_eq!(hit.dense_score, Some(1.0));
                    }
                })
            })
            .collect::<Vec<_>>();
        for join in joins {
            join.join().unwrap();
        }
    }

    #[test]
    fn component_identity_is_rfc8785_canonical_and_cross_language_stable() {
        let object = |path: &str, digit: char| ObjectSummary {
            path: path.into(),
            rows: 1,
            bytes: 2,
            sha256: digit.to_string().repeat(64),
            order_sha256: None,
        };
        let manifest = FastIndexManifest {
            schema_version: "livefire.rag.fast-index/2".into(),
            component_sha256: String::new(),
            source: SourceBinding {
                snapshot_sha256: "a".repeat(64),
                mapping_sha256: "b".repeat(64),
            },
            build_scope: BuildScope::Sample,
            complete: false,
            documents: ObjectSummary {
                order_sha256: Some("c".repeat(64)),
                ..object("documents.parquet", 'd')
            },
            occurrences: object("occurrences.parquet", 'e'),
            vectors: VectorSummary {
                path: "vectors.f32".into(),
                count: 1,
                bytes: 72,
                sha256: "f".repeat(64),
                dimensions: 2,
                dtype: "f32le".into(),
                header_bytes: VECTOR_HEADER_BYTES,
                document_order_sha256: "c".repeat(64),
            },
            lexical: LexicalSummary {
                path: "lexical/index.json".into(),
                document_count: 1,
                bytes: 10,
                sha256: "1".repeat(64),
                schema: None,
                tokenizer: "ascii_camel_lower_v1".into(),
                k1: 1.2,
                b: 0.75,
            },
            occurrence_lookup: OccurrenceLookupSummary {
                path: "occurrence-index.sqlite3".into(),
                rows: 1,
                bytes: 4096,
                sha256: "2".repeat(64),
                schema: OCCURRENCE_LOOKUP_SCHEMA.into(),
            },
            embedding_profile: profile(),
            pipeline_provenance: None,
            test_only: false,
        };
        assert_eq!(
            manifest_component_sha256(&manifest).unwrap(),
            "e24c25ab06821dbca8686404d30d69407c7aa6d1c0954c47dc300e57f40f9b30"
        );
    }
}
