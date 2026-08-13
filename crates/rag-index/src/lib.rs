//! Portable fast-index writer and exact dense/lexical/fused reader.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::{ArrowWriter, ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use rag_embedding::{EmbeddingProfile, validate_embedding_profile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const VECTOR_MAGIC: [u8; 8] = *b"LFRAGV1\0";
pub const VECTOR_HEADER_BYTES: u32 = 64;
pub const MAX_RETURNED_OCCURRENCES_PER_HIT: usize = 50;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectSummary {
    pub path: String,
    pub rows: u64,
    pub order_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSummary {
    pub path: String,
    pub count: u64,
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
    pub tokenizer: String,
    pub k1: f64,
    pub b: f64,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastIndexManifest {
    pub schema_version: String,
    pub source: SourceBinding,
    pub build_scope: BuildScope,
    pub complete: bool,
    pub documents: ObjectSummary,
    pub occurrences: ObjectSummary,
    pub vectors: VectorSummary,
    pub lexical: LexicalSummary,
    pub embedding_profile: EmbeddingProfile,
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

pub fn write_fast_index(
    output: &Path,
    source: SourceBinding,
    build_scope: BuildScope,
    documents: &[FastDocument],
    occurrences: &[FastOccurrence],
    vectors: &[Vec<f32>],
    embedding_profile: EmbeddingProfile,
) -> Result<FastIndexManifest> {
    if output.exists() || documents.is_empty() || documents.len() != vectors.len() {
        return Err(IndexError::Invalid("output/documents/vectors"));
    }
    validate_order(&source, documents, occurrences, vectors, &embedding_profile)?;
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
        write_documents(&staging.join("documents.parquet"), documents)?;
        write_occurrences(&staging.join("occurrences.parquet"), occurrences)?;
        let order_sha = document_order_sha256(documents);
        write_vectors(
            &staging.join("vectors.f32"),
            vectors,
            embedding_profile.dimensions,
            &order_sha,
        )?;
        let lexical = build_lexical(documents);
        fs::create_dir(staging.join("lexical"))?;
        let mut lexical_bytes = serde_json::to_vec(&lexical)?;
        lexical_bytes.push(b'\n');
        fs::write(staging.join("lexical/index.json"), lexical_bytes)?;
        let manifest = FastIndexManifest {
            schema_version: "livefire.rag.fast-index/1".into(),
            source,
            build_scope: build_scope.clone(),
            complete: matches!(build_scope, BuildScope::Full),
            documents: ObjectSummary {
                path: "documents.parquet".into(),
                rows: documents.len() as u64,
                order_sha256: Some(order_sha.clone()),
            },
            occurrences: ObjectSummary {
                path: "occurrences.parquet".into(),
                rows: occurrences.len() as u64,
                order_sha256: None,
            },
            vectors: VectorSummary {
                path: "vectors.f32".into(),
                count: vectors.len() as u64,
                dimensions: embedding_profile.dimensions,
                dtype: "f32le".into(),
                header_bytes: VECTOR_HEADER_BYTES,
                document_order_sha256: order_sha,
            },
            lexical: LexicalSummary {
                path: "lexical/index.json".into(),
                document_count: documents.len() as u64,
                tokenizer: "ascii_camel_lower_v1".into(),
                k1: 1.2,
                b: 0.75,
            },
            embedding_profile,
        };
        let mut bytes = serde_json::to_vec_pretty(&manifest)?;
        bytes.push(b'\n');
        fs::write(staging.join("index.json"), bytes)?;
        let report = BuildReport {
            schema_version: "livefire.rag.fast-build-report/1".into(),
            source: manifest.source.clone(),
            build_scope: manifest.build_scope.clone(),
            complete: manifest.complete,
            document_count: documents.len() as u64,
            occurrence_count: occurrences.len() as u64,
            vector_count: vectors.len() as u64,
            embedding_profile_sha256: manifest.embedding_profile.sha256.clone(),
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
    documents_file: File,
    occurrences_file: File,
    lexical_bytes: Vec<u8>,
    vectors_file: File,
    data: OnceLock<SearchData>,
}

struct SearchData {
    documents: Vec<FastDocument>,
    lexical: LexicalIndex,
}

impl FastIndex {
    /// Fast open validates metadata and vector/document pairing without replaying the parent.
    pub fn open(root: &Path) -> Result<Self> {
        let root = fs::canonicalize(root)?;
        let manifest: FastIndexManifest =
            serde_json::from_slice(&fs::read(root.join("index.json"))?)?;
        if manifest.schema_version != "livefire.rag.fast-index/1" {
            return Err(IndexError::Invalid("manifest version"));
        }
        if manifest.complete != matches!(manifest.build_scope, BuildScope::Full)
            || manifest.vectors.count != manifest.documents.rows
            || manifest.vectors.dimensions != manifest.embedding_profile.dimensions
            || manifest.vectors.dtype != "f32le"
            || manifest.vectors.header_bytes != VECTOR_HEADER_BYTES
            || manifest.lexical.document_count != manifest.documents.rows
            || manifest.lexical.tokenizer != "ascii_camel_lower_v1"
            || !manifest.lexical.k1.is_finite()
            || manifest.lexical.k1 <= 0.0
            || !manifest.lexical.b.is_finite()
            || !(0.0..=1.0).contains(&manifest.lexical.b)
            || validate_embedding_profile(&manifest.embedding_profile).is_err()
        {
            return Err(IndexError::Invalid("manifest bindings"));
        }
        let documents_path = safe_artifact(&root, &manifest.documents.path)?;
        let occurrences_path = safe_artifact(&root, &manifest.occurrences.path)?;
        if parquet_row_count(&documents_path)? != manifest.documents.rows
            || parquet_row_count(&occurrences_path)? != manifest.occurrences.rows
        {
            return Err(IndexError::Invalid("manifest row binding"));
        }
        let vectors_path = safe_artifact(&root, &manifest.vectors.path)?;
        validate_vector_file(&vectors_path, &manifest.vectors)?;
        let lexical_path = safe_artifact(&root, &manifest.lexical.path)?;
        let documents_file = File::open(&documents_path)?;
        let occurrences_file = File::open(&occurrences_path)?;
        let vectors_file = File::open(&vectors_path)?;
        let lexical_bytes = fs::read(&lexical_path)?;
        Ok(Self {
            root,
            manifest,
            documents_file,
            occurrences_file,
            lexical_bytes,
            vectors_file,
            data: OnceLock::new(),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn search(
        &self,
        mode: SearchMode,
        query_text: &str,
        query_vector: Option<&[f32]>,
        filters: &SearchFilters,
        top_n: usize,
    ) -> Result<Vec<SearchHit>> {
        if top_n == 0 || top_n > 100 {
            return Err(IndexError::Invalid("top_n"));
        }
        let data = self.search_data()?;
        let eligible = self.eligible(data, filters)?;
        let dense = if matches!(mode, SearchMode::Dense | SearchMode::Fused) {
            Some(self.dense_scores(
                data,
                query_vector.ok_or(IndexError::Invalid("query vector"))?,
                &eligible,
            )?)
        } else {
            None
        };
        let lexical = if matches!(mode, SearchMode::Lexical | SearchMode::Fused) {
            Some(data.lexical.scores(query_text, &eligible))
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
        let occurrences_by_document =
            read_matching_occurrences(self.occurrences_file.try_clone()?, &selected_ids, filters)?;
        Ok(selected
            .into_iter()
            .enumerate()
            .map(|(index, (id, score))| SearchHit {
                rank: index + 1,
                semantic_text: data
                    .documents
                    .iter()
                    .find(|document| document.document_id == id)
                    .expect("ranked document exists")
                    .semantic_text
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

    fn search_data(&self) -> Result<&SearchData> {
        if let Some(data) = self.data.get() {
            return Ok(data);
        }
        let documents = read_documents(self.documents_file.try_clone()?)?;
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
        validate_occurrence_closure(
            self.occurrences_file.try_clone()?,
            &self.manifest.source,
            &documents,
        )?;
        let lexical: LexicalIndex = serde_json::from_slice(&self.lexical_bytes)?;
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
        let _ = self.data.set(SearchData { documents, lexical });
        self.data
            .get()
            .ok_or(IndexError::Invalid("search data initialization"))
    }

    fn eligible(&self, data: &SearchData, filters: &SearchFilters) -> Result<BTreeSet<String>> {
        if filters.relations.is_empty()
            && filters.time_start_ms.is_none()
            && filters.time_end_ms.is_none()
        {
            return Ok(data
                .documents
                .iter()
                .map(|document| document.document_id.clone())
                .collect());
        }
        let mut eligible = BTreeSet::new();
        visit_filter_rows(
            self.occurrences_file.try_clone()?,
            |document_id, event_time_ms, relation| {
                if filter_values_match(event_time_ms, &relation, filters) {
                    eligible.insert(document_id);
                }
            },
        )?;
        Ok(eligible)
    }

    fn dense_scores(
        &self,
        data: &SearchData,
        query: &[f32],
        eligible: &BTreeSet<String>,
    ) -> Result<Vec<(String, f64)>> {
        let dimensions = self.manifest.vectors.dimensions as usize;
        if query.len() != dimensions || query.iter().any(|value| !value.is_finite()) {
            return Err(IndexError::Invalid("query vector"));
        }
        if self.manifest.embedding_profile.normalization == "l2" {
            let norm = query
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                .sqrt();
            if (norm - 1.0).abs() > 1.0e-4 {
                return Err(IndexError::Invalid("query vector normalization"));
            }
        }
        let mut vector_file = self.vectors_file.try_clone()?;
        vector_file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::with_capacity(
            dimensions
                .checked_mul(4)
                .and_then(|bytes| bytes.checked_mul(16))
                .ok_or(IndexError::Invalid("vector buffer"))?,
            vector_file,
        );
        let mut header = [0_u8; VECTOR_HEADER_BYTES as usize];
        reader.read_exact(&mut header)?;
        let mut row = vec![0_u8; dimensions * 4];
        let mut scores = Vec::with_capacity(eligible.len());
        for document in &data.documents {
            reader.read_exact(&mut row)?;
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
            if !eligible.contains(&document.document_id) {
                continue;
            }
            scores.push((document.document_id.clone(), score));
        }
        sort_scores(&mut scores);
        Ok(scores)
    }
}

fn filter_values_match(
    event_time_ms: Option<u64>,
    relation: &str,
    filters: &SearchFilters,
) -> bool {
    (filters.relations.is_empty() || filters.relations.contains(relation))
        && filters
            .time_start_ms
            .is_none_or(|start| event_time_ms.is_some_and(|value| value >= start))
        && filters
            .time_end_ms
            .is_none_or(|end| event_time_ms.is_some_and(|value| value < end))
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
    fn scores(&self, query: &str, eligible: &BTreeSet<String>) -> Vec<(String, f64)> {
        let terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        let mut scores = self
            .documents
            .iter()
            .filter(|doc| eligible.contains(&doc.document_id))
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

fn build_lexical(documents: &[FastDocument]) -> LexicalIndex {
    let mut document_frequency = BTreeMap::<String, u64>::new();
    let mut lexical_documents = Vec::new();
    let mut total_length = 0_u64;
    for document in documents {
        let tokens = tokenize(&document.semantic_text);
        total_length += tokens.len() as u64;
        let mut terms = BTreeMap::<String, u64>::new();
        for token in tokens {
            *terms.entry(token).or_default() += 1;
        }
        for term in terms.keys() {
            *document_frequency.entry(term.clone()).or_default() += 1;
        }
        lexical_documents.push(LexicalDocument {
            document_id: document.document_id.clone(),
            length: terms.values().sum(),
            terms,
        });
    }
    LexicalIndex {
        document_count: documents.len() as u64,
        average_length: total_length as f64 / documents.len().max(1) as f64,
        document_frequency,
        documents: lexical_documents,
    }
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

fn validate_order(
    source: &SourceBinding,
    documents: &[FastDocument],
    occurrences: &[FastOccurrence],
    vectors: &[Vec<f32>],
    profile: &EmbeddingProfile,
) -> Result<()> {
    let dimensions = profile.dimensions as usize;
    let ids = documents
        .iter()
        .map(|item| &item.document_id)
        .collect::<BTreeSet<_>>();
    let mut occurrence_ids = BTreeSet::new();
    let mut occurrence_counts = BTreeMap::<&str, u64>::new();
    if decode_sha256(&source.snapshot_sha256).is_err()
        || decode_sha256(&source.mapping_sha256).is_err()
        || ids.len() != documents.len()
        || documents
            .iter()
            .any(|item| item.document_id.is_empty() || item.document_id.contains('\0'))
        || documents
            .iter()
            .enumerate()
            .any(|(index, item)| item.vector_ordinal != index as u64)
        || occurrences.iter().any(|item| {
            *occurrence_counts.entry(&item.document_id).or_default() += 1;
            item.occurrence_id.is_empty()
                || !occurrence_ids.insert(&item.occurrence_id)
                || item.event_id.is_empty()
                || item.support_ref.is_empty()
                || item.snapshot_sha256 != source.snapshot_sha256
                || item.mapping_sha256 != source.mapping_sha256
                || !ids.contains(&item.document_id)
        })
        || documents.iter().any(|item| {
            occurrence_counts
                .get(item.document_id.as_str())
                .copied()
                .unwrap_or(0)
                != item.occurrence_count
                || item.occurrence_count == 0
        })
        || vectors.iter().any(|vector| {
            vector.len() != dimensions || vector.iter().any(|value| !value.is_finite())
        })
    {
        return Err(IndexError::Invalid(
            "document/vector/occurrence association",
        ));
    }
    if profile.normalization == "l2"
        && vectors.iter().any(|vector| {
            let norm = vector
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                .sqrt();
            (norm - 1.0).abs() > 1.0e-4
        })
    {
        return Err(IndexError::Invalid("vector normalization"));
    }
    Ok(())
}

fn validate_occurrence_closure(
    file: File,
    source: &SourceBinding,
    documents: &[FastDocument],
) -> Result<()> {
    let document_counts = documents
        .iter()
        .map(|document| (document.document_id.as_str(), document.occurrence_count))
        .collect::<BTreeMap<_, _>>();
    let mut actual_counts = BTreeMap::<String, u64>::new();
    let mut occurrence_ids = BTreeSet::new();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::leaves(builder.parquet_schema(), [0, 1, 5, 6, 7, 8]);
    let reader = builder
        .with_projection(projection)
        .with_batch_size(8192)
        .build()?;
    for batch in reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let occurrence_id = text(&batch, "occurrence_id", row)?;
            let document_id = text(&batch, "document_id", row)?;
            if occurrence_id.is_empty()
                || !occurrence_ids.insert(occurrence_id)
                || !document_counts.contains_key(document_id.as_str())
                || text(&batch, "snapshot_sha256", row)? != source.snapshot_sha256
                || text(&batch, "mapping_sha256", row)? != source.mapping_sha256
                || text(&batch, "event_id", row)?.is_empty()
                || text(&batch, "support_ref", row)?.is_empty()
            {
                return Err(IndexError::Corrupt("occurrence source closure"));
            }
            *actual_counts.entry(document_id).or_default() += 1;
        }
    }
    if documents.iter().any(|document| {
        document.occurrence_count == 0
            || actual_counts
                .get(&document.document_id)
                .copied()
                .unwrap_or(0)
                != document.occurrence_count
    }) {
        return Err(IndexError::Corrupt("occurrence count closure"));
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

fn write_vectors(
    path: &Path,
    vectors: &[Vec<f32>],
    dimensions: u32,
    order_sha: &str,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&VECTOR_MAGIC)?;
    writer.write_all(&VECTOR_HEADER_BYTES.to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&[1_u8, 0_u8])?;
    writer.write_all(&(vectors.len() as u64).to_le_bytes())?;
    writer.write_all(&dimensions.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&decode_sha256(order_sha)?)?;
    for vector in vectors {
        for value in vector {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
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

fn write_documents(path: &Path, rows: &[FastDocument]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("document_id", DataType::Utf8, false),
        Field::new("document_sha256", DataType::Utf8, false),
        Field::new("document_kind", DataType::Utf8, false),
        Field::new("semantic_text", DataType::Utf8, false),
        Field::new("facets_json", DataType::Utf8, false),
        Field::new("relations_json", DataType::Utf8, false),
        Field::new("occurrence_count", DataType::UInt64, false),
        Field::new("vector_ordinal", DataType::UInt64, false),
    ]));
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
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(parquet_properties()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_occurrences(path: &Path, rows: &[FastOccurrence]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("occurrence_id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("event_time_ms", DataType::UInt64, true),
        Field::new("relation", DataType::Utf8, false),
        Field::new("exact_attributes_json", DataType::Utf8, false),
        Field::new("snapshot_sha256", DataType::Utf8, false),
        Field::new("mapping_sha256", DataType::Utf8, false),
        Field::new("event_id", DataType::Utf8, false),
        Field::new("support_ref", DataType::Utf8, false),
    ]));
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
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(parquet_properties()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn read_documents(file: File) -> Result<Vec<FastDocument>> {
    read_parquet(file, |batch, row| {
        Ok(FastDocument {
            document_id: text(batch, "document_id", row)?,
            document_sha256: text(batch, "document_sha256", row)?,
            document_kind: text(batch, "document_kind", row)?,
            semantic_text: text(batch, "semantic_text", row)?,
            facets_json: text(batch, "facets_json", row)?,
            relations_json: text(batch, "relations_json", row)?,
            occurrence_count: number(batch, "occurrence_count", row)?,
            vector_ordinal: number(batch, "vector_ordinal", row)?,
        })
    })
}
fn visit_filter_rows<F: FnMut(String, Option<u64>, String)>(
    file: File,
    mut visit: F,
) -> Result<()> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::leaves(builder.parquet_schema(), [1, 2, 3]);
    let reader = builder
        .with_projection(projection)
        .with_batch_size(8192)
        .build()?;
    for batch in reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            visit(
                text(&batch, "document_id", row)?,
                optional_number(&batch, "event_time_ms", row)?,
                text(&batch, "relation", row)?,
            );
        }
    }
    Ok(())
}

fn read_matching_occurrences(
    file: File,
    document_ids: &BTreeSet<String>,
    filters: &SearchFilters,
) -> Result<BTreeMap<String, MatchedOccurrences>> {
    let mut matched = BTreeMap::<String, MatchedOccurrences>::new();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let projection = ProjectionMask::leaves(builder.parquet_schema(), [1, 2, 3, 5, 6, 7, 8]);
    let reader = builder
        .with_projection(projection)
        .with_batch_size(8192)
        .build()?;
    for batch in reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let document_id = text(&batch, "document_id", row)?;
            let event_time_ms = optional_number(&batch, "event_time_ms", row)?;
            let relation = text(&batch, "relation", row)?;
            if document_ids.contains(&document_id)
                && filter_values_match(event_time_ms, &relation, filters)
            {
                let entry = matched.entry(document_id).or_default();
                entry.eligible_count += 1;
                if entry.rows.len() < MAX_RETURNED_OCCURRENCES_PER_HIT {
                    entry.rows.push(EvidenceOccurrence {
                        event_time_ms,
                        relation,
                        snapshot_sha256: text(&batch, "snapshot_sha256", row)?,
                        mapping_sha256: text(&batch, "mapping_sha256", row)?,
                        event_id: text(&batch, "event_id", row)?,
                        support_ref: text(&batch, "support_ref", row)?,
                    });
                }
            }
        }
    }
    Ok(matched)
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
        assert!(out.join("build-report.json").is_file());
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
    fn open_is_metadata_only_and_search_loads_corpus_objects() {
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
        let index = FastIndex::open(&out).expect("metadata-scale open succeeds");
        assert!(
            index
                .search(
                    SearchMode::Lexical,
                    "logging",
                    None,
                    &SearchFilters::default(),
                    1,
                )
                .is_err()
        );
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

    #[test]
    fn rejects_foreign_occurrence_source_and_noncontiguous_vector_ordinals() {
        let (_root, out) = built_index();
        let path = out.join("occurrences.parquet");
        let mut rows = read_parquet(File::open(&path).unwrap(), |batch, row| {
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
        })
        .unwrap();
        rows[0].snapshot_sha256 = "f".repeat(64);
        fs::remove_file(&path).unwrap();
        write_occurrences(&path, &rows).unwrap();
        let index = FastIndex::open(&out).unwrap();
        assert!(matches!(
            index.search(
                SearchMode::Lexical,
                "logging",
                None,
                &SearchFilters::default(),
                1
            ),
            Err(IndexError::Corrupt("occurrence source closure"))
        ));

        let (_root, out) = built_index();
        let path = out.join("documents.parquet");
        let mut documents = read_documents(File::open(&path).unwrap()).unwrap();
        documents[0].vector_ordinal = 10;
        documents[1].vector_ordinal = 11;
        fs::remove_file(&path).unwrap();
        write_documents(&path, &documents).unwrap();
        let index = FastIndex::open(&out).unwrap();
        assert!(matches!(
            index.search(
                SearchMode::Lexical,
                "logging",
                None,
                &SearchFilters::default(),
                1
            ),
            Err(IndexError::Corrupt("manifest row/order binding"))
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
        let index = FastIndex::open(&out).unwrap();
        assert!(matches!(
            index.search(
                SearchMode::Dense,
                "",
                Some(&[1.0, 0.0]),
                &SearchFilters::default(),
                1
            ),
            Err(IndexError::Corrupt("non-finite vector value"))
        ));

        let (_root, out) = built_index();
        let path = out.join("lexical/index.json");
        let mut lexical: LexicalIndex = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        lexical.documents[0].document_id = "foreign".into();
        fs::write(&path, serde_json::to_vec(&lexical).unwrap()).unwrap();
        let index = FastIndex::open(&out).unwrap();
        assert!(matches!(
            index.search(
                SearchMode::Lexical,
                "logging",
                None,
                &SearchFilters::default(),
                1
            ),
            Err(IndexError::Corrupt("lexical document association"))
        ));
    }
}
