//! Portable, content-bound contracts for per-dataset preparation and embedding.

#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
};

use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest as ShaDigest, Sha256};
use thiserror::Error;

pub const PREPARED_CORPUS_SCHEMA: &str = "livefire.rag.prepared-corpus/1";
pub const EMBEDDING_PLAN_SCHEMA: &str = "livefire.rag.embedding-plan/1";
pub const EMBEDDING_PLAN_V2_SCHEMA: &str = "livefire.rag.embedding-plan/2";
pub const VECTOR_RECEIPT_SCHEMA: &str = "livefire.rag.vector-result-receipt/1";
pub const DERIVED_VECTOR_RECEIPT_SCHEMA: &str = "livefire.rag.vector-result-receipt/2";
pub const RESULT_SET_SCHEMA: &str = "livefire.rag.embedding-result-set/1";
pub const TEST_RESULT_SET_SCHEMA: &str = "livefire.rag.embedding-result-set/2";
pub const DERIVED_RESULT_SET_SCHEMA: &str = "livefire.rag.embedding-result-set/3";
pub const TEST_VECTOR_EXECUTOR_ID: &str =
    "livefire.rag.embedding-executor.deterministic-test-vectors";
pub const DERIVED_VECTOR_EXECUTOR_ID: &str = "livefire.rag.embedding-executor.prefix-l2-derivation";
pub const PREFIX_L2_DERIVATION_POLICY: &str = "prefix_then_l2_normalize_v1";
pub const BENCHMARK_SELECTION_SCHEMA: &str = "livefire.rag.benchmark-selection/1";
pub const DATASET_CATALOGUE_SCHEMA: &str = "livefire.rag.dataset-catalogue/1";
pub const RUNPOD_EMBEDDING_BUNDLE_SCHEMA: &str = "livefire.rag.runpod-embedding-bundle/1";
pub const RUNPOD_WORKER_ATTEMPT_SCHEMA: &str = "livefire.rag.runpod-worker-attempt/1";
pub const RUNPOD_RUN_REPORT_SCHEMA: &str = "livefire.rag.runpod-run-report/1";
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
static ATOMIC_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

mod benchmark_selection;
mod cloud;
mod dataset_catalogue;
mod query_vector_set;
mod runpod_conformance;
mod runpod_storage_challenge;
mod token_plan;

pub use benchmark_selection::{
    BenchmarkLengthStratum, BenchmarkPreparedCorpusIdentity, BenchmarkPublishedCorpus,
    BenchmarkSelectionCandidate, BenchmarkSelectionManifest, BenchmarkSelectionPolicy,
    BenchmarkSelectionRow, BenchmarkSelectionTarget, BenchmarkStratumQuota, BenchmarkTargetQuota,
    STANDARD_BENCHMARK_SIZES, bind_benchmark_prepared_corpus, build_benchmark_selection_manifest,
    select_benchmark_documents,
};
pub use cloud::{
    CloudComponentArtifact, CloudObjectRef, CloudPreparedDocumentArtifact,
    RunpodAcceleratorIdentity, RunpodBundleArtifacts, RunpodEmbeddingBundle,
    RunpodExecutionIdentity, RunpodExpectedQueryVectorOutput, RunpodExpectedTaskOutput,
    RunpodMachineIdentity, RunpodQueryVectorSetOutput, RunpodRunReport, RunpodSelectedAttempt,
    RunpodTaskOutput, RunpodWorkerAssignment, RunpodWorkerAttemptMarker, WorkerAttemptOutcome,
    build_runpod_embedding_bundle, build_runpod_run_report,
};
pub use dataset_catalogue::{
    CatalogueArtifactRef, CatalogueDatasetEntry, CatalogueMode, DatasetCatalogue,
    RelationOverlapAllowance, validate_dataset_pipeline_binding,
};
pub use query_vector_set::{
    PackedQueryVectors, QUERY_VECTOR_SET_MANIFEST, QUERY_VECTOR_SET_PLAN, QUERY_VECTOR_SET_SCHEMA,
    QUERY_VECTOR_SET_VECTORS, QueryVectorArtifact, QueryVectorExecutionBinding,
    QueryVectorPlanQuery, QueryVectorRow, QueryVectorSetInput, QueryVectorSetManifest,
    SealedQueryVectorSet, query_vector_plan_queries, write_query_vector_set,
};
pub use runpod_conformance::{
    EMBEDDING_POLICY_V3_CONFORMANCE_MODE, EmbeddingPolicyV3ConformanceFields,
    RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA, RUNPOD_TEI_CONFORMANCE_CANDIDATE_SCHEMA,
    RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA, RunpodExecutorImageBuildReceipt,
    RunpodTeiAcceleratorPolicy, RunpodTeiArtifactObject, RunpodTeiBoundArtifact,
    RunpodTeiBoundBuildReceipt, RunpodTeiConformanceCandidate, RunpodTeiConformanceFixture,
    RunpodTeiConformanceOutcome, RunpodTeiConformanceResult, RunpodTeiExecutionIdentity,
    RunpodTeiImageIdentity, RunpodTeiLoadPolicy, RunpodTeiMachineIdentity,
    RunpodTeiNormalizedOutput, RunpodTeiTokenizerIdentity, model_artifact_set_digest,
    seal_embedding_policy_v3_conformance,
};
pub use runpod_storage_challenge::{
    RUNPOD_STORAGE_CHALLENGE_RESPONSE_SCHEMA, RunpodStorageChallengeResponse,
};
pub use token_plan::{
    DOCUMENT_TOKEN_COUNTS_PATH, DocumentTokenCountsObject, EmbeddingInputSliceV2, EmbeddingPlanV2,
    EmbeddingTaskV2, ExactTokenizer, ExecutableTokenizerRef, TokenBalanceOptions, TokenStatistics,
    TokenizerArtifactFormat, build_token_balanced_plan, build_token_balanced_plan_with_counts,
    decode_document_token_counts, derive_embedding_plan_v2, document_token_counts_digest,
    encode_document_token_counts, format_document_input_exact, token_statistics,
};

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("invalid pipeline contract: {0}")]
    Invalid(&'static str),
    #[error("pipeline I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("pipeline JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("pipeline tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("pipeline Arrow failed: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("pipeline Parquet failed: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

/// A canonical lowercase SHA-256 value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Digest(String);

impl Digest {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(PipelineError::Invalid("SHA-256 digest"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Digest {
    type Err = PipelineError;
    fn from_str(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// A slash-separated path that cannot escape its artifact root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct SafeRelativePath(String);

impl SafeRelativePath {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty()
            || value.contains('\\')
            || value.contains('\0')
            || value.contains(':')
            || value
                .split('/')
                .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
        {
            return Err(PipelineError::Invalid("safe relative path"));
        }
        let path = Path::new(&value);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(PipelineError::Invalid("safe relative path"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn join_to(&self, root: &Path) -> PathBuf {
        root.join(&self.0)
    }
}

impl fmt::Display for SafeRelativePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SafeRelativePath {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentRef {
    pub id: String,
    pub version: String,
    pub sha256: Digest,
}

impl ComponentRef {
    pub fn validate(&self) -> Result<()> {
        require_text(&self.id)?;
        require_text(&self.version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetIdentity {
    pub id: String,
    pub version: String,
    pub source_snapshot: ComponentRef,
    pub mapping: ComponentRef,
    /// Additional source-admission identities required to interpret this
    /// snapshot. Older prepared corpora omit the field and retain identical
    /// serialized bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_admission: Vec<ComponentRef>,
    pub included_relations: Vec<String>,
    pub excluded_relations: Vec<String>,
    pub structured_only_relations: Vec<String>,
}

impl DatasetIdentity {
    pub fn validate(&self) -> Result<()> {
        require_text(&self.id)?;
        require_text(&self.version)?;
        self.source_snapshot.validate()?;
        self.mapping.validate()?;
        for component in &self.source_admission {
            component.validate()?;
        }
        if !self
            .source_admission
            .windows(2)
            .all(|pair| pair[0].id < pair[1].id)
        {
            return Err(PipelineError::Invalid(
                "source admission components must be sorted and unique by id",
            ));
        }
        let mut all = BTreeSet::new();
        for relations in [
            &self.included_relations,
            &self.excluded_relations,
            &self.structured_only_relations,
        ] {
            require_sorted_unique(relations)?;
            for relation in relations {
                if !all.insert(relation) {
                    return Err(PipelineError::Invalid("relation scopes overlap"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    Activity,
    State,
    Detection,
}

impl DocumentKind {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Activity => "activity",
            Self::State => "state",
            Self::Detection => "detection",
        }
    }
}

/// Fields accepted by the existing fast-index document type, without coupling
/// this portable crate to that implementation crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDocumentFields {
    pub document_id: String,
    pub document_sha256: String,
    pub document_kind: String,
    pub semantic_text: String,
    pub facets_json: String,
    pub relations_json: String,
    pub occurrence_count: u64,
    pub vector_ordinal: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedDocumentRow {
    pub document_ordinal: u64,
    pub document_id: String,
    pub document_sha256: Digest,
    pub semantic_text_sha256: Digest,
    pub semantic_text: String,
    pub document_kind: DocumentKind,
    pub primary_relation: String,
    pub facets_json: String,
    pub relations_json: String,
    pub occurrence_count: u64,
}

impl PreparedDocumentRow {
    pub fn validate(&self) -> Result<()> {
        require_safe_u64(self.document_ordinal)?;
        require_safe_u64(self.occurrence_count)?;
        require_text(&self.document_id)?;
        require_text(&self.semantic_text)?;
        require_text(&self.primary_relation)?;
        validate_canonical_json(&self.facets_json)?;
        validate_canonical_json(&self.relations_json)?;
        if digest_bytes(self.semantic_text.as_bytes()) != self.semantic_text_sha256 {
            return Err(PipelineError::Invalid("semantic text digest"));
        }
        if self.occurrence_count == 0 {
            return Err(PipelineError::Invalid("document occurrence count"));
        }
        Ok(())
    }

    #[must_use]
    pub fn into_index_fields(self) -> IndexDocumentFields {
        IndexDocumentFields {
            vector_ordinal: self.document_ordinal,
            document_id: self.document_id,
            document_sha256: self.document_sha256.to_string(),
            document_kind: self.document_kind.as_str().to_owned(),
            semantic_text: self.semantic_text,
            facets_json: self.facets_json,
            relations_json: self.relations_json,
            occurrence_count: self.occurrence_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedOccurrenceRow {
    pub occurrence_id: String,
    pub document_id: String,
    pub event_time_ms: Option<u64>,
    pub relation: String,
    pub source_row_ordinal: u64,
    pub exact_attributes_json: String,
    pub snapshot_sha256: Digest,
    pub mapping_sha256: Digest,
    pub event_id: String,
    pub support_ref: String,
}

impl PreparedOccurrenceRow {
    pub fn validate(&self) -> Result<()> {
        require_safe_u64(self.source_row_ordinal)?;
        if let Some(value) = self.event_time_ms {
            require_safe_u64(value)?;
        }
        for text in [
            &self.occurrence_id,
            &self.document_id,
            &self.relation,
            &self.event_id,
            &self.support_ref,
        ] {
            require_text(text)?;
        }
        validate_canonical_json(&self.exact_attributes_json)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectEntry {
    pub path: SafeRelativePath,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: Digest,
    pub logical_order_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedDocumentObject {
    #[serde(flatten)]
    pub object: ObjectEntry,
    pub ordinal: u32,
    pub first_document_id: String,
    pub last_document_id: String,
    pub embedding_input_order_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedOccurrenceObject {
    #[serde(flatten)]
    pub object: ObjectEntry,
    pub ordinal: u32,
    pub relation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationAccounting {
    pub source_rows: u64,
    pub searchable_occurrences: u64,
    pub selected_occurrences: u64,
    pub excluded_rows: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedCorpusManifest {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub dataset: DatasetIdentity,
    pub projection_policy: ComponentRef,
    pub document_schema: ComponentRef,
    pub occurrence_schema: ComponentRef,
    pub preparation_implementation: ComponentRef,
    pub document_count: u64,
    pub occurrence_count: u64,
    pub document_order_sha256: Digest,
    pub embedding_input_order_sha256: Digest,
    pub documents: Vec<PreparedDocumentObject>,
    pub occurrences: Vec<PreparedOccurrenceObject>,
    pub relation_accounting: BTreeMap<String, RelationAccounting>,
}

impl PreparedCorpusManifest {
    pub fn validate(&self) -> Result<()> {
        require_safe_u64(self.document_count)?;
        require_safe_u64(self.occurrence_count)?;
        if self.schema_version != PREPARED_CORPUS_SCHEMA {
            return Err(PipelineError::Invalid("prepared manifest schema"));
        }
        self.dataset.validate()?;
        self.projection_policy.validate()?;
        self.document_schema.validate()?;
        self.occurrence_schema.validate()?;
        self.preparation_implementation.validate()?;
        validate_objects(&self.documents, self.document_count)?;
        validate_occurrence_objects(
            &self.occurrences,
            self.occurrence_count,
            &self.dataset.included_relations,
        )?;
        let mut all_paths = BTreeSet::new();
        if self
            .documents
            .iter()
            .map(|object| &object.object.path)
            .chain(self.occurrences.iter().map(|object| &object.object.path))
            .any(|path| !all_paths.insert(path))
        {
            return Err(PipelineError::Invalid("duplicate prepared object path"));
        }
        validate_relation_accounting(self)?;
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("prepared component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProfileRef {
    #[serde(flatten)]
    pub component: ComponentRef,
    pub model_artifact: ComponentRef,
    pub tokenizer: ComponentRef,
    pub maximum_input_tokens: u32,
    pub pooling: String,
    pub normalization: String,
    pub dimensions: u32,
    pub dtype: String,
    pub document_format: String,
}

impl EmbeddingProfileRef {
    pub fn validate(&self) -> Result<()> {
        self.component.validate()?;
        self.model_artifact.validate()?;
        self.tokenizer.validate()?;
        if self.maximum_input_tokens == 0 || self.dimensions == 0 {
            return Err(PipelineError::Invalid("embedding profile dimensions"));
        }
        if !matches!(self.normalization.as_str(), "l2" | "none") || self.dtype != "f32le" {
            return Err(PipelineError::Invalid("embedding profile format"));
        }
        require_text(&self.pooling)?;
        require_text(&self.document_format)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInputSlice {
    pub path: SafeRelativePath,
    pub object_sha256: Digest,
    pub row_offset: u64,
    pub rows: u64,
    pub embedding_input_order_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingTask {
    pub task_id: String,
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub input_slices: Vec<EmbeddingInputSlice>,
    pub embedding_input_order_sha256: Digest,
    pub result_path: SafeRelativePath,
    pub receipt_path: SafeRelativePath,
}

impl EmbeddingTask {
    #[must_use]
    pub fn row_count(&self) -> u64 {
        self.ordinal_end.saturating_sub(self.ordinal_start)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingPlan {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub dataset: DatasetIdentity,
    pub embedding_profile: EmbeddingProfileRef,
    pub document_count: u64,
    pub document_order_sha256: Digest,
    pub tasks: Vec<EmbeddingTask>,
}

impl EmbeddingPlan {
    pub fn validate(&self) -> Result<()> {
        require_safe_u64(self.document_count)?;
        if self.schema_version != EMBEDDING_PLAN_SCHEMA {
            return Err(PipelineError::Invalid("embedding plan schema"));
        }
        self.dataset.validate()?;
        self.embedding_profile.validate()?;
        validate_tasks(&self.tasks, self.document_count)?;
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("embedding plan component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }

    #[must_use]
    pub fn task(&self, id: &str) -> Option<&EmbeddingTask> {
        self.tasks.iter().find(|task| task.task_id == id)
    }

    /// Validate all plan bindings and task order against the prepared corpus
    /// manifest and the exact rows that will be sent to the model.
    pub fn validate_against_prepared(
        &self,
        prepared: &PreparedCorpusManifest,
        documents: &[PreparedDocumentRow],
    ) -> Result<()> {
        self.validate_manifest_binding(prepared)?;
        validate_prepared_documents(prepared, documents)?;
        let mut object_starts = BTreeMap::new();
        let mut object_start = 0_u64;
        for object in &prepared.documents {
            object_starts.insert(object.object.path.clone(), (object_start, object));
            object_start = object_start
                .checked_add(object.object.rows)
                .ok_or(PipelineError::Invalid("document object ordinal"))?;
        }
        for task in &self.tasks {
            let start = usize::try_from(task.ordinal_start)
                .map_err(|_| PipelineError::Invalid("task ordinal"))?;
            let end = usize::try_from(task.ordinal_end)
                .map_err(|_| PipelineError::Invalid("task ordinal"))?;
            let task_rows = documents
                .get(start..end)
                .ok_or(PipelineError::Invalid("task row range"))?;
            if embedding_input_order_digest(task_rows) != task.embedding_input_order_sha256 {
                return Err(PipelineError::Invalid("task input order"));
            }
            let mut next_slice_ordinal = task.ordinal_start;
            for slice in &task.input_slices {
                let (object_start, object) = object_starts
                    .get(&slice.path)
                    .ok_or(PipelineError::Invalid("task input object"))?;
                let slice_end = slice
                    .row_offset
                    .checked_add(slice.rows)
                    .ok_or(PipelineError::Invalid("task input slice binding"))?;
                let global_start = object_start
                    .checked_add(slice.row_offset)
                    .ok_or(PipelineError::Invalid("task input slice binding"))?;
                let global_end = object_start
                    .checked_add(slice_end)
                    .ok_or(PipelineError::Invalid("task input slice binding"))?;
                let global_start_usize = usize::try_from(global_start)
                    .map_err(|_| PipelineError::Invalid("task input slice binding"))?;
                let global_end_usize = usize::try_from(global_end)
                    .map_err(|_| PipelineError::Invalid("task input slice binding"))?;
                let slice_rows = documents
                    .get(global_start_usize..global_end_usize)
                    .ok_or(PipelineError::Invalid("task input slice binding"))?;
                if object.object.sha256 != slice.object_sha256
                    || slice_end > object.object.rows
                    || global_start != next_slice_ordinal
                    || global_end > task.ordinal_end
                    || embedding_input_order_digest(slice_rows)
                        != slice.embedding_input_order_sha256
                {
                    return Err(PipelineError::Invalid("task input slice binding"));
                }
                next_slice_ordinal = global_end;
            }
            if next_slice_ordinal != task.ordinal_end {
                return Err(PipelineError::Invalid("task input slice coverage"));
            }
        }
        Ok(())
    }

    /// Validate bindings that can be checked without materializing prepared
    /// rows. Streaming consumers must additionally verify each object's file
    /// digest and row/order metadata while reading it.
    pub fn validate_manifest_binding(&self, prepared: &PreparedCorpusManifest) -> Result<()> {
        self.validate()?;
        prepared.validate()?;
        if self.prepared_corpus_sha256 != prepared.component_sha256
            || self.dataset != prepared.dataset
            || self.document_count != prepared.document_count
            || self.document_order_sha256 != prepared.document_order_sha256
        {
            return Err(PipelineError::Invalid("plan prepared-corpus binding"));
        }
        validate_task_slices_against_manifest(&self.tasks, &prepared.documents)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorObject {
    pub path: SafeRelativePath,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: Digest,
    pub dimensions: u32,
    pub dtype: String,
    pub embedding_input_order_sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorReceipt {
    pub implementation: ComponentRef,
    pub runtime: ComponentRef,
    pub returned_model: String,
    pub requests: u64,
    pub retries: u64,
    /// Conservative UTF-8 byte upper bound used for the first LM Studio
    /// executor. Exact tokenizer counts require the later tokenizer worker.
    pub input_bytes_upper_bound: u64,
    pub elapsed_ms: u64,
    pub conformance_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedVectorBinding {
    pub parent_embedding_profile_sha256: Digest,
    pub parent_result_set_sha256: Digest,
    pub parent_receipt_sha256: Digest,
    pub parent_vector_sha256: Digest,
    pub parent_dimensions: u32,
    pub transformation: String,
}

impl DerivedVectorBinding {
    pub fn validate(&self, target_dimensions: u32) -> Result<()> {
        if self.parent_dimensions <= target_dimensions
            || self.transformation != PREFIX_L2_DERIVATION_POLICY
        {
            return Err(PipelineError::Invalid("derived vector binding"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorResultReceipt {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub plan_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub embedding_profile_sha256: Digest,
    pub task_id: String,
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub embedding_input_order_sha256: Digest,
    pub vector: VectorObject,
    pub executor: ExecutorReceipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<DerivedVectorBinding>,
    pub finite_values_validated: bool,
    pub normalization_validated: bool,
}

impl VectorResultReceipt {
    #[must_use]
    pub fn test_only(&self) -> bool {
        self.executor.implementation.id == TEST_VECTOR_EXECUTOR_ID
    }

    #[must_use]
    pub fn derived(&self) -> bool {
        self.derivation.is_some()
    }

    pub fn validate_against(&self, plan: &EmbeddingPlan) -> Result<()> {
        for value in [
            self.ordinal_start,
            self.ordinal_end,
            self.vector.rows,
            self.vector.bytes,
            self.executor.requests,
            self.executor.retries,
            self.executor.input_bytes_upper_bound,
            self.executor.elapsed_ms,
        ] {
            require_safe_u64(value)?;
        }
        if !matches!(
            (self.schema_version.as_str(), self.derived()),
            (VECTOR_RECEIPT_SCHEMA, false) | (DERIVED_VECTOR_RECEIPT_SCHEMA, true)
        ) {
            return Err(PipelineError::Invalid("vector receipt schema"));
        }
        let task = plan
            .task(&self.task_id)
            .ok_or(PipelineError::Invalid("receipt task"))?;
        if self.plan_sha256 != plan.component_sha256
            || self.prepared_corpus_sha256 != plan.prepared_corpus_sha256
            || self.embedding_profile_sha256 != plan.embedding_profile.component.sha256
            || self.ordinal_start != task.ordinal_start
            || self.ordinal_end != task.ordinal_end
            || self.embedding_input_order_sha256 != task.embedding_input_order_sha256
            || self.vector.path != task.result_path
            || self.vector.rows != task.row_count()
            || self.vector.dimensions != plan.embedding_profile.dimensions
            || self.vector.dtype != plan.embedding_profile.dtype
            || self.vector.embedding_input_order_sha256 != task.embedding_input_order_sha256
        {
            return Err(PipelineError::Invalid("receipt plan binding"));
        }
        if !self.finite_values_validated || !self.normalization_validated {
            return Err(PipelineError::Invalid("receipt validation flags"));
        }
        self.executor.implementation.validate()?;
        self.executor.runtime.validate()?;
        require_text(&self.executor.returned_model)?;
        if let Some(derivation) = &self.derivation {
            derivation.validate(self.vector.dimensions)?;
        }
        let invalid_executor = if self.test_only() {
            self.executor.requests != 0
                || self.executor.retries != 0
                || self.executor.conformance_passed
        } else if self.derived() {
            self.executor.implementation.id != DERIVED_VECTOR_EXECUTOR_ID
                || self.executor.requests != 0
                || self.executor.retries != 0
                || self.executor.conformance_passed
        } else {
            !self.executor.conformance_passed
                || self.executor.retries > self.executor.requests
                || self.executor.requests == 0
        };
        if invalid_executor {
            return Err(PipelineError::Invalid("receipt executor validation"));
        }
        let expected_bytes = 64_u64
            .checked_add(
                self.vector
                    .rows
                    .checked_mul(u64::from(self.vector.dimensions))
                    .and_then(|values| values.checked_mul(4))
                    .ok_or(PipelineError::Invalid("receipt vector byte length"))?,
            )
            .ok_or(PipelineError::Invalid("receipt vector byte length"))?;
        if self.vector.bytes != expected_bytes {
            return Err(PipelineError::Invalid("receipt vector byte length"));
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("receipt component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptEntry {
    pub task_id: String,
    pub path: SafeRelativePath,
    pub sha256: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingResultSetManifest {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub plan_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub embedding_profile_sha256: Digest,
    pub document_count: u64,
    pub document_order_sha256: Digest,
    pub receipts: Vec<ReceiptEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<DerivedResultSetBinding>,
    /// Synthetic vectors are useful for checking the full artifact chain but
    /// must never be admitted as model-produced search data.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub test_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedResultSetBinding {
    pub parent_embedding_profile_sha256: Digest,
    pub parent_result_set_sha256: Digest,
    pub parent_dimensions: u32,
    pub transformation: String,
}

impl DerivedResultSetBinding {
    pub fn validate(&self, target_dimensions: u32) -> Result<()> {
        if self.parent_dimensions <= target_dimensions
            || self.transformation != PREFIX_L2_DERIVATION_POLICY
        {
            return Err(PipelineError::Invalid("derived result-set binding"));
        }
        Ok(())
    }
}

impl EmbeddingResultSetManifest {
    pub fn validate(&self, plan: &EmbeddingPlan, loaded: &[VectorResultReceipt]) -> Result<()> {
        require_safe_u64(self.document_count)?;
        if !matches!(
            (
                self.schema_version.as_str(),
                self.test_only,
                self.derivation.is_some()
            ),
            (RESULT_SET_SCHEMA, false, false)
                | (TEST_RESULT_SET_SCHEMA, true, false)
                | (DERIVED_RESULT_SET_SCHEMA, false, true)
        ) || self.plan_sha256 != plan.component_sha256
            || self.prepared_corpus_sha256 != plan.prepared_corpus_sha256
            || self.embedding_profile_sha256 != plan.embedding_profile.component.sha256
            || self.document_count != plan.document_count
            || self.document_order_sha256 != plan.document_order_sha256
        {
            return Err(PipelineError::Invalid("result set plan binding"));
        }
        if self.receipts.len() != plan.tasks.len() || loaded.len() != plan.tasks.len() {
            return Err(PipelineError::Invalid("result set coverage"));
        }
        if let Some(derivation) = &self.derivation {
            derivation.validate(plan.embedding_profile.dimensions)?;
        }
        let entries: BTreeMap<_, _> = self
            .receipts
            .iter()
            .map(|entry| (entry.task_id.as_str(), entry))
            .collect();
        if entries.len() != self.receipts.len() {
            return Err(PipelineError::Invalid("duplicate result task"));
        }
        let values: BTreeMap<_, _> = loaded
            .iter()
            .map(|receipt| (receipt.task_id.as_str(), receipt))
            .collect();
        if values.len() != loaded.len() {
            return Err(PipelineError::Invalid("duplicate receipt task"));
        }
        for task in &plan.tasks {
            let entry = entries
                .get(task.task_id.as_str())
                .ok_or(PipelineError::Invalid("missing result task"))?;
            let receipt = values
                .get(task.task_id.as_str())
                .ok_or(PipelineError::Invalid("missing receipt"))?;
            receipt.validate_against(plan)?;
            if receipt.test_only() != self.test_only {
                return Err(PipelineError::Invalid("result set test-only binding"));
            }
            if receipt.derivation.as_ref().map(|value| {
                (
                    &value.parent_embedding_profile_sha256,
                    &value.parent_result_set_sha256,
                    value.parent_dimensions,
                    value.transformation.as_str(),
                )
            }) != self.derivation.as_ref().map(|value| {
                (
                    &value.parent_embedding_profile_sha256,
                    &value.parent_result_set_sha256,
                    value.parent_dimensions,
                    value.transformation.as_str(),
                )
            }) {
                return Err(PipelineError::Invalid("result derivation binding"));
            }
            if entry.path != task.receipt_path || entry.sha256 != receipt.component_sha256 {
                return Err(PipelineError::Invalid("result receipt binding"));
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("result set component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }
}

pub fn digest_bytes(bytes: &[u8]) -> Digest {
    Digest(format!("{:x}", Sha256::digest(bytes)))
}

pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    serde_json_canonicalizer::to_vec(&value).map_err(PipelineError::Json)
}

pub fn canonical_digest<T: Serialize>(value: &T) -> Result<Digest> {
    Ok(digest_bytes(&canonical_json_bytes(value)?))
}

/// Compute a component digest after omitting its top-level self digest, matching
/// the repository's RFC 8785 component identity convention.
pub fn component_digest<T: Serialize>(value: &T) -> Result<Digest> {
    let mut value = serde_json::to_value(value)?;
    let object = value
        .as_object_mut()
        .ok_or(PipelineError::Invalid("component object"))?;
    if object.remove("component_sha256").is_none() {
        return Err(PipelineError::Invalid("component digest field"));
    }
    canonical_digest(&value)
}

pub fn document_order_digest<'a>(ids: impl IntoIterator<Item = &'a str>) -> Digest {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.as_bytes());
        hasher.update([0]);
    }
    Digest(format!("{:x}", hasher.finalize()))
}

pub fn embedding_input_order_digest<'a>(
    rows: impl IntoIterator<Item = &'a PreparedDocumentRow>,
) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"livefire.rag.embedding-input-order/1\0");
    for row in rows {
        for field in [
            row.document_id.as_str(),
            row.document_sha256.as_str(),
            row.semantic_text_sha256.as_str(),
        ] {
            hasher.update(field.as_bytes());
            hasher.update([0]);
        }
    }
    Digest(format!("{:x}", hasher.finalize()))
}

/// Validate complete, canonical prepared document coverage against a manifest.
pub fn validate_prepared_documents(
    manifest: &PreparedCorpusManifest,
    rows: &[PreparedDocumentRow],
) -> Result<()> {
    if u64::try_from(rows.len()).ok() != Some(manifest.document_count) {
        return Err(PipelineError::Invalid("prepared document count"));
    }
    for (ordinal, row) in rows.iter().enumerate() {
        row.validate()?;
        if usize::try_from(row.document_ordinal).ok() != Some(ordinal)
            || ordinal > 0 && rows[ordinal - 1].document_id >= row.document_id
        {
            return Err(PipelineError::Invalid("prepared document order"));
        }
    }
    if document_order_digest(rows.iter().map(|row| row.document_id.as_str()))
        != manifest.document_order_sha256
        || embedding_input_order_digest(rows) != manifest.embedding_input_order_sha256
    {
        return Err(PipelineError::Invalid("prepared document order digest"));
    }
    let mut offset = 0_usize;
    for object in &manifest.documents {
        let count = usize::try_from(object.object.rows)
            .map_err(|_| PipelineError::Invalid("document object rows"))?;
        let end = offset
            .checked_add(count)
            .ok_or(PipelineError::Invalid("document object rows"))?;
        let object_rows = rows
            .get(offset..end)
            .ok_or(PipelineError::Invalid("document object rows"))?;
        let first = object_rows
            .first()
            .ok_or(PipelineError::Invalid("document object rows"))?;
        let last = object_rows
            .last()
            .ok_or(PipelineError::Invalid("document object rows"))?;
        if first.document_id != object.first_document_id
            || last.document_id != object.last_document_id
            || canonical_digest(&object_rows)? != object.object.logical_order_sha256
            || embedding_input_order_digest(object_rows) != object.embedding_input_order_sha256
        {
            return Err(PipelineError::Invalid("document object metadata"));
        }
        offset = end;
    }
    if offset != rows.len() {
        return Err(PipelineError::Invalid("document object coverage"));
    }
    Ok(())
}

/// Validate occurrence source bindings, closure, and per-document counts.
pub fn validate_prepared_occurrences(
    manifest: &PreparedCorpusManifest,
    documents: &[PreparedDocumentRow],
    occurrences: &[PreparedOccurrenceRow],
) -> Result<()> {
    if u64::try_from(occurrences.len()).ok() != Some(manifest.occurrence_count) {
        return Err(PipelineError::Invalid("prepared occurrence count"));
    }
    let mut counts = BTreeMap::<&str, u64>::new();
    let mut occurrence_ids = BTreeSet::new();
    for row in occurrences {
        row.validate()?;
        if row.snapshot_sha256 != manifest.dataset.source_snapshot.sha256
            || row.mapping_sha256 != manifest.dataset.mapping.sha256
            || !manifest.dataset.included_relations.contains(&row.relation)
        {
            return Err(PipelineError::Invalid("prepared occurrence source binding"));
        }
        if !occurrence_ids.insert(row.occurrence_id.as_str()) {
            return Err(PipelineError::Invalid("prepared occurrence order"));
        }
        *counts.entry(&row.document_id).or_default() += 1;
    }
    if documents.iter().any(|document| {
        counts.remove(document.document_id.as_str()) != Some(document.occurrence_count)
    }) || !counts.is_empty()
    {
        return Err(PipelineError::Invalid("document occurrence closure"));
    }
    let mut offset = 0_usize;
    for object in &manifest.occurrences {
        let count = usize::try_from(object.object.rows)
            .map_err(|_| PipelineError::Invalid("occurrence object rows"))?;
        let end = offset
            .checked_add(count)
            .ok_or(PipelineError::Invalid("occurrence object rows"))?;
        let object_rows = occurrences
            .get(offset..end)
            .ok_or(PipelineError::Invalid("occurrence object rows"))?;
        if object_rows
            .iter()
            .any(|row| row.relation != object.relation)
            || object_rows
                .windows(2)
                .any(|rows| rows[0].source_row_ordinal >= rows[1].source_row_ordinal)
            || canonical_digest(&object_rows)? != object.object.logical_order_sha256
        {
            return Err(PipelineError::Invalid("occurrence object metadata"));
        }
        offset = end;
    }
    if offset != occurrences.len() {
        return Err(PipelineError::Invalid("occurrence object coverage"));
    }
    Ok(())
}

pub fn write_canonical_json(path: &Path, value: &impl Serialize) -> Result<()> {
    atomic_write(path, &canonical_json_bytes(value)?)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Resolve an existing artifact and prove that symlink resolution did not
/// escape the supplied root.
pub fn resolve_existing_artifact(root: &Path, relative: &SafeRelativePath) -> Result<PathBuf> {
    let root = fs::canonicalize(root)?;
    let artifact = fs::canonicalize(relative.join_to(&root))?;
    if artifact == root || !artifact.starts_with(&root) {
        return Err(PipelineError::Invalid("artifact path containment"));
    }
    Ok(artifact)
}

/// Resolve and verify an immutable manifest object against its physical byte
/// length and SHA-256 digest.
pub fn validate_object_file(root: &Path, object: &ObjectEntry) -> Result<PathBuf> {
    let path = resolve_existing_artifact(root, &object.path)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() != object.bytes {
        return Err(PipelineError::Invalid("artifact object metadata"));
    }
    let mut file = File::open(&path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    if Digest(format!("{:x}", hasher.finalize())) != object.sha256 {
        return Err(PipelineError::Invalid("artifact object digest"));
    }
    Ok(path)
}

/// Resolve a not-yet-created artifact below an existing root. Existing path
/// components must be real directories rather than symlinks, preventing a
/// manifest path from redirecting a subsequent create outside the root.
pub fn resolve_output_artifact(root: &Path, relative: &SafeRelativePath) -> Result<PathBuf> {
    let root = fs::canonicalize(root)?;
    let mut candidate = root.clone();
    let components = Path::new(relative.as_str())
        .components()
        .collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(PipelineError::Invalid("artifact path containment"));
        };
        candidate.push(name);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || index + 1 < components.len() && !metadata.is_dir()
                {
                    return Err(PipelineError::Invalid("artifact path containment"));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    if candidate == root || !candidate.starts_with(&root) {
        return Err(PipelineError::Invalid("artifact path containment"));
    }
    Ok(candidate)
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or(PipelineError::Invalid("output parent"))?;
    fs::create_dir_all(parent)?;
    let process_id = std::process::id();
    let sequence = ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(".livefire-rag-atomic-{process_id}-{sequence}-"))
        .suffix(".partial")
        .tempfile_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| PipelineError::Io(error.error))?;
    Ok(())
}

/// Removes only temporary files owned by [`atomic_write`]. A killed process
/// can leave these files behind; they are never complete pipeline artifacts.
/// Callers that can run concurrently should first check the process id now
/// recorded in every new temporary file name.
pub fn remove_stale_atomic_writes(directory: &Path) -> Result<()> {
    if !directory.try_exists()? {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or(PipelineError::Invalid("atomic temporary path is not UTF-8"))?;
        if name.starts_with(".livefire-rag-atomic-") && name.ends_with(".partial") {
            if !entry.file_type()?.is_file() {
                return Err(PipelineError::Invalid(
                    "atomic temporary path is not a regular file",
                ));
            }
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

static DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A sibling staging directory that becomes visible through one rename.
#[derive(Debug)]
pub struct AtomicDirectory {
    staging: PathBuf,
    destination: PathBuf,
    published: bool,
}

impl AtomicDirectory {
    pub fn new(destination: &Path) -> Result<Self> {
        if destination.exists() {
            return Err(PipelineError::Invalid("publish destination exists"));
        }
        let parent = destination
            .parent()
            .ok_or(PipelineError::Invalid("publish parent"))?;
        fs::create_dir_all(parent)?;
        let name = destination
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or(PipelineError::Invalid("publish name"))?;
        let sequence = DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging = parent.join(format!(".{name}.partial-{}-{sequence}", std::process::id()));
        fs::create_dir(&staging)?;
        Ok(Self {
            staging,
            destination: destination.to_owned(),
            published: false,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.staging
    }

    pub fn publish(mut self) -> Result<PathBuf> {
        fs::rename(&self.staging, &self.destination)?;
        self.published = true;
        Ok(self.destination.clone())
    }

    /// Publishes one fully assembled child directory. This lets callers use
    /// tools that require a non-existent output path while keeping their final
    /// directory hidden until all companion files have been written.
    pub fn publish_child(mut self, child_name: &str) -> Result<PathBuf> {
        if child_name.is_empty()
            || child_name == "."
            || child_name == ".."
            || child_name.contains('/')
            || child_name.contains('\\')
        {
            return Err(PipelineError::Invalid("publish child name"));
        }
        let child = self.staging.join(child_name);
        if !child.is_dir() {
            return Err(PipelineError::Invalid("publish child directory"));
        }
        fs::rename(&child, &self.destination)?;
        fs::remove_dir(&self.staging)?;
        self.published = true;
        Ok(self.destination.clone())
    }
}

impl Drop for AtomicDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

fn require_text(value: &str) -> Result<()> {
    if value.is_empty() {
        Err(PipelineError::Invalid("non-empty text"))
    } else {
        Ok(())
    }
}

fn require_safe_u64(value: u64) -> Result<()> {
    if value > MAX_SAFE_JSON_INTEGER {
        Err(PipelineError::Invalid("RFC 8785 safe integer"))
    } else {
        Ok(())
    }
}

fn require_sorted_unique(values: &[String]) -> Result<()> {
    if values.iter().any(String::is_empty) || values.windows(2).any(|pair| pair[0] >= pair[1]) {
        Err(PipelineError::Invalid("sorted unique values"))
    } else {
        Ok(())
    }
}

fn validate_canonical_json(text: &str) -> Result<()> {
    let value: Value = serde_json::from_str(text)?;
    let canonical = String::from_utf8(canonical_json_bytes(&value)?)
        .map_err(|_| PipelineError::Invalid("canonical JSON UTF-8"))?;
    if canonical == text {
        Ok(())
    } else {
        Err(PipelineError::Invalid("canonical JSON text"))
    }
}

fn validate_relation_accounting(manifest: &PreparedCorpusManifest) -> Result<()> {
    let expected = manifest
        .dataset
        .included_relations
        .iter()
        .chain(manifest.dataset.excluded_relations.iter())
        .chain(manifest.dataset.structured_only_relations.iter())
        .collect::<BTreeSet<_>>();
    if manifest.relation_accounting.keys().collect::<BTreeSet<_>>() != expected {
        return Err(PipelineError::Invalid("relation accounting coverage"));
    }
    let mut selected_total = 0_u64;
    for (relation, accounting) in &manifest.relation_accounting {
        for value in [
            accounting.source_rows,
            accounting.searchable_occurrences,
            accounting.selected_occurrences,
            accounting.excluded_rows,
        ] {
            require_safe_u64(value)?;
        }
        if accounting.selected_occurrences > accounting.searchable_occurrences
            || accounting.excluded_rows > accounting.source_rows
        {
            return Err(PipelineError::Invalid("relation accounting bounds"));
        }
        if manifest.dataset.included_relations.contains(relation) {
            if accounting.selected_occurrences != accounting.searchable_occurrences
                || accounting
                    .searchable_occurrences
                    .checked_add(accounting.excluded_rows)
                    != Some(accounting.source_rows)
            {
                return Err(PipelineError::Invalid("included relation accounting"));
            }
            selected_total = selected_total
                .checked_add(accounting.selected_occurrences)
                .ok_or(PipelineError::Invalid("relation accounting total"))?;
        } else if accounting.searchable_occurrences != 0
            || accounting.selected_occurrences != 0
            || accounting.excluded_rows != accounting.source_rows
        {
            return Err(PipelineError::Invalid("excluded relation accounting"));
        }
    }
    if selected_total != manifest.occurrence_count {
        return Err(PipelineError::Invalid("relation accounting total"));
    }
    Ok(())
}

fn validate_objects(objects: &[PreparedDocumentObject], expected_rows: u64) -> Result<()> {
    let mut rows = 0_u64;
    let mut previous_last: Option<&str> = None;
    let mut paths = BTreeSet::new();
    for (index, object) in objects.iter().enumerate() {
        require_safe_u64(object.object.rows)?;
        require_safe_u64(object.object.bytes)?;
        if object.ordinal as usize != index
            || object.object.rows == 0
            || object.object.bytes == 0
            || object.first_document_id.is_empty()
            || object.last_document_id < object.first_document_id
            || previous_last.is_some_and(|last| last >= object.first_document_id.as_str())
            || !paths.insert(object.object.path.clone())
        {
            return Err(PipelineError::Invalid("document object order"));
        }
        rows = rows
            .checked_add(object.object.rows)
            .ok_or(PipelineError::Invalid("document row total"))?;
        previous_last = Some(&object.last_document_id);
    }
    if rows != expected_rows || (expected_rows > 0) != !objects.is_empty() {
        return Err(PipelineError::Invalid("document object coverage"));
    }
    Ok(())
}

fn validate_occurrence_objects(
    objects: &[PreparedOccurrenceObject],
    expected_rows: u64,
    included_relations: &[String],
) -> Result<()> {
    let mut rows = 0_u64;
    let mut paths = BTreeSet::new();
    let mut previous_relation_index = None;
    for (index, object) in objects.iter().enumerate() {
        require_safe_u64(object.object.rows)?;
        require_safe_u64(object.object.bytes)?;
        let relation_index = included_relations
            .iter()
            .enumerate()
            .find_map(|(relation_index, relation)| {
                (relation == &object.relation).then_some(relation_index)
            })
            .ok_or(PipelineError::Invalid("occurrence object relation order"))?;
        if object.ordinal as usize != index
            || object.object.rows == 0
            || object.object.bytes == 0
            || object.relation.is_empty()
            || !paths.insert(object.object.path.clone())
            || previous_relation_index.is_some_and(|previous| relation_index < previous)
        {
            return Err(PipelineError::Invalid("occurrence object order"));
        }
        rows = rows
            .checked_add(object.object.rows)
            .ok_or(PipelineError::Invalid("occurrence row total"))?;
        previous_relation_index = Some(relation_index);
    }
    if rows != expected_rows || (expected_rows > 0) != !objects.is_empty() {
        return Err(PipelineError::Invalid("occurrence object coverage"));
    }
    Ok(())
}

fn validate_tasks(tasks: &[EmbeddingTask], document_count: u64) -> Result<()> {
    let mut next = 0_u64;
    let mut ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for task in tasks {
        require_safe_u64(task.ordinal_start)?;
        require_safe_u64(task.ordinal_end)?;
        let slice_rows = task.input_slices.iter().try_fold(0_u64, |sum, slice| {
            require_safe_u64(slice.row_offset)?;
            require_safe_u64(slice.rows)?;
            if slice.rows == 0 {
                return Err(PipelineError::Invalid("empty task slice"));
            }
            sum.checked_add(slice.rows)
                .ok_or(PipelineError::Invalid("task slice total"))
        })?;
        if task.task_id.is_empty()
            || task.ordinal_start != next
            || task.ordinal_end <= task.ordinal_start
            || task.row_count() != slice_rows
            || task.input_slices.is_empty()
            || !ids.insert(&task.task_id)
            || !paths.insert(task.result_path.clone())
            || !paths.insert(task.receipt_path.clone())
        {
            return Err(PipelineError::Invalid("embedding task coverage"));
        }
        next = task.ordinal_end;
    }
    if next != document_count || (document_count > 0) != !tasks.is_empty() {
        return Err(PipelineError::Invalid("embedding plan coverage"));
    }
    Ok(())
}

fn validate_task_slices_against_manifest(
    tasks: &[EmbeddingTask],
    objects: &[PreparedDocumentObject],
) -> Result<()> {
    let mut object_starts = BTreeMap::new();
    let mut object_start = 0_u64;
    for object in objects {
        object_starts.insert(object.object.path.clone(), (object_start, object));
        object_start = object_start
            .checked_add(object.object.rows)
            .ok_or(PipelineError::Invalid("document object ordinal"))?;
    }
    for task in tasks {
        let mut next = task.ordinal_start;
        for slice in &task.input_slices {
            let (start, object) = object_starts
                .get(&slice.path)
                .ok_or(PipelineError::Invalid("task input object"))?;
            let slice_end = slice
                .row_offset
                .checked_add(slice.rows)
                .ok_or(PipelineError::Invalid("task input slice binding"))?;
            let global_start = start
                .checked_add(slice.row_offset)
                .ok_or(PipelineError::Invalid("task input slice binding"))?;
            let global_end = start
                .checked_add(slice_end)
                .ok_or(PipelineError::Invalid("task input slice binding"))?;
            if slice.object_sha256 != object.object.sha256
                || slice_end > object.object.rows
                || global_start != next
                || global_end > task.ordinal_end
            {
                return Err(PipelineError::Invalid("task input slice binding"));
            }
            next = global_end;
        }
        if next != task.ordinal_end {
            return Err(PipelineError::Invalid("task input slice coverage"));
        }
    }
    Ok(())
}

pub fn prepared_document_schema() -> Schema {
    Schema::new(vec![
        Field::new("document_ordinal", DataType::UInt64, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("document_sha256", DataType::Utf8, false),
        Field::new("semantic_text_sha256", DataType::Utf8, false),
        Field::new("semantic_text", DataType::Utf8, false),
        Field::new("document_kind", DataType::Utf8, false),
        Field::new("primary_relation", DataType::Utf8, false),
        Field::new("facets_json", DataType::Utf8, false),
        Field::new("relations_json", DataType::Utf8, false),
        Field::new("occurrence_count", DataType::UInt64, false),
    ])
}

pub fn prepared_occurrence_schema() -> Schema {
    Schema::new(vec![
        Field::new("occurrence_id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("event_time_ms", DataType::UInt64, true),
        Field::new("relation", DataType::Utf8, false),
        Field::new("source_row_ordinal", DataType::UInt64, false),
        Field::new("exact_attributes_json", DataType::Utf8, false),
        Field::new("snapshot_sha256", DataType::Utf8, false),
        Field::new("mapping_sha256", DataType::Utf8, false),
        Field::new("event_id", DataType::Utf8, false),
        Field::new("support_ref", DataType::Utf8, false),
    ])
}

pub fn write_prepared_documents(path: &Path, rows: &[PreparedDocumentRow]) -> Result<()> {
    for row in rows {
        row.validate()?;
    }
    let schema = std::sync::Arc::new(prepared_document_schema());
    let strings = |values: Vec<&str>| std::sync::Arc::new(StringArray::from(values)) as _;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.document_ordinal),
            )),
            strings(rows.iter().map(|r| r.document_id.as_str()).collect()),
            strings(rows.iter().map(|r| r.document_sha256.as_str()).collect()),
            strings(
                rows.iter()
                    .map(|r| r.semantic_text_sha256.as_str())
                    .collect(),
            ),
            strings(rows.iter().map(|r| r.semantic_text.as_str()).collect()),
            strings(rows.iter().map(|r| r.document_kind.as_str()).collect()),
            strings(rows.iter().map(|r| r.primary_relation.as_str()).collect()),
            strings(rows.iter().map(|r| r.facets_json.as_str()).collect()),
            strings(rows.iter().map(|r| r.relations_json.as_str()).collect()),
            std::sync::Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.occurrence_count),
            )),
        ],
    )?;
    write_batch(path, schema, &batch)
}

pub fn read_prepared_documents(path: &Path) -> Result<Vec<PreparedDocumentRow>> {
    let mut result = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    for batch in reader {
        let batch = batch?;
        if batch.schema().as_ref() != &prepared_document_schema() {
            return Err(PipelineError::Invalid("prepared document Parquet schema"));
        }
        let u64s = |i| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or(PipelineError::Invalid("document column"))
        };
        let strings = |i| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(PipelineError::Invalid("document column"))
        };
        let ordinals = u64s(0)?;
        let ids = strings(1)?;
        let document_hashes = strings(2)?;
        let text_hashes = strings(3)?;
        let texts = strings(4)?;
        let kinds = strings(5)?;
        let primary = strings(6)?;
        let facets = strings(7)?;
        let relations = strings(8)?;
        let counts = u64s(9)?;
        for i in 0..batch.num_rows() {
            let row = PreparedDocumentRow {
                document_ordinal: ordinals.value(i),
                document_id: ids.value(i).to_owned(),
                document_sha256: Digest::new(document_hashes.value(i))?,
                semantic_text_sha256: Digest::new(text_hashes.value(i))?,
                semantic_text: texts.value(i).to_owned(),
                document_kind: match kinds.value(i) {
                    "activity" => DocumentKind::Activity,
                    "state" => DocumentKind::State,
                    "detection" => DocumentKind::Detection,
                    _ => return Err(PipelineError::Invalid("document kind")),
                },
                primary_relation: primary.value(i).to_owned(),
                facets_json: facets.value(i).to_owned(),
                relations_json: relations.value(i).to_owned(),
                occurrence_count: counts.value(i),
            };
            row.validate()?;
            result.push(row);
        }
    }
    Ok(result)
}

pub fn write_prepared_occurrences(path: &Path, rows: &[PreparedOccurrenceRow]) -> Result<()> {
    for row in rows {
        row.validate()?;
    }
    let schema = std::sync::Arc::new(prepared_occurrence_schema());
    let strings = |values: Vec<&str>| std::sync::Arc::new(StringArray::from(values)) as _;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            strings(rows.iter().map(|row| row.occurrence_id.as_str()).collect()),
            strings(rows.iter().map(|row| row.document_id.as_str()).collect()),
            std::sync::Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.event_time_ms).collect::<Vec<_>>(),
            )),
            strings(rows.iter().map(|row| row.relation.as_str()).collect()),
            std::sync::Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|row| row.source_row_ordinal),
            )),
            strings(
                rows.iter()
                    .map(|row| row.exact_attributes_json.as_str())
                    .collect(),
            ),
            strings(
                rows.iter()
                    .map(|row| row.snapshot_sha256.as_str())
                    .collect(),
            ),
            strings(rows.iter().map(|row| row.mapping_sha256.as_str()).collect()),
            strings(rows.iter().map(|row| row.event_id.as_str()).collect()),
            strings(rows.iter().map(|row| row.support_ref.as_str()).collect()),
        ],
    )?;
    write_batch(path, schema, &batch)
}

pub fn read_prepared_occurrences(path: &Path) -> Result<Vec<PreparedOccurrenceRow>> {
    let mut result = Vec::new();
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    for batch in reader {
        let batch = batch?;
        if batch.schema().as_ref() != &prepared_occurrence_schema() {
            return Err(PipelineError::Invalid("prepared occurrence Parquet schema"));
        }
        let u64s = |index| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or(PipelineError::Invalid("occurrence column"))
        };
        let strings = |index| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(PipelineError::Invalid("occurrence column"))
        };
        let occurrence_ids = strings(0)?;
        let document_ids = strings(1)?;
        let times = u64s(2)?;
        let relations = strings(3)?;
        let ordinals = u64s(4)?;
        let attributes = strings(5)?;
        let snapshots = strings(6)?;
        let mappings = strings(7)?;
        let event_ids = strings(8)?;
        let support_refs = strings(9)?;
        for index in 0..batch.num_rows() {
            let row = PreparedOccurrenceRow {
                occurrence_id: occurrence_ids.value(index).to_owned(),
                document_id: document_ids.value(index).to_owned(),
                event_time_ms: (!times.is_null(index)).then(|| times.value(index)),
                relation: relations.value(index).to_owned(),
                source_row_ordinal: ordinals.value(index),
                exact_attributes_json: attributes.value(index).to_owned(),
                snapshot_sha256: Digest::new(snapshots.value(index))?,
                mapping_sha256: Digest::new(mappings.value(index))?,
                event_id: event_ids.value(index).to_owned(),
                support_ref: support_refs.value(index).to_owned(),
            };
            row.validate()?;
            result.push(row);
        }
    }
    Ok(result)
}

fn write_batch(path: &Path, schema: std::sync::Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = tempfile::NamedTempFile::new_in(
        path.parent()
            .ok_or(PipelineError::Invalid("Parquet parent"))?,
    )?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(temporary.reopen()?, schema, Some(properties))?;
    writer.write(batch)?;
    writer.close()?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| PipelineError::Io(error.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> Digest {
        Digest::new(character.to_string().repeat(64)).unwrap()
    }
    fn component(id: &str, character: char) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: digest(character),
        }
    }
    fn dataset() -> DatasetIdentity {
        DatasetIdentity {
            id: "dataset-a".into(),
            version: "1".into(),
            source_snapshot: component("snapshot", 'a'),
            mapping: component("mapping", 'b'),
            source_admission: vec![],
            included_relations: vec!["events".into()],
            excluded_relations: vec!["network".into()],
            structured_only_relations: vec!["metrics".into()],
        }
    }
    fn profile() -> EmbeddingProfileRef {
        EmbeddingProfileRef {
            component: component("profile", 'c'),
            model_artifact: component("model", 'd'),
            tokenizer: component("tokenizer", 'e'),
            maximum_input_tokens: 8192,
            pooling: "last_token".into(),
            normalization: "l2".into(),
            dimensions: 4,
            dtype: "f32le".into(),
            document_format: "{semantic_text}".into(),
        }
    }
    fn task(id: &str, start: u64, end: u64, character: char) -> EmbeddingTask {
        EmbeddingTask {
            task_id: id.into(),
            ordinal_start: start,
            ordinal_end: end,
            input_slices: vec![EmbeddingInputSlice {
                path: SafeRelativePath::new("documents/part.parquet").unwrap(),
                object_sha256: digest('f'),
                row_offset: start,
                rows: end - start,
                embedding_input_order_sha256: digest(character),
            }],
            embedding_input_order_sha256: digest(character),
            result_path: SafeRelativePath::new(format!("parts/{id}.f32")).unwrap(),
            receipt_path: SafeRelativePath::new(format!("receipts/{id}.json")).unwrap(),
        }
    }
    fn plan(tasks: Vec<EmbeddingTask>, count: u64) -> EmbeddingPlan {
        let mut plan = EmbeddingPlan {
            schema_version: EMBEDDING_PLAN_SCHEMA.into(),
            component_sha256: digest('0'),
            prepared_corpus_sha256: digest('a'),
            dataset: dataset(),
            embedding_profile: profile(),
            document_count: count,
            document_order_sha256: digest('b'),
            tasks,
        };
        plan.seal().unwrap();
        plan
    }

    #[test]
    fn rejects_unsafe_paths() {
        for path in [
            "",
            "/root",
            "../escape",
            "a/../b",
            "./a",
            "a\\b",
            "C:/x",
            "a//b",
            "a/",
        ] {
            assert!(SafeRelativePath::new(path).is_err(), "{path}");
        }
        assert!(SafeRelativePath::new("documents/part-000001.parquet").is_ok());
    }

    #[test]
    fn rejects_missing_and_overlapping_tasks() {
        let missing = EmbeddingPlan {
            tasks: vec![task("a", 0, 2, '1'), task("b", 3, 4, '2')],
            ..plan(vec![task("seed", 0, 4, '3')], 4)
        };
        assert!(missing.validate().is_err());
        let overlap = EmbeddingPlan {
            tasks: vec![task("a", 0, 3, '1'), task("b", 2, 4, '2')],
            ..plan(vec![task("seed", 0, 4, '3')], 4)
        };
        assert!(overlap.validate().is_err());
    }

    #[test]
    fn receipt_rejects_wrong_profile_corpus_and_order() {
        let plan = plan(vec![task("a", 0, 2, '1')], 2);
        let mut receipt = VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: digest('0'),
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: "a".into(),
            ordinal_start: 0,
            ordinal_end: 2,
            embedding_input_order_sha256: digest('1'),
            vector: VectorObject {
                path: SafeRelativePath::new("parts/a.f32").unwrap(),
                rows: 2,
                bytes: 96,
                sha256: digest('9'),
                dimensions: 4,
                dtype: "f32le".into(),
                embedding_input_order_sha256: digest('1'),
            },
            executor: ExecutorReceipt {
                implementation: component("executor", '5'),
                runtime: component("runtime", '6'),
                returned_model: "model".into(),
                requests: 1,
                retries: 0,
                input_bytes_upper_bound: 2,
                elapsed_ms: 1,
                conformance_passed: true,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal().unwrap();
        assert!(receipt.validate_against(&plan).is_ok());
        for mutation in 0..3 {
            let mut wrong = receipt.clone();
            match mutation {
                0 => wrong.prepared_corpus_sha256 = digest('7'),
                1 => wrong.embedding_profile_sha256 = digest('7'),
                _ => wrong.embedding_input_order_sha256 = digest('7'),
            };
            wrong.seal().unwrap();
            assert!(wrong.validate_against(&plan).is_err());
        }
        let mut wrong_bytes = receipt.clone();
        wrong_bytes.vector.bytes += 4;
        wrong_bytes.seal().unwrap();
        assert!(wrong_bytes.validate_against(&plan).is_err());
        let mut unconformant = receipt;
        unconformant.executor.conformance_passed = false;
        unconformant.seal().unwrap();
        assert!(unconformant.validate_against(&plan).is_err());
    }

    #[test]
    fn canonical_digest_is_deterministic_across_map_order() {
        let first: Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        let second: Value = serde_json::from_str(r#"{"a":1,"b":2}"#).unwrap();
        assert_eq!(
            canonical_digest(&first).unwrap(),
            canonical_digest(&second).unwrap()
        );
    }

    #[test]
    fn component_digest_omits_its_self_digest() {
        let value = serde_json::json!({
            "schema_version": "example/1",
            "component_sha256": "ignored",
            "value": 7
        });
        let material = serde_json::json!({"schema_version": "example/1", "value": 7});
        assert_eq!(
            component_digest(&value).unwrap(),
            canonical_digest(&material).unwrap()
        );
    }

    #[test]
    fn dataset_source_admission_is_optional_but_sorted_and_unique() {
        let legacy = dataset();
        let bytes = serde_json::to_vec(&legacy).unwrap();
        assert!(
            !String::from_utf8(bytes)
                .unwrap()
                .contains("source_admission")
        );

        let mut current = dataset();
        current.source_admission = vec![
            component("relation-contract", 'c'),
            component("snapshot-capabilities", 'd'),
        ];
        current.validate().unwrap();
        current.source_admission.reverse();
        assert!(matches!(
            current.validate(),
            Err(PipelineError::Invalid(
                "source admission components must be sorted and unique by id"
            ))
        ));
    }

    #[test]
    fn exact_slice_validation_rejects_wrong_offsets_and_slice_digests() {
        let text_a = "alpha";
        let text_b = "beta";
        let rows = vec![
            PreparedDocumentRow {
                document_ordinal: 0,
                document_id: "a".into(),
                document_sha256: digest('1'),
                semantic_text_sha256: digest_bytes(text_a.as_bytes()),
                semantic_text: text_a.into(),
                document_kind: DocumentKind::Activity,
                primary_relation: "events".into(),
                facets_json: "{}".into(),
                relations_json: "[\"events\"]".into(),
                occurrence_count: 1,
            },
            PreparedDocumentRow {
                document_ordinal: 1,
                document_id: "b".into(),
                document_sha256: digest('2'),
                semantic_text_sha256: digest_bytes(text_b.as_bytes()),
                semantic_text: text_b.into(),
                document_kind: DocumentKind::Activity,
                primary_relation: "events".into(),
                facets_json: "{}".into(),
                relations_json: "[\"events\"]".into(),
                occurrence_count: 1,
            },
        ];
        let path = SafeRelativePath::new("documents/part.parquet").unwrap();
        let object_sha = digest('f');
        let mut manifest = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: digest('0'),
            dataset: dataset(),
            projection_policy: component("projection", '3'),
            document_schema: component("documents", '4'),
            occurrence_schema: component("occurrences", '5'),
            preparation_implementation: component("prepare", '6'),
            document_count: 2,
            occurrence_count: 2,
            document_order_sha256: document_order_digest(
                rows.iter().map(|row| row.document_id.as_str()),
            ),
            embedding_input_order_sha256: embedding_input_order_digest(&rows),
            documents: vec![PreparedDocumentObject {
                object: ObjectEntry {
                    path: path.clone(),
                    rows: 2,
                    bytes: 1,
                    sha256: object_sha.clone(),
                    logical_order_sha256: canonical_digest(&rows).unwrap(),
                },
                ordinal: 0,
                first_document_id: "a".into(),
                last_document_id: "b".into(),
                embedding_input_order_sha256: embedding_input_order_digest(&rows),
            }],
            occurrences: vec![PreparedOccurrenceObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("occurrences/events/part.parquet").unwrap(),
                    rows: 2,
                    bytes: 1,
                    sha256: digest('7'),
                    logical_order_sha256: digest('8'),
                },
                ordinal: 0,
                relation: "events".into(),
            }],
            relation_accounting: BTreeMap::from([
                (
                    "events".into(),
                    RelationAccounting {
                        source_rows: 2,
                        searchable_occurrences: 2,
                        selected_occurrences: 2,
                        excluded_rows: 0,
                    },
                ),
                (
                    "network".into(),
                    RelationAccounting {
                        source_rows: 1,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 1,
                    },
                ),
                (
                    "metrics".into(),
                    RelationAccounting {
                        source_rows: 1,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 1,
                    },
                ),
            ]),
        };
        manifest.seal().unwrap();
        let all_order = embedding_input_order_digest(&rows);
        let mut valid = plan(
            vec![EmbeddingTask {
                task_id: "all".into(),
                ordinal_start: 0,
                ordinal_end: 2,
                input_slices: vec![EmbeddingInputSlice {
                    path: path.clone(),
                    object_sha256: object_sha,
                    row_offset: 0,
                    rows: 2,
                    embedding_input_order_sha256: all_order.clone(),
                }],
                embedding_input_order_sha256: all_order,
                result_path: SafeRelativePath::new("parts/all.f32").unwrap(),
                receipt_path: SafeRelativePath::new("receipts/all.json").unwrap(),
            }],
            2,
        );
        valid.prepared_corpus_sha256 = manifest.component_sha256.clone();
        valid.dataset = manifest.dataset.clone();
        valid.document_order_sha256 = manifest.document_order_sha256.clone();
        valid.seal().unwrap();
        assert!(valid.validate_against_prepared(&manifest, &rows).is_ok());

        let mut wrong_offset = valid.clone();
        wrong_offset.tasks[0].input_slices[0].row_offset = 1;
        wrong_offset.seal().unwrap();
        assert!(
            wrong_offset
                .validate_against_prepared(&manifest, &rows)
                .is_err()
        );
        let mut wrong_digest = valid;
        wrong_digest.tasks[0].input_slices[0].embedding_input_order_sha256 = digest('9');
        wrong_digest.seal().unwrap();
        assert!(
            wrong_digest
                .validate_against_prepared(&manifest, &rows)
                .is_err()
        );
    }

    #[test]
    fn manifest_rejects_extra_accounting_and_wrong_object_metadata() {
        let mut identity = dataset();
        identity.excluded_relations.clear();
        identity.structured_only_relations.clear();
        let mut manifest = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: digest('0'),
            dataset: identity,
            projection_policy: component("projection", '3'),
            document_schema: component("documents", '4'),
            occurrence_schema: component("occurrences", '5'),
            preparation_implementation: component("prepare", '6'),
            document_count: 0,
            occurrence_count: 0,
            document_order_sha256: document_order_digest(std::iter::empty()),
            embedding_input_order_sha256: embedding_input_order_digest(std::iter::empty()),
            documents: vec![],
            occurrences: vec![],
            relation_accounting: BTreeMap::from([
                (
                    "events".into(),
                    RelationAccounting {
                        source_rows: 0,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 0,
                    },
                ),
                (
                    "extra".into(),
                    RelationAccounting {
                        source_rows: 0,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 0,
                    },
                ),
            ]),
        };
        manifest.component_sha256 = component_digest(&manifest).unwrap();
        assert!(manifest.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_resolution_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("object"), b"outside").unwrap();
        symlink(outside.path(), root.path().join("escape")).unwrap();
        let relative = SafeRelativePath::new("escape/object").unwrap();
        assert!(resolve_existing_artifact(root.path(), &relative).is_err());
        assert!(resolve_output_artifact(root.path(), &relative).is_err());
    }

    #[test]
    fn prepared_document_parquet_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("part.parquet");
        let text = "hello";
        let row = PreparedDocumentRow {
            document_ordinal: 0,
            document_id: "doc-1".into(),
            document_sha256: digest('a'),
            semantic_text_sha256: digest_bytes(text.as_bytes()),
            semantic_text: text.into(),
            document_kind: DocumentKind::Activity,
            primary_relation: "events".into(),
            facets_json: "{}".into(),
            relations_json: "[\"events\"]".into(),
            occurrence_count: 1,
        };
        write_prepared_documents(&path, std::slice::from_ref(&row)).unwrap();
        assert_eq!(read_prepared_documents(&path).unwrap(), vec![row]);
    }

    #[test]
    fn prepared_occurrence_parquet_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("part.parquet");
        let rows = vec![
            PreparedOccurrenceRow {
                occurrence_id: "occ-1".into(),
                document_id: "doc-1".into(),
                event_time_ms: Some(42),
                relation: "events".into(),
                source_row_ordinal: 3,
                exact_attributes_json: "{}".into(),
                snapshot_sha256: digest('a'),
                mapping_sha256: digest('b'),
                event_id: "evt-1".into(),
                support_ref: "support-1".into(),
            },
            PreparedOccurrenceRow {
                occurrence_id: "occ-2".into(),
                document_id: "doc-1".into(),
                event_time_ms: None,
                relation: "events".into(),
                source_row_ordinal: 4,
                exact_attributes_json: "{\"a\":1}".into(),
                snapshot_sha256: digest('a'),
                mapping_sha256: digest('b'),
                event_id: "evt-2".into(),
                support_ref: "support-2".into(),
            },
        ];
        write_prepared_occurrences(&path, &rows).unwrap();
        assert_eq!(read_prepared_occurrences(&path).unwrap(), rows);
    }

    #[test]
    fn atomic_directory_publishes_complete_tree() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("index");
        let staging = AtomicDirectory::new(&destination).unwrap();
        fs::write(staging.path().join("manifest.json"), b"{}").unwrap();
        staging.publish().unwrap();
        assert_eq!(fs::read(destination.join("manifest.json")).unwrap(), b"{}");
    }

    #[test]
    fn atomic_directory_publishes_completed_child_only_at_the_end() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("index");
        let staging = AtomicDirectory::new(&destination).unwrap();
        let child = staging.path().join("assembled");
        fs::create_dir(&child).unwrap();
        fs::write(child.join("index.json"), b"{}").unwrap();
        assert!(!destination.exists());
        staging.publish_child("assembled").unwrap();
        assert_eq!(fs::read(destination.join("index.json")).unwrap(), b"{}");
    }

    #[test]
    fn stale_atomic_writes_are_owned_and_restart_cleanable() {
        let root = tempfile::tempdir().unwrap();
        let stale = root
            .path()
            .join(".livefire-rag-atomic-4294967295-0-crash.partial");
        let unrelated = root.path().join(".tmp-user-file");
        fs::write(&stale, b"incomplete").unwrap();
        fs::write(&unrelated, b"keep").unwrap();
        remove_stale_atomic_writes(root.path()).unwrap();
        assert!(!stale.exists());
        assert!(unrelated.exists());
    }
}
