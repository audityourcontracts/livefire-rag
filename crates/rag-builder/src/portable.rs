//! Dataset-oriented prepare, embed, and assemble commands.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow_array::RecordBatch;
use rag_embedding::{
    AtomicFilePublication, BearerAuthorization, EmbeddingShardExpectation, EmbeddingShardMetadata,
    EmbeddingShardWriter, EmbeddingTaskOptions, EmbeddingTaskPartPreparation, EmbeddingTaskReport,
    LmStudioEmbedder, RetryPolicy, TaskSelection, TeiCheckpointProfileV3, TeiEmbedder,
    VectorDerivation, adapt_model_vector, complete_embedding_task_part_recovery, decode_sha256_hex,
    execute_embedding_task_reported, format_document_input, parse_bound_embedding_profile,
    parse_embedding_profile, parse_tei_checkpoint_profile_v3, prepare_embedding_task_part,
    restore_quarantined_embedding_task_part, validate_vector, verify_embedding_task_part,
};
use rag_index::{
    BuildScope, FastDocument, FastIndexManifest, OrderedVectorShard, PipelineIndexOptions,
    PipelineProvenance, SourceBinding, documents_from_parquet_shards,
    occurrences_from_parquet_shards, vectors_from_embedding_shards,
    write_bound_fast_index_from_streams, write_bound_scalable_fast_index_from_streams,
};
use rag_ocsf::{
    AdmittedParquetObject, LocalSnapshotReader, OcsfRowGroup, OcsfSnapshot, SnapshotReader,
};
use rag_pipeline::{
    AtomicDirectory, BenchmarkLengthStratum, BenchmarkPublishedCorpus, BenchmarkSelectionCandidate,
    BenchmarkSelectionPolicy, BenchmarkSelectionRow, BenchmarkStratumQuota, BenchmarkTargetQuota,
    ComponentRef, DERIVED_RESULT_SET_SCHEMA, DERIVED_VECTOR_EXECUTOR_ID,
    DERIVED_VECTOR_RECEIPT_SCHEMA, DatasetIdentity, DerivedResultSetBinding, DerivedVectorBinding,
    Digest, DocumentKind, EMBEDDING_PLAN_SCHEMA, EMBEDDING_PLAN_V2_SCHEMA, EmbeddingInputSliceV2,
    EmbeddingPlanV2, EmbeddingProfileRef, EmbeddingResultSetManifest, EmbeddingTaskV2,
    ExactTokenizer, ExecutableTokenizerRef, ExecutorReceipt, ObjectEntry,
    PREFIX_L2_DERIVATION_POLICY, PREPARED_CORPUS_SCHEMA, PreparedCorpusManifest,
    PreparedDocumentObject, PreparedDocumentRow, PreparedOccurrenceObject, PreparedOccurrenceRow,
    RESULT_SET_SCHEMA, ReceiptEntry, RelationAccounting, STANDARD_BENCHMARK_SIZES,
    SafeRelativePath, TEST_RESULT_SET_SCHEMA, TEST_VECTOR_EXECUTOR_ID, TokenBalanceOptions,
    TokenizerArtifactFormat, VECTOR_RECEIPT_SCHEMA, VectorObject, VectorResultReceipt,
    atomic_write, bind_benchmark_prepared_corpus, build_benchmark_selection_manifest,
    build_token_balanced_plan_with_counts, canonical_digest, canonical_json_bytes,
    component_digest, derive_embedding_plan_v2, digest_bytes, document_order_digest,
    embedding_input_order_digest, read_json, read_prepared_documents, read_prepared_occurrences,
    resolve_existing_artifact, resolve_output_artifact, select_benchmark_documents,
    validate_prepared_documents, write_canonical_json, write_prepared_documents,
    write_prepared_occurrences,
};
use rag_projection::{
    ComponentRef as ProjectionComponentRef, ProjectionContext, ProjectionInput, ProjectionOutput,
    project, project_document_summary, project_m45_command,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as ShaDigest, Sha256};

use super::{
    Error, IndexFormat, RecoveryAction, Result, fast_document, parse_event_time_ms, strings,
};
use crate::report::{
    GitState, LmStudioContext, LocalRunContext, MachineContext, ObservationStatus, ResourceUsage,
    RunArtifactSizes, TaskArtifactSizes, TransportByteAccounting, run_artifact_sizes,
    task_artifact_sizes,
};

const MANIFEST_FILE: &str = "manifest.json";
const EMBEDDING_PROFILE_FILE: &str = "embedding-profile.json";
const DOCUMENT_SCHEMA_BYTES: &[u8] =
    include_bytes!("../../../specs/prepared-document-row.v1.schema.json");
const OCCURRENCE_SCHEMA_BYTES: &[u8] =
    include_bytes!("../../../specs/prepared-occurrence-row.v1.schema.json");
const PROJECTION_POLICY_BYTES: &[u8] =
    include_bytes!("../../../specs/evidence-projection-policy.v2.json");
const M45_COMMAND_PROJECTION_POLICY_BYTES: &[u8] =
    include_bytes!("../../../specs/m45-command-projection-policy.v1.json");
const PREPARATION_SOURCE_BYTES: &[u8] = include_bytes!("portable.rs");
const OCCURRENCE_SHARD_ROWS: usize = 8_192;
const DEFAULT_DOCUMENT_RUN_ROWS: usize = 100_000;
const MIN_CONFIGURED_DOCUMENT_RUN_ROWS: usize = 1_024;
const MAX_DOCUMENT_RUN_ROWS: usize = 600_000;
const MAX_BENCHMARK_CANDIDATES: usize = 600_000;
const DOCUMENT_RUN_ROWS_ENV: &str = "LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS";
const MAX_CENSUS_WORKERS: usize = 64;
const MAX_PREPARE_WORKERS: usize = 64;
const MIN_PARALLEL_BATCH_ROWS: usize = 1_024;
const MIN_BATCH_RANGE_ROWS: usize = 256;

#[derive(Clone, Copy)]
struct BatchProjectionExecution<'a> {
    worker_pool: &'a rayon::ThreadPool,
    workers: usize,
}

#[derive(Clone, Copy)]
struct PreparedProjectionExecution<'a> {
    batch: BatchProjectionExecution<'a>,
    projection: PreparationProjection,
}

pub(crate) fn default_census_workers() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(8)
}

pub(crate) fn default_prepare_workers() -> usize {
    default_census_workers()
}

/// Project bounded waves of admitted Parquet row groups in parallel, then
/// merge each result in original file order. This is shared by census and
/// benchmark preparation so concurrency cannot leak into artifact ordering.
fn map_row_groups_in_source_order<T, Project, Merge>(
    worker_pool: &rayon::ThreadPool,
    object: &AdmittedParquetObject,
    workers: usize,
    project: Project,
    mut merge: Merge,
) -> Result<()>
where
    T: Send,
    Project: Fn(&OcsfRowGroup, usize) -> Result<T> + Sync,
    Merge: FnMut(&OcsfRowGroup, T) -> Result<()>,
{
    for group_wave in object.row_groups().chunks(workers) {
        let batch_workers = if group_wave.len() < workers {
            workers
        } else {
            1
        };
        let partials = worker_pool.install(|| {
            group_wave
                .par_iter()
                .map(|group| project(group, batch_workers))
                .collect::<Vec<_>>()
        });
        for (group, partial) in group_wave.iter().zip(partials) {
            merge(group, partial?)?;
        }
    }
    Ok(())
}

/// Project borrowed row ranges from one decoded Arrow batch on the existing
/// bounded worker pool, then merge range results in source order. No batch or
/// Arrow value buffer is cloned.
fn map_batch_ranges_in_source_order<T, Project, Merge>(
    worker_pool: &rayon::ThreadPool,
    rows: usize,
    workers: usize,
    project: Project,
    mut merge: Merge,
) -> Result<()>
where
    T: Send,
    Project: Fn(usize, std::ops::Range<usize>) -> Result<T> + Sync,
    Merge: FnMut(usize, std::ops::Range<usize>, T) -> Result<()>,
{
    if rows == 0 {
        return Ok(());
    }
    let target_ranges = workers.saturating_mul(2).max(1);
    let range_rows = if workers > 1 && rows >= MIN_PARALLEL_BATCH_ROWS {
        rows.div_ceil(target_ranges).max(MIN_BATCH_RANGE_ROWS)
    } else {
        rows
    };
    let ranges = (0..rows)
        .step_by(range_rows)
        .map(|start| start..start.saturating_add(range_rows).min(rows))
        .collect::<Vec<_>>();
    let partials = if ranges.len() == 1 {
        vec![project(0, ranges[0].clone())]
    } else {
        worker_pool.install(|| {
            ranges
                .par_iter()
                .cloned()
                .enumerate()
                .map(|(ordinal, range)| project(ordinal, range))
                .collect::<Vec<_>>()
        })
    };
    for (ordinal, (range, partial)) in ranges.into_iter().zip(partials).enumerate() {
        merge(ordinal, range, partial?)?;
    }
    Ok(())
}

pub(crate) struct PrepareOptions {
    pub snapshot: PathBuf,
    pub dataset_id: String,
    pub dataset_version: String,
    pub relations: Vec<String>,
    pub out: PathBuf,
    pub document_shard_rows: usize,
    pub workers: usize,
}

pub(crate) struct PrepareCommandsOptions {
    pub snapshot: PathBuf,
    pub out: PathBuf,
    pub document_shard_rows: usize,
    pub workers: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparationProjection {
    Generic,
    M45Commands,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct M45CommandSourceContract {
    snapshot_receipt_schema: u8,
    snapshot_manifest_schema: u8,
    snapshot_id: String,
    snapshot_version: String,
    snapshot_sha256: Digest,
    mapping_id: String,
    mapping_version: String,
    mapping_sha256: Digest,
    relation_contract_sha256: Digest,
    snapshot_capabilities_sha256: Digest,
    admitted_relations: Vec<String>,
    authority: String,
    event_reference: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct M45CommandAdmittedIdentity {
    snapshot_manifest_schema: u8,
    snapshot_id: String,
    snapshot_version: String,
    snapshot_sha256: Digest,
    mapping_id: String,
    mapping_version: String,
    mapping_sha256: Digest,
    relation_contract_sha256: Digest,
    snapshot_capabilities_sha256: Digest,
}

pub(crate) struct PrepareBenchmarkOptions {
    pub snapshot: PathBuf,
    pub dataset_id: String,
    pub dataset_version: String,
    pub relations: Vec<String>,
    pub out: PathBuf,
    pub document_shard_rows: usize,
    pub selection_seed: String,
    pub workers: usize,
}

pub(crate) struct CensusOptions {
    pub snapshot: PathBuf,
    pub relations: Vec<String>,
    pub out: Option<PathBuf>,
    pub workers: usize,
}

pub(crate) struct VerifyTokenizerOptions {
    pub tokenizer_json: PathBuf,
    pub tokenizer_ref: PathBuf,
    pub fixture: PathBuf,
}

pub(crate) struct PlanOptions {
    pub prepared: PathBuf,
    pub embedding_profile: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_ref: PathBuf,
    pub out: PathBuf,
    pub maximum_task_tokens: u64,
    pub maximum_task_documents: u32,
}

pub(crate) struct TeiPlanOptions {
    pub prepared: PathBuf,
    pub embedding_policy: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_ref: PathBuf,
    pub out: PathBuf,
    pub maximum_task_tokens: u64,
    pub maximum_task_documents: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TeiWorkerReportContextV2 {
    pub schema_version: String,
    pub execution_identity: EmbeddingExecutionIdentityV2,
    pub git: GitState,
    pub machine: MachineContext,
    pub accelerator: AcceleratorContextV2,
    pub backend: EmbeddingBackendContextV2,
    pub resource_usage: ResourceUsageV2,
}

pub(crate) struct TeiEmbedOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_policy: PathBuf,
    pub conformance_fixture: PathBuf,
    pub out: PathBuf,
    pub batch_size: usize,
    pub requests_in_flight: usize,
    pub task_range: Option<String>,
    pub worker: TeiWorkerReportContextV2,
}

pub(crate) struct EmbedOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub embedding_endpoint: String,
    pub out: PathBuf,
    pub batch_size: usize,
    pub requests_in_flight: usize,
    pub task_range: Option<String>,
}

pub(crate) struct FinalizeOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub embeddings: PathBuf,
}

pub(crate) struct DeriveEmbeddingsOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub embeddings: PathBuf,
    pub dimensions: u32,
    pub out: PathBuf,
}

pub(crate) struct TestEmbedOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub out: PathBuf,
}

pub(crate) struct RecoveryOptions {
    pub plan: PathBuf,
    pub embedding_profile: PathBuf,
    pub embeddings: PathBuf,
    pub task_id: String,
    pub action: RecoveryAction,
}

pub(crate) struct AssembleOptions {
    pub prepared: PathBuf,
    pub plan: PathBuf,
    pub embeddings: PathBuf,
    pub embedding_profile: PathBuf,
    pub out: PathBuf,
    pub index_format: IndexFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskRunOutcome {
    Executed,
    Reused,
    TestGenerated,
    Derived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BuilderEmbeddingTaskReport {
    schema_version: String,
    plan_sha256: Digest,
    source_snapshot_sha256: Digest,
    prepared_corpus_sha256: Digest,
    embedding_profile_sha256: Digest,
    tokenizer_sha256: Digest,
    task_id: String,
    task_index: usize,
    ordinal_start: u64,
    ordinal_end: u64,
    document_count: u64,
    token_count: u64,
    receipt_sha256: Digest,
    outcome: TaskRunOutcome,
    started_unix_ms: Option<u64>,
    finished_unix_ms: Option<u64>,
    git: GitState,
    machine: MachineContext,
    lm_studio: LmStudioContext,
    transport_bytes: TransportByteAccounting,
    resource_usage: ResourceUsage,
    artifact_sizes: TaskArtifactSizes,
    execution: Option<EmbeddingTaskReport>,
}

/// Exact execution material shared by every task in a portable backend run.
/// Host identity is deliberately separate so RunPod may schedule the same
/// sealed execution on different machines. The certified accelerator class is
/// part of this identity because conformance is hardware-specific.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingExecutionIdentityV2 {
    pub backend_kind: String,
    pub executor_image: ComponentRef,
    pub executor_image_build: ComponentRef,
    pub runtime: ComponentRef,
    pub worker_binary: ComponentRef,
    pub model_artifact: ComponentRef,
    pub embedding_profile: ComponentRef,
    pub returned_model: String,
    pub accelerator: EmbeddingAcceleratorPolicyV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingAcceleratorPolicyV2 {
    pub provider: String,
    pub model: String,
    pub architecture: String,
    pub compute_capability: String,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingBackendContextV2 {
    pub status: ObservationStatus,
    pub kind: String,
    pub version: Option<String>,
    pub endpoint_kind: String,
    pub batch_size: usize,
    pub requests_in_flight: usize,
    pub cold_load_micros: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AcceleratorContextV2 {
    pub status: ObservationStatus,
    pub provider: Option<String>,
    pub machine_id: Option<String>,
    pub model: Option<String>,
    pub architecture: Option<String>,
    pub compute_capability: Option<String>,
    pub count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResourceUsageV2 {
    pub status: ObservationStatus,
    pub worker_peak_rss_bytes: Option<u64>,
    pub backend_peak_rss_bytes: Option<u64>,
}

/// Backend-neutral task report used by TEI and remote workers. V1 remains the
/// stable LM Studio wire contract and is still emitted by the local path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuilderEmbeddingTaskReportV2 {
    pub schema_version: String,
    pub plan_sha256: Digest,
    pub source_snapshot_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub embedding_profile_sha256: Digest,
    pub tokenizer_sha256: Digest,
    pub task_id: String,
    pub task_index: usize,
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub document_count: u64,
    pub token_count: u64,
    pub receipt_sha256: Digest,
    pub outcome: TaskRunOutcome,
    pub started_unix_ms: Option<u64>,
    pub finished_unix_ms: Option<u64>,
    pub execution_identity: EmbeddingExecutionIdentityV2,
    pub git: GitState,
    pub machine: MachineContext,
    pub accelerator: AcceleratorContextV2,
    pub backend: EmbeddingBackendContextV2,
    pub transport_bytes: TransportByteAccounting,
    pub resource_usage: ResourceUsageV2,
    pub artifact_sizes: TaskArtifactSizes,
    pub execution: Option<EmbeddingTaskReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValidatedEmbeddingTaskReport {
    V1(Box<BuilderEmbeddingTaskReport>),
    V2(Box<BuilderEmbeddingTaskReportV2>),
}

struct TaskRunDetails {
    outcome: TaskRunOutcome,
    started_unix_ms: Option<u64>,
    finished_unix_ms: Option<u64>,
    execution: Option<EmbeddingTaskReport>,
}

struct TaskReportBindings<'a> {
    prepared: &'a PreparedCorpusManifest,
    profile: &'a rag_embedding::EmbeddingProfile,
    receipt: &'a VectorResultReceipt,
    vector_path: &'a Path,
    receipt_path: &'a Path,
    run_context: &'a LocalRunContext,
    batch_size: usize,
    requests_in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRunSummary {
    schema_version: String,
    status: String,
    source_snapshot_sha256: Digest,
    prepared_corpus_sha256: Digest,
    plan_sha256: Digest,
    embedding_profile_sha256: Digest,
    tokenizer_sha256: Digest,
    git: GitState,
    machine: MachineContext,
    lm_studio: LmStudioContext,
    resource_usage: ResourceUsage,
    artifact_sizes: RunArtifactSizes,
    tasks: usize,
    documents: u64,
    tokens: u64,
    unique_input_text_bytes: u64,
    sent_input_text_bytes: Option<u64>,
    vector_payload_bytes: u64,
    vector_shard_bytes: u64,
    transport_bytes: TransportByteAccounting,
    requests: u64,
    retries: u64,
    execution_reports_complete: bool,
    calendar_span_micros: Option<u64>,
    wall_time_micros: Option<u64>,
    task_elapsed_micros_sum: Option<u64>,
    request_elapsed_micros: Option<u64>,
    retry_backoff_micros: Option<u64>,
    peak_in_flight: Option<usize>,
    documents_per_second: Option<f64>,
    tokens_per_second: Option<f64>,
    request_latency_micros: RequestLatencySummary,
    length_bucket_throughput: Vec<LengthBucketThroughput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingWorkerProvenanceV2 {
    task_id: String,
    git: GitState,
    machine: MachineContext,
    accelerator: AcceleratorContextV2,
    backend: EmbeddingBackendContextV2,
    resource_usage: ResourceUsageV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRunAggregateV2 {
    artifact_sizes: RunArtifactSizes,
    tasks: usize,
    documents: u64,
    tokens: u64,
    unique_input_text_bytes: u64,
    sent_input_text_bytes: Option<u64>,
    vector_payload_bytes: u64,
    vector_shard_bytes: u64,
    transport_bytes: TransportByteAccounting,
    requests: u64,
    retries: u64,
    execution_reports_complete: bool,
    calendar_span_micros: Option<u64>,
    active_time_micros: Option<u64>,
    task_elapsed_micros_sum: Option<u64>,
    request_elapsed_micros: Option<u64>,
    retry_backoff_micros: Option<u64>,
    peak_in_flight_per_worker: Option<usize>,
    documents_per_active_second: Option<f64>,
    tokens_per_active_second: Option<f64>,
    request_latency_micros: RequestLatencySummary,
    length_bucket_throughput: Vec<LengthBucketThroughput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRunSummaryV2 {
    schema_version: String,
    status: String,
    source_snapshot_sha256: Digest,
    prepared_corpus_sha256: Digest,
    plan_sha256: Digest,
    embedding_profile_sha256: Digest,
    tokenizer_sha256: Digest,
    execution_identity: EmbeddingExecutionIdentityV2,
    workers: Vec<EmbeddingWorkerProvenanceV2>,
    aggregate: EmbeddingRunAggregateV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum EmbeddingRunSummaryContract {
    V1(Box<EmbeddingRunSummary>),
    V2(Box<EmbeddingRunSummaryV2>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestLatencySummary {
    p50: Option<u64>,
    p95: Option<u64>,
    samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LengthBucketThroughput {
    basis: String,
    minimum_tokens: u32,
    maximum_tokens: Option<u32>,
    documents: u64,
    tokens: u64,
    shared_wall_time_micros: Option<u64>,
    documents_per_second: Option<f64>,
    tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RecoveryArtifactState {
    Absent,
    Valid,
    Invalid,
    Orphan,
    Quarantined,
}

#[derive(Debug)]
struct TaskArtifactInspection {
    part: RecoveryArtifactState,
    receipt: RecoveryArtifactState,
    report: RecoveryArtifactState,
    receipt_value: Option<VectorResultReceipt>,
}

#[derive(Debug, Serialize)]
struct TaskRecoveryReport {
    schema_version: &'static str,
    action: &'static str,
    plan_sha256: Digest,
    embedding_profile_sha256: Digest,
    task_id: String,
    task_index: usize,
    complete: bool,
    part: RecoveryArtifactState,
    receipt: RecoveryArtifactState,
    report: RecoveryArtifactState,
    changed_artifacts: Vec<String>,
    model_contacted: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct DocumentAccumulator {
    document: FastDocument,
    primary_relation: String,
    relations: BTreeSet<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentRunRow {
    document_id: String,
    accumulated: DocumentAccumulator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DocumentRunStats {
    run_count: usize,
    maximum_buffered_documents: usize,
}

struct SortedDocumentRuns {
    root: PathBuf,
    maximum_rows: usize,
    buffered: BTreeMap<String, DocumentAccumulator>,
    paths: Vec<PathBuf>,
    maximum_buffered_documents: usize,
    cleaned: bool,
}

struct DocumentRunReader {
    reader: BufReader<File>,
    line: String,
    previous_document_id: Option<String>,
}

impl SortedDocumentRuns {
    fn new(root: PathBuf, maximum_rows: usize) -> Result<Self> {
        if maximum_rows == 0 || maximum_rows > MAX_DOCUMENT_RUN_ROWS {
            return Err(Error::AccountingClosure(
                "document run rows must be between 1 and 600000",
            ));
        }
        if root.exists() {
            return Err(Error::AccountingClosure(
                "document run staging directory already exists",
            ));
        }
        fs::create_dir(&root)?;
        Ok(Self {
            root,
            maximum_rows,
            buffered: BTreeMap::new(),
            paths: Vec::new(),
            maximum_buffered_documents: 0,
            cleaned: false,
        })
    }

    fn add_all(&mut self, documents: BTreeMap<String, DocumentAccumulator>) -> Result<()> {
        for (document_id, accumulated) in documents {
            self.add(document_id, accumulated)?;
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.paths.is_empty() && self.buffered.is_empty()
    }

    fn add(&mut self, document_id: String, accumulated: DocumentAccumulator) -> Result<()> {
        if let Some(existing) = self.buffered.get_mut(&document_id) {
            merge_document_accumulators(&document_id, existing, accumulated)?;
            return Ok(());
        }
        if self.buffered.len() == self.maximum_rows {
            self.flush()?;
        }
        self.buffered.insert(document_id, accumulated);
        self.maximum_buffered_documents = self.maximum_buffered_documents.max(self.buffered.len());
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        let path = self
            .root
            .join(format!("part-{:06}.jsonl", self.paths.len()));
        let mut writer = BufWriter::new(File::create(&path)?);
        for (document_id, accumulated) in std::mem::take(&mut self.buffered) {
            serde_json::to_writer(
                &mut writer,
                &DocumentRunRow {
                    document_id,
                    accumulated,
                },
            )?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        self.paths.push(path);
        Ok(())
    }

    fn merge<F>(&mut self, mut visit: F) -> Result<DocumentRunStats>
    where
        F: FnMut(DocumentAccumulator) -> Result<()>,
    {
        self.flush()?;
        let stats = DocumentRunStats {
            run_count: self.paths.len(),
            maximum_buffered_documents: self.maximum_buffered_documents,
        };
        let mut readers = self
            .paths
            .iter()
            .map(|path| DocumentRunReader::open(path))
            .collect::<Result<Vec<_>>>()?;
        let mut heap = BinaryHeap::<Reverse<(String, usize)>>::new();
        let mut current = Vec::with_capacity(readers.len());
        for (run_ordinal, reader) in readers.iter_mut().enumerate() {
            let row = reader.next()?;
            if let Some(row) = &row {
                heap.push(Reverse((row.document_id.clone(), run_ordinal)));
            }
            current.push(row);
        }
        while let Some(Reverse((document_id, run_ordinal))) = heap.pop() {
            let mut merged = current[run_ordinal]
                .take()
                .ok_or(Error::AccountingClosure(
                    "document run heap is inconsistent",
                ))?
                .accumulated;
            advance_document_run(run_ordinal, &mut readers, &mut current, &mut heap)?;
            while heap
                .peek()
                .is_some_and(|Reverse((next_id, _))| next_id == &document_id)
            {
                let Reverse((_, duplicate_run_ordinal)) = heap.pop().ok_or(
                    Error::AccountingClosure("document run heap is inconsistent"),
                )?;
                let duplicate = current[duplicate_run_ordinal]
                    .take()
                    .ok_or(Error::AccountingClosure(
                        "document run heap is inconsistent",
                    ))?
                    .accumulated;
                merge_document_accumulators(&document_id, &mut merged, duplicate)?;
                advance_document_run(duplicate_run_ordinal, &mut readers, &mut current, &mut heap)?;
            }
            visit(merged)?;
        }
        fs::remove_dir_all(&self.root)?;
        self.cleaned = true;
        Ok(stats)
    }
}

impl Drop for SortedDocumentRuns {
    fn drop(&mut self) {
        if !self.cleaned {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

impl DocumentRunReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
            line: String::new(),
            previous_document_id: None,
        })
    }

    fn next(&mut self) -> Result<Option<DocumentRunRow>> {
        self.line.clear();
        if self.reader.read_line(&mut self.line)? == 0 {
            return Ok(None);
        }
        let row: DocumentRunRow = serde_json::from_str(self.line.trim_end_matches(['\r', '\n']))?;
        if row.document_id != row.accumulated.document.document_id
            || self
                .previous_document_id
                .as_ref()
                .is_some_and(|previous| previous >= &row.document_id)
        {
            return Err(Error::AccountingClosure(
                "document run rows are not strictly ordered",
            ));
        }
        self.previous_document_id = Some(row.document_id.clone());
        Ok(Some(row))
    }
}

fn advance_document_run(
    run_ordinal: usize,
    readers: &mut [DocumentRunReader],
    current: &mut [Option<DocumentRunRow>],
    heap: &mut BinaryHeap<Reverse<(String, usize)>>,
) -> Result<()> {
    let row = readers[run_ordinal].next()?;
    if let Some(row) = &row {
        heap.push(Reverse((row.document_id.clone(), run_ordinal)));
    }
    current[run_ordinal] = row;
    Ok(())
}

fn merge_document_accumulators(
    document_id: &str,
    existing: &mut DocumentAccumulator,
    accumulated: DocumentAccumulator,
) -> Result<()> {
    if existing.document.document_sha256 != accumulated.document.document_sha256
        || existing.document.semantic_text != accumulated.document.semantic_text
        || existing.document.facets_json != accumulated.document.facets_json
    {
        return Err(Error::InconsistentDocument(document_id.to_owned()));
    }
    existing.document.occurrence_count = existing
        .document
        .occurrence_count
        .checked_add(accumulated.document.occurrence_count)
        .ok_or(Error::CountOverflow)?;
    if accumulated.primary_relation < existing.primary_relation {
        existing.primary_relation = accumulated.primary_relation;
    }
    existing.relations.extend(accumulated.relations);
    Ok(())
}

#[derive(Debug, PartialEq)]
struct PreparedRowGroupProjection {
    ordinal: usize,
    source_rows: u64,
    documents: BTreeMap<String, DocumentAccumulator>,
    occurrences: Vec<PreparedOccurrenceRow>,
}

#[derive(Debug, PartialEq)]
struct PreparedBatchProjection {
    source_rows: u64,
    documents: BTreeMap<String, DocumentAccumulator>,
    occurrences: Vec<PreparedOccurrenceRow>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkCandidateAccumulator {
    candidate: BenchmarkSelectionCandidate,
}

#[derive(Debug, PartialEq)]
struct BenchmarkCandidateRowGroup {
    ordinal: usize,
    source_rows: u64,
    candidates: BTreeMap<String, BenchmarkCandidateAccumulator>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkCandidateBatchProjection {
    source_rows: u64,
    candidates: BTreeMap<String, BenchmarkCandidateAccumulator>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkSelectedOccurrence {
    selection_rank: u64,
    document: FastDocument,
    occurrence: PreparedOccurrenceRow,
}

#[derive(Debug, PartialEq)]
struct BenchmarkOccurrenceRowGroup {
    ordinal: usize,
    source_rows: u64,
    selected: Vec<BenchmarkSelectedOccurrence>,
}

#[derive(Debug, PartialEq)]
struct BenchmarkOccurrenceBatchProjection {
    source_rows: u64,
    selected: Vec<BenchmarkSelectedOccurrence>,
}

#[derive(Debug)]
struct BenchmarkCorpusStage {
    document_count: u64,
    root: PathBuf,
    occurrence_objects: Vec<PreparedOccurrenceObject>,
    occurrence_count: u64,
    selected_occurrences: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct CensusRelationReport {
    source_rows: u64,
    semantic_occurrences: u64,
    structured_only_occurrences: u64,
    distinct_documents: u64,
    document_kinds: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct CorpusCensusReport {
    schema_version: &'static str,
    component_sha256: Digest,
    source_snapshot: ComponentRef,
    mapping: ComponentRef,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    source_admission: Vec<ComponentRef>,
    projection_policy: ComponentRef,
    relations_counted: Vec<String>,
    source_rows: u64,
    semantic_occurrences: u64,
    structured_only_occurrences: u64,
    distinct_documents: u64,
    document_order_sha256: Digest,
    document_kinds: BTreeMap<String, u64>,
    relations: BTreeMap<String, CensusRelationReport>,
}

const TOKENIZER_PARITY_FIXTURE_SCHEMA: &str = "livefire.rag.tokenizer-parity-fixture/1";

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerParityFixture {
    schema_version: String,
    source: TokenizerParitySource,
    cases: Vec<TokenizerParityCase>,
    generated_cases: Vec<GeneratedTokenizerParityCase>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerParitySource {
    runtime: String,
    model_file: String,
    model_revision: String,
    source_tokenizer_json_revision: String,
    source_tokenizer_json_sha256: Digest,
    executable_tokenizer_json_sha256: Digest,
    add_special_tokens: bool,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerParityCase {
    name: String,
    input: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedTokenizerParityCase {
    name: String,
    repeat: String,
    count: u64,
    token_count: u64,
    token_ids_u32le_sha256: Digest,
}

#[derive(Debug, Serialize)]
struct TokenizerVerificationReport {
    schema_version: &'static str,
    component_sha256: Digest,
    status: &'static str,
    fixture_sha256: Digest,
    tokenizer_reference_sha256: Digest,
    executable_tokenizer: ComponentRef,
    target_tokenizer: ComponentRef,
    model_revision: String,
    source_tokenizer_json_revision: String,
    source_tokenizer_json_sha256: Digest,
    direct_cases: u64,
    generated_cases: u64,
    maximum_input_boundary_cases: u64,
    verified_inputs: u64,
    verified_tokens: u64,
}

#[derive(Debug, PartialEq)]
struct CensusRowGroupReport {
    ordinal: usize,
    source_rows: u64,
    semantic_occurrences: u64,
    structured_only_occurrences: u64,
    documents: BTreeMap<String, (String, String)>,
}

#[derive(Debug)]
struct CensusBatchProjection {
    source_rows: u64,
    semantic_occurrences: u64,
    structured_only_occurrences: u64,
    documents: BTreeMap<String, (String, String)>,
}

pub(crate) fn census(options: CensusOptions) -> Result<()> {
    let report = build_corpus_census_report(&options)?;
    if let Some(path) = options.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_canonical_json(&path, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn build_corpus_census_report(options: &CensusOptions) -> Result<CorpusCensusReport> {
    if options.workers == 0 || options.workers > MAX_CENSUS_WORKERS {
        return Err(Error::AccountingClosure(
            "census workers must be between 1 and 64",
        ));
    }
    let worker_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .thread_name(|ordinal| format!("rag-census-{ordinal}"))
        .build()
        .map_err(|_| Error::AccountingClosure("census worker pool could not start"))?;
    let reader = LocalSnapshotReader::open(&options.snapshot)?;
    let identity = reader.identity();
    let mut selected_relations = options.relations.clone();
    selected_relations.sort();
    selected_relations.dedup();
    let available_relations = reader
        .typed_relations()
        .map(|relation| relation.name.as_str())
        .collect::<BTreeSet<_>>();
    if selected_relations
        .iter()
        .any(|relation| !available_relations.contains(relation.as_str()))
    {
        return Err(Error::AccountingClosure("census relation scope is invalid"));
    }
    let context = ProjectionContext {
        snapshot: ProjectionComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ProjectionComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };
    let mut relations = BTreeMap::new();
    let mut documents = BTreeMap::<String, (String, String)>::new();
    let mut document_kinds = BTreeMap::<String, u64>::new();
    let mut total_source_rows = 0_u64;
    let mut total_semantic_occurrences = 0_u64;
    let mut total_structured_only = 0_u64;

    for relation in reader.typed_relations().filter(|relation| {
        selected_relations.is_empty() || selected_relations.contains(&relation.name)
    }) {
        let mut source_rows = 0_u64;
        let mut semantic_occurrences = 0_u64;
        let mut structured_only_occurrences = 0_u64;
        let mut relation_documents = BTreeMap::<String, String>::new();
        let admitted = reader.admit_object(relation)?;
        if relation.name == "ocsf_ext_livefire_system_metric" {
            // This relation is classified from its admitted schema, not its JSON values.
            // Admission above still verifies the source object's digest and footer.
            source_rows = relation.rows;
            structured_only_occurrences = relation.rows;
        } else {
            // Work in fixed-size waves. At most `workers` row groups and their
            // decoded Arrow batches are live at once; results are then merged
            // in admitted file order so worker scheduling cannot change output.
            map_row_groups_in_source_order(
                &worker_pool,
                &admitted,
                options.workers,
                |group, batch_workers| {
                    census_row_group(
                        &admitted,
                        group.ordinal,
                        &context,
                        &worker_pool,
                        batch_workers,
                    )
                },
                |group, partial| {
                    if partial.ordinal != group.ordinal || partial.source_rows != group.rows {
                        return Err(Error::AccountingClosure(
                            "census row-group row counts do not close",
                        ));
                    }
                    source_rows = source_rows
                        .checked_add(partial.source_rows)
                        .ok_or(Error::CountOverflow)?;
                    semantic_occurrences = semantic_occurrences
                        .checked_add(partial.semantic_occurrences)
                        .ok_or(Error::CountOverflow)?;
                    structured_only_occurrences = structured_only_occurrences
                        .checked_add(partial.structured_only_occurrences)
                        .ok_or(Error::CountOverflow)?;
                    for (document_id, (document_sha256, document_kind)) in partial.documents {
                        merge_census_document(
                            &mut documents,
                            &mut document_kinds,
                            &document_id,
                            &document_sha256,
                            &document_kind,
                        )?;
                        if let Some(existing_kind) =
                            relation_documents.insert(document_id.clone(), document_kind.clone())
                            && existing_kind != document_kind
                        {
                            return Err(Error::InconsistentDocument(document_id));
                        }
                    }
                    Ok(())
                },
            )?;
        }
        if source_rows != relation.rows
            || source_rows
                != semantic_occurrences
                    .checked_add(structured_only_occurrences)
                    .ok_or(Error::CountOverflow)?
        {
            return Err(Error::AccountingClosure(
                "census relation row counts do not close",
            ));
        }
        let mut relation_kinds = BTreeMap::<String, u64>::new();
        for kind in relation_documents.values() {
            *relation_kinds.entry(kind.clone()).or_default() += 1;
        }
        total_source_rows = total_source_rows
            .checked_add(source_rows)
            .ok_or(Error::CountOverflow)?;
        total_semantic_occurrences = total_semantic_occurrences
            .checked_add(semantic_occurrences)
            .ok_or(Error::CountOverflow)?;
        total_structured_only = total_structured_only
            .checked_add(structured_only_occurrences)
            .ok_or(Error::CountOverflow)?;
        relations.insert(
            relation.name.clone(),
            CensusRelationReport {
                source_rows,
                semantic_occurrences,
                structured_only_occurrences,
                distinct_documents: relation_documents.len() as u64,
                document_kinds: relation_kinds,
            },
        );
    }
    if total_source_rows
        != total_semantic_occurrences
            .checked_add(total_structured_only)
            .ok_or(Error::CountOverflow)?
    {
        return Err(Error::AccountingClosure(
            "census source counts do not close",
        ));
    }
    let document_order_sha256 = document_order_digest(documents.keys().map(String::as_str));
    let relations_counted = relations.keys().cloned().collect::<Vec<_>>();
    let mut report = CorpusCensusReport {
        schema_version: "livefire.rag.corpus-census/1",
        component_sha256: zero_digest()?,
        source_snapshot: component(
            &identity.snapshot_id,
            &identity.snapshot_version,
            identity.snapshot_sha256.as_str(),
        )?,
        mapping: component(
            &identity.mapping_id,
            &identity.mapping_version,
            identity.mapping_sha256.as_str(),
        )?,
        source_admission: source_admission_components(identity)?,
        projection_policy: projection_policy_component()?,
        relations_counted,
        source_rows: total_source_rows,
        semantic_occurrences: total_semantic_occurrences,
        structured_only_occurrences: total_structured_only,
        distinct_documents: documents.len() as u64,
        document_order_sha256,
        document_kinds,
        relations,
    };
    report.component_sha256 = component_digest(&report)?;
    Ok(report)
}

fn census_row_group(
    admitted: &AdmittedParquetObject,
    row_group_ordinal: usize,
    context: &ProjectionContext,
    worker_pool: &rayon::ThreadPool,
    batch_workers: usize,
) -> Result<CensusRowGroupReport> {
    let mut report = CensusRowGroupReport {
        ordinal: row_group_ordinal,
        source_rows: 0,
        semantic_occurrences: 0,
        structured_only_occurrences: 0,
        documents: BTreeMap::new(),
    };
    for batch in admitted.scan_row_group(row_group_ordinal, &["typed_event_json"])? {
        let batch = batch?;
        let rows = strings(&batch, "typed_event_json")?;
        map_batch_ranges_in_source_order(
            worker_pool,
            batch.num_rows(),
            batch_workers,
            |_range_ordinal, range| {
                let mut partial = CensusBatchProjection {
                    source_rows: 0,
                    semantic_occurrences: 0,
                    structured_only_occurrences: 0,
                    documents: BTreeMap::new(),
                };
                for row in range {
                    partial.source_rows = partial
                        .source_rows
                        .checked_add(1)
                        .ok_or(Error::CountOverflow)?;
                    let summary = project_document_summary(
                        &admitted.relation().name,
                        rows.value(row),
                        context,
                    )?;
                    let Some(document) = summary.document else {
                        partial.structured_only_occurrences = partial
                            .structured_only_occurrences
                            .checked_add(1)
                            .ok_or(Error::CountOverflow)?;
                        continue;
                    };
                    partial.semantic_occurrences = partial
                        .semantic_occurrences
                        .checked_add(1)
                        .ok_or(Error::CountOverflow)?;
                    let document_sha256 = sha256_bytes(&serde_json::to_vec(&document)?);
                    let document_kind = match document.document_kind {
                        rag_projection::DocumentKind::Activity => "activity",
                        rag_projection::DocumentKind::State => "state",
                        rag_projection::DocumentKind::Detection => "detection",
                        rag_projection::DocumentKind::StructuredOnly => "structured_only",
                    }
                    .to_owned();
                    if let Some((existing_sha256, existing_kind)) =
                        partial.documents.get(&document.document_id)
                        && (existing_sha256 != &document_sha256 || existing_kind != &document_kind)
                    {
                        return Err(Error::InconsistentDocument(document.document_id));
                    }
                    partial
                        .documents
                        .insert(document.document_id, (document_sha256, document_kind));
                }
                Ok(partial)
            },
            |_range_ordinal, range, partial| {
                if partial.source_rows
                    != u64::try_from(range.len()).map_err(|_| Error::CountOverflow)?
                    || partial.source_rows
                        != partial
                            .semantic_occurrences
                            .checked_add(partial.structured_only_occurrences)
                            .ok_or(Error::CountOverflow)?
                {
                    return Err(Error::AccountingClosure(
                        "census batch range row counts do not close",
                    ));
                }
                report.source_rows = report
                    .source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                report.semantic_occurrences = report
                    .semantic_occurrences
                    .checked_add(partial.semantic_occurrences)
                    .ok_or(Error::CountOverflow)?;
                report.structured_only_occurrences = report
                    .structured_only_occurrences
                    .checked_add(partial.structured_only_occurrences)
                    .ok_or(Error::CountOverflow)?;
                for (document_id, (document_sha256, document_kind)) in partial.documents {
                    if let Some((existing_sha256, existing_kind)) =
                        report.documents.get(&document_id)
                        && (existing_sha256 != &document_sha256 || existing_kind != &document_kind)
                    {
                        return Err(Error::InconsistentDocument(document_id));
                    }
                    report
                        .documents
                        .insert(document_id, (document_sha256, document_kind));
                }
                Ok(())
            },
        )?;
    }
    if report.source_rows
        != report
            .semantic_occurrences
            .checked_add(report.structured_only_occurrences)
            .ok_or(Error::CountOverflow)?
    {
        return Err(Error::AccountingClosure(
            "census row-group row counts do not close",
        ));
    }
    Ok(report)
}

fn merge_census_document(
    documents: &mut BTreeMap<String, (String, String)>,
    document_kinds: &mut BTreeMap<String, u64>,
    document_id: &str,
    document_sha256: &str,
    document_kind: &str,
) -> Result<()> {
    if let Some((existing_sha256, existing_kind)) = documents.get(document_id) {
        if existing_sha256 != document_sha256 || existing_kind != document_kind {
            return Err(Error::InconsistentDocument(document_id.to_owned()));
        }
    } else {
        documents.insert(
            document_id.to_owned(),
            (document_sha256.to_owned(), document_kind.to_owned()),
        );
        *document_kinds.entry(document_kind.to_owned()).or_default() += 1;
    }
    Ok(())
}

pub(crate) fn verify_tokenizer(options: VerifyTokenizerOptions) -> Result<()> {
    let report = build_tokenizer_verification_report(&options)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn build_tokenizer_verification_report(
    options: &VerifyTokenizerOptions,
) -> Result<TokenizerVerificationReport> {
    let reference_bytes = fs::read(&options.tokenizer_ref)?;
    let fixture_bytes = fs::read(&options.fixture)?;
    if reference_bytes.len() > 64 * 1024 || fixture_bytes.len() > 8 * 1024 * 1024 {
        return Err(Error::AccountingClosure(
            "tokenizer reference or parity fixture is too large",
        ));
    }
    let reference: ExecutableTokenizerRef = serde_json::from_slice(&reference_bytes)?;
    reference.validate()?;
    let fixture: TokenizerParityFixture = serde_json::from_slice(&fixture_bytes)?;
    if fixture.schema_version != TOKENIZER_PARITY_FIXTURE_SCHEMA
        || fixture.source.runtime.is_empty()
        || fixture.source.model_file.is_empty()
        || fixture.source.model_revision.is_empty()
        || fixture.source.source_tokenizer_json_revision.is_empty()
        || fixture.cases.is_empty()
        || fixture.generated_cases.is_empty()
        || fixture.cases.len() > 10_000
        || fixture.generated_cases.len() > 1_024
        || fixture.source.model_revision != reference.model_revision
        || fixture.source.executable_tokenizer_json_sha256 != reference.artifact.sha256
        || fixture.source.add_special_tokens != reference.add_special_tokens
    {
        return Err(Error::AccountingClosure(
            "tokenizer parity fixture is not bound to the tokenizer reference",
        ));
    }

    let tokenizer_bytes = fs::read(&options.tokenizer_json)?;
    let tokenizer = ExactTokenizer::from_bytes(reference.clone(), &tokenizer_bytes)?;
    let mut names = BTreeSet::new();
    let mut verified_tokens = 0_u64;
    for case in &fixture.cases {
        if case.name.is_empty() || !names.insert(case.name.as_str()) || case.token_ids.is_empty() {
            return Err(Error::AccountingClosure(
                "tokenizer parity direct case is invalid",
            ));
        }
        let actual = tokenizer.token_ids(&case.input)?;
        if actual != case.token_ids {
            return Err(Error::AccountingClosure(
                "tokenizer parity direct token IDs differ",
            ));
        }
        verified_tokens = verified_tokens
            .checked_add(u64::try_from(actual.len()).map_err(|_| Error::CountOverflow)?)
            .ok_or(Error::CountOverflow)?;
    }

    let maximum_input_bytes = reference.maximum_input_bytes;
    let mut boundary_cases = 0_u64;
    for case in &fixture.generated_cases {
        if case.name.is_empty()
            || !names.insert(case.name.as_str())
            || case.repeat.is_empty()
            || case.count == 0
            || case.token_count == 0
        {
            return Err(Error::AccountingClosure(
                "tokenizer parity generated case is invalid",
            ));
        }
        let repeat_bytes = u64::try_from(case.repeat.len()).map_err(|_| Error::CountOverflow)?;
        let input_bytes = repeat_bytes
            .checked_mul(case.count)
            .ok_or(Error::CountOverflow)?;
        if input_bytes > maximum_input_bytes {
            return Err(Error::AccountingClosure(
                "tokenizer parity generated input exceeds the reference byte limit",
            ));
        }
        if input_bytes == maximum_input_bytes {
            boundary_cases = boundary_cases.checked_add(1).ok_or(Error::CountOverflow)?;
        }
        let repetitions = usize::try_from(case.count).map_err(|_| Error::CountOverflow)?;
        let input = case.repeat.repeat(repetitions);
        if u64::try_from(input.len()).map_err(|_| Error::CountOverflow)? != input_bytes {
            return Err(Error::AccountingClosure(
                "tokenizer parity generated input byte count differs",
            ));
        }
        let actual = tokenizer.token_ids(&input)?;
        let actual_count = u64::try_from(actual.len()).map_err(|_| Error::CountOverflow)?;
        if actual_count != case.token_count {
            return Err(Error::AccountingClosure(
                "tokenizer parity generated token count differs",
            ));
        }
        let mut token_bytes = Vec::with_capacity(
            actual
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(Error::CountOverflow)?,
        );
        for token_id in &actual {
            token_bytes.extend_from_slice(&token_id.to_le_bytes());
        }
        if digest_bytes(&token_bytes) != case.token_ids_u32le_sha256 {
            return Err(Error::AccountingClosure(
                "tokenizer parity generated token digest differs",
            ));
        }
        verified_tokens = verified_tokens
            .checked_add(actual_count)
            .ok_or(Error::CountOverflow)?;
    }
    if boundary_cases == 0 {
        return Err(Error::AccountingClosure(
            "tokenizer parity fixture has no maximum-input boundary case",
        ));
    }

    let direct_cases = u64::try_from(fixture.cases.len()).map_err(|_| Error::CountOverflow)?;
    let generated_cases =
        u64::try_from(fixture.generated_cases.len()).map_err(|_| Error::CountOverflow)?;
    let mut report = TokenizerVerificationReport {
        schema_version: "livefire.rag.tokenizer-verification/1",
        component_sha256: zero_digest()?,
        status: "passed",
        fixture_sha256: digest_bytes(&fixture_bytes),
        tokenizer_reference_sha256: digest_bytes(&reference_bytes),
        executable_tokenizer: reference.artifact,
        target_tokenizer: reference.target_tokenizer,
        model_revision: reference.model_revision,
        source_tokenizer_json_revision: fixture.source.source_tokenizer_json_revision,
        source_tokenizer_json_sha256: fixture.source.source_tokenizer_json_sha256,
        direct_cases,
        generated_cases,
        maximum_input_boundary_cases: boundary_cases,
        verified_inputs: direct_cases
            .checked_add(generated_cases)
            .ok_or(Error::CountOverflow)?,
        verified_tokens,
    };
    report.component_sha256 = component_digest(&report)?;
    Ok(report)
}

pub(crate) fn prepare(options: PrepareOptions) -> Result<()> {
    let document_run_rows = match std::env::var(DOCUMENT_RUN_ROWS_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::AccountingClosure("LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be an integer")
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_DOCUMENT_RUN_ROWS,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::AccountingClosure(
                "LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be UTF-8",
            ));
        }
    };
    if !(MIN_CONFIGURED_DOCUMENT_RUN_ROWS..=MAX_DOCUMENT_RUN_ROWS).contains(&document_run_rows) {
        return Err(Error::AccountingClosure(
            "LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be between 1024 and 600000",
        ));
    }
    prepare_with_document_run_rows(options, document_run_rows, PreparationProjection::Generic)
}

pub(crate) fn prepare_commands(options: PrepareCommandsOptions) -> Result<()> {
    let document_run_rows = match std::env::var(DOCUMENT_RUN_ROWS_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::AccountingClosure("LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be an integer")
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_DOCUMENT_RUN_ROWS,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::AccountingClosure(
                "LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be UTF-8",
            ));
        }
    };
    if !(MIN_CONFIGURED_DOCUMENT_RUN_ROWS..=MAX_DOCUMENT_RUN_ROWS).contains(&document_run_rows) {
        return Err(Error::AccountingClosure(
            "LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS must be between 1024 and 600000",
        ));
    }
    prepare_with_document_run_rows(
        PrepareOptions {
            snapshot: options.snapshot,
            dataset_id: "livefire.rag.m45-command-evidence".to_owned(),
            dataset_version: "1".to_owned(),
            relations: vec![
                "ocsf_api_activity".to_owned(),
                "ocsf_event_log_activity".to_owned(),
                "ocsf_process_activity".to_owned(),
            ],
            out: options.out,
            document_shard_rows: options.document_shard_rows,
            workers: options.workers,
        },
        document_run_rows,
        PreparationProjection::M45Commands,
    )
}

fn prepare_with_document_run_rows(
    options: PrepareOptions,
    document_run_rows: usize,
    projection: PreparationProjection,
) -> Result<()> {
    if options.dataset_id.is_empty()
        || options.dataset_version.is_empty()
        || options.document_shard_rows == 0
        || options.document_shard_rows > 65_536
        || options.workers == 0
        || options.workers > MAX_PREPARE_WORKERS
    {
        return Err(Error::AccountingClosure(
            "invalid dataset preparation options",
        ));
    }
    let worker_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .thread_name(|ordinal| format!("rag-prepare-{ordinal}"))
        .build()
        .map_err(|_| Error::AccountingClosure("preparation worker pool could not start"))?;
    let reader = LocalSnapshotReader::open(&options.snapshot)?;
    let identity = reader.identity();
    if projection == PreparationProjection::M45Commands {
        validate_m45_command_source(identity)?;
    }
    let mut included = options.relations;
    included.sort();
    included.dedup();
    if included.is_empty() || included.iter().any(|relation| relation.is_empty()) {
        return Err(Error::AccountingClosure("dataset relation scope is empty"));
    }
    let available = reader
        .typed_relations()
        .map(|relation| (relation.name.as_str(), relation))
        .collect::<BTreeMap<_, _>>();
    if included
        .iter()
        .any(|relation| !available.contains_key(relation.as_str()))
        || included
            .iter()
            .any(|relation| relation == "ocsf_ext_livefire_system_metric")
    {
        return Err(Error::AccountingClosure(
            "dataset relation scope is invalid",
        ));
    }

    let context = ProjectionContext {
        snapshot: ProjectionComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ProjectionComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };
    let staging = AtomicDirectory::new(&options.out)?;
    let root = staging.path();
    let mut document_runs =
        SortedDocumentRuns::new(root.join(".document-runs"), document_run_rows)?;
    let mut searchable_by_relation = BTreeMap::<String, u64>::new();
    let mut occurrence_objects = Vec::new();
    let mut occurrence_count = 0_u64;

    for relation_name in &included {
        let relation = available[relation_name.as_str()];
        let admitted = reader.admit_object(relation)?;
        let mut relation_source_rows = 0_u64;
        let mut occurrence_buffer = Vec::with_capacity(OCCURRENCE_SHARD_ROWS);
        let mut relation_part = 0_u64;
        map_row_groups_in_source_order(
            &worker_pool,
            &admitted,
            options.workers,
            |group, batch_workers| {
                project_prepared_row_group(
                    &admitted,
                    group.ordinal,
                    group.first_row,
                    relation_name,
                    &context,
                    PreparedProjectionExecution {
                        batch: BatchProjectionExecution {
                            worker_pool: &worker_pool,
                            workers: batch_workers,
                        },
                        projection,
                    },
                )
            },
            |group, partial| {
                if partial.ordinal != group.ordinal || partial.source_rows != group.rows {
                    return Err(Error::AccountingClosure(
                        "prepared row-group rows do not close",
                    ));
                }
                relation_source_rows = relation_source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                occurrence_buffer.extend(partial.occurrences);
                document_runs.add_all(partial.documents)?;
                flush_occurrence_shards(
                    root,
                    relation_name,
                    &mut occurrence_buffer,
                    &mut relation_part,
                    &mut occurrence_objects,
                    false,
                )
            },
        )?;
        if relation_source_rows != relation.rows {
            return Err(Error::AccountingClosure(
                "prepared relation source rows do not close",
            ));
        }
        let relation_occurrence_count = relation_part
            .checked_mul(OCCURRENCE_SHARD_ROWS as u64)
            .and_then(|count| count.checked_add(occurrence_buffer.len() as u64))
            .ok_or(Error::CountOverflow)?;
        flush_occurrence_shards(
            root,
            relation_name,
            &mut occurrence_buffer,
            &mut relation_part,
            &mut occurrence_objects,
            true,
        )?;
        occurrence_count = occurrence_count
            .checked_add(relation_occurrence_count)
            .ok_or(Error::CountOverflow)?;
        searchable_by_relation.insert(relation_name.clone(), relation_occurrence_count);
    }
    if document_runs.is_empty() {
        return Err(Error::AccountingClosure(
            "dataset produced no searchable documents",
        ));
    }

    let mut document_objects = Vec::new();
    let mut document_buffer = Vec::with_capacity(options.document_shard_rows);
    let mut document_count = 0_u64;
    let mut document_order_hasher = Sha256::new();
    let mut embedding_input_order_hasher = Sha256::new();
    embedding_input_order_hasher.update(b"livefire.rag.embedding-input-order/1\0");
    let mut previous_document_id: Option<String> = None;
    document_runs.merge(|mut accumulated| {
        let document_ordinal = document_count;
        accumulated.document.vector_ordinal = document_ordinal;
        accumulated.document.relations_json = canonical_string(&accumulated.relations)?;
        let row = PreparedDocumentRow {
            document_ordinal,
            document_id: accumulated.document.document_id,
            document_sha256: Digest::new(accumulated.document.document_sha256)?,
            semantic_text_sha256: digest_bytes(accumulated.document.semantic_text.as_bytes()),
            semantic_text: accumulated.document.semantic_text,
            document_kind: match accumulated.document.document_kind.as_str() {
                "activity" => DocumentKind::Activity,
                "state" => DocumentKind::State,
                "detection" => DocumentKind::Detection,
                _ => return Err(Error::AccountingClosure("unknown projected document kind")),
            },
            primary_relation: accumulated.primary_relation,
            facets_json: accumulated.document.facets_json,
            relations_json: accumulated.document.relations_json,
            occurrence_count: accumulated.document.occurrence_count,
        };
        if previous_document_id
            .as_ref()
            .is_some_and(|previous| previous >= &row.document_id)
        {
            return Err(Error::AccountingClosure(
                "merged prepared documents are not strictly ordered",
            ));
        }
        previous_document_id = Some(row.document_id.clone());
        document_order_hasher.update(row.document_id.as_bytes());
        document_order_hasher.update([0]);
        for field in [
            row.document_id.as_str(),
            row.document_sha256.as_str(),
            row.semantic_text_sha256.as_str(),
        ] {
            embedding_input_order_hasher.update(field.as_bytes());
            embedding_input_order_hasher.update([0]);
        }
        document_count = document_count.checked_add(1).ok_or(Error::CountOverflow)?;
        document_buffer.push(row);
        flush_document_shard(
            root,
            &mut document_buffer,
            options.document_shard_rows,
            &mut document_objects,
            false,
        )
    })?;
    flush_document_shard(
        root,
        &mut document_buffer,
        options.document_shard_rows,
        &mut document_objects,
        true,
    )?;
    let all_typed = reader
        .typed_relations()
        .map(|relation| relation.name.clone())
        .collect::<Vec<_>>();
    let structured_only = all_typed
        .iter()
        .filter(|relation| relation.as_str() == "ocsf_ext_livefire_system_metric")
        .cloned()
        .collect::<Vec<_>>();
    let excluded = all_typed
        .iter()
        .filter(|relation| !included.contains(relation) && !structured_only.contains(relation))
        .cloned()
        .collect::<Vec<_>>();
    let dataset = DatasetIdentity {
        id: options.dataset_id,
        version: options.dataset_version,
        source_snapshot: component(
            &identity.snapshot_id,
            &identity.snapshot_version,
            identity.snapshot_sha256.as_str(),
        )?,
        mapping: component(
            &identity.mapping_id,
            &identity.mapping_version,
            identity.mapping_sha256.as_str(),
        )?,
        source_admission: source_admission_components(identity)?,
        included_relations: included.clone(),
        excluded_relations: excluded.clone(),
        structured_only_relations: structured_only.clone(),
    };

    // Object ordinals are global and contiguous across relation directories.
    for (ordinal, object) in occurrence_objects.iter_mut().enumerate() {
        object.ordinal = ordinal as u32;
    }

    let mut relation_accounting = BTreeMap::new();
    for relation in reader.typed_relations() {
        let searchable = searchable_by_relation
            .get(&relation.name)
            .copied()
            .unwrap_or(0);
        relation_accounting.insert(
            relation.name.clone(),
            RelationAccounting {
                source_rows: relation.rows,
                searchable_occurrences: searchable,
                selected_occurrences: searchable,
                excluded_rows: if included.contains(&relation.name) {
                    relation
                        .rows
                        .checked_sub(searchable)
                        .ok_or(Error::AccountingClosure(
                            "searchable rows exceed source rows",
                        ))?
                } else {
                    relation.rows
                },
            },
        );
    }
    let mut manifest = PreparedCorpusManifest {
        schema_version: PREPARED_CORPUS_SCHEMA.into(),
        component_sha256: zero_digest()?,
        dataset,
        projection_policy: match projection {
            PreparationProjection::Generic => projection_policy_component()?,
            PreparationProjection::M45Commands => m45_command_projection_policy_component()?,
        },
        document_schema: component(
            "livefire.rag.prepared-document-row",
            "1",
            &sha256_bytes(DOCUMENT_SCHEMA_BYTES),
        )?,
        occurrence_schema: component(
            "livefire.rag.prepared-occurrence-row",
            "1",
            &sha256_bytes(OCCURRENCE_SCHEMA_BYTES),
        )?,
        preparation_implementation: component(
            match projection {
                PreparationProjection::Generic => "livefire.rag.portable-preparation",
                PreparationProjection::M45Commands => "livefire.rag.m45-command-preparation",
            },
            env!("CARGO_PKG_VERSION"),
            &sha256_bytes(PREPARATION_SOURCE_BYTES),
        )?,
        document_count,
        occurrence_count,
        document_order_sha256: Digest::new(format!("{:x}", document_order_hasher.finalize()))?,
        embedding_input_order_sha256: Digest::new(format!(
            "{:x}",
            embedding_input_order_hasher.finalize()
        ))?,
        documents: document_objects,
        occurrences: occurrence_objects,
        relation_accounting,
    };
    manifest.seal()?;
    validate_streamed_prepared_documents(root, &manifest)?;
    write_canonical_json(&root.join(MANIFEST_FILE), &manifest)?;
    write_canonical_json(
        &root.join("accounting.json"),
        &json!({
            "schema_version": "livefire.rag.prepared-accounting/1",
            "dataset": manifest.dataset,
            "documents": manifest.document_count,
            "occurrences": manifest.occurrence_count,
            "preparation_document_grouping": "bounded_sorted_runs_v1",
            "occurrence_shard_rows": OCCURRENCE_SHARD_ROWS,
            "relations": manifest.relation_accounting,
        }),
    )?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

/// Build the three standard performance corpora without first publishing a
/// complete prepared copy of the source. The first pass records the candidate
/// universe; the second pass writes exact rows only for selected documents.
pub(crate) fn prepare_benchmark(options: PrepareBenchmarkOptions) -> Result<()> {
    if options.dataset_id.is_empty()
        || options.dataset_version.is_empty()
        || options.selection_seed.is_empty()
        || options.document_shard_rows == 0
        || options.document_shard_rows > 65_536
        || options.workers == 0
        || options.workers > MAX_PREPARE_WORKERS
    {
        return Err(Error::AccountingClosure(
            "invalid benchmark preparation options",
        ));
    }
    let worker_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .thread_name(|ordinal| format!("rag-prepare-{ordinal}"))
        .build()
        .map_err(|_| Error::AccountingClosure("preparation worker pool could not start"))?;
    let reader = LocalSnapshotReader::open(&options.snapshot)?;
    let identity = reader.identity();
    let mut included = options.relations;
    included.sort();
    included.dedup();
    let available = reader
        .typed_relations()
        .map(|relation| (relation.name.as_str(), relation))
        .collect::<BTreeMap<_, _>>();
    if included.is_empty()
        || included.iter().any(|relation| relation.is_empty())
        || included
            .iter()
            .any(|relation| !available.contains_key(relation.as_str()))
        || included
            .iter()
            .any(|relation| relation == "ocsf_ext_livefire_system_metric")
    {
        return Err(Error::AccountingClosure(
            "benchmark relation scope is invalid",
        ));
    }

    let context = projection_context(identity);
    let dataset = portable_dataset_identity(
        &reader,
        identity,
        options.dataset_id,
        options.dataset_version,
        &included,
    )?;
    let projection_policy = projection_policy_component()?;
    let admitted = included
        .iter()
        .map(|relation_name| {
            Ok((
                relation_name.clone(),
                reader.admit_object(available[relation_name.as_str()])?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;

    let mut candidate_map = BTreeMap::<String, BenchmarkCandidateAccumulator>::new();
    for relation_name in &included {
        let object = &admitted[relation_name];
        let mut relation_source_rows = 0_u64;
        map_row_groups_in_source_order(
            &worker_pool,
            object,
            options.workers,
            |group, batch_workers| {
                benchmark_candidate_row_group(
                    object,
                    group.ordinal,
                    relation_name,
                    &context,
                    &worker_pool,
                    batch_workers,
                )
            },
            |group, partial| {
                if partial.ordinal != group.ordinal || partial.source_rows != group.rows {
                    return Err(Error::AccountingClosure(
                        "benchmark candidate row-group rows do not close",
                    ));
                }
                relation_source_rows = relation_source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                for accumulated in partial.candidates.into_values() {
                    merge_benchmark_candidate(&mut candidate_map, accumulated.candidate)?;
                }
                Ok(())
            },
        )?;
        if relation_source_rows != object.relation().rows {
            return Err(Error::AccountingClosure(
                "benchmark candidate source rows do not close",
            ));
        }
    }
    let candidates = candidate_map
        .values()
        .map(|entry| entry.candidate.clone())
        .collect::<Vec<_>>();
    let length_strata = benchmark_length_strata(&candidates)?;
    let mut policy = benchmark_selection_policy(
        &dataset,
        &length_strata,
        &candidates,
        options.selection_seed,
    )?;
    policy.seal(&dataset)?;
    let (candidate_count, candidate_universe_sha256, selections) =
        select_benchmark_documents(&dataset, &projection_policy, &policy, &candidates)?;
    let selected_ranks = selections
        .iter()
        .map(|row| (row.document_id.as_str(), row.selection_rank))
        .collect::<BTreeMap<_, _>>();

    let staging = AtomicDirectory::new(&options.out)?;
    let mut stages = STANDARD_BENCHMARK_SIZES
        .into_iter()
        .map(|document_count| {
            let root = staging.path().join(format!("prepared-{document_count:05}"));
            fs::create_dir_all(&root)?;
            Ok(BenchmarkCorpusStage {
                document_count,
                root,
                occurrence_objects: Vec::new(),
                occurrence_count: 0,
                selected_occurrences: BTreeMap::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut selected_documents = BTreeMap::<String, DocumentAccumulator>::new();

    for relation_name in &included {
        let object = &admitted[relation_name];
        let mut relation_source_rows = 0_u64;
        let mut buffers = stages
            .iter()
            .map(|_| Vec::<PreparedOccurrenceRow>::with_capacity(OCCURRENCE_SHARD_ROWS))
            .collect::<Vec<_>>();
        let mut relation_parts = vec![0_u64; stages.len()];
        // Projection is parallel within a bounded wave of admitted row
        // groups. Only selected rows are retained in each fragment. Fragments
        // are merged and written in source row-group order, so worker timing
        // cannot change occurrence order or Parquet shard bytes.
        map_row_groups_in_source_order(
            &worker_pool,
            object,
            options.workers,
            |group, batch_workers| {
                benchmark_occurrence_row_group(
                    object,
                    group.ordinal,
                    group.first_row,
                    relation_name,
                    &context,
                    &selected_ranks,
                    BatchProjectionExecution {
                        worker_pool: &worker_pool,
                        workers: batch_workers,
                    },
                )
            },
            |group, partial| {
                if partial.ordinal != group.ordinal || partial.source_rows != group.rows {
                    return Err(Error::AccountingClosure(
                        "benchmark occurrence row-group rows do not close",
                    ));
                }
                relation_source_rows = relation_source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                for selected_occurrence in partial.selected {
                    merge_benchmark_selected_occurrence(
                        relation_name,
                        selected_occurrence,
                        &selections,
                        &mut selected_documents,
                        &mut stages,
                        &mut buffers,
                        &mut relation_parts,
                    )?;
                }
                Ok(())
            },
        )?;
        if relation_source_rows != object.relation().rows {
            return Err(Error::AccountingClosure(
                "benchmark source rows do not close",
            ));
        }
        for (stage_index, stage) in stages.iter_mut().enumerate() {
            flush_occurrence_shards(
                &stage.root,
                relation_name,
                &mut buffers[stage_index],
                &mut relation_parts[stage_index],
                &mut stage.occurrence_objects,
                true,
            )?;
            stage.occurrence_count = stage
                .occurrence_count
                .checked_add(
                    stage
                        .selected_occurrences
                        .get(relation_name)
                        .copied()
                        .unwrap_or(0),
                )
                .ok_or(Error::CountOverflow)?;
        }
    }
    if selected_documents.len() != STANDARD_BENCHMARK_SIZES[2] as usize {
        return Err(Error::AccountingClosure(
            "selected benchmark documents do not close",
        ));
    }

    let mut published = Vec::<BenchmarkPublishedCorpus>::new();
    let mut prepared_for_validation = Vec::new();
    for stage in &mut stages {
        for (ordinal, object) in stage.occurrence_objects.iter_mut().enumerate() {
            object.ordinal = u32::try_from(ordinal)
                .map_err(|_| Error::AccountingClosure("too many occurrence shards"))?;
        }
        let (manifest, documents) = write_benchmark_prepared_corpus(
            stage,
            &reader,
            &dataset,
            &projection_policy,
            &selected_ranks,
            &selected_documents,
            options.document_shard_rows,
        )?;
        let binding = bind_benchmark_prepared_corpus(
            stage.document_count,
            &selections,
            &dataset,
            &projection_policy,
            &manifest,
            &documents,
        )?;
        published.push(binding);
        prepared_for_validation.push((manifest, documents));
    }

    let manifest = build_benchmark_selection_manifest(
        dataset,
        projection_policy,
        component(
            "livefire.rag.benchmark-preparation",
            env!("CARGO_PKG_VERSION"),
            &sha256_bytes(PREPARATION_SOURCE_BYTES),
        )?,
        policy,
        candidate_count,
        candidate_universe_sha256,
        selections,
        published,
    )?;
    manifest.validate_against_candidates(&candidates)?;
    let validation_inputs = prepared_for_validation
        .iter()
        .map(|(prepared, documents)| (prepared, documents.as_slice()))
        .collect::<Vec<_>>();
    manifest.validate_prepared_corpora(&validation_inputs)?;
    write_canonical_json(&staging.path().join("selection-manifest.json"), &manifest)?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn benchmark_candidate_row_group(
    object: &AdmittedParquetObject,
    row_group_ordinal: usize,
    relation_name: &str,
    context: &ProjectionContext,
    worker_pool: &rayon::ThreadPool,
    batch_workers: usize,
) -> Result<BenchmarkCandidateRowGroup> {
    let mut source_rows = 0_u64;
    let mut candidates = BTreeMap::new();
    for batch in object.scan_row_group(row_group_ordinal, &["typed_event_json"])? {
        let batch = batch?;
        let json_rows = strings(&batch, "typed_event_json")?;
        map_batch_ranges_in_source_order(
            worker_pool,
            batch.num_rows(),
            batch_workers,
            |_range_ordinal, range| {
                let mut partial = BenchmarkCandidateBatchProjection {
                    source_rows: 0,
                    candidates: BTreeMap::new(),
                };
                for row in range {
                    partial.source_rows = partial
                        .source_rows
                        .checked_add(1)
                        .ok_or(Error::CountOverflow)?;
                    let summary =
                        project_document_summary(relation_name, json_rows.value(row), context)?;
                    let Some(document) = summary.document else {
                        continue;
                    };
                    let document_sha256 = sha256_bytes(&serde_json::to_vec(&document)?);
                    let candidate = BenchmarkSelectionCandidate {
                        document_id: document.document_id.clone(),
                        document_sha256: Digest::new(document_sha256)?,
                        semantic_text_sha256: digest_bytes(document.semantic_text.as_bytes()),
                        semantic_text_utf8_bytes: u64::try_from(document.semantic_text.len())
                            .map_err(|_| Error::CountOverflow)?,
                        primary_relation: relation_name.to_owned(),
                    };
                    merge_benchmark_candidate(&mut partial.candidates, candidate)?;
                }
                Ok(partial)
            },
            |_range_ordinal, range, partial| {
                if partial.source_rows
                    != u64::try_from(range.len()).map_err(|_| Error::CountOverflow)?
                {
                    return Err(Error::AccountingClosure(
                        "benchmark candidate batch range rows do not close",
                    ));
                }
                source_rows = source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                for accumulated in partial.candidates.into_values() {
                    merge_benchmark_candidate(&mut candidates, accumulated.candidate)?;
                }
                Ok(())
            },
        )?;
    }
    Ok(BenchmarkCandidateRowGroup {
        ordinal: row_group_ordinal,
        source_rows,
        candidates,
    })
}

fn merge_benchmark_candidate(
    candidates: &mut BTreeMap<String, BenchmarkCandidateAccumulator>,
    candidate: BenchmarkSelectionCandidate,
) -> Result<()> {
    match candidates.get(&candidate.document_id) {
        Some(existing)
            if existing.candidate.document_sha256 != candidate.document_sha256
                || existing.candidate.semantic_text_sha256 != candidate.semantic_text_sha256 =>
        {
            Err(Error::InconsistentDocument(candidate.document_id))
        }
        Some(_) => Ok(()),
        None => {
            if candidates.len() >= MAX_BENCHMARK_CANDIDATES {
                return Err(Error::AccountingClosure(
                    "benchmark candidate limit exceeded",
                ));
            }
            candidates.insert(
                candidate.document_id.clone(),
                BenchmarkCandidateAccumulator { candidate },
            );
            Ok(())
        }
    }
}

fn benchmark_occurrence_row_group(
    object: &AdmittedParquetObject,
    row_group_ordinal: usize,
    first_source_row: u64,
    relation_name: &str,
    context: &ProjectionContext,
    selected_ranks: &BTreeMap<&str, u64>,
    execution: BatchProjectionExecution<'_>,
) -> Result<BenchmarkOccurrenceRowGroup> {
    let mut source_rows = 0_u64;
    let mut selected = Vec::new();
    for batch in object.scan_row_group(
        row_group_ordinal,
        &["event_id", "typed_event_json", "support_ref"],
    )? {
        let batch = batch?;
        let event_ids = strings(&batch, "event_id")?;
        let json_rows = strings(&batch, "typed_event_json")?;
        let support = strings(&batch, "support_ref")?;
        let batch_first_row = first_source_row
            .checked_add(source_rows)
            .ok_or(Error::CountOverflow)?;
        map_batch_ranges_in_source_order(
            execution.worker_pool,
            batch.num_rows(),
            execution.workers,
            |_range_ordinal, range| {
                let mut partial = BenchmarkOccurrenceBatchProjection {
                    source_rows: 0,
                    selected: Vec::new(),
                };
                for row in range {
                    let source_row_ordinal = batch_first_row
                        .checked_add(u64::try_from(row).map_err(|_| Error::CountOverflow)?)
                        .ok_or(Error::CountOverflow)?;
                    partial.source_rows = partial
                        .source_rows
                        .checked_add(1)
                        .ok_or(Error::CountOverflow)?;
                    let summary =
                        project_document_summary(relation_name, json_rows.value(row), context)?;
                    let Some(summary_document) = summary.document else {
                        continue;
                    };
                    let Some(&selection_rank) =
                        selected_ranks.get(summary_document.document_id.as_str())
                    else {
                        continue;
                    };
                    let projected = project(ProjectionInput {
                        relation_name,
                        event_id: event_ids.value(row),
                        typed_event_json: json_rows.value(row),
                        support_ref: support.value(row),
                        context,
                    })?;
                    let document = projected.document.as_ref().ok_or(Error::AccountingClosure(
                        "selected projection became structured-only",
                    ))?;
                    let document_sha256 = sha256_bytes(&serde_json::to_vec(document)?);
                    let mut fast = fast_document(document.clone(), document_sha256)?;
                    fast.facets_json = canonical_string(&document.facets)?;
                    let occurrence = prepared_occurrence(
                        relation_name,
                        source_row_ordinal,
                        event_ids.value(row),
                        support.value(row),
                        context,
                        &projected,
                    )?;
                    partial.selected.push(BenchmarkSelectedOccurrence {
                        selection_rank,
                        document: fast,
                        occurrence,
                    });
                }
                Ok(partial)
            },
            |_range_ordinal, range, partial| {
                if partial.source_rows
                    != u64::try_from(range.len()).map_err(|_| Error::CountOverflow)?
                {
                    return Err(Error::AccountingClosure(
                        "benchmark occurrence batch range rows do not close",
                    ));
                }
                source_rows = source_rows
                    .checked_add(partial.source_rows)
                    .ok_or(Error::CountOverflow)?;
                selected.extend(partial.selected);
                Ok(())
            },
        )?;
    }
    Ok(BenchmarkOccurrenceRowGroup {
        ordinal: row_group_ordinal,
        source_rows,
        selected,
    })
}

fn merge_benchmark_selected_occurrence(
    relation_name: &str,
    selected_occurrence: BenchmarkSelectedOccurrence,
    selections: &[BenchmarkSelectionRow],
    selected_documents: &mut BTreeMap<String, DocumentAccumulator>,
    stages: &mut [BenchmarkCorpusStage],
    buffers: &mut [Vec<PreparedOccurrenceRow>],
    relation_parts: &mut [u64],
) -> Result<()> {
    let selection = selections
        .get(
            usize::try_from(selected_occurrence.selection_rank)
                .map_err(|_| Error::CountOverflow)?,
        )
        .ok_or(Error::AccountingClosure(
            "selected benchmark rank is outside the selection",
        ))?;
    let document = selected_occurrence.document;
    if document.document_id != selection.document_id
        || document.document_sha256 != selection.document_sha256.as_str()
    {
        return Err(Error::InconsistentDocument(document.document_id));
    }
    let entry = selected_documents
        .entry(document.document_id.clone())
        .or_insert_with(|| DocumentAccumulator {
            document: document.clone(),
            primary_relation: relation_name.to_owned(),
            relations: BTreeSet::from([relation_name.to_owned()]),
        });
    if entry.document.document_sha256 != document.document_sha256
        || entry.document.semantic_text != document.semantic_text
        || entry.document.facets_json != document.facets_json
    {
        return Err(Error::InconsistentDocument(document.document_id));
    }
    entry.document.occurrence_count = entry
        .document
        .occurrence_count
        .checked_add(1)
        .ok_or(Error::CountOverflow)?;
    entry.relations.insert(relation_name.to_owned());
    for (stage_index, stage) in stages.iter_mut().enumerate() {
        if selected_occurrence.selection_rank >= stage.document_count {
            continue;
        }
        buffers[stage_index].push(selected_occurrence.occurrence.clone());
        let selected_count = stage
            .selected_occurrences
            .entry(relation_name.to_owned())
            .or_default();
        *selected_count = selected_count.checked_add(1).ok_or(Error::CountOverflow)?;
        flush_occurrence_shards(
            &stage.root,
            relation_name,
            &mut buffers[stage_index],
            &mut relation_parts[stage_index],
            &mut stage.occurrence_objects,
            false,
        )?;
    }
    Ok(())
}

fn projection_context(identity: &OcsfSnapshot) -> ProjectionContext {
    ProjectionContext {
        snapshot: ProjectionComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ProjectionComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    }
}

fn portable_dataset_identity(
    reader: &LocalSnapshotReader,
    identity: &OcsfSnapshot,
    id: String,
    version: String,
    included: &[String],
) -> Result<DatasetIdentity> {
    let all_typed = reader
        .typed_relations()
        .map(|relation| relation.name.clone())
        .collect::<Vec<_>>();
    let structured_only = all_typed
        .iter()
        .filter(|relation| relation.as_str() == "ocsf_ext_livefire_system_metric")
        .cloned()
        .collect::<Vec<_>>();
    let excluded = all_typed
        .iter()
        .filter(|relation| !included.contains(relation) && !structured_only.contains(relation))
        .cloned()
        .collect::<Vec<_>>();
    Ok(DatasetIdentity {
        id,
        version,
        source_snapshot: component(
            &identity.snapshot_id,
            &identity.snapshot_version,
            identity.snapshot_sha256.as_str(),
        )?,
        mapping: component(
            &identity.mapping_id,
            &identity.mapping_version,
            identity.mapping_sha256.as_str(),
        )?,
        source_admission: source_admission_components(identity)?,
        included_relations: included.to_vec(),
        excluded_relations: excluded,
        structured_only_relations: structured_only,
    })
}

fn source_admission_components(identity: &OcsfSnapshot) -> Result<Vec<ComponentRef>> {
    let Some(capability_sha256) = &identity.snapshot_capabilities_sha256 else {
        return Ok(Vec::new());
    };
    let mut components = vec![
        component(
            &identity.relation_contract_id,
            &identity.relation_contract_version,
            identity.relation_contract_sha256.as_str(),
        )?,
        component(
            "livefire.ocsf.snapshot-capabilities",
            "1",
            capability_sha256.as_str(),
        )?,
    ];
    components.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(components)
}

fn benchmark_length_strata(
    candidates: &[BenchmarkSelectionCandidate],
) -> Result<Vec<BenchmarkLengthStratum>> {
    if candidates.len() < STANDARD_BENCHMARK_SIZES[2] as usize {
        return Err(Error::AccountingClosure(
            "benchmark needs at least 10,000 documents",
        ));
    }
    let mut lengths = candidates
        .iter()
        .map(|candidate| candidate.semantic_text_utf8_bytes)
        .collect::<Vec<_>>();
    lengths.sort_unstable();
    let maximum = *lengths
        .last()
        .ok_or(Error::AccountingClosure("benchmark has no text lengths"))?;
    if maximum == 0 {
        return Err(Error::AccountingClosure("benchmark semantic text is empty"));
    }
    let last_index = lengths.len() - 1;
    let mut upper_bounds = [50_usize, 90, 95, 99]
        .into_iter()
        .map(|percent| lengths[last_index.saturating_mul(percent) / 100])
        .filter(|upper| *upper < maximum)
        .collect::<BTreeSet<_>>();
    // Give the observed maximum its own final stratum. This guarantees that
    // the 10,000-document corpus covers the longest source text when capacity
    // permits, even when several percentile boundaries are identical.
    upper_bounds.insert(maximum - 1);

    let mut minimum = 0_u64;
    let mut strata = Vec::new();
    for upper in upper_bounds {
        if upper < minimum {
            continue;
        }
        strata.push(BenchmarkLengthStratum {
            id: format!("bytes_{minimum}_to_{upper}"),
            minimum_utf8_bytes: minimum,
            maximum_utf8_bytes: Some(upper),
        });
        minimum = upper.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    strata.push(BenchmarkLengthStratum {
        id: format!("bytes_{minimum}_and_up"),
        minimum_utf8_bytes: minimum,
        maximum_utf8_bytes: None,
    });
    Ok(strata)
}

fn benchmark_selection_policy(
    dataset: &DatasetIdentity,
    strata: &[BenchmarkLengthStratum],
    candidates: &[BenchmarkSelectionCandidate],
    selection_seed: String,
) -> Result<BenchmarkSelectionPolicy> {
    let cells = dataset
        .included_relations
        .iter()
        .flat_map(|relation| {
            strata
                .iter()
                .map(move |stratum| (relation.clone(), stratum.id.clone()))
        })
        .collect::<Vec<_>>();
    let mut capacities = vec![0_u64; cells.len()];
    for candidate in candidates {
        let stratum_index = strata
            .iter()
            .position(|stratum| {
                candidate.semantic_text_utf8_bytes >= stratum.minimum_utf8_bytes
                    && stratum
                        .maximum_utf8_bytes
                        .is_none_or(|maximum| candidate.semantic_text_utf8_bytes <= maximum)
            })
            .ok_or(Error::AccountingClosure(
                "candidate length is outside benchmark strata",
            ))?;
        let relation_index = dataset
            .included_relations
            .iter()
            .position(|relation| relation == &candidate.primary_relation)
            .ok_or(Error::AccountingClosure(
                "candidate relation is outside benchmark scope",
            ))?;
        let cell_index = relation_index
            .checked_mul(strata.len())
            .and_then(|index| index.checked_add(stratum_index))
            .ok_or(Error::CountOverflow)?;
        capacities[cell_index] = capacities[cell_index]
            .checked_add(1)
            .ok_or(Error::CountOverflow)?;
    }
    let capacity = capacities.iter().try_fold(0_u64, |total, value| {
        total.checked_add(*value).ok_or(Error::CountOverflow)
    })?;
    if capacity < STANDARD_BENCHMARK_SIZES[2] {
        return Err(Error::AccountingClosure(
            "benchmark needs at least 10,000 documents",
        ));
    }

    let mut allocated = vec![0_u64; cells.len()];
    let mut allocated_total = 0_u64;
    let mut targets = Vec::new();
    for document_count in STANDARD_BENCHMARK_SIZES {
        while allocated_total < document_count {
            let mut progressed = false;
            for relation_index in 0..dataset.included_relations.len() {
                if allocated_total == document_count {
                    break;
                }
                let first = relation_index * strata.len();
                let chosen = (first..first + strata.len())
                    .filter(|index| allocated[*index] < capacities[*index])
                    .min_by_key(|index| (allocated[*index], *index));
                let Some(index) = chosen else {
                    continue;
                };
                allocated[index] = allocated[index]
                    .checked_add(1)
                    .ok_or(Error::CountOverflow)?;
                allocated_total = allocated_total.checked_add(1).ok_or(Error::CountOverflow)?;
                progressed = true;
            }
            if !progressed {
                return Err(Error::AccountingClosure(
                    "benchmark quota exceeds candidate capacity",
                ));
            }
        }
        targets.push(BenchmarkTargetQuota {
            document_count,
            quotas: cells
                .iter()
                .zip(&allocated)
                .map(
                    |((relation, length_stratum), documents)| BenchmarkStratumQuota {
                        relation: relation.clone(),
                        length_stratum: length_stratum.clone(),
                        documents: *documents,
                    },
                )
                .collect(),
        });
    }
    Ok(BenchmarkSelectionPolicy {
        schema_version: "livefire.rag.benchmark-selection-policy/1".into(),
        component_sha256: zero_digest()?,
        algorithm: "staged_stratified_sha256_v1".into(),
        selection_seed,
        length_strata: strata.to_vec(),
        targets,
    })
}

fn prepared_occurrence(
    relation: &str,
    source_row_ordinal: u64,
    event_id: &str,
    support_ref: &str,
    context: &ProjectionContext,
    projected: &ProjectionOutput,
) -> Result<PreparedOccurrenceRow> {
    let document = projected
        .document
        .as_ref()
        .ok_or(Error::AccountingClosure("selected document is absent"))?;
    let event_time_ms = parse_event_time_ms(
        projected.occurrence.event_time.as_deref(),
        projected.occurrence.event_time_availability,
    )?;
    let mut occurrence_hasher = Sha256::new();
    occurrence_hasher.update(context.snapshot.sha256.as_bytes());
    occurrence_hasher.update([0]);
    occurrence_hasher.update(relation.as_bytes());
    occurrence_hasher.update([0]);
    occurrence_hasher.update(event_id.as_bytes());
    Ok(PreparedOccurrenceRow {
        occurrence_id: format!("occ-{:x}", occurrence_hasher.finalize()),
        document_id: document.document_id.clone(),
        event_time_ms,
        relation: relation.to_owned(),
        source_row_ordinal,
        exact_attributes_json: canonical_string(&projected.occurrence.exact_attributes)?,
        snapshot_sha256: Digest::new(context.snapshot.sha256.clone())?,
        mapping_sha256: Digest::new(context.mapping_pack.sha256.clone())?,
        event_id: event_id.to_owned(),
        support_ref: support_ref.to_owned(),
    })
}

#[allow(clippy::too_many_arguments)]
fn write_benchmark_prepared_corpus(
    stage: &BenchmarkCorpusStage,
    reader: &LocalSnapshotReader,
    dataset: &DatasetIdentity,
    projection_policy: &ComponentRef,
    selected_ranks: &BTreeMap<&str, u64>,
    selected_documents: &BTreeMap<String, DocumentAccumulator>,
    document_shard_rows: usize,
) -> Result<(PreparedCorpusManifest, Vec<PreparedDocumentRow>)> {
    let mut documents = Vec::with_capacity(stage.document_count as usize);
    for (document_id, accumulated) in selected_documents {
        if selected_ranks
            .get(document_id.as_str())
            .is_none_or(|rank| *rank >= stage.document_count)
        {
            continue;
        }
        let mut fast = accumulated.document.clone();
        fast.vector_ordinal = documents.len() as u64;
        fast.relations_json = canonical_string(&accumulated.relations)?;
        documents.push(PreparedDocumentRow {
            document_ordinal: fast.vector_ordinal,
            document_id: fast.document_id,
            document_sha256: Digest::new(fast.document_sha256)?,
            semantic_text_sha256: digest_bytes(fast.semantic_text.as_bytes()),
            semantic_text: fast.semantic_text,
            document_kind: match fast.document_kind.as_str() {
                "activity" => DocumentKind::Activity,
                "state" => DocumentKind::State,
                "detection" => DocumentKind::Detection,
                _ => return Err(Error::AccountingClosure("unknown projected document kind")),
            },
            primary_relation: accumulated.primary_relation.clone(),
            facets_json: fast.facets_json,
            relations_json: fast.relations_json,
            occurrence_count: fast.occurrence_count,
        });
    }
    if documents.len() as u64 != stage.document_count {
        return Err(Error::AccountingClosure(
            "benchmark prepared document count differs",
        ));
    }
    let mut document_objects = Vec::new();
    for (ordinal, rows) in documents.chunks(document_shard_rows).enumerate() {
        let relative = format!("documents/part-{ordinal:06}.parquet");
        let path = stage.root.join(&relative);
        write_prepared_documents(&path, rows)?;
        document_objects.push(PreparedDocumentObject {
            object: object_entry(
                &relative,
                &path,
                rows.len() as u64,
                canonical_digest(&rows)?,
            )?,
            ordinal: u32::try_from(ordinal)
                .map_err(|_| Error::AccountingClosure("too many document shards"))?,
            first_document_id: rows[0].document_id.clone(),
            last_document_id: rows[rows.len() - 1].document_id.clone(),
            embedding_input_order_sha256: embedding_input_order_digest(rows),
        });
    }
    let mut relation_accounting = BTreeMap::new();
    for relation in reader.typed_relations() {
        let selected = stage
            .selected_occurrences
            .get(&relation.name)
            .copied()
            .unwrap_or(0);
        if selected > relation.rows {
            return Err(Error::AccountingClosure(
                "benchmark selected occurrences exceed source rows",
            ));
        }
        let included = dataset.included_relations.contains(&relation.name);
        relation_accounting.insert(
            relation.name.clone(),
            RelationAccounting {
                source_rows: relation.rows,
                searchable_occurrences: if included { selected } else { 0 },
                selected_occurrences: if included { selected } else { 0 },
                excluded_rows: if included {
                    relation.rows - selected
                } else {
                    relation.rows
                },
            },
        );
    }
    let mut manifest = PreparedCorpusManifest {
        schema_version: PREPARED_CORPUS_SCHEMA.into(),
        component_sha256: zero_digest()?,
        dataset: dataset.clone(),
        projection_policy: projection_policy.clone(),
        document_schema: component(
            "livefire.rag.prepared-document-row",
            "1",
            &sha256_bytes(DOCUMENT_SCHEMA_BYTES),
        )?,
        occurrence_schema: component(
            "livefire.rag.prepared-occurrence-row",
            "1",
            &sha256_bytes(OCCURRENCE_SCHEMA_BYTES),
        )?,
        preparation_implementation: component(
            "livefire.rag.benchmark-preparation",
            env!("CARGO_PKG_VERSION"),
            &sha256_bytes(PREPARATION_SOURCE_BYTES),
        )?,
        document_count: documents.len() as u64,
        occurrence_count: stage.occurrence_count,
        document_order_sha256: document_order_digest(
            documents.iter().map(|row| row.document_id.as_str()),
        ),
        embedding_input_order_sha256: embedding_input_order_digest(&documents),
        documents: document_objects,
        occurrences: stage.occurrence_objects.clone(),
        relation_accounting,
    };
    manifest.seal()?;
    validate_prepared_documents(&manifest, &documents)?;
    write_canonical_json(&stage.root.join(MANIFEST_FILE), &manifest)?;
    write_canonical_json(
        &stage.root.join("accounting.json"),
        &json!({
            "schema_version": "livefire.rag.prepared-accounting/1",
            "dataset": manifest.dataset,
            "documents": manifest.document_count,
            "occurrences": manifest.occurrence_count,
            "preparation_document_grouping": "selected_in_memory_btree_v1",
            "preparation_document_limit": MAX_BENCHMARK_CANDIDATES,
            "occurrence_shard_rows": OCCURRENCE_SHARD_ROWS,
            "benchmark_selection_size": stage.document_count,
            "relations": manifest.relation_accounting,
        }),
    )?;
    Ok((manifest, documents))
}

pub(crate) fn plan_embeddings(options: PlanOptions) -> Result<()> {
    let prepared = load_prepared_documents_only(&options.prepared)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let compact = parse_embedding_profile(&profile_bytes)?;
    let profile = profile_ref(&profile_bytes, &compact)?;
    let tokenizer_bytes = fs::read(&options.tokenizer_json)?;
    let tokenizer: ExecutableTokenizerRef = read_json(&options.tokenizer_ref)?;
    let documents = load_all_prepared_documents(&options.prepared, &prepared)?;
    let (plan, token_counts) = build_token_balanced_plan_with_counts(
        &prepared,
        &documents,
        profile,
        tokenizer,
        &tokenizer_bytes,
        TokenBalanceOptions {
            maximum_task_tokens: options.maximum_task_tokens,
            maximum_task_documents: options.maximum_task_documents,
        },
    )?;
    plan.validate_with_tokenizer(&prepared, &documents, &tokenizer_bytes)?;
    let staging = AtomicDirectory::new(&options.out)?;
    plan.write_document_token_counts(staging.path(), &token_counts)?;
    write_canonical_json(&staging.path().join("plan.json"), &plan)?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

/// Build an exact-token plan from a complete embedding-policy/3 contract.
/// The compact profile digest is the digest of the original policy bytes, so
/// no runtime or conformance field can change without changing the plan.
pub(crate) fn plan_embeddings_tei(options: TeiPlanOptions) -> Result<()> {
    let prepared = load_prepared_documents_only(&options.prepared)?;
    let policy_bytes = fs::read(&options.embedding_policy)?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let compact = policy.embedding_profile(&policy_bytes)?;
    let profile = tei_profile_ref(&policy, &compact)?;
    let tokenizer_bytes = fs::read(&options.tokenizer_json)?;
    let tokenizer: ExecutableTokenizerRef = read_json(&options.tokenizer_ref)?;
    validate_tei_tokenizer_inputs(&policy, &tokenizer, &tokenizer_bytes)?;
    let documents = load_all_prepared_documents(&options.prepared, &prepared)?;
    let (plan, token_counts) = build_token_balanced_plan_with_counts(
        &prepared,
        &documents,
        profile,
        tokenizer,
        &tokenizer_bytes,
        TokenBalanceOptions {
            maximum_task_tokens: options.maximum_task_tokens,
            maximum_task_documents: options.maximum_task_documents,
        },
    )?;
    plan.validate_with_tokenizer(&prepared, &documents, &tokenizer_bytes)?;
    let staging = AtomicDirectory::new(&options.out)?;
    plan.write_document_token_counts(staging.path(), &token_counts)?;
    write_canonical_json(&staging.path().join("plan.json"), &plan)?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

pub(crate) async fn embed(options: EmbedOptions) -> Result<()> {
    let prepared = load_prepared_documents_only(&options.prepared)?;
    let plan = load_embedding_plan_v2(&options.plan)?;
    plan.validate_manifest_binding(&prepared)?;
    let selection = parse_task_selection(options.task_range.as_deref())?;
    let range = selection.resolve(plan.tasks.len())?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    if profile.vector_derivation.is_some() {
        return Err(Error::AccountingClosure(
            "derived profiles must use derive-embeddings, not model embedding",
        ));
    }
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    for directory in ["parts", "receipts", "reports"] {
        fs::create_dir_all(options.out.join(directory))?;
    }
    let embedder = Arc::new(LmStudioEmbedder::with_timeout(
        &options.embedding_endpoint,
        &profile.model,
        Duration::from_secs(300),
    )?);
    let run_context = LocalRunContext::observe();
    let mut conformance_validated = false;
    let mut task_documents = TaskDocumentLoaderV2::new(&options.prepared, &prepared);
    let mut executed = 0_u64;
    let mut reused = 0_u64;
    for task_index in range.clone() {
        let task = &plan.tasks[task_index];
        let vector_path = resolve_output_artifact(&options.out, &task.result_path)?;
        let receipt_path = resolve_output_artifact(&options.out, &task.receipt_path)?;
        let report_path = task_report_path(&options.out, task)?;
        let expected = task_shard_expectation_v2(task, profile.dimensions)?;

        if let Some(receipt) = validate_completed_embedding_task_v2(
            &receipt_path,
            &vector_path,
            &report_path,
            task,
            &plan,
            &profile,
            &runtime,
        )? {
            ensure_task_report(
                &report_path,
                task_index,
                task,
                &plan,
                TaskReportBindings {
                    prepared: &prepared,
                    profile: &profile,
                    receipt: &receipt,
                    vector_path: &vector_path,
                    receipt_path: &receipt_path,
                    run_context: &run_context,
                    batch_size: options.batch_size,
                    requests_in_flight: options.requests_in_flight,
                },
                TaskRunDetails {
                    outcome: TaskRunOutcome::Reused,
                    started_unix_ms: None,
                    finished_unix_ms: None,
                    execution: None,
                },
            )?;
            complete_embedding_task_part_recovery(
                &vector_path,
                expected,
                &profile.normalization,
                Some(decode_sha256_hex(receipt.vector.sha256.as_str())?),
            )?;
            complete_regular_file_recovery(&receipt_path)?;
            complete_regular_file_recovery(&report_path)?;
            reused = reused.checked_add(1).ok_or(Error::CountOverflow)?;
            continue;
        }
        if report_path.try_exists()? {
            quarantine_regular_file(&report_path)?;
        }

        // This checks a pre-existing part before any request. Invalid bytes
        // move to a deterministic sibling quarantine; valid orphan bytes are
        // re-executed and must match exactly before a receipt can be created.
        let _preparation =
            prepare_embedding_task_part(&vector_path, expected, &profile.normalization, None)?;
        if !conformance_validated {
            validate_lmstudio_conformance(&embedder, &profile_bytes, &profile).await?;
            conformance_validated = true;
        }
        let rows = task_documents.load(task)?;
        let texts = rows
            .iter()
            .map(|row| {
                format_document_input(&plan.embedding_profile.document_format, &row.semantic_text)
            })
            .collect::<rag_embedding::Result<Vec<_>>>()?;
        let input_bytes = texts.iter().try_fold(0_u64, |total, input| {
            total
                .checked_add(input.len() as u64)
                .ok_or(Error::CountOverflow)
        })?;
        let began_unix_ms = unix_time_ms()?;
        let began = Instant::now();
        let (stats, execution_report) = execute_embedding_task_reported(
            Arc::clone(&embedder),
            &profile,
            &texts,
            &vector_path,
            expected.order_sha256,
            EmbeddingTaskOptions {
                batch_size: options.batch_size,
                max_in_flight: options.requests_in_flight,
                retry: RetryPolicy::default(),
            },
        )
        .await?;
        let verified =
            verify_embedding_task_part(&vector_path, expected, &profile.normalization, None)?;
        let mut receipt = VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: zero_digest()?,
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: task.task_id.clone(),
            ordinal_start: task.ordinal_start,
            ordinal_end: task.ordinal_end,
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: task.result_path.clone(),
                rows: task.row_count(),
                bytes: verified.bytes,
                sha256: Digest::new(
                    verified
                        .sha256
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                )?,
                dimensions: profile.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: component(
                    "livefire.rag.embedding-executor.lmstudio",
                    env!("CARGO_PKG_VERSION"),
                    &sha256_bytes(include_bytes!("../../../crates/rag-embedding/src/task.rs")),
                )?,
                runtime: runtime.clone(),
                returned_model: stats.returned_model.clone(),
                requests: stats.requests as u64,
                retries: stats.retries as u64,
                input_bytes_upper_bound: input_bytes,
                elapsed_ms: began.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                conformance_passed: true,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal()?;
        receipt.validate_against_v2(&plan)?;
        write_canonical_json(&receipt_path, &receipt)?;
        let finished_unix_ms = unix_time_ms()?;
        ensure_task_report(
            &report_path,
            task_index,
            task,
            &plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &profile,
                receipt: &receipt,
                vector_path: &vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: options.batch_size,
                requests_in_flight: options.requests_in_flight,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::Executed,
                started_unix_ms: Some(began_unix_ms),
                finished_unix_ms: Some(finished_unix_ms),
                execution: Some(execution_report),
            },
        )?;
        complete_embedding_task_part_recovery(
            &vector_path,
            expected,
            &profile.normalization,
            Some(verified.sha256),
        )?;
        complete_regular_file_recovery(&receipt_path)?;
        complete_regular_file_recovery(&report_path)?;
        executed = executed.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": "livefire.rag.embedding-run-progress/1",
            "plan_sha256": plan.component_sha256,
            "task_range": {"start": range.start, "end": range.end},
            "tasks_executed": executed,
            "tasks_reused": reused,
            "finalized": false,
            "next": "run rag finalize-embeddings after every task range is complete"
        }))?
    );
    Ok(())
}

/// Execute TEI tasks using only the exact checkpoint policy and a pre-built
/// loopback client. Construction of the client remains with the local worker
/// so credentials and endpoint details never enter durable artifacts.
pub(crate) async fn embed_tei_with_embedder(
    options: TeiEmbedOptions,
    embedder: Arc<TeiEmbedder>,
) -> Result<()> {
    let prepared = load_prepared_documents_only(&options.prepared)?;
    let plan = load_embedding_plan_v2(&options.plan)?;
    plan.validate_manifest_binding(&prepared)?;
    let range = parse_task_selection(options.task_range.as_deref())?.resolve(plan.tasks.len())?;
    let policy_bytes = fs::read(&options.embedding_policy)?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let profile = policy.embedding_profile(&policy_bytes)?;
    validate_plan_profile_fields(&plan.embedding_profile, &policy_bytes, &profile)?;
    validate_tei_worker_context(&options.worker, &policy, &plan)?;
    if options.batch_size == 0
        || options.batch_size > policy.batching.maximum_batch_items as usize
        || options.requests_in_flight == 0
        || options.worker.backend.batch_size != options.batch_size
        || options.worker.backend.requests_in_flight != options.requests_in_flight
    {
        return Err(Error::AccountingClosure("TEI task concurrency is invalid"));
    }
    for directory in ["parts", "receipts", "reports"] {
        fs::create_dir_all(options.out.join(directory))?;
    }
    let fixture_bytes = fs::read(&options.conformance_fixture)?;
    let mut conformance_validated = false;
    let mut task_documents = TaskDocumentLoaderV2::new(&options.prepared, &prepared);
    let mut executed = 0_u64;
    let mut reused = 0_u64;
    for task_index in range.clone() {
        let task = &plan.tasks[task_index];
        let vector_path = resolve_output_artifact(&options.out, &task.result_path)?;
        let receipt_path = resolve_output_artifact(&options.out, &task.receipt_path)?;
        let report_path = task_report_path(&options.out, task)?;
        let expected = task_shard_expectation_v2(task, profile.dimensions)?;
        if let Some(receipt) = validate_completed_embedding_task_v2(
            &receipt_path,
            &vector_path,
            &report_path,
            task,
            &plan,
            &profile,
            &options.worker.execution_identity.runtime,
        )? {
            let reusable = read_validated_task_report(
                &report_path,
                task_index,
                task,
                &plan,
                Some(&prepared),
                &profile,
                &receipt,
            )
            .is_ok_and(|report| reusable_tei_report(&report, &options.worker.execution_identity));
            if reusable {
                complete_embedding_task_part_recovery(
                    &vector_path,
                    expected,
                    &profile.normalization,
                    Some(decode_sha256_hex(receipt.vector.sha256.as_str())?),
                )?;
                complete_regular_file_recovery(&receipt_path)?;
                complete_regular_file_recovery(&report_path)?;
                reused = reused.checked_add(1).ok_or(Error::CountOverflow)?;
                continue;
            }
            quarantine_regular_file(&receipt_path)?;
            if report_path.try_exists()? {
                quarantine_regular_file(&report_path)?;
            }
        }
        let _ = prepare_embedding_task_part(&vector_path, expected, &profile.normalization, None)?;
        if !conformance_validated {
            embedder
                .checkpoint_conformance_probe(&fixture_bytes)
                .await?;
            conformance_validated = true;
        }
        let rows = task_documents.load(task)?;
        let texts = rows
            .iter()
            .map(|row| {
                format_document_input(&plan.embedding_profile.document_format, &row.semantic_text)
            })
            .collect::<rag_embedding::Result<Vec<_>>>()?;
        let input_bytes = texts.iter().try_fold(0_u64, |total, input| {
            total
                .checked_add(input.len() as u64)
                .ok_or(Error::CountOverflow)
        })?;
        let began_unix_ms = unix_time_ms()?;
        let began = Instant::now();
        let (stats, execution) = execute_embedding_task_reported(
            Arc::clone(&embedder),
            &profile,
            &texts,
            &vector_path,
            expected.order_sha256,
            EmbeddingTaskOptions {
                batch_size: options.batch_size,
                max_in_flight: options.requests_in_flight,
                retry: RetryPolicy::default(),
            },
        )
        .await?;
        let verified =
            verify_embedding_task_part(&vector_path, expected, &profile.normalization, None)?;
        let mut receipt = VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: zero_digest()?,
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: task.task_id.clone(),
            ordinal_start: task.ordinal_start,
            ordinal_end: task.ordinal_end,
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: task.result_path.clone(),
                rows: task.row_count(),
                bytes: verified.bytes,
                sha256: Digest::new(
                    verified
                        .sha256
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                )?,
                dimensions: profile.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: options.worker.execution_identity.worker_binary.clone(),
                runtime: options.worker.execution_identity.runtime.clone(),
                returned_model: stats.returned_model,
                requests: stats.requests as u64,
                retries: stats.retries as u64,
                input_bytes_upper_bound: input_bytes,
                elapsed_ms: began.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                conformance_passed: true,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal()?;
        receipt.validate_against_v2(&plan)?;
        write_canonical_json(&receipt_path, &receipt)?;
        write_tei_task_report_v2(
            &report_path,
            task_index,
            task,
            &plan,
            &prepared,
            &receipt,
            &vector_path,
            &receipt_path,
            &options.worker,
            began_unix_ms,
            unix_time_ms()?,
            execution,
        )?;
        complete_embedding_task_part_recovery(
            &vector_path,
            expected,
            &profile.normalization,
            Some(verified.sha256),
        )?;
        complete_regular_file_recovery(&receipt_path)?;
        complete_regular_file_recovery(&report_path)?;
        executed = executed.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": "livefire.rag.embedding-run-progress/2",
            "plan_sha256": plan.component_sha256,
            "task_range": {"start": range.start, "end": range.end},
            "tasks_executed": executed,
            "tasks_reused": reused,
            "finalized": false
        }))?
    );
    Ok(())
}

fn reusable_tei_report(
    report: &ValidatedEmbeddingTaskReport,
    execution: &EmbeddingExecutionIdentityV2,
) -> bool {
    matches!(report, ValidatedEmbeddingTaskReport::V2(report)
        if report.execution_identity == *execution)
}

pub(crate) async fn embed_tei(options: TeiEmbedOptions, endpoint: &str) -> Result<()> {
    let policy_bytes = fs::read(&options.embedding_policy)?;
    let embedder = Arc::new(TeiEmbedder::checkpoint_profile_loopback(
        endpoint,
        &policy_bytes,
        BearerAuthorization::None,
    )?);
    embed_tei_with_embedder(options, embedder).await
}

/// Produce a complete embedding artifact chain without contacting a model.
/// The sparse unit vectors are deliberately useful only for end-to-end file,
/// ordering, assembly, and refusal tests. Every downstream artifact carries a
/// test-only marker, and normal search/provider paths refuse the final index.
pub(crate) fn test_embed(options: TestEmbedOptions) -> Result<()> {
    let prepared = load_prepared_documents_only(&options.prepared)?;
    let plan = load_embedding_plan_v2(&options.plan)?;
    plan.validate_manifest_binding(&prepared)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    if profile.dimensions != 4_096
        || profile.normalization != "l2"
        || plan.embedding_profile.dtype != "f32le"
    {
        return Err(Error::AccountingClosure(
            "test vectors require a 4096-d l2-normalized f32 profile",
        ));
    }
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let staging = AtomicDirectory::new(&options.out)?;
    for directory in ["parts", "receipts", "reports"] {
        fs::create_dir_all(staging.path().join(directory))?;
    }
    let run_context = LocalRunContext::deterministic_test_vectors();
    let implementation = component(
        TEST_VECTOR_EXECUTOR_ID,
        env!("CARGO_PKG_VERSION"),
        &sha256_bytes(b"livefire.rag.deterministic-test-vectors/1"),
    )?;
    let mut task_documents = TaskDocumentLoaderV2::new(&options.prepared, &prepared);
    for (task_index, task) in plan.tasks.iter().enumerate() {
        let rows = task_documents.load(task)?;
        let vector_path = resolve_output_artifact(staging.path(), &task.result_path)?;
        let receipt_path = resolve_output_artifact(staging.path(), &task.receipt_path)?;
        let report_path = task_report_path(staging.path(), task)?;
        let expected = task_shard_expectation_v2(task, profile.dimensions)?;
        let publication = AtomicFilePublication::new(&vector_path)?;
        let mut writer = EmbeddingShardWriter::create(
            publication.staging_path(),
            EmbeddingShardMetadata::from(expected),
        )?;
        let mut input_bytes = 0_u64;
        for row in &rows {
            let input =
                format_document_input(&plan.embedding_profile.document_format, &row.semantic_text)?;
            input_bytes = input_bytes
                .checked_add(u64::try_from(input.len()).map_err(|_| Error::CountOverflow)?)
                .ok_or(Error::CountOverflow)?;
            writer.write_vector(&deterministic_test_vector(row)?)?;
        }
        writer.finish()?;
        publication.commit()?;
        let verified =
            verify_embedding_task_part(&vector_path, expected, &profile.normalization, None)?;
        let mut receipt = VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: zero_digest()?,
            plan_sha256: plan.component_sha256.clone(),
            prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            task_id: task.task_id.clone(),
            ordinal_start: task.ordinal_start,
            ordinal_end: task.ordinal_end,
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: task.result_path.clone(),
                rows: task.row_count(),
                bytes: verified.bytes,
                sha256: Digest::new(
                    verified
                        .sha256
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                )?,
                dimensions: profile.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: implementation.clone(),
                runtime: runtime.clone(),
                returned_model: profile.model.clone(),
                requests: 0,
                retries: 0,
                input_bytes_upper_bound: input_bytes,
                elapsed_ms: 0,
                conformance_passed: false,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal()?;
        receipt.validate_against_v2(&plan)?;
        write_canonical_json(&receipt_path, &receipt)?;
        ensure_task_report(
            &report_path,
            task_index,
            task,
            &plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &profile,
                receipt: &receipt,
                vector_path: &vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: 0,
                requests_in_flight: 0,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::TestGenerated,
                started_unix_ms: None,
                finished_unix_ms: None,
                execution: None,
            },
        )?;
    }
    finalize_embeddings(FinalizeOptions {
        prepared: options.prepared,
        plan: options.plan,
        embedding_profile: options.embedding_profile,
        embeddings: staging.path().to_owned(),
    })?;
    staging.publish()?;
    Ok(())
}

fn deterministic_test_vector(row: &PreparedDocumentRow) -> Result<Vec<f32>> {
    const NONZERO_VALUES: usize = 16;
    const VALUE: f32 = 0.25;
    let mut seed = Sha256::new();
    seed.update(b"livefire.rag.deterministic-test-vector/1\0");
    seed.update(row.document_ordinal.to_le_bytes());
    seed.update(row.document_id.as_bytes());
    seed.update([0]);
    seed.update(row.document_sha256.as_str().as_bytes());
    seed.update([0]);
    seed.update(row.semantic_text_sha256.as_str().as_bytes());
    let seed = seed.finalize();
    let mut vector = vec![0.0_f32; 4_096];
    let mut selected = BTreeSet::new();
    let mut counter = 0_u32;
    while selected.len() < NONZERO_VALUES {
        let mut block = Sha256::new();
        block.update(b"livefire.rag.deterministic-test-vector-expand/1\0");
        block.update(seed);
        block.update(counter.to_le_bytes());
        let block = block.finalize();
        for pair in block.chunks_exact(2) {
            let raw = u16::from_le_bytes([pair[0], pair[1]]);
            let index = usize::from(raw & 0x0fff);
            if selected.insert(index) {
                vector[index] = if raw & 0x8000 == 0 { VALUE } else { -VALUE };
                if selected.len() == NONZERO_VALUES {
                    break;
                }
            }
        }
        counter = counter.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    Ok(vector)
}

pub(crate) fn finalize_embeddings(options: FinalizeOptions) -> Result<()> {
    let prepared = load_prepared(&options.prepared)?;
    let plan = load_embedding_plan_v2(&options.plan)?;
    plan.validate_manifest_binding(&prepared)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    let bound_profile_path = ensure_embedding_profile_copy(&options.embeddings, &profile_bytes)?;
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let (entries, receipts, reports) = validate_complete_embedding_tasks_v2(
        &options.embeddings,
        &plan,
        &prepared,
        &profile,
        &runtime,
    )?;
    let test_only = receipts.first().is_some_and(VectorResultReceipt::test_only);
    if receipts
        .iter()
        .any(|receipt| receipt.test_only() != test_only)
    {
        return Err(Error::AccountingClosure(
            "embedding result mixes model and test-only vectors",
        ));
    }
    let mut result_set = EmbeddingResultSetManifest {
        schema_version: if test_only {
            TEST_RESULT_SET_SCHEMA.into()
        } else {
            RESULT_SET_SCHEMA.into()
        },
        component_sha256: zero_digest()?,
        plan_sha256: plan.component_sha256.clone(),
        prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        document_count: plan.document_count,
        document_order_sha256: plan.document_order_sha256.clone(),
        receipts: entries,
        derivation: None,
        test_only,
    };
    result_set.seal()?;
    result_set.validate_v2(&plan, &receipts)?;
    let plan_root = if options.plan.is_dir() {
        options.plan.as_path()
    } else {
        options
            .plan
            .parent()
            .ok_or(Error::AccountingClosure("embedding plan parent is absent"))?
    };
    let token_counts = plan.read_document_token_counts(plan_root)?;
    let summary = embedding_run_summary(
        &plan,
        &prepared,
        &receipts,
        &reports,
        &token_counts,
        embedding_run_artifact_sizes(
            &options.prepared,
            &options.plan,
            &bound_profile_path,
            &options.embeddings,
            &plan,
        )?,
    )?;
    let summary_path = options.embeddings.join("summary.json");
    if summary_path.try_exists()? {
        let existing: EmbeddingRunSummaryContract = read_json(&summary_path)?;
        validate_embedding_run_summary(&existing, &summary)?;
    } else {
        write_canonical_json(&summary_path, &summary)?;
    }
    let manifest_path = options.embeddings.join(MANIFEST_FILE);
    if manifest_path.try_exists()? {
        let existing: EmbeddingResultSetManifest = read_json(&manifest_path)?;
        existing.validate_v2(&plan, &receipts)?;
        if existing != result_set {
            return Err(Error::AccountingClosure(
                "existing embedding result manifest differs",
            ));
        }
    } else {
        write_canonical_json(&manifest_path, &result_set)?;
    }
    validate_embedding_artifact_coverage_v2(&options.embeddings, &plan, true)?;
    println!("{}", serde_json::to_string_pretty(&result_set)?);
    Ok(())
}

/// Derive a smaller Matryoshka-style vector set from a completed 4,096-value
/// result. This is a local file transformation: it never creates an embedder
/// or sends text to a model service.
pub(crate) fn derive_embeddings(options: DeriveEmbeddingsOptions) -> Result<()> {
    if !matches!(options.dimensions, 1_024 | 2_048) {
        return Err(Error::AccountingClosure(
            "derived dimensions must be 1024 or 2048",
        ));
    }
    let prepared = load_prepared(&options.prepared)?;
    let source_plan = load_embedding_plan_v2(&options.plan)?;
    source_plan.validate_manifest_binding(&prepared)?;
    let source_profile_bytes = fs::read(&options.embedding_profile)?;
    let source_profile = parse_bound_portable_profile(
        &source_profile_bytes,
        source_plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(
        &source_plan.embedding_profile,
        &source_profile_bytes,
        &source_profile,
    )?;
    if source_profile.dimensions != 4_096 || source_profile.vector_derivation.is_some() {
        return Err(Error::AccountingClosure(
            "source profile must be an original 4096-dimensional profile",
        ));
    }
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&source_profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let (_, source_receipts, _) = validate_complete_embedding_tasks_v2(
        &options.embeddings,
        &source_plan,
        &prepared,
        &source_profile,
        &runtime,
    )?;
    let source_result: EmbeddingResultSetManifest =
        read_json(&options.embeddings.join(MANIFEST_FILE))?;
    source_result.validate_v2(&source_plan, &source_receipts)?;
    if source_result.test_only || source_result.derivation.is_some() {
        return Err(Error::AccountingClosure(
            "source result must contain original model vectors",
        ));
    }

    let mut derived_profile_value: Value = serde_json::from_slice(&source_profile_bytes)?;
    let profile_object = derived_profile_value
        .as_object_mut()
        .ok_or(Error::AccountingClosure(
            "embedding profile must be an object",
        ))?;
    profile_object.insert(
        "schema_version".into(),
        json!("livefire.rag.embedding-policy/2"),
    );
    profile_object.insert("dimensions".into(), json!(options.dimensions));
    let parent_conformance_sha256 = profile_object
        .get("conformance")
        .ok_or(Error::AccountingClosure(
            "parent embedding conformance contract is absent",
        ))
        .and_then(|value| canonical_digest(value).map_err(Error::from))?;
    profile_object.insert(
        "conformance".into(),
        json!({
            "mode": "parent_bound_local_derivation",
            "parent_conformance_sha256": parent_conformance_sha256,
            "transformation": PREFIX_L2_DERIVATION_POLICY,
            "output_dimensions": options.dimensions,
            "normalization": "l2",
        }),
    );
    if let Some(output_processing) = profile_object
        .get_mut("output_processing")
        .and_then(Value::as_object_mut)
    {
        output_processing.insert(
            "client_normalization".into(),
            json!(PREFIX_L2_DERIVATION_POLICY),
        );
    }
    profile_object.insert(
        "vector_derivation".into(),
        json!({
            "parent_embedding_profile_sha256": source_profile.sha256,
            "parent_dimensions": source_profile.dimensions,
            "transformation": PREFIX_L2_DERIVATION_POLICY,
        }),
    );
    let derived_profile_bytes = canonical_json_bytes(&derived_profile_value)?;
    let derived_profile = parse_embedding_profile(&derived_profile_bytes)?;
    if derived_profile.vector_derivation
        != Some(VectorDerivation {
            parent_embedding_profile_sha256: source_profile.sha256.clone(),
            parent_dimensions: 4_096,
            transformation: PREFIX_L2_DERIVATION_POLICY.into(),
        })
    {
        return Err(Error::AccountingClosure(
            "derived embedding profile binding differs",
        ));
    }
    let derived_profile_ref = profile_ref(&derived_profile_bytes, &derived_profile)?;
    let derived_plan = derive_embedding_plan_v2(&source_plan, derived_profile_ref)?;
    let token_counts = source_plan.read_document_token_counts(if options.plan.is_dir() {
        &options.plan
    } else {
        options
            .plan
            .parent()
            .ok_or(Error::AccountingClosure("embedding plan parent is absent"))?
    })?;

    let staging = AtomicDirectory::new(&options.out)?;
    let plan_root = staging.path().join("plan");
    let result_root = staging.path().join("results");
    fs::create_dir_all(&plan_root)?;
    fs::create_dir_all(&result_root)?;
    derived_plan.write_document_token_counts(&plan_root, &token_counts)?;
    write_canonical_json(&plan_root.join("plan.json"), &derived_plan)?;
    atomic_write(
        &staging.path().join(EMBEDDING_PROFILE_FILE),
        &derived_profile_bytes,
    )?;
    atomic_write(
        &result_root.join(EMBEDDING_PROFILE_FILE),
        &derived_profile_bytes,
    )?;

    let implementation = component(
        DERIVED_VECTOR_EXECUTOR_ID,
        env!("CARGO_PKG_VERSION"),
        &sha256_bytes(include_bytes!("portable.rs")),
    )?;
    let run_context = LocalRunContext::deterministic_test_vectors();
    let mut receipts = Vec::with_capacity(derived_plan.tasks.len());
    let mut entries = Vec::with_capacity(derived_plan.tasks.len());
    for (task_index, ((source_task, target_task), source_receipt)) in source_plan
        .tasks
        .iter()
        .zip(&derived_plan.tasks)
        .zip(&source_receipts)
        .enumerate()
    {
        if source_task.ordinal_start != target_task.ordinal_start
            || source_task.ordinal_end != target_task.ordinal_end
            || source_task.embedding_input_order_sha256 != target_task.embedding_input_order_sha256
        {
            return Err(Error::AccountingClosure("derived task order differs"));
        }
        let source_vector_path =
            resolve_existing_artifact(&options.embeddings, &source_task.result_path)?;
        let target_vector_path = resolve_output_artifact(&result_root, &target_task.result_path)?;
        if let Some(parent) = target_vector_path.parent() {
            fs::create_dir_all(parent)?;
        }
        File::create(&target_vector_path)?;
        let source_shard = rag_embedding::EmbeddingShard::open_expected(
            &source_vector_path,
            task_shard_expectation_v2(source_task, source_profile.dimensions)?,
        )?;
        let mut writer = EmbeddingShardWriter::create(
            &target_vector_path,
            EmbeddingShardMetadata {
                row_count: target_task.row_count(),
                dimensions: options.dimensions,
                order_sha256: decode_sha256_hex(target_task.embedding_input_order_sha256.as_str())?,
            },
        )?;
        for vector in source_shard.vectors()? {
            writer.write_vector(&adapt_model_vector(&derived_profile, vector?)?)?;
        }
        writer.finish()?;
        let verified = verify_embedding_task_part(
            &target_vector_path,
            task_shard_expectation_v2(target_task, options.dimensions)?,
            "l2",
            None,
        )?;
        let mut receipt = VectorResultReceipt {
            schema_version: DERIVED_VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: zero_digest()?,
            plan_sha256: derived_plan.component_sha256.clone(),
            prepared_corpus_sha256: derived_plan.prepared_corpus_sha256.clone(),
            embedding_profile_sha256: derived_plan.embedding_profile.component.sha256.clone(),
            task_id: target_task.task_id.clone(),
            ordinal_start: target_task.ordinal_start,
            ordinal_end: target_task.ordinal_end,
            embedding_input_order_sha256: target_task.embedding_input_order_sha256.clone(),
            vector: VectorObject {
                path: target_task.result_path.clone(),
                rows: target_task.row_count(),
                bytes: verified.bytes,
                sha256: Digest::new(
                    verified
                        .sha256
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>(),
                )?,
                dimensions: options.dimensions,
                dtype: "f32le".into(),
                embedding_input_order_sha256: target_task.embedding_input_order_sha256.clone(),
            },
            executor: ExecutorReceipt {
                implementation: implementation.clone(),
                runtime: runtime.clone(),
                returned_model: derived_profile.model.clone(),
                requests: 0,
                retries: 0,
                input_bytes_upper_bound: 0,
                elapsed_ms: 0,
                conformance_passed: false,
            },
            derivation: Some(DerivedVectorBinding {
                parent_embedding_profile_sha256: source_plan
                    .embedding_profile
                    .component
                    .sha256
                    .clone(),
                parent_result_set_sha256: source_result.component_sha256.clone(),
                parent_receipt_sha256: source_receipt.component_sha256.clone(),
                parent_vector_sha256: source_receipt.vector.sha256.clone(),
                parent_dimensions: source_profile.dimensions,
                transformation: PREFIX_L2_DERIVATION_POLICY.into(),
            }),
            finite_values_validated: true,
            normalization_validated: true,
        };
        receipt.seal()?;
        receipt.validate_against_v2(&derived_plan)?;
        let receipt_path = resolve_output_artifact(&result_root, &target_task.receipt_path)?;
        write_canonical_json(&receipt_path, &receipt)?;
        ensure_task_report(
            &task_report_path(&result_root, target_task)?,
            task_index,
            target_task,
            &derived_plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &derived_profile,
                receipt: &receipt,
                vector_path: &target_vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: 0,
                requests_in_flight: 0,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::Derived,
                started_unix_ms: None,
                finished_unix_ms: None,
                execution: None,
            },
        )?;
        entries.push(ReceiptEntry {
            task_id: target_task.task_id.clone(),
            path: target_task.receipt_path.clone(),
            sha256: receipt.component_sha256.clone(),
        });
        receipts.push(receipt);
    }
    let derivation = DerivedResultSetBinding {
        parent_embedding_profile_sha256: source_plan.embedding_profile.component.sha256.clone(),
        parent_result_set_sha256: source_result.component_sha256.clone(),
        parent_dimensions: source_profile.dimensions,
        transformation: PREFIX_L2_DERIVATION_POLICY.into(),
    };
    let mut result = EmbeddingResultSetManifest {
        schema_version: DERIVED_RESULT_SET_SCHEMA.into(),
        component_sha256: zero_digest()?,
        plan_sha256: derived_plan.component_sha256.clone(),
        prepared_corpus_sha256: derived_plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: derived_plan.embedding_profile.component.sha256.clone(),
        document_count: derived_plan.document_count,
        document_order_sha256: derived_plan.document_order_sha256.clone(),
        receipts: entries,
        derivation: Some(derivation),
        test_only: false,
    };
    result.seal()?;
    result.validate_v2(&derived_plan, &receipts)?;
    write_canonical_json(&result_root.join(MANIFEST_FILE), &result)?;
    let (_, validated_receipts, reports) = validate_complete_embedding_tasks_v2(
        &result_root,
        &derived_plan,
        &prepared,
        &derived_profile,
        &runtime,
    )?;
    let summary = embedding_run_summary(
        &derived_plan,
        &prepared,
        &validated_receipts,
        &reports,
        &token_counts,
        embedding_run_artifact_sizes(
            &options.prepared,
            &plan_root,
            &result_root.join(EMBEDDING_PROFILE_FILE),
            &result_root,
            &derived_plan,
        )?,
    )?;
    write_canonical_json(&result_root.join("summary.json"), &summary)?;
    validate_embedding_artifact_coverage_v2(&result_root, &derived_plan, true)?;
    staging.publish()?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn ensure_embedding_profile_copy(root: &Path, profile_bytes: &[u8]) -> Result<PathBuf> {
    fs::create_dir_all(root)?;
    let path = root.join(EMBEDDING_PROFILE_FILE);
    if path.try_exists()? {
        if fs::read(&path)? != profile_bytes {
            return Err(Error::AccountingClosure(
                "finalized embedding profile copy differs",
            ));
        }
    } else {
        atomic_write(&path, profile_bytes)?;
    }
    Ok(path)
}

pub(crate) fn recover_embedding_task(options: RecoveryOptions) -> Result<()> {
    let plan = load_embedding_plan_v2(&options.plan)?;
    let (task_index, task) = plan
        .tasks
        .iter()
        .enumerate()
        .find(|(_, task)| task.task_id == options.task_id)
        .ok_or(Error::UnknownEmbeddingTask)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let mut changed_artifacts = Vec::new();
    let initial = inspect_embedding_task_artifacts(
        &options.embeddings,
        &plan,
        task_index,
        task,
        &profile,
        &runtime,
    )?;
    match options.action {
        RecoveryAction::Verify => {}
        RecoveryAction::Quarantine => quarantine_embedding_task_artifacts(
            &options.embeddings,
            task,
            &profile,
            &initial,
            &mut changed_artifacts,
        )?,
        RecoveryAction::Restore => {
            restore_embedding_task_artifacts(&options.embeddings, task, &mut changed_artifacts)?
        }
    }
    let inspected = inspect_embedding_task_artifacts(
        &options.embeddings,
        &plan,
        task_index,
        task,
        &profile,
        &runtime,
    )?;
    let complete = task_inspection_complete(&inspected);
    let report = TaskRecoveryReport {
        schema_version: "livefire.rag.embedding-task-recovery/1",
        action: match options.action {
            RecoveryAction::Verify => "verify",
            RecoveryAction::Quarantine => "quarantine",
            RecoveryAction::Restore => "restore",
        },
        plan_sha256: plan.component_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        task_id: task.task_id.clone(),
        task_index,
        complete,
        part: inspected.part,
        receipt: inspected.receipt,
        report: inspected.report,
        changed_artifacts,
        model_contacted: false,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if matches!(options.action, RecoveryAction::Verify) && !complete {
        return Err(Error::EmbeddingTaskIncomplete);
    }
    Ok(())
}

fn inspect_embedding_task_artifacts(
    root: &Path,
    plan: &EmbeddingPlanV2,
    task_index: usize,
    task: &EmbeddingTaskV2,
    profile: &rag_embedding::EmbeddingProfile,
    runtime: &ComponentRef,
) -> Result<TaskArtifactInspection> {
    let part_path = resolve_output_artifact(root, &task.result_path)?;
    let receipt_path = resolve_output_artifact(root, &task.receipt_path)?;
    let report_path = task_report_path(root, task)?;
    let receipt_exists = regular_file_exists(&receipt_path)?;
    let receipt_value = if receipt_exists {
        read_json::<VectorResultReceipt>(&receipt_path)
            .ok()
            .filter(|receipt| {
                receipt.validate_against_v2(plan).is_ok()
                    && receipt.executor.returned_model == profile.model
                    && &receipt.executor.runtime == runtime
            })
    } else {
        None
    };
    let receipt = if receipt_value.is_some() {
        RecoveryArtifactState::Valid
    } else if receipt_exists {
        RecoveryArtifactState::Invalid
    } else if quarantine_exists(&receipt_path)? {
        RecoveryArtifactState::Quarantined
    } else {
        RecoveryArtifactState::Absent
    };
    let part_exists = regular_file_exists(&part_path)?;
    let part_valid = part_exists
        && verify_embedding_task_part(
            &part_path,
            task_shard_expectation_v2(task, profile.dimensions)?,
            &profile.normalization,
            receipt_value
                .as_ref()
                .and_then(|receipt| decode_sha256_hex(receipt.vector.sha256.as_str()).ok()),
        )
        .is_ok();
    let part = if part_valid && receipt_value.is_some() {
        RecoveryArtifactState::Valid
    } else if part_valid {
        RecoveryArtifactState::Orphan
    } else if part_exists {
        RecoveryArtifactState::Invalid
    } else if quarantine_exists(&part_path)? {
        RecoveryArtifactState::Quarantined
    } else {
        RecoveryArtifactState::Absent
    };
    let report_exists = regular_file_exists(&report_path)?;
    let report_valid = report_exists
        && receipt_value.as_ref().is_some_and(|receipt| {
            read_json::<BuilderEmbeddingTaskReport>(&report_path).is_ok_and(|report| {
                validate_task_report(&report, task_index, task, plan, None, profile, receipt)
                    .is_ok()
            })
        });
    let report = if report_valid {
        RecoveryArtifactState::Valid
    } else if report_exists && receipt_value.is_none() {
        RecoveryArtifactState::Orphan
    } else if report_exists {
        RecoveryArtifactState::Invalid
    } else if quarantine_exists(&report_path)? {
        RecoveryArtifactState::Quarantined
    } else {
        RecoveryArtifactState::Absent
    };
    Ok(TaskArtifactInspection {
        part,
        receipt,
        report,
        receipt_value,
    })
}

fn task_inspection_complete(inspection: &TaskArtifactInspection) -> bool {
    matches!(inspection.part, RecoveryArtifactState::Valid)
        && matches!(inspection.receipt, RecoveryArtifactState::Valid)
        && matches!(inspection.report, RecoveryArtifactState::Valid)
}

fn quarantine_embedding_task_artifacts(
    root: &Path,
    task: &EmbeddingTaskV2,
    profile: &rag_embedding::EmbeddingProfile,
    inspection: &TaskArtifactInspection,
    changed: &mut Vec<String>,
) -> Result<()> {
    if task_inspection_complete(inspection) {
        return Ok(());
    }
    let part_path = resolve_output_artifact(root, &task.result_path)?;
    let receipt_path = resolve_output_artifact(root, &task.receipt_path)?;
    let report_path = task_report_path(root, task)?;
    if matches!(
        inspection.part,
        RecoveryArtifactState::Invalid | RecoveryArtifactState::Orphan
    ) {
        if matches!(inspection.part, RecoveryArtifactState::Invalid) {
            let state = prepare_embedding_task_part(
                &part_path,
                task_shard_expectation_v2(task, profile.dimensions)?,
                &profile.normalization,
                inspection
                    .receipt_value
                    .as_ref()
                    .and_then(|receipt| decode_sha256_hex(receipt.vector.sha256.as_str()).ok()),
            )?;
            if !matches!(state, EmbeddingTaskPartPreparation::Quarantined { .. }) {
                return Err(Error::AccountingClosure(
                    "invalid embedding part was not quarantined",
                ));
            }
        } else {
            quarantine_regular_file(&part_path)?;
        }
        changed.push(task.result_path.as_str().to_owned());
    }
    let part_is_usable = matches!(inspection.part, RecoveryArtifactState::Valid);
    if regular_file_exists(&receipt_path)?
        && (!matches!(inspection.receipt, RecoveryArtifactState::Valid) || !part_is_usable)
    {
        quarantine_regular_file(&receipt_path)?;
        changed.push(task.receipt_path.as_str().to_owned());
    }
    if regular_file_exists(&report_path)?
        && (!matches!(inspection.report, RecoveryArtifactState::Valid)
            || !part_is_usable
            || !matches!(inspection.receipt, RecoveryArtifactState::Valid))
    {
        quarantine_regular_file(&report_path)?;
        changed.push(format!("reports/{}.json", task.task_id));
    }
    Ok(())
}

fn restore_embedding_task_artifacts(
    root: &Path,
    task: &EmbeddingTaskV2,
    changed: &mut Vec<String>,
) -> Result<()> {
    let part_path = resolve_output_artifact(root, &task.result_path)?;
    let receipt_path = resolve_output_artifact(root, &task.receipt_path)?;
    let report_path = task_report_path(root, task)?;
    if quarantine_exists(&part_path)? {
        restore_quarantined_embedding_task_part(&part_path)?;
        changed.push(task.result_path.as_str().to_owned());
    }
    for (path, relative) in [
        (&receipt_path, task.receipt_path.as_str().to_owned()),
        (&report_path, format!("reports/{}.json", task.task_id)),
    ] {
        if quarantine_exists(path)? {
            restore_quarantined_regular_file(path)?;
            changed.push(relative);
        }
    }
    Ok(())
}

fn regular_file_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(Error::AccountingClosure(
            "embedding recovery artifact is not a regular file",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn quarantine_path(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().ok_or(Error::AccountingClosure(
        "embedding recovery artifact parent is absent",
    ))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(Error::AccountingClosure(
            "embedding recovery artifact name is invalid",
        ))?;
    Ok(parent.join(format!(".{name}.quarantine")))
}

fn quarantine_exists(path: &Path) -> Result<bool> {
    regular_file_exists(&quarantine_path(path)?)
}

fn quarantine_regular_file(path: &Path) -> Result<()> {
    if !regular_file_exists(path)? || quarantine_exists(path)? {
        return Err(Error::AccountingClosure(
            "embedding recovery quarantine state conflicts",
        ));
    }
    fs::rename(path, quarantine_path(path)?)?;
    sync_parent_directory(path)
}

fn restore_quarantined_regular_file(path: &Path) -> Result<()> {
    if regular_file_exists(path)? || !quarantine_exists(path)? {
        return Err(Error::AccountingClosure(
            "embedding recovery restore state conflicts",
        ));
    }
    fs::rename(quarantine_path(path)?, path)?;
    sync_parent_directory(path)
}

fn complete_regular_file_recovery(path: &Path) -> Result<()> {
    if !regular_file_exists(path)? {
        return Err(Error::AccountingClosure(
            "embedding recovery replacement is absent",
        ));
    }
    let quarantine = quarantine_path(path)?;
    if regular_file_exists(&quarantine)? {
        fs::remove_file(quarantine)?;
        sync_parent_directory(path)?;
    }
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<()> {
    File::open(path.parent().ok_or(Error::AccountingClosure(
        "embedding recovery artifact parent is absent",
    ))?)?
    .sync_all()?;
    Ok(())
}

pub(crate) fn assemble(options: AssembleOptions) -> Result<()> {
    let prepared = load_prepared(&options.prepared)?;
    let plan = load_embedding_plan_v2(&options.plan)?;
    let result_set: EmbeddingResultSetManifest =
        read_json(&options.embeddings.join(MANIFEST_FILE))?;
    validate_embedding_artifact_coverage_v2(&options.embeddings, &plan, true)?;
    plan.validate_manifest_binding(&prepared)?;
    let profile_bytes = fs::read(&options.embedding_profile)?;
    let bound_profile_path = options.embeddings.join(EMBEDDING_PROFILE_FILE);
    if fs::read(&bound_profile_path)? != profile_bytes {
        return Err(Error::AccountingClosure(
            "finalized embedding profile copy differs from assembly input",
        ));
    }
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let (_, receipts, reports) = validate_complete_embedding_tasks_v2(
        &options.embeddings,
        &plan,
        &prepared,
        &profile,
        &runtime,
    )?;
    result_set.validate_v2(&plan, &receipts)?;
    let plan_root = if options.plan.is_dir() {
        options.plan.as_path()
    } else {
        options
            .plan
            .parent()
            .ok_or(Error::AccountingClosure("embedding plan parent is absent"))?
    };
    let token_counts = plan.read_document_token_counts(plan_root)?;
    let expected_summary = embedding_run_summary(
        &plan,
        &prepared,
        &receipts,
        &reports,
        &token_counts,
        embedding_run_artifact_sizes(
            &options.prepared,
            &options.plan,
            &bound_profile_path,
            &options.embeddings,
            &plan,
        )?,
    )?;
    let actual_summary: EmbeddingRunSummaryContract =
        read_json(&options.embeddings.join("summary.json"))?;
    validate_embedding_run_summary(&actual_summary, &expected_summary)?;
    let vector_shards = plan
        .tasks
        .iter()
        .map(|task| {
            Ok(OrderedVectorShard {
                path: resolve_existing_artifact(&options.embeddings, &task.result_path)?,
                first_vector_ordinal: task.ordinal_start,
                vector_count: task.row_count(),
                dimensions: profile.dimensions,
                order_sha256: task.embedding_input_order_sha256.to_string(),
            })
        })
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let vectors = vectors_from_embedding_shards(vector_shards)?;
    let document_paths = prepared
        .documents
        .iter()
        .map(|object| resolve_existing_artifact(&options.prepared, &object.object.path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let occurrence_paths = prepared
        .occurrences
        .iter()
        .map(|object| resolve_existing_artifact(&options.prepared, &object.object.path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let document_rows = documents_from_parquet_shards(document_paths);
    let occurrence_rows = occurrences_from_parquet_shards(occurrence_paths);
    let staging = AtomicDirectory::new(&options.out)?;
    let staged_index = staging.path().join("assembled-index");
    let pipeline_provenance = PipelineProvenance {
        dataset_sha256: canonical_digest(&prepared.dataset)?.to_string(),
        prepared_corpus_sha256: prepared.component_sha256.to_string(),
        embedding_plan_sha256: plan.component_sha256.to_string(),
        embedding_result_set_sha256: result_set.component_sha256.to_string(),
    };
    // Neither physical format has an explicit dataset-scope field. Keep the
    // output partial so a dataset miss is never represented as corpus-wide.
    let manifest = match options.index_format {
        IndexFormat::LegacyJsonV2 => write_bound_fast_index_from_streams(
            &staged_index,
            document_rows,
            occurrence_rows,
            vectors,
            PipelineIndexOptions {
                source: SourceBinding {
                    snapshot_sha256: prepared.dataset.source_snapshot.sha256.to_string(),
                    mapping_sha256: prepared.dataset.mapping.sha256.to_string(),
                },
                build_scope: BuildScope::Sample,
                embedding_profile: profile,
                provenance: pipeline_provenance,
                test_only: result_set.test_only,
            },
        )?,
        IndexFormat::SqliteV3 => write_bound_scalable_fast_index_from_streams(
            &staged_index,
            document_rows,
            occurrence_rows,
            vectors,
            PipelineIndexOptions {
                source: SourceBinding {
                    snapshot_sha256: prepared.dataset.source_snapshot.sha256.to_string(),
                    mapping_sha256: prepared.dataset.mapping.sha256.to_string(),
                },
                build_scope: BuildScope::Sample,
                embedding_profile: profile,
                provenance: pipeline_provenance,
                test_only: result_set.test_only,
            },
        )?,
    };
    write_portable_build_report(&staged_index, &prepared, &manifest)?;
    staging.publish_child("assembled-index")?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn write_portable_build_report(
    out: &Path,
    prepared: &PreparedCorpusManifest,
    manifest: &FastIndexManifest,
) -> Result<()> {
    let accounting = portable_dataset_accounting(prepared, manifest.occurrences.rows)?;
    let mut report = json!({
        "schema_version": if manifest.test_only { "livefire.rag.fast-build-report/2" } else { "livefire.rag.fast-build-report/1" },
        "source": manifest.source,
        "build_scope": manifest.build_scope,
        "complete": manifest.complete,
        "document_count": manifest.documents.rows,
        "occurrence_count": manifest.occurrences.rows,
        "vector_count": manifest.vectors.count,
        "embedding_profile_sha256": manifest.embedding_profile.sha256,
        // Embedding happened in the previous portable stage. These counters
        // describe work done by assembly, so both are zero.
        "cache_hits": 0,
        "embedded": 0,
        "accounting": accounting,
    });
    if manifest.test_only {
        report["test_only"] = Value::Bool(true);
    }
    write_canonical_json(&out.join("build-report.json"), &report)?;
    Ok(())
}

fn portable_dataset_accounting(
    prepared: &PreparedCorpusManifest,
    indexed_occurrences: u64,
) -> Result<Value> {
    let mut source_rows_by_relation = BTreeMap::new();
    let mut structured_only_by_relation = BTreeMap::new();
    let mut excluded_by_scope_by_relation = BTreeMap::new();
    let mut source_rows = 0_u64;
    let mut structured_only = 0_u64;
    let mut excluded_by_scope = 0_u64;
    for (relation, accounting) in &prepared.relation_accounting {
        source_rows = source_rows
            .checked_add(accounting.source_rows)
            .ok_or(Error::CountOverflow)?;
        source_rows_by_relation.insert(relation.clone(), accounting.source_rows);
        if prepared
            .dataset
            .structured_only_relations
            .contains(relation)
        {
            structured_only = structured_only
                .checked_add(accounting.source_rows)
                .ok_or(Error::CountOverflow)?;
            structured_only_by_relation.insert(relation.clone(), accounting.source_rows);
        } else if prepared.dataset.excluded_relations.contains(relation) {
            excluded_by_scope = excluded_by_scope
                .checked_add(accounting.source_rows)
                .ok_or(Error::CountOverflow)?;
            excluded_by_scope_by_relation.insert(relation.clone(), accounting.source_rows);
        } else if prepared.dataset.included_relations.contains(relation)
            && accounting.excluded_rows > 0
        {
            // Included relations can still contain malformed, unknown, or
            // deliberately structured-only rows. They were inspected during
            // preparation but do not have searchable document projections.
            structured_only = structured_only
                .checked_add(accounting.excluded_rows)
                .ok_or(Error::CountOverflow)?;
            structured_only_by_relation.insert(relation.clone(), accounting.excluded_rows);
        }
    }
    let closed = indexed_occurrences
        .checked_add(structured_only)
        .and_then(|value| value.checked_add(excluded_by_scope))
        .ok_or(Error::CountOverflow)?;
    if source_rows != closed {
        return Err(Error::AccountingClosure(
            "portable dataset source accounting does not close",
        ));
    }
    Ok(json!({
            "coverage_semantics": "dataset_scope_only_not_source_corpus_coverage",
            "semantic_source_coverage_complete": false,
            "dataset_id": prepared.dataset.id,
            "dataset_version": prepared.dataset.version,
            "source_records": source_rows,
            "source_records_by_relation": source_rows_by_relation,
            "indexed_occurrences": indexed_occurrences,
            "structured_only_occurrences": structured_only,
            "structured_only_by_relation": structured_only_by_relation,
            "excluded_by_scope_occurrences": excluded_by_scope,
            "excluded_by_scope_by_relation": excluded_by_scope_by_relation,
    }))
}

fn project_prepared_batch(
    batch: &RecordBatch,
    relation: &str,
    context: &ProjectionContext,
    source_row_ordinal: &mut u64,
    documents: &mut BTreeMap<String, DocumentAccumulator>,
    occurrences: &mut Vec<PreparedOccurrenceRow>,
    execution: PreparedProjectionExecution<'_>,
) -> Result<()> {
    let event_ids = strings(batch, "event_id")?;
    let json_rows = strings(batch, "typed_event_json")?;
    let support = strings(batch, "support_ref")?;
    let batch_first_row = *source_row_ordinal;
    map_batch_ranges_in_source_order(
        execution.batch.worker_pool,
        batch.num_rows(),
        execution.batch.workers,
        |_range_ordinal, range| {
            let mut partial = PreparedBatchProjection {
                source_rows: 0,
                documents: BTreeMap::new(),
                occurrences: Vec::new(),
            };
            for row in range {
                let ordinal = batch_first_row
                    .checked_add(u64::try_from(row).map_err(|_| Error::CountOverflow)?)
                    .ok_or(Error::CountOverflow)?;
                partial.source_rows = partial
                    .source_rows
                    .checked_add(1)
                    .ok_or(Error::CountOverflow)?;
                let input = ProjectionInput {
                    relation_name: relation,
                    event_id: event_ids.value(row),
                    typed_event_json: json_rows.value(row),
                    support_ref: support.value(row),
                    context,
                };
                let projected = match execution.projection {
                    PreparationProjection::Generic => Some(project(input)?),
                    PreparationProjection::M45Commands => project_m45_command(input)?,
                };
                let Some(projected) = projected else {
                    continue;
                };
                let Some(document) = projected.document else {
                    continue;
                };
                let document_sha256 = sha256_bytes(&serde_json::to_vec(&document)?);
                let mut fast = fast_document(document.clone(), document_sha256.clone())?;
                fast.facets_json = canonical_string(&document.facets)?;
                let entry = partial
                    .documents
                    .entry(document.document_id.clone())
                    .or_insert_with(|| DocumentAccumulator {
                        document: fast.clone(),
                        primary_relation: relation.to_owned(),
                        relations: BTreeSet::from([relation.to_owned()]),
                    });
                if entry.document.document_sha256 != document_sha256
                    || entry.document.semantic_text != fast.semantic_text
                    || entry.document.facets_json != fast.facets_json
                {
                    return Err(Error::InconsistentDocument(document.document_id));
                }
                entry.document.occurrence_count = entry
                    .document
                    .occurrence_count
                    .checked_add(1)
                    .ok_or(Error::CountOverflow)?;
                entry.relations.insert(relation.to_owned());
                let event_time_ms = parse_event_time_ms(
                    projected.occurrence.event_time.as_deref(),
                    projected.occurrence.event_time_availability,
                )?;
                let mut occurrence_hasher = Sha256::new();
                occurrence_hasher.update(context.snapshot.sha256.as_bytes());
                occurrence_hasher.update([0]);
                occurrence_hasher.update(relation.as_bytes());
                occurrence_hasher.update([0]);
                occurrence_hasher.update(event_ids.value(row).as_bytes());
                partial.occurrences.push(PreparedOccurrenceRow {
                    occurrence_id: format!("occ-{:x}", occurrence_hasher.finalize()),
                    document_id: document.document_id,
                    event_time_ms,
                    relation: relation.to_owned(),
                    source_row_ordinal: ordinal,
                    exact_attributes_json: canonical_string(
                        &projected.occurrence.exact_attributes,
                    )?,
                    snapshot_sha256: Digest::new(context.snapshot.sha256.clone())?,
                    mapping_sha256: Digest::new(context.mapping_pack.sha256.clone())?,
                    event_id: event_ids.value(row).to_owned(),
                    support_ref: support.value(row).to_owned(),
                });
            }
            Ok(partial)
        },
        |_range_ordinal, range, partial| {
            if partial.source_rows
                != u64::try_from(range.len()).map_err(|_| Error::CountOverflow)?
            {
                return Err(Error::AccountingClosure(
                    "prepared batch range rows do not close",
                ));
            }
            merge_prepared_projection(
                documents,
                occurrences,
                partial.documents,
                partial.occurrences,
            )
        },
    )?;
    *source_row_ordinal = batch_first_row
        .checked_add(u64::try_from(batch.num_rows()).map_err(|_| Error::CountOverflow)?)
        .ok_or(Error::CountOverflow)?;
    Ok(())
}

fn project_prepared_row_group(
    object: &AdmittedParquetObject,
    row_group_ordinal: usize,
    first_source_row: u64,
    relation: &str,
    context: &ProjectionContext,
    execution: PreparedProjectionExecution<'_>,
) -> Result<PreparedRowGroupProjection> {
    let mut next_source_row = first_source_row;
    let mut documents = BTreeMap::new();
    let mut occurrences = Vec::new();
    for batch in object.scan_row_group(
        row_group_ordinal,
        &["event_id", "typed_event_json", "support_ref"],
    )? {
        project_prepared_batch(
            &batch?,
            relation,
            context,
            &mut next_source_row,
            &mut documents,
            &mut occurrences,
            execution,
        )?;
    }
    Ok(PreparedRowGroupProjection {
        ordinal: row_group_ordinal,
        source_rows: next_source_row
            .checked_sub(first_source_row)
            .ok_or(Error::CountOverflow)?,
        documents,
        occurrences,
    })
}

fn merge_prepared_projection(
    documents: &mut BTreeMap<String, DocumentAccumulator>,
    occurrences: &mut Vec<PreparedOccurrenceRow>,
    partial_documents: BTreeMap<String, DocumentAccumulator>,
    partial_occurrences: Vec<PreparedOccurrenceRow>,
) -> Result<()> {
    for (document_id, accumulated) in partial_documents {
        if let Some(existing) = documents.get_mut(&document_id) {
            merge_document_accumulators(&document_id, existing, accumulated)?;
        } else {
            documents.insert(document_id, accumulated);
        }
    }
    occurrences.extend(partial_occurrences);
    Ok(())
}

fn flush_document_shard(
    root: &Path,
    buffer: &mut Vec<PreparedDocumentRow>,
    shard_rows: usize,
    objects: &mut Vec<PreparedDocumentObject>,
    flush_remainder: bool,
) -> Result<()> {
    while buffer.len() >= shard_rows || flush_remainder && !buffer.is_empty() {
        let rows_to_write = buffer.len().min(shard_rows);
        let remainder = buffer.split_off(rows_to_write);
        let rows = std::mem::replace(buffer, remainder);
        let ordinal = objects.len();
        let relative = format!("documents/part-{ordinal:06}.parquet");
        let path = root.join(&relative);
        write_prepared_documents(&path, &rows)?;
        objects.push(PreparedDocumentObject {
            object: object_entry(
                &relative,
                &path,
                rows.len() as u64,
                canonical_digest(&rows)?,
            )?,
            ordinal: u32::try_from(ordinal)
                .map_err(|_| Error::AccountingClosure("too many document shards"))?,
            first_document_id: rows[0].document_id.clone(),
            last_document_id: rows[rows.len() - 1].document_id.clone(),
            embedding_input_order_sha256: embedding_input_order_digest(&rows),
        });
    }
    Ok(())
}

fn validate_streamed_prepared_documents(
    root: &Path,
    manifest: &PreparedCorpusManifest,
) -> Result<()> {
    let mut count = 0_u64;
    let mut previous_document_id: Option<String> = None;
    let mut document_order_hasher = Sha256::new();
    let mut embedding_input_order_hasher = Sha256::new();
    embedding_input_order_hasher.update(b"livefire.rag.embedding-input-order/1\0");
    for (ordinal, object) in manifest.documents.iter().enumerate() {
        if usize::try_from(object.ordinal).ok() != Some(ordinal) {
            return Err(Error::AccountingClosure(
                "prepared document object ordinals differ",
            ));
        }
        let path = resolve_existing_artifact(root, &object.object.path)?;
        if fs::metadata(&path)?.len() != object.object.bytes
            || file_digest(&path)? != object.object.sha256
        {
            return Err(Error::AccountingClosure("prepared object digest differs"));
        }
        let rows = read_prepared_documents(&path)?;
        if rows.len() as u64 != object.object.rows
            || rows.first().map(|row| &row.document_id) != Some(&object.first_document_id)
            || rows.last().map(|row| &row.document_id) != Some(&object.last_document_id)
            || canonical_digest(&rows)? != object.object.logical_order_sha256
            || embedding_input_order_digest(&rows) != object.embedding_input_order_sha256
        {
            return Err(Error::AccountingClosure(
                "prepared document object metadata differs",
            ));
        }
        for row in &rows {
            if row.document_ordinal != count
                || previous_document_id
                    .as_ref()
                    .is_some_and(|previous| previous >= &row.document_id)
            {
                return Err(Error::AccountingClosure(
                    "prepared document rows are not strictly ordered",
                ));
            }
            previous_document_id = Some(row.document_id.clone());
            document_order_hasher.update(row.document_id.as_bytes());
            document_order_hasher.update([0]);
            for field in [
                row.document_id.as_str(),
                row.document_sha256.as_str(),
                row.semantic_text_sha256.as_str(),
            ] {
                embedding_input_order_hasher.update(field.as_bytes());
                embedding_input_order_hasher.update([0]);
            }
            count = count.checked_add(1).ok_or(Error::CountOverflow)?;
        }
    }
    if count != manifest.document_count
        || Digest::new(format!("{:x}", document_order_hasher.finalize()))?
            != manifest.document_order_sha256
        || Digest::new(format!("{:x}", embedding_input_order_hasher.finalize()))?
            != manifest.embedding_input_order_sha256
    {
        return Err(Error::AccountingClosure(
            "prepared document stream metadata differs",
        ));
    }
    Ok(())
}

fn flush_occurrence_shards(
    root: &Path,
    relation: &str,
    buffer: &mut Vec<PreparedOccurrenceRow>,
    relation_part: &mut u64,
    objects: &mut Vec<PreparedOccurrenceObject>,
    flush_remainder: bool,
) -> Result<()> {
    while buffer.len() >= OCCURRENCE_SHARD_ROWS || flush_remainder && !buffer.is_empty() {
        let rows_to_write = buffer.len().min(OCCURRENCE_SHARD_ROWS);
        let remainder = buffer.split_off(rows_to_write);
        let rows = std::mem::replace(buffer, remainder);
        let relative = format!("occurrences/{relation}/part-{:06}.parquet", *relation_part);
        let path = root.join(&relative);
        write_prepared_occurrences(&path, &rows)?;
        objects.push(PreparedOccurrenceObject {
            object: object_entry(
                &relative,
                &path,
                rows.len() as u64,
                canonical_digest(&rows)?,
            )?,
            // The caller rewrites global ordinals after all relation shards
            // are known. Keep this checked relation-local value until then.
            ordinal: u32::try_from(*relation_part)
                .map_err(|_| Error::AccountingClosure("too many occurrence shards"))?,
            relation: relation.to_owned(),
        });
        *relation_part = relation_part.checked_add(1).ok_or(Error::CountOverflow)?;
    }
    Ok(())
}

pub(crate) fn load_prepared(root: &Path) -> Result<PreparedCorpusManifest> {
    let manifest = load_prepared_documents_only(root)?;
    verify_prepared_occurrence_objects(root, &manifest)?;
    Ok(manifest)
}

/// Verify every object in a prepared dataset without changing the dataset.
pub(crate) fn verify_prepared(root: &Path) -> Result<()> {
    let manifest = load_prepared(root)?;
    load_all_prepared_documents(root, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

/// Load the prepared manifest and verify only the document files needed by
/// planning and embedding. Occurrence files belong to assembly and full
/// verification, so this path deliberately never opens them.
fn load_prepared_documents_only(root: &Path) -> Result<PreparedCorpusManifest> {
    let manifest: PreparedCorpusManifest = read_json(&root.join(MANIFEST_FILE))?;
    manifest.validate()?;
    verify_prepared_document_objects(root, &manifest)?;
    Ok(manifest)
}

fn load_all_prepared_documents(
    root: &Path,
    manifest: &PreparedCorpusManifest,
) -> Result<Vec<PreparedDocumentRow>> {
    let capacity = usize::try_from(manifest.document_count)
        .map_err(|_| Error::AccountingClosure("prepared document count overflow"))?;
    let mut rows = Vec::with_capacity(capacity);
    for object in &manifest.documents {
        rows.extend(read_prepared_documents(&resolve_existing_artifact(
            root,
            &object.object.path,
        )?)?);
    }
    validate_prepared_documents(manifest, &rows)?;
    Ok(rows)
}

pub(crate) fn load_embedding_plan_v2(root_or_file: &Path) -> Result<EmbeddingPlanV2> {
    let path = manifest_or_file(root_or_file, "plan.json");
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let schema =
        value
            .get("schema_version")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure(
                "embedding plan schema version is absent",
            ))?;
    if schema == EMBEDDING_PLAN_SCHEMA {
        return Err(Error::UnsupportedPlanVersion(
            "embedding plan v1 cannot be executed; rerun plan-embeddings with a pinned tokenizer to create v2"
                .into(),
        ));
    }
    if schema != EMBEDDING_PLAN_V2_SCHEMA {
        return Err(Error::UnsupportedPlanVersion(format!(
            "unsupported embedding plan schema: {schema}"
        )));
    }
    let plan: EmbeddingPlanV2 = serde_json::from_value(value)?;
    plan.validate()?;
    let plan_root = if root_or_file.is_dir() {
        root_or_file
    } else {
        path.parent()
            .ok_or(Error::AccountingClosure("embedding plan parent is absent"))?
    };
    plan.read_document_token_counts(plan_root)?;
    Ok(plan)
}

/// Re-open a finalized result directory and prove that its manifest, receipts,
/// vector parts, and task reports still match the prepared corpus and plan.
/// The index profile supplies the compact model fields used when the parts
/// were assembled; no model server is contacted.
pub(crate) fn load_completed_embedding_result_set(
    root: &Path,
    prepared_root: &Path,
    plan_root: &Path,
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    expected_profile: &rag_embedding::EmbeddingProfile,
) -> Result<EmbeddingResultSetManifest> {
    validate_embedding_artifact_coverage_v2(root, plan, true)?;
    let result_set: EmbeddingResultSetManifest = read_json(&root.join(MANIFEST_FILE))?;
    let bound_profile_path = root.join(EMBEDDING_PROFILE_FILE);
    let profile_bytes = fs::read(&bound_profile_path)?;
    let profile = parse_bound_portable_profile(
        &profile_bytes,
        plan.embedding_profile.component.sha256.as_str(),
    )?;
    validate_plan_profile_fields(&plan.embedding_profile, &profile_bytes, &profile)?;
    if &profile != expected_profile {
        return Err(Error::AccountingClosure(
            "catalogue embedding profile differs from finalized result",
        ));
    }
    let runtime = component_from_value(
        serde_json::from_slice::<Value>(&profile_bytes)?
            .get("runtime")
            .ok_or(Error::AccountingClosure(
                "embedding runtime component is absent",
            ))?,
    )?;
    let (_, receipts, reports) =
        validate_complete_embedding_tasks_v2(root, plan, prepared, &profile, &runtime)?;
    result_set.validate_v2(plan, &receipts)?;
    let token_counts = plan.read_document_token_counts(plan_root)?;
    let expected_summary = embedding_run_summary(
        plan,
        prepared,
        &receipts,
        &reports,
        &token_counts,
        embedding_run_artifact_sizes(prepared_root, plan_root, &bound_profile_path, root, plan)?,
    )?;
    let actual_summary: EmbeddingRunSummaryContract = read_json(&root.join("summary.json"))?;
    validate_embedding_run_summary(&actual_summary, &expected_summary)?;
    Ok(result_set)
}

fn validate_embedding_run_summary(
    actual: &EmbeddingRunSummaryContract,
    expected: &EmbeddingRunSummaryContract,
) -> Result<()> {
    if actual != expected {
        return Err(Error::AccountingClosure(
            "embedding run summary differs from task reports",
        ));
    }
    Ok(())
}

fn parse_task_selection(value: Option<&str>) -> Result<TaskSelection> {
    let Some(value) = value else {
        return Ok(TaskSelection::All);
    };
    let (start, end) = value
        .split_once("..")
        .filter(|(start, end)| {
            !start.is_empty() && !end.is_empty() && !start.contains('.') && !end.contains('.')
        })
        .ok_or(Error::InvalidTaskRange)?;
    let start = start
        .parse::<usize>()
        .map_err(|_| Error::InvalidTaskRange)?;
    let end = end.parse::<usize>().map_err(|_| Error::InvalidTaskRange)?;
    if start >= end {
        return Err(Error::InvalidTaskRange);
    }
    Ok(TaskSelection::Range { start, end })
}

fn verify_prepared_document_objects(root: &Path, manifest: &PreparedCorpusManifest) -> Result<()> {
    for object in manifest.documents.iter().map(|value| &value.object) {
        let path = resolve_existing_artifact(root, &object.path)?;
        if fs::metadata(&path)?.len() != object.bytes || file_digest(&path)? != object.sha256 {
            return Err(Error::AccountingClosure("prepared object digest differs"));
        }
    }
    Ok(())
}

fn verify_prepared_occurrence_objects(
    root: &Path,
    manifest: &PreparedCorpusManifest,
) -> Result<()> {
    let mut previous_occurrence_order: Option<(String, u64)> = None;
    let mut occurrence_rows = 0_u64;
    for object in &manifest.occurrences {
        let path = resolve_existing_artifact(root, &object.object.path)?;
        if fs::metadata(&path)?.len() != object.object.bytes
            || file_digest(&path)? != object.object.sha256
        {
            return Err(Error::AccountingClosure("prepared object digest differs"));
        }
        let rows = read_prepared_occurrences(&path)?;
        if rows.len() as u64 != object.object.rows
            || rows.iter().any(|row| row.validate().is_err())
            || rows.iter().any(|row| {
                row.relation != object.relation
                    || row.snapshot_sha256 != manifest.dataset.source_snapshot.sha256
                    || row.mapping_sha256 != manifest.dataset.mapping.sha256
            })
            || canonical_digest(&rows)? != object.object.logical_order_sha256
        {
            return Err(Error::AccountingClosure(
                "prepared occurrence object metadata differs",
            ));
        }
        for row in &rows {
            let order = (row.relation.clone(), row.source_row_ordinal);
            if previous_occurrence_order
                .as_ref()
                .is_some_and(|previous| previous >= &order)
            {
                return Err(Error::AccountingClosure(
                    "prepared occurrence object order differs",
                ));
            }
            previous_occurrence_order = Some(order);
        }
        occurrence_rows = occurrence_rows
            .checked_add(rows.len() as u64)
            .ok_or(Error::CountOverflow)?;
    }
    if occurrence_rows != manifest.occurrence_count {
        return Err(Error::AccountingClosure(
            "prepared occurrence object coverage differs",
        ));
    }
    Ok(())
}

struct TaskDocumentLoaderV2<'a> {
    root: &'a Path,
    prepared: &'a PreparedCorpusManifest,
    cached: Option<(SafeRelativePath, Vec<PreparedDocumentRow>)>,
}

impl<'a> TaskDocumentLoaderV2<'a> {
    fn new(root: &'a Path, prepared: &'a PreparedCorpusManifest) -> Self {
        Self {
            root,
            prepared,
            cached: None,
        }
    }

    fn load(&mut self, task: &EmbeddingTaskV2) -> Result<Vec<PreparedDocumentRow>> {
        let capacity = usize::try_from(task.row_count())
            .map_err(|_| Error::AccountingClosure("task row count overflow"))?;
        let mut task_rows = Vec::with_capacity(capacity);
        for slice in &task.input_slices {
            self.load_slice(slice, &mut task_rows)?;
        }
        if task_rows.len() != capacity
            || task_rows.iter().enumerate().any(|(offset, row)| {
                row.document_ordinal != task.ordinal_start.saturating_add(offset as u64)
            })
            || embedding_input_order_digest(&task_rows) != task.embedding_input_order_sha256
        {
            return Err(Error::AccountingClosure(
                "embedding task rows differ from prepared order",
            ));
        }
        Ok(task_rows)
    }

    fn load_slice(
        &mut self,
        slice: &EmbeddingInputSliceV2,
        task_rows: &mut Vec<PreparedDocumentRow>,
    ) -> Result<()> {
        let object = self
            .prepared
            .documents
            .iter()
            .find(|object| object.object.path == slice.path)
            .ok_or(Error::AccountingClosure(
                "embedding task input object is absent",
            ))?;
        if object.object.sha256 != slice.object_sha256 {
            return Err(Error::AccountingClosure(
                "embedding task input object digest differs",
            ));
        }
        if self
            .cached
            .as_ref()
            .is_none_or(|(path, _)| path != &slice.path)
        {
            let path = resolve_existing_artifact(self.root, &slice.path)?;
            if fs::metadata(&path)?.len() != object.object.bytes
                || file_digest(&path)? != object.object.sha256
            {
                return Err(Error::AccountingClosure(
                    "prepared document object digest differs",
                ));
            }
            let rows = read_prepared_documents(&path)?;
            if rows.len() as u64 != object.object.rows
                || rows.iter().any(|row| row.validate().is_err())
                || rows.first().map(|row| &row.document_id) != Some(&object.first_document_id)
                || rows.last().map(|row| &row.document_id) != Some(&object.last_document_id)
                || canonical_digest(&rows)? != object.object.logical_order_sha256
                || embedding_input_order_digest(&rows) != object.embedding_input_order_sha256
            {
                return Err(Error::AccountingClosure(
                    "prepared document object metadata differs",
                ));
            }
            self.cached = Some((slice.path.clone(), rows));
        }
        let rows = &self.cached.as_ref().expect("cache was populated").1;
        let start = usize::try_from(slice.row_offset)
            .map_err(|_| Error::AccountingClosure("task slice offset overflow"))?;
        let count = usize::try_from(slice.rows)
            .map_err(|_| Error::AccountingClosure("task slice row count overflow"))?;
        let end = start
            .checked_add(count)
            .ok_or(Error::AccountingClosure("task slice range overflow"))?;
        let selected = rows.get(start..end).ok_or(Error::AccountingClosure(
            "task slice exceeds prepared object",
        ))?;
        if embedding_input_order_digest(selected) != slice.embedding_input_order_sha256 {
            return Err(Error::AccountingClosure(
                "task slice input order digest differs",
            ));
        }
        task_rows.extend_from_slice(selected);
        Ok(())
    }
}

fn task_shard_expectation_v2(
    task: &EmbeddingTaskV2,
    dimensions: u32,
) -> Result<EmbeddingShardExpectation> {
    Ok(EmbeddingShardExpectation {
        row_count: task.row_count(),
        dimensions,
        order_sha256: decode_sha256_hex(task.embedding_input_order_sha256.as_str())?,
    })
}

fn task_report_path(root: &Path, task: &EmbeddingTaskV2) -> Result<PathBuf> {
    let path = SafeRelativePath::new(format!("reports/{}.json", task.task_id))?;
    resolve_output_artifact(root, &path).map_err(Error::from)
}

fn ensure_task_report(
    path: &Path,
    task_index: usize,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    binding: TaskReportBindings<'_>,
    details: TaskRunDetails,
) -> Result<()> {
    let transport_bytes = TransportByteAccounting {
        status: if details.execution.is_some() {
            ObservationStatus::Partial
        } else {
            ObservationStatus::NotMeasured
        },
        request_body_bytes: None,
        response_body_bytes: None,
        submitted_text_bytes: details
            .execution
            .as_ref()
            .map(|execution| execution.sent_input_text_bytes),
        decoded_vector_bytes: details
            .execution
            .as_ref()
            .map(|execution| execution.vector_bytes),
    };
    let report = BuilderEmbeddingTaskReport {
        schema_version: "livefire.rag.embedding-task-run-report/1".into(),
        plan_sha256: plan.component_sha256.clone(),
        source_snapshot_sha256: binding.prepared.dataset.source_snapshot.sha256.clone(),
        prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        tokenizer_sha256: plan.executable_tokenizer.artifact.sha256.clone(),
        task_id: task.task_id.clone(),
        task_index,
        ordinal_start: task.ordinal_start,
        ordinal_end: task.ordinal_end,
        document_count: task.row_count(),
        token_count: task.token_count,
        receipt_sha256: binding.receipt.component_sha256.clone(),
        outcome: details.outcome,
        started_unix_ms: details.started_unix_ms,
        finished_unix_ms: details.finished_unix_ms,
        git: binding.run_context.git.clone(),
        machine: binding.run_context.machine.clone(),
        lm_studio: if binding.receipt.derived() {
            LmStudioContext::derived_vectors(&binding.profile.model)
        } else if binding.receipt.test_only() {
            LmStudioContext::deterministic_test_vectors(&binding.profile.model)
        } else {
            LmStudioContext::embedding(
                &binding.profile.model,
                &binding.receipt.executor.returned_model,
                binding.batch_size,
                binding.requests_in_flight,
            )
        },
        transport_bytes,
        resource_usage: binding.run_context.resources.clone(),
        artifact_sizes: task_artifact_sizes(binding.vector_path, binding.receipt_path),
        execution: details.execution,
    };
    if path.try_exists()? {
        let existing = read_json::<BuilderEmbeddingTaskReport>(path);
        if existing.as_ref().is_ok_and(|existing| {
            validate_task_report(
                existing,
                task_index,
                task,
                plan,
                Some(binding.prepared),
                binding.profile,
                binding.receipt,
            )
            .is_ok()
        }) {
            return Ok(());
        }
        quarantine_regular_file(path)?;
    }
    write_canonical_json(path, &report)?;
    Ok(())
}

fn validate_task_report(
    report: &BuilderEmbeddingTaskReport,
    task_index: usize,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    prepared: Option<&PreparedCorpusManifest>,
    profile: &rag_embedding::EmbeddingProfile,
    receipt: &VectorResultReceipt,
) -> Result<()> {
    if report.schema_version != "livefire.rag.embedding-task-run-report/1"
        || report.plan_sha256 != plan.component_sha256
        || report.prepared_corpus_sha256 != plan.prepared_corpus_sha256
        || report.embedding_profile_sha256 != plan.embedding_profile.component.sha256
        || report.tokenizer_sha256 != plan.executable_tokenizer.artifact.sha256
        || prepared.is_some_and(|prepared| {
            report.source_snapshot_sha256 != prepared.dataset.source_snapshot.sha256
        })
        || report.lm_studio.configured_model != profile.model
        || report.lm_studio.returned_model != receipt.executor.returned_model
        || report.task_id != task.task_id
        || report.task_index != task_index
        || report.ordinal_start != task.ordinal_start
        || report.ordinal_end != task.ordinal_end
        || report.document_count != task.row_count()
        || report.token_count != task.token_count
        || report.receipt_sha256 != receipt.component_sha256
        || report.started_unix_ms.is_some() != report.finished_unix_ms.is_some()
        || report
            .started_unix_ms
            .zip(report.finished_unix_ms)
            .is_some_and(|(start, end)| start > end)
        || report.execution.is_some() != report.started_unix_ms.is_some()
        || report.transport_bytes.request_body_bytes.is_some()
        || report.transport_bytes.response_body_bytes.is_some()
        || report
            .execution
            .as_ref()
            .map(|execution| execution.sent_input_text_bytes)
            != report.transport_bytes.submitted_text_bytes
        || report
            .execution
            .as_ref()
            .map(|execution| execution.vector_bytes)
            != report.transport_bytes.decoded_vector_bytes
        || report.artifact_sizes.vector_shard_bytes != Some(receipt.vector.bytes)
        || report.artifact_sizes.receipt_bytes.is_none()
        || report.artifact_sizes.task_report_bytes.is_some()
        || report.execution.as_ref().is_some_and(|execution| {
            execution.rows != task.row_count()
                || execution.shard_bytes != receipt.vector.bytes
                || execution.attempts as u64 != receipt.executor.requests
                || execution.retries as u64 != receipt.executor.retries
        })
    {
        return Err(Error::AccountingClosure(
            "embedding task report binding differs",
        ));
    }
    Ok(())
}

fn validate_task_report_v2(
    report: &BuilderEmbeddingTaskReportV2,
    task_index: usize,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    prepared: Option<&PreparedCorpusManifest>,
    profile: &rag_embedding::EmbeddingProfile,
    receipt: &VectorResultReceipt,
) -> Result<()> {
    let identity = &report.execution_identity;
    for component in [
        &identity.executor_image,
        &identity.executor_image_build,
        &identity.runtime,
        &identity.worker_binary,
        &identity.model_artifact,
        &identity.embedding_profile,
    ] {
        component.validate()?;
    }
    if report.schema_version != "livefire.rag.embedding-task-run-report/2"
        || report.plan_sha256 != plan.component_sha256
        || report.prepared_corpus_sha256 != plan.prepared_corpus_sha256
        || report.embedding_profile_sha256 != plan.embedding_profile.component.sha256
        || report.tokenizer_sha256 != plan.executable_tokenizer.artifact.sha256
        || prepared.is_some_and(|prepared| {
            report.source_snapshot_sha256 != prepared.dataset.source_snapshot.sha256
        })
        || identity.backend_kind.is_empty()
        || identity.backend_kind != report.backend.kind
        || identity.embedding_profile != plan.embedding_profile.component
        || identity.model_artifact != plan.embedding_profile.model_artifact
        || identity.runtime != receipt.executor.runtime
        || identity.worker_binary != receipt.executor.implementation
        || identity.returned_model != profile.model
        || identity.returned_model != receipt.executor.returned_model
        || report.accelerator.status != ObservationStatus::Observed
        || report.accelerator.provider.as_deref() != Some(identity.accelerator.provider.as_str())
        || report.accelerator.model.as_deref() != Some(identity.accelerator.model.as_str())
        || report.accelerator.architecture.as_deref()
            != Some(identity.accelerator.architecture.as_str())
        || report.accelerator.compute_capability.as_deref()
            != Some(identity.accelerator.compute_capability.as_str())
        || report.accelerator.count != Some(identity.accelerator.count)
        || identity.accelerator.provider.is_empty()
        || identity.accelerator.model.is_empty()
        || identity.accelerator.architecture.is_empty()
        || identity.accelerator.compute_capability.is_empty()
        || identity.accelerator.count != 1
        || report.backend.endpoint_kind.is_empty()
        || report.backend.batch_size == 0
        || report.backend.requests_in_flight == 0
        || report.accelerator.count == Some(0)
        || report.task_id != task.task_id
        || report.task_index != task_index
        || report.ordinal_start != task.ordinal_start
        || report.ordinal_end != task.ordinal_end
        || report.document_count != task.row_count()
        || report.token_count != task.token_count
        || report.receipt_sha256 != receipt.component_sha256
        || report.started_unix_ms.is_some() != report.finished_unix_ms.is_some()
        || report
            .started_unix_ms
            .zip(report.finished_unix_ms)
            .is_some_and(|(start, end)| start > end)
        || report.execution.is_some() != report.started_unix_ms.is_some()
        || report.transport_bytes.request_body_bytes.is_some()
        || report.transport_bytes.response_body_bytes.is_some()
        || report
            .execution
            .as_ref()
            .map(|execution| execution.sent_input_text_bytes)
            != report.transport_bytes.submitted_text_bytes
        || report
            .execution
            .as_ref()
            .map(|execution| execution.vector_bytes)
            != report.transport_bytes.decoded_vector_bytes
        || report.artifact_sizes.vector_shard_bytes != Some(receipt.vector.bytes)
        || report.artifact_sizes.receipt_bytes.is_none()
        || report.artifact_sizes.task_report_bytes.is_some()
        || report.execution.as_ref().is_some_and(|execution| {
            execution.rows != task.row_count()
                || execution.shard_bytes != receipt.vector.bytes
                || execution.attempts as u64 != receipt.executor.requests
                || execution.retries as u64 != receipt.executor.retries
        })
    {
        return Err(Error::AccountingClosure(
            "embedding task report v2 binding differs",
        ));
    }
    Ok(())
}

fn validate_tei_worker_context(
    worker: &TeiWorkerReportContextV2,
    policy: &TeiCheckpointProfileV3,
    plan: &EmbeddingPlanV2,
) -> Result<()> {
    let identity = &worker.execution_identity;
    let expected_image = component(
        &policy.executor_image.component.id,
        &policy.executor_image.component.version,
        &policy.executor_image.component.sha256,
    )?;
    let expected_runtime = component(
        &policy.runtime.id,
        &policy.runtime.version,
        &policy.runtime.sha256,
    )?;
    let expected_image_build = component(
        &policy.executor_image_build.id,
        &policy.executor_image_build.version,
        &policy.executor_image_build.sha256,
    )?;
    let expected_accelerator = EmbeddingAcceleratorPolicyV2 {
        provider: policy.accelerator.provider.clone(),
        model: policy.accelerator.gpu_model_id.clone(),
        architecture: policy.accelerator.architecture_image_class.clone(),
        compute_capability: policy.accelerator.compute_capability.clone(),
        count: policy.accelerator.gpu_count,
    };
    if worker.schema_version != "livefire.rag.tei-worker-report-context/1"
        || identity.backend_kind != "tei"
        || identity.executor_image != expected_image
        || identity.executor_image_build != expected_image_build
        || identity.runtime != expected_runtime
        || identity.model_artifact != plan.embedding_profile.model_artifact
        || identity.embedding_profile != plan.embedding_profile.component
        || identity.returned_model != policy.api_model_key
        || identity.accelerator != expected_accelerator
        || worker.backend.kind != "tei"
        || worker.backend.batch_size == 0
        || worker.backend.requests_in_flight == 0
        || worker.accelerator.status != ObservationStatus::Observed
        || worker.accelerator.provider.as_deref() != Some(expected_accelerator.provider.as_str())
        || worker.accelerator.model.as_deref() != Some(expected_accelerator.model.as_str())
        || worker.accelerator.architecture.as_deref()
            != Some(expected_accelerator.architecture.as_str())
        || worker.accelerator.compute_capability.as_deref()
            != Some(expected_accelerator.compute_capability.as_str())
        || worker.accelerator.count != Some(1)
    {
        return Err(Error::AccountingClosure(
            "TEI worker execution context differs from embedding-policy/3",
        ));
    }
    identity.worker_binary.validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_tei_task_report_v2(
    path: &Path,
    task_index: usize,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    receipt: &VectorResultReceipt,
    vector_path: &Path,
    receipt_path: &Path,
    worker: &TeiWorkerReportContextV2,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    execution: EmbeddingTaskReport,
) -> Result<()> {
    let report = BuilderEmbeddingTaskReportV2 {
        schema_version: "livefire.rag.embedding-task-run-report/2".into(),
        plan_sha256: plan.component_sha256.clone(),
        source_snapshot_sha256: prepared.dataset.source_snapshot.sha256.clone(),
        prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        tokenizer_sha256: plan.executable_tokenizer.artifact.sha256.clone(),
        task_id: task.task_id.clone(),
        task_index,
        ordinal_start: task.ordinal_start,
        ordinal_end: task.ordinal_end,
        document_count: task.row_count(),
        token_count: task.token_count,
        receipt_sha256: receipt.component_sha256.clone(),
        outcome: TaskRunOutcome::Executed,
        started_unix_ms: Some(started_unix_ms),
        finished_unix_ms: Some(finished_unix_ms),
        execution_identity: worker.execution_identity.clone(),
        git: worker.git.clone(),
        machine: worker.machine.clone(),
        accelerator: worker.accelerator.clone(),
        backend: worker.backend.clone(),
        transport_bytes: TransportByteAccounting {
            status: ObservationStatus::Partial,
            request_body_bytes: None,
            response_body_bytes: None,
            submitted_text_bytes: Some(execution.sent_input_text_bytes),
            decoded_vector_bytes: Some(execution.vector_bytes),
        },
        resource_usage: worker.resource_usage.clone(),
        artifact_sizes: task_artifact_sizes(vector_path, receipt_path),
        execution: Some(execution),
    };
    if path.try_exists()? {
        quarantine_regular_file(path)?;
    }
    write_canonical_json(path, &report)?;
    Ok(())
}

fn unix_time_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::AccountingClosure("system time precedes Unix epoch"))?
        .as_millis()
        .try_into()
        .map_err(|_| Error::CountOverflow)
}

fn validate_completed_embedding_task_v2(
    receipt_path: &Path,
    vector_path: &Path,
    report_path: &Path,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    profile: &rag_embedding::EmbeddingProfile,
    runtime: &ComponentRef,
) -> Result<Option<VectorResultReceipt>> {
    if !receipt_path.try_exists()? {
        return Ok(None);
    }
    let receipt = match read_json::<VectorResultReceipt>(receipt_path) {
        Ok(receipt) => receipt,
        Err(_) => {
            quarantine_regular_file(receipt_path)?;
            if regular_file_exists(report_path)? {
                quarantine_regular_file(report_path)?;
            }
            return Ok(None);
        }
    };
    let expected = task_shard_expectation_v2(task, profile.dimensions)?;
    let valid = receipt.validate_against_v2(plan).is_ok()
        && receipt.executor.returned_model == profile.model
        && &receipt.executor.runtime == runtime
        && prepare_embedding_task_part(
            vector_path,
            expected,
            &profile.normalization,
            decode_sha256_hex(receipt.vector.sha256.as_str()).ok(),
        )
        .is_ok_and(|state| matches!(state, EmbeddingTaskPartPreparation::Verified { .. }));
    if valid {
        Ok(Some(receipt))
    } else {
        quarantine_regular_file(receipt_path)?;
        if regular_file_exists(report_path)? {
            quarantine_regular_file(report_path)?;
        }
        Ok(None)
    }
}

fn validate_complete_embedding_tasks_v2(
    root: &Path,
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    profile: &rag_embedding::EmbeddingProfile,
    runtime: &ComponentRef,
) -> Result<(
    Vec<ReceiptEntry>,
    Vec<VectorResultReceipt>,
    Vec<ValidatedEmbeddingTaskReport>,
)> {
    let mut entries = Vec::with_capacity(plan.tasks.len());
    let mut receipts = Vec::with_capacity(plan.tasks.len());
    let mut reports = Vec::with_capacity(plan.tasks.len());
    for (task_index, task) in plan.tasks.iter().enumerate() {
        let receipt_path = resolve_existing_artifact(root, &task.receipt_path)?;
        let vector_path = resolve_existing_artifact(root, &task.result_path)?;
        let receipt: VectorResultReceipt = read_json(&receipt_path)?;
        receipt.validate_against_v2(plan)?;
        if receipt.executor.returned_model != profile.model || &receipt.executor.runtime != runtime
        {
            return Err(Error::AccountingClosure(
                "embedding receipt runtime differs from profile",
            ));
        }
        verify_embedding_task_part(
            &vector_path,
            task_shard_expectation_v2(task, profile.dimensions)?,
            &profile.normalization,
            Some(decode_sha256_hex(receipt.vector.sha256.as_str())?),
        )?;
        let report_path = task_report_path(root, task)?;
        if !report_path.try_exists()? {
            return Err(Error::AccountingClosure("embedding task report is absent"));
        }
        let report = read_validated_task_report(
            &report_path,
            task_index,
            task,
            plan,
            Some(prepared),
            profile,
            &receipt,
        )?;
        entries.push(ReceiptEntry {
            task_id: task.task_id.clone(),
            path: task.receipt_path.clone(),
            sha256: receipt.component_sha256.clone(),
        });
        receipts.push(receipt);
        reports.push(report);
    }
    Ok((entries, receipts, reports))
}

fn read_validated_task_report(
    path: &Path,
    task_index: usize,
    task: &EmbeddingTaskV2,
    plan: &EmbeddingPlanV2,
    prepared: Option<&PreparedCorpusManifest>,
    profile: &rag_embedding::EmbeddingProfile,
    receipt: &VectorResultReceipt,
) -> Result<ValidatedEmbeddingTaskReport> {
    let value: Value = read_json(path)?;
    let report = decode_task_report(value)?;
    match &report {
        ValidatedEmbeddingTaskReport::V1(report) => {
            validate_task_report(report, task_index, task, plan, prepared, profile, receipt)?;
        }
        ValidatedEmbeddingTaskReport::V2(report) => {
            validate_task_report_v2(report, task_index, task, plan, prepared, profile, receipt)?;
        }
    }
    Ok(report)
}

fn decode_task_report(value: Value) -> Result<ValidatedEmbeddingTaskReport> {
    match value.get("schema_version").and_then(Value::as_str) {
        Some("livefire.rag.embedding-task-run-report/1") => {
            let report: BuilderEmbeddingTaskReport = serde_json::from_value(value)?;
            Ok(ValidatedEmbeddingTaskReport::V1(Box::new(report)))
        }
        Some("livefire.rag.embedding-task-run-report/2") => {
            let report: BuilderEmbeddingTaskReportV2 = serde_json::from_value(value)?;
            Ok(ValidatedEmbeddingTaskReport::V2(Box::new(report)))
        }
        _ => Err(Error::AccountingClosure(
            "embedding task report schema is unsupported",
        )),
    }
}

fn embedding_run_summary(
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    receipts: &[VectorResultReceipt],
    reports: &[ValidatedEmbeddingTaskReport],
    token_counts: &[u32],
    artifact_sizes: RunArtifactSizes,
) -> Result<EmbeddingRunSummaryContract> {
    if reports
        .iter()
        .all(|report| matches!(report, ValidatedEmbeddingTaskReport::V1(_)))
    {
        let reports = reports
            .iter()
            .map(|report| match report {
                ValidatedEmbeddingTaskReport::V1(report) => (**report).clone(),
                ValidatedEmbeddingTaskReport::V2(_) => unreachable!("variant checked"),
            })
            .collect::<Vec<_>>();
        return embedding_run_summary_v1(
            plan,
            prepared,
            receipts,
            &reports,
            token_counts,
            artifact_sizes,
        )
        .map(Box::new)
        .map(EmbeddingRunSummaryContract::V1);
    }
    if reports
        .iter()
        .all(|report| matches!(report, ValidatedEmbeddingTaskReport::V2(_)))
    {
        let reports = reports
            .iter()
            .map(|report| match report {
                ValidatedEmbeddingTaskReport::V2(report) => (**report).clone(),
                ValidatedEmbeddingTaskReport::V1(_) => unreachable!("variant checked"),
            })
            .collect::<Vec<_>>();
        return embedding_run_summary_v2(
            plan,
            prepared,
            receipts,
            &reports,
            token_counts,
            artifact_sizes,
        )
        .map(Box::new)
        .map(EmbeddingRunSummaryContract::V2);
    }
    Err(Error::AccountingClosure(
        "embedding task report schema versions are mixed",
    ))
}

fn embedding_run_summary_v1(
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    receipts: &[VectorResultReceipt],
    reports: &[BuilderEmbeddingTaskReport],
    token_counts: &[u32],
    artifact_sizes: RunArtifactSizes,
) -> Result<EmbeddingRunSummary> {
    if receipts.len() != plan.tasks.len() || reports.len() != plan.tasks.len() {
        return Err(Error::AccountingClosure(
            "embedding summary task coverage differs",
        ));
    }
    plan.validate_document_token_counts(token_counts)?;
    let sum = |values: Vec<u64>| -> Result<u64> {
        values.into_iter().try_fold(0_u64, |total, value| {
            total.checked_add(value).ok_or(Error::CountOverflow)
        })
    };
    let token_count = sum(plan.tasks.iter().map(|task| task.token_count).collect())?;
    let unique_input_text_bytes = sum(receipts
        .iter()
        .map(|receipt| receipt.executor.input_bytes_upper_bound)
        .collect())?;
    let vector_bytes = sum(receipts
        .iter()
        .map(|receipt| {
            receipt
                .vector
                .rows
                .checked_mul(u64::from(receipt.vector.dimensions))
                .and_then(|values| values.checked_mul(4))
                .ok_or(Error::CountOverflow)
        })
        .collect::<Result<Vec<_>>>()?)?;
    let shard_bytes = sum(receipts
        .iter()
        .map(|receipt| receipt.vector.bytes)
        .collect())?;
    let requests = sum(receipts
        .iter()
        .map(|receipt| receipt.executor.requests)
        .collect())?;
    let retries = sum(receipts
        .iter()
        .map(|receipt| receipt.executor.retries)
        .collect())?;
    let execution_complete = reports.iter().all(|report| report.execution.is_some());
    let execution_reports = reports
        .iter()
        .filter_map(|report| report.execution.as_ref())
        .collect::<Vec<_>>();
    let sent_input_text_bytes = execution_complete
        .then(|| {
            sum(execution_reports
                .iter()
                .map(|report| report.sent_input_text_bytes)
                .collect())
        })
        .transpose()?;
    let task_elapsed_micros_sum = execution_complete
        .then(|| {
            sum(execution_reports
                .iter()
                .map(|report| report.elapsed_micros)
                .collect())
        })
        .transpose()?;
    let request_elapsed_micros = execution_complete
        .then(|| {
            sum(execution_reports
                .iter()
                .map(|report| report.request_elapsed_micros)
                .collect())
        })
        .transpose()?;
    let retry_backoff_micros = execution_complete
        .then(|| {
            sum(execution_reports
                .iter()
                .map(|report| report.retry_backoff_micros)
                .collect())
        })
        .transpose()?;
    let peak_in_flight = execution_complete.then(|| {
        execution_reports
            .iter()
            .map(|report| report.peak_in_flight)
            .max()
            .unwrap_or(0)
    });
    let (calendar_span_micros, wall_time_micros) = if execution_complete {
        execution_time_bounds(reports, task_elapsed_micros_sum)?
    } else {
        (None, None)
    };
    let mut request_latencies = execution_reports
        .iter()
        .flat_map(|report| &report.batch_reports)
        .flat_map(|batch| &batch.attempts)
        .map(|attempt| attempt.elapsed_micros)
        .collect::<Vec<_>>();
    request_latencies.sort_unstable();
    let docs_per_second =
        wall_time_micros.map(|elapsed| plan.document_count as f64 * 1_000_000_f64 / elapsed as f64);
    let tokens_per_second =
        wall_time_micros.map(|elapsed| token_count as f64 * 1_000_000_f64 / elapsed as f64);
    let first_report = homogeneous_report_context(reports)?;
    Ok(EmbeddingRunSummary {
        schema_version: "livefire.rag.embedding-run-summary/1".into(),
        status: "finalized".into(),
        source_snapshot_sha256: prepared.dataset.source_snapshot.sha256.clone(),
        prepared_corpus_sha256: plan.prepared_corpus_sha256.clone(),
        plan_sha256: plan.component_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        tokenizer_sha256: plan.executable_tokenizer.artifact.sha256.clone(),
        git: first_report.git.clone(),
        machine: first_report.machine.clone(),
        lm_studio: first_report.lm_studio.clone(),
        resource_usage: first_report.resource_usage.clone(),
        artifact_sizes,
        tasks: plan.tasks.len(),
        documents: plan.document_count,
        tokens: token_count,
        unique_input_text_bytes,
        sent_input_text_bytes,
        vector_payload_bytes: vector_bytes,
        vector_shard_bytes: shard_bytes,
        transport_bytes: TransportByteAccounting {
            status: if execution_complete {
                ObservationStatus::Partial
            } else {
                ObservationStatus::NotMeasured
            },
            request_body_bytes: None,
            response_body_bytes: None,
            submitted_text_bytes: sent_input_text_bytes,
            decoded_vector_bytes: execution_complete.then_some(vector_bytes),
        },
        requests,
        retries,
        execution_reports_complete: execution_complete,
        calendar_span_micros,
        wall_time_micros,
        task_elapsed_micros_sum,
        request_elapsed_micros,
        retry_backoff_micros,
        peak_in_flight,
        documents_per_second: docs_per_second,
        tokens_per_second,
        request_latency_micros: RequestLatencySummary {
            p50: percentile(&request_latencies, 50),
            p95: percentile(&request_latencies, 95),
            samples: request_latencies.len(),
        },
        length_bucket_throughput: length_bucket_throughput(token_counts, wall_time_micros)?,
    })
}

fn embedding_run_summary_v2(
    plan: &EmbeddingPlanV2,
    prepared: &PreparedCorpusManifest,
    receipts: &[VectorResultReceipt],
    reports: &[BuilderEmbeddingTaskReportV2],
    token_counts: &[u32],
    artifact_sizes: RunArtifactSizes,
) -> Result<EmbeddingRunSummaryV2> {
    let execution_identity = homogeneous_execution_identity_v2(reports)?;

    // Reuse the established aggregate calculations with deliberately uniform
    // placeholder provenance. Only timings and counters are consumed below;
    // the v2 summary publishes every task's real worker provenance separately.
    let metric_reports = reports
        .iter()
        .map(metric_report_from_v2)
        .collect::<Vec<_>>();
    let legacy = embedding_run_summary_v1(
        plan,
        prepared,
        receipts,
        &metric_reports,
        token_counts,
        artifact_sizes,
    )?;
    let workers = reports
        .iter()
        .map(|report| EmbeddingWorkerProvenanceV2 {
            task_id: report.task_id.clone(),
            git: report.git.clone(),
            machine: report.machine.clone(),
            accelerator: report.accelerator.clone(),
            backend: report.backend.clone(),
            resource_usage: report.resource_usage.clone(),
        })
        .collect();
    Ok(EmbeddingRunSummaryV2 {
        schema_version: "livefire.rag.embedding-run-summary/2".into(),
        status: "finalized".into(),
        source_snapshot_sha256: legacy.source_snapshot_sha256,
        prepared_corpus_sha256: legacy.prepared_corpus_sha256,
        plan_sha256: legacy.plan_sha256,
        embedding_profile_sha256: legacy.embedding_profile_sha256,
        tokenizer_sha256: legacy.tokenizer_sha256,
        execution_identity: execution_identity.clone(),
        workers,
        aggregate: EmbeddingRunAggregateV2 {
            artifact_sizes: legacy.artifact_sizes,
            tasks: legacy.tasks,
            documents: legacy.documents,
            tokens: legacy.tokens,
            unique_input_text_bytes: legacy.unique_input_text_bytes,
            sent_input_text_bytes: legacy.sent_input_text_bytes,
            vector_payload_bytes: legacy.vector_payload_bytes,
            vector_shard_bytes: legacy.vector_shard_bytes,
            transport_bytes: legacy.transport_bytes,
            requests: legacy.requests,
            retries: legacy.retries,
            execution_reports_complete: legacy.execution_reports_complete,
            calendar_span_micros: legacy.calendar_span_micros,
            active_time_micros: legacy.wall_time_micros,
            task_elapsed_micros_sum: legacy.task_elapsed_micros_sum,
            request_elapsed_micros: legacy.request_elapsed_micros,
            retry_backoff_micros: legacy.retry_backoff_micros,
            peak_in_flight_per_worker: legacy.peak_in_flight,
            documents_per_active_second: legacy.documents_per_second,
            tokens_per_active_second: legacy.tokens_per_second,
            request_latency_micros: legacy.request_latency_micros,
            length_bucket_throughput: legacy.length_bucket_throughput,
        },
    })
}

fn homogeneous_execution_identity_v2(
    reports: &[BuilderEmbeddingTaskReportV2],
) -> Result<&EmbeddingExecutionIdentityV2> {
    let first = reports.first().ok_or(Error::AccountingClosure(
        "embedding summary has no task report",
    ))?;
    if reports.iter().any(|report| {
        report.execution_identity != first.execution_identity
            || report.accelerator.status != ObservationStatus::Observed
            || report.accelerator.provider.as_deref()
                != Some(first.execution_identity.accelerator.provider.as_str())
            || report.accelerator.model.as_deref()
                != Some(first.execution_identity.accelerator.model.as_str())
            || report.accelerator.architecture.as_deref()
                != Some(first.execution_identity.accelerator.architecture.as_str())
            || report.accelerator.compute_capability.as_deref()
                != Some(
                    first
                        .execution_identity
                        .accelerator
                        .compute_capability
                        .as_str(),
                )
            || report.accelerator.count != Some(first.execution_identity.accelerator.count)
    }) {
        return Err(Error::AccountingClosure(
            "embedding task reports have different sealed execution identity",
        ));
    }
    Ok(&first.execution_identity)
}

fn metric_report_from_v2(report: &BuilderEmbeddingTaskReportV2) -> BuilderEmbeddingTaskReport {
    BuilderEmbeddingTaskReport {
        schema_version: "livefire.rag.embedding-task-run-report/1".into(),
        plan_sha256: report.plan_sha256.clone(),
        source_snapshot_sha256: report.source_snapshot_sha256.clone(),
        prepared_corpus_sha256: report.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: report.embedding_profile_sha256.clone(),
        tokenizer_sha256: report.tokenizer_sha256.clone(),
        task_id: report.task_id.clone(),
        task_index: report.task_index,
        ordinal_start: report.ordinal_start,
        ordinal_end: report.ordinal_end,
        document_count: report.document_count,
        token_count: report.token_count,
        receipt_sha256: report.receipt_sha256.clone(),
        outcome: report.outcome,
        started_unix_ms: report.started_unix_ms,
        finished_unix_ms: report.finished_unix_ms,
        git: GitState {
            status: ObservationStatus::NotMeasured,
            commit: None,
            working_tree_dirty: None,
        },
        machine: MachineContext {
            status: ObservationStatus::NotMeasured,
            operating_system: None,
            operating_system_version: None,
            architecture: None,
            cpu_model: None,
            logical_cpu_count: None,
            ram_bytes: None,
        },
        lm_studio: LmStudioContext {
            status: ObservationStatus::NotMeasured,
            version: None,
            configured_model: report.execution_identity.returned_model.clone(),
            returned_model: report.execution_identity.returned_model.clone(),
            endpoint_kind: "portable_backend_aggregate_only".into(),
            batch_size: None,
            requests_in_flight: None,
            cold_load_micros: None,
        },
        transport_bytes: report.transport_bytes.clone(),
        resource_usage: ResourceUsage {
            status: ObservationStatus::NotMeasured,
            rust_peak_rss_bytes: None,
            lm_studio_peak_rss_bytes: None,
        },
        artifact_sizes: report.artifact_sizes.clone(),
        execution: report.execution.clone(),
    }
}

fn embedding_run_artifact_sizes(
    prepared: &Path,
    plan_path: &Path,
    profile: &Path,
    embeddings: &Path,
    plan: &EmbeddingPlanV2,
) -> Result<RunArtifactSizes> {
    let vector_shards = plan
        .tasks
        .iter()
        .map(|task| resolve_existing_artifact(embeddings, &task.result_path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let receipts = plan
        .tasks
        .iter()
        .map(|task| resolve_existing_artifact(embeddings, &task.receipt_path))
        .collect::<rag_pipeline::Result<Vec<_>>>()?;
    let task_reports = plan
        .tasks
        .iter()
        .map(|task| task_report_path(embeddings, task))
        .collect::<Result<Vec<_>>>()?;
    let prepared_root = artifact_root(prepared)?;
    let plan_root = artifact_root(plan_path)?;
    Ok(run_artifact_sizes(
        &prepared_root,
        &plan_root,
        profile,
        &vector_shards,
        &receipts,
        &task_reports,
    ))
}

fn artifact_root(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        Ok(path.to_owned())
    } else {
        path.parent()
            .map(Path::to_owned)
            .ok_or(Error::AccountingClosure("artifact root is absent"))
    }
}

fn homogeneous_report_context(
    reports: &[BuilderEmbeddingTaskReport],
) -> Result<&BuilderEmbeddingTaskReport> {
    let first_report = reports.first().ok_or(Error::AccountingClosure(
        "embedding summary has no task report",
    ))?;
    if reports.iter().skip(1).any(|report| {
        report.git != first_report.git
            || report.machine != first_report.machine
            || report.lm_studio != first_report.lm_studio
            || report.resource_usage != first_report.resource_usage
    }) {
        return Err(Error::AccountingClosure(
            "embedding task reports have different execution provenance",
        ));
    }
    Ok(first_report)
}

fn execution_time_bounds(
    reports: &[BuilderEmbeddingTaskReport],
    task_elapsed_micros_sum: Option<u64>,
) -> Result<(Option<u64>, Option<u64>)> {
    let mut intervals = reports
        .iter()
        .map(|report| {
            report
                .started_unix_ms
                .zip(report.finished_unix_ms)
                .ok_or(Error::AccountingClosure(
                    "embedding task report execution interval is absent",
                ))
        })
        .collect::<Result<Vec<_>>>()?;
    intervals.sort_unstable();
    let Some(&(first_start, first_end)) = intervals.first() else {
        return Ok((None, None));
    };
    let last_end = intervals
        .iter()
        .map(|(_, end)| *end)
        .max()
        .ok_or(Error::AccountingClosure(
            "embedding task report execution interval is absent",
        ))?;
    let calendar_span_micros = last_end
        .checked_sub(first_start)
        .and_then(|elapsed| elapsed.checked_mul(1_000))
        .filter(|elapsed| *elapsed > 0);

    let mut active_millis = 0_u64;
    let mut active_start = first_start;
    let mut active_end = first_end;
    for &(start, end) in intervals.iter().skip(1) {
        if start <= active_end {
            active_end = active_end.max(end);
        } else {
            active_millis = active_millis
                .checked_add(
                    active_end
                        .checked_sub(active_start)
                        .ok_or(Error::CountOverflow)?,
                )
                .ok_or(Error::CountOverflow)?;
            active_start = start;
            active_end = end;
        }
    }
    active_millis = active_millis
        .checked_add(
            active_end
                .checked_sub(active_start)
                .ok_or(Error::CountOverflow)?,
        )
        .ok_or(Error::CountOverflow)?;
    let active_time_micros = active_millis
        .checked_mul(1_000)
        .filter(|elapsed| *elapsed > 0)
        .or_else(|| {
            reports
                .iter()
                .filter_map(|report| report.execution.as_ref())
                .map(|execution| execution.elapsed_micros)
                .max()
                .filter(|elapsed| *elapsed > 0)
        })
        .or(task_elapsed_micros_sum.filter(|elapsed| *elapsed > 0));
    Ok((
        calendar_span_micros.or(active_time_micros),
        active_time_micros,
    ))
}

fn length_bucket_throughput(
    token_counts: &[u32],
    wall_time_micros: Option<u64>,
) -> Result<Vec<LengthBucketThroughput>> {
    const BOUNDS: &[(u32, Option<u32>)] = &[
        (1, Some(128)),
        (129, Some(256)),
        (257, Some(512)),
        (513, Some(1_024)),
        (1_025, None),
    ];
    BOUNDS
        .iter()
        .map(|(minimum, maximum)| {
            let mut selected = token_counts.iter().copied().filter(|count| {
                count >= minimum && maximum.is_none_or(|maximum| *count <= maximum)
            });
            let (documents, tokens) = selected.try_fold(
                (0_u64, 0_u64),
                |(documents, tokens), count| -> Result<(u64, u64)> {
                    Ok((
                        documents.checked_add(1).ok_or(Error::CountOverflow)?,
                        tokens
                            .checked_add(u64::from(count))
                            .ok_or(Error::CountOverflow)?,
                    ))
                },
            )?;
            let seconds = wall_time_micros
                .filter(|elapsed| *elapsed > 0)
                .map(|elapsed| elapsed as f64 / 1_000_000_f64);
            Ok(LengthBucketThroughput {
                basis: "exact_model_input_tokens".into(),
                minimum_tokens: *minimum,
                maximum_tokens: *maximum,
                documents,
                tokens,
                shared_wall_time_micros: wall_time_micros,
                documents_per_second: seconds.map(|seconds| documents as f64 / seconds),
                tokens_per_second: seconds.map(|seconds| tokens as f64 / seconds),
            })
        })
        .collect()
}

fn percentile(sorted: &[u64], percentile: usize) -> Option<u64> {
    if sorted.is_empty() || !(1..=100).contains(&percentile) {
        return None;
    }
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted.get(rank.saturating_sub(1)).copied()
}

fn validate_embedding_artifact_coverage_v2(
    root: &Path,
    plan: &EmbeddingPlanV2,
    require_manifest: bool,
) -> Result<()> {
    let mut expected = BTreeSet::new();
    if require_manifest {
        expected.insert(MANIFEST_FILE.to_owned());
        expected.insert("summary.json".to_owned());
        expected.insert(EMBEDDING_PROFILE_FILE.to_owned());
    }
    for task in &plan.tasks {
        expected.insert(task.result_path.as_str().to_owned());
        expected.insert(task.receipt_path.as_str().to_owned());
        expected.insert(format!("reports/{}.json", task.task_id));
    }
    let canonical_root = fs::canonicalize(root)?;
    let mut pending = vec![canonical_root.clone()];
    let mut actual = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(Error::AccountingClosure(
                    "embedding artifact tree contains a symlink",
                ));
            }
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(&canonical_root)
                    .map_err(|_| Error::AccountingClosure("embedding artifact path escaped"))?
                    .to_str()
                    .ok_or(Error::AccountingClosure(
                        "embedding artifact path is not UTF-8",
                    ))?
                    .to_owned();
                if let Some(process_id) = internal_embedding_stage_process_id(&relative, plan) {
                    if !process_is_active(process_id) {
                        fs::remove_file(&path)?;
                        sync_parent_directory(&path)?;
                    }
                    continue;
                }
                actual.insert(relative);
            } else {
                return Err(Error::AccountingClosure(
                    "embedding artifact tree contains an unsupported entry",
                ));
            }
        }
    }
    if actual != expected {
        return Err(Error::AccountingClosure(
            "embedding artifact coverage differs from plan",
        ));
    }
    Ok(())
}

/// Verify that a cloud-fetched, not-yet-finalized embedding directory contains
/// exactly the vector, receipt, and task-report files declared by the plan.
pub(crate) fn validate_unfinalized_embedding_artifacts(
    root: &Path,
    plan: &EmbeddingPlanV2,
) -> Result<()> {
    validate_embedding_artifact_coverage_v2(root, plan, false)
}

fn internal_embedding_stage_process_id(relative: &str, plan: &EmbeddingPlanV2) -> Option<u32> {
    let relative_path = Path::new(relative);
    let parent = relative_path.parent()?;
    let file_name = relative_path.file_name()?.to_str()?;
    let atomic_parent_allowed = parent.as_os_str().is_empty()
        || plan.tasks.iter().any(|task| {
            Path::new(task.receipt_path.as_str()).parent() == Some(parent)
                || Path::new(&format!("reports/{}.json", task.task_id)).parent() == Some(parent)
        });
    if atomic_parent_allowed
        && let Some(body) = file_name
            .strip_prefix(".livefire-rag-atomic-")
            .and_then(|body| body.strip_suffix(".partial"))
    {
        let mut fields = body.splitn(3, '-');
        let process_id = fields.next()?.parse().ok()?;
        fields.next()?.parse::<u64>().ok()?;
        if fields.next().is_some_and(|random| !random.is_empty()) {
            return Some(process_id);
        }
    }
    for task in &plan.tasks {
        let result_path = Path::new(task.result_path.as_str());
        if result_path.parent() != Some(parent) {
            continue;
        }
        let destination = result_path.file_name()?.to_str()?;
        let body = file_name
            .strip_prefix(&format!(".{destination}."))
            .and_then(|body| body.strip_suffix(".partial"));
        let Some(body) = body else {
            continue;
        };
        let mut fields = body.split('.');
        let process_id = fields.next()?.parse().ok()?;
        fields.next()?.parse::<u64>().ok()?;
        if fields.next().is_none() {
            return Some(process_id);
        }
    }
    None
}

fn process_is_active(process_id: u32) -> bool {
    if process_id == std::process::id() {
        return true;
    }
    #[cfg(unix)]
    {
        match std::process::Command::new("kill")
            .args(["-0", &process_id.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) => status.success(),
            Err(_) => true,
        }
    }
    #[cfg(not(unix))]
    {
        // Retain the file when this platform cannot prove its owner exited.
        true
    }
}

fn profile_ref(
    bytes: &[u8],
    compact: &rag_embedding::EmbeddingProfile,
) -> Result<EmbeddingProfileRef> {
    let value: Value = serde_json::from_slice(bytes)?;
    let model = value
        .get("model_artifact_set")
        .ok_or(Error::AccountingClosure(
            "embedding model component is absent",
        ))?;
    let tokenizer = value.get("tokenizer").ok_or(Error::AccountingClosure(
        "embedding tokenizer component is absent",
    ))?;
    Ok(EmbeddingProfileRef {
        component: component(&compact.id, &compact.version, &compact.sha256)?,
        model_artifact: component_from_value(model)?,
        tokenizer: component_from_value(tokenizer)?,
        maximum_input_tokens: value
            .get("maximum_tokens")
            .and_then(Value::as_u64)
            .and_then(|number| u32::try_from(number).ok())
            .ok_or(Error::AccountingClosure(
                "embedding maximum tokens are absent",
            ))?,
        pooling: value
            .get("pooling")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        normalization: compact.normalization.clone(),
        dimensions: compact.dimensions,
        dtype: "f32le".into(),
        document_format: value
            .get("document_prefix")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
            + "{semantic_text}",
    })
}

fn tei_profile_ref(
    policy: &TeiCheckpointProfileV3,
    compact: &rag_embedding::EmbeddingProfile,
) -> Result<EmbeddingProfileRef> {
    Ok(EmbeddingProfileRef {
        component: component(&compact.id, &compact.version, &compact.sha256)?,
        model_artifact: component(
            &policy.model_artifact_set.id,
            &policy.model_artifact_set.version,
            &policy.model_artifact_set.sha256,
        )?,
        tokenizer: component(
            &policy.tokenizer.id,
            &policy.tokenizer.version,
            &policy.tokenizer.sha256,
        )?,
        maximum_input_tokens: policy.maximum_tokens,
        pooling: policy.pooling.clone(),
        normalization: policy.normalization.clone(),
        dimensions: policy.dimensions,
        dtype: policy.stored_vector_dtype.clone(),
        document_format: policy.document_format.clone(),
    })
}

fn validate_tei_tokenizer_inputs(
    policy: &TeiCheckpointProfileV3,
    tokenizer: &ExecutableTokenizerRef,
    tokenizer_bytes: &[u8],
) -> Result<()> {
    tokenizer.validate()?;
    let target = component(
        &policy.tokenizer.id,
        &policy.tokenizer.version,
        &policy.tokenizer.sha256,
    )?;
    if tokenizer.artifact.version != policy.executable_tokenizer.revision
        || tokenizer.artifact.id != format!("{}-json", policy.tokenizer.id)
        || tokenizer.artifact.sha256.as_str() != policy.executable_tokenizer.object.sha256
        || tokenizer.format != TokenizerArtifactFormat::HuggingFaceTokenizerJson
        || tokenizer.model_revision != policy.model_revision
        || tokenizer.target_tokenizer != target
        || tokenizer.add_special_tokens != policy.executable_tokenizer.add_special_tokens
        || tokenizer.maximum_input_bytes != 16_384
        || tokenizer_bytes.len() as u64 != policy.executable_tokenizer.object.bytes
        || sha256_bytes(tokenizer_bytes) != policy.executable_tokenizer.object.sha256
    {
        return Err(Error::AccountingClosure(
            "TEI tokenizer files differ from embedding-policy/3",
        ));
    }
    Ok(())
}

fn parse_bound_portable_profile(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<rag_embedding::EmbeddingProfile> {
    let value: Value = serde_json::from_slice(bytes)?;
    if value.get("schema_version").and_then(Value::as_str)
        == Some(rag_embedding::TEI_CHECKPOINT_PROFILE_SCHEMA_V3)
    {
        if sha256_bytes(bytes) != expected_sha256 {
            return Err(Error::AccountingClosure(
                "embedding profile byte digest differs",
            ));
        }
        let policy = parse_tei_checkpoint_profile_v3(bytes)?;
        return Ok(policy.embedding_profile(bytes)?);
    }
    Ok(parse_bound_embedding_profile(bytes, expected_sha256)?)
}

fn validate_plan_profile_fields(
    planned: &EmbeddingProfileRef,
    bytes: &[u8],
    compact: &rag_embedding::EmbeddingProfile,
) -> Result<()> {
    let value: Value = serde_json::from_slice(bytes)?;
    let expected = if value.get("schema_version").and_then(Value::as_str)
        == Some(rag_embedding::TEI_CHECKPOINT_PROFILE_SCHEMA_V3)
    {
        tei_profile_ref(&parse_tei_checkpoint_profile_v3(bytes)?, compact)?
    } else {
        profile_ref(bytes, compact)?
    };
    if expected != *planned {
        return Err(Error::AccountingClosure(
            "embedding plan profile fields differ from profile bytes",
        ));
    }
    Ok(())
}

async fn validate_lmstudio_conformance(
    embedder: &LmStudioEmbedder,
    profile_bytes: &[u8],
    profile: &rag_embedding::EmbeddingProfile,
) -> Result<()> {
    const FIXTURES: [&[u8]; 2] = [
        include_bytes!("../../../fixtures/embedding-conformance.v1.json"),
        include_bytes!("../../../fixtures/generic-evidence-embedding-conformance.v1.json"),
    ];
    let profile_value: Value = serde_json::from_slice(profile_bytes)?;
    let conformance = profile_value
        .get("conformance")
        .and_then(Value::as_object)
        .ok_or(Error::AccountingClosure(
            "embedding conformance contract is absent",
        ))?;
    let fixture_sha256 = conformance
        .get("fixture_sha256")
        .and_then(Value::as_str)
        .ok_or(Error::AccountingClosure(
            "embedding conformance fixture digest is absent",
        ))?;
    let expected_output_sha256 = conformance
        .get("normalized_output_sha256")
        .and_then(Value::as_str)
        .ok_or(Error::AccountingClosure(
            "embedding conformance output digest is absent",
        ))?;
    let fixture_bytes = FIXTURES
        .into_iter()
        .find(|bytes| sha256_bytes(bytes) == fixture_sha256)
        .ok_or(Error::AccountingClosure(
            "embedding conformance fixture is unavailable",
        ))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)?;
    let request =
        fixture
            .get("request")
            .and_then(Value::as_object)
            .ok_or(Error::AccountingClosure(
                "embedding conformance request is absent",
            ))?;
    if request.get("model").and_then(Value::as_str) != Some(profile.model.as_str()) {
        return Err(Error::AccountingClosure(
            "embedding conformance model differs from profile",
        ));
    }
    let inputs = request
        .get("input")
        .and_then(Value::as_array)
        .ok_or(Error::AccountingClosure(
            "embedding conformance inputs are absent",
        ))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or(Error::AccountingClosure(
                    "embedding conformance input is invalid",
                ))
        })
        .collect::<Result<Vec<_>>>()?;
    let dimensions = usize::try_from(profile.dimensions)
        .map_err(|_| Error::AccountingClosure("embedding dimensions overflow"))?;
    for _ in 0..2 {
        let (batch, normalized_sha256) = embedder.conformance_probe(&inputs).await?;
        if batch.returned_model != profile.model || batch.vectors.len() != inputs.len() {
            return Err(Error::AccountingClosure(
                "embedding conformance response identity differs",
            ));
        }
        for vector in &batch.vectors {
            validate_vector(vector, dimensions, &profile.normalization)?;
        }
        if normalized_sha256 != expected_output_sha256 {
            return Err(Error::AccountingClosure(
                "embedding conformance output digest differs",
            ));
        }
    }
    Ok(())
}

fn component_from_value(value: &Value) -> Result<ComponentRef> {
    component(
        value
            .get("id")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component ID is absent"))?,
        value
            .get("version")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component version is absent"))?,
        value
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or(Error::AccountingClosure("component digest is absent"))?,
    )
}

fn projection_policy_component() -> Result<ComponentRef> {
    let material: Value = serde_json::from_slice(PROJECTION_POLICY_BYTES)?;
    let digest = canonical_digest(&material)?;
    component(
        "livefire.rag.generic-evidence-projection-policy",
        "2",
        digest.as_str(),
    )
}

fn m45_command_projection_policy_component() -> Result<ComponentRef> {
    let material: Value = serde_json::from_slice(M45_COMMAND_PROJECTION_POLICY_BYTES)?;
    validate_m45_command_policy_material(&material)?;
    let digest = canonical_digest(&material)?;
    component(
        "livefire.rag.m45-command-projection-policy",
        "1",
        digest.as_str(),
    )
}

fn m45_command_source_contract() -> Result<M45CommandSourceContract> {
    let material: Value = serde_json::from_slice(M45_COMMAND_PROJECTION_POLICY_BYTES)?;
    validate_m45_command_policy_material(&material)
}

fn validate_m45_command_policy_material(material: &Value) -> Result<M45CommandSourceContract> {
    if material.get("schema_version").and_then(Value::as_str)
        != Some("livefire.rag.m45-command-projection-policy/1")
    {
        return Err(Error::AccountingClosure(
            "M45 command projection policy schema is invalid",
        ));
    }
    let contract: M45CommandSourceContract =
        serde_json::from_value(material.get("source_contract").cloned().ok_or(
            Error::AccountingClosure("M45 command source contract is absent"),
        )?)?;
    let expected_relations = [
        "ocsf_api_activity",
        "ocsf_event_log_activity",
        "ocsf_process_activity",
    ];
    if contract.snapshot_receipt_schema != 2
        || contract.snapshot_manifest_schema != 3
        || contract.snapshot_id.is_empty()
        || contract.snapshot_version.is_empty()
        || contract.mapping_id.is_empty()
        || contract.mapping_version.is_empty()
        || contract.authority.is_empty()
        || contract.event_reference.is_empty()
        || contract
            .admitted_relations
            .iter()
            .map(String::as_str)
            .ne(expected_relations)
    {
        return Err(Error::AccountingClosure(
            "M45 command source contract is invalid",
        ));
    }
    Ok(contract)
}

fn validate_m45_command_source(identity: &OcsfSnapshot) -> Result<()> {
    let capabilities =
        identity
            .snapshot_capabilities_sha256
            .as_ref()
            .ok_or(Error::AccountingClosure(
                "command preparation requires the exact admitted M45 OCSF snapshot",
            ))?;
    let admitted = M45CommandAdmittedIdentity {
        snapshot_manifest_schema: identity.schema_version,
        snapshot_id: identity.snapshot_id.clone(),
        snapshot_version: identity.snapshot_version.clone(),
        snapshot_sha256: Digest::new(identity.snapshot_sha256.as_str())?,
        mapping_id: identity.mapping_id.clone(),
        mapping_version: identity.mapping_version.clone(),
        mapping_sha256: Digest::new(identity.mapping_sha256.as_str())?,
        relation_contract_sha256: Digest::new(identity.relation_contract_sha256.as_str())?,
        snapshot_capabilities_sha256: Digest::new(capabilities.as_str())?,
    };
    validate_m45_command_admitted_identity(&m45_command_source_contract()?, &admitted)
}

fn validate_m45_command_admitted_identity(
    contract: &M45CommandSourceContract,
    admitted: &M45CommandAdmittedIdentity,
) -> Result<()> {
    if admitted.snapshot_manifest_schema != contract.snapshot_manifest_schema
        || admitted.snapshot_id != contract.snapshot_id
        || admitted.snapshot_version != contract.snapshot_version
        || admitted.snapshot_sha256 != contract.snapshot_sha256
        || admitted.mapping_id != contract.mapping_id
        || admitted.mapping_version != contract.mapping_version
        || admitted.mapping_sha256 != contract.mapping_sha256
        || admitted.relation_contract_sha256 != contract.relation_contract_sha256
        || admitted.snapshot_capabilities_sha256 != contract.snapshot_capabilities_sha256
    {
        return Err(Error::AccountingClosure(
            "command preparation requires the exact admitted M45 OCSF snapshot",
        ));
    }
    Ok(())
}

fn component(id: &str, version: &str, sha256: &str) -> Result<ComponentRef> {
    let value = ComponentRef {
        id: id.to_owned(),
        version: version.to_owned(),
        sha256: Digest::new(sha256)?,
    };
    value.validate()?;
    Ok(value)
}

fn object_entry(
    relative: &str,
    path: &Path,
    rows: u64,
    logical_order_sha256: Digest,
) -> Result<ObjectEntry> {
    Ok(ObjectEntry {
        path: SafeRelativePath::new(relative)?,
        rows,
        bytes: fs::metadata(path)?.len(),
        sha256: file_digest(path)?,
        logical_order_sha256,
    })
}

fn file_digest(path: &Path) -> Result<Digest> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Digest::new(format!("{:x}", hasher.finalize())).map_err(Error::from)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn zero_digest() -> Result<Digest> {
    Ok(Digest::new("0".repeat(64))?)
}

fn canonical_string(value: &impl serde::Serialize) -> Result<String> {
    String::from_utf8(canonical_json_bytes(value)?)
        .map_err(|_| Error::AccountingClosure("canonical JSON is not UTF-8"))
}

fn manifest_or_file(root: &Path, name: &str) -> PathBuf {
    if root.is_dir() {
        root.join(name)
    } else {
        root.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use arrow_array::{ArrayRef, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
    use rag_pipeline::{EmbeddingInputSliceV2, read_prepared_occurrences};

    const TEST_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"é":2,"<unk>":4},"unk_token":"<unk>"}}"#;

    fn benchmark_component(id: &str) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: digest_bytes(id.as_bytes()),
        }
    }

    fn exact_m45_command_identity(
        contract: &M45CommandSourceContract,
    ) -> M45CommandAdmittedIdentity {
        M45CommandAdmittedIdentity {
            snapshot_manifest_schema: contract.snapshot_manifest_schema,
            snapshot_id: contract.snapshot_id.clone(),
            snapshot_version: contract.snapshot_version.clone(),
            snapshot_sha256: contract.snapshot_sha256.clone(),
            mapping_id: contract.mapping_id.clone(),
            mapping_version: contract.mapping_version.clone(),
            mapping_sha256: contract.mapping_sha256.clone(),
            relation_contract_sha256: contract.relation_contract_sha256.clone(),
            snapshot_capabilities_sha256: contract.snapshot_capabilities_sha256.clone(),
        }
    }

    fn assert_m45_command_identity_rejected(identity: &M45CommandAdmittedIdentity) {
        let contract = m45_command_source_contract().unwrap();
        assert!(matches!(
            validate_m45_command_admitted_identity(&contract, identity),
            Err(Error::AccountingClosure(
                "command preparation requires the exact admitted M45 OCSF snapshot"
            ))
        ));
    }

    #[test]
    fn m45_command_policy_declares_and_accepts_the_exact_released_identity() {
        let contract = m45_command_source_contract().unwrap();
        assert_eq!(contract.snapshot_receipt_schema, 2);
        assert_eq!(contract.snapshot_manifest_schema, 3);
        assert_eq!(contract.snapshot_id, "botsv3-ocsf-normalized-snapshot");
        assert_eq!(contract.snapshot_version, "45");
        assert_eq!(
            contract.snapshot_sha256.as_str(),
            "23077f2605cb4d0ca7f1a857dd0c540d990911197c21a80c886fc1099f6e7d10"
        );
        assert_eq!(contract.mapping_id, "botsv3-ocsf-m45");
        assert_eq!(contract.mapping_version, "1");
        assert_eq!(
            contract.mapping_sha256.as_str(),
            "641e479d5d830edef80c4e57c8048eed9b26710d35a18101e9441065f4337bb7"
        );
        assert_eq!(
            contract.relation_contract_sha256.as_str(),
            "a40656d2b8e233326157a40c08a257bffe8ef2b97ca76ff62740fbef43eca549"
        );
        assert_eq!(
            contract.snapshot_capabilities_sha256.as_str(),
            "d9e7e485213c09abb9862f8620cebc410649bc8241688ae21c53721958493e1b"
        );
        validate_m45_command_admitted_identity(&contract, &exact_m45_command_identity(&contract))
            .unwrap();
    }

    #[test]
    fn m45_command_gate_rejects_each_wrong_immutable_source_identity() {
        let contract = m45_command_source_contract().unwrap();

        let mut identity = exact_m45_command_identity(&contract);
        identity.snapshot_manifest_schema = 2;
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.snapshot_id = "another.snapshot".into();
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.snapshot_version = "46".into();
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.snapshot_sha256 = digest_bytes(b"wrong snapshot");
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.mapping_id = "another.mapping".into();
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.mapping_version = "2".into();
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.mapping_sha256 = digest_bytes(b"wrong mapping");
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.relation_contract_sha256 = digest_bytes(b"wrong relation contract");
        assert_m45_command_identity_rejected(&identity);

        let mut identity = exact_m45_command_identity(&contract);
        identity.snapshot_capabilities_sha256 = digest_bytes(b"wrong capabilities");
        assert_m45_command_identity_rejected(&identity);
    }

    struct CensusFixture {
        root: tempfile::TempDir,
    }

    impl CensusFixture {
        fn write() -> Self {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir(root.path().join("semantic")).unwrap();

            let core_schema = Arc::new(Schema::new(vec![Field::new(
                "support_ref",
                DataType::Utf8,
                false,
            )]));
            for relation in [
                "event_facets",
                "entities",
                "observables",
                "participants",
                "event_observables",
                "relationships",
            ] {
                let batch = RecordBatch::try_new(
                    Arc::clone(&core_schema),
                    vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
                )
                .unwrap();
                write_census_parquet(
                    &root.path().join(format!("semantic/{relation}.parquet")),
                    Arc::clone(&core_schema),
                    batch,
                    1,
                );
            }

            let events_schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Utf8, false),
                Field::new("event_time_ms", DataType::UInt64, false),
                Field::new("support_ref", DataType::Utf8, false),
            ]));
            let events = RecordBatch::try_new(
                Arc::clone(&events_schema),
                vec![
                    Arc::new(StringArray::from(vec!["e1", "e2", "e3", "e4"])) as ArrayRef,
                    Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["s1", "s2", "s3", "s4"])) as ArrayRef,
                ],
            )
            .unwrap();
            write_census_parquet(
                &root.path().join("semantic/events.parquet"),
                events_schema,
                events,
                1,
            );

            let typed_schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Utf8, false),
                Field::new("typed_event_json", DataType::Utf8, false),
                Field::new("support_ref", DataType::Utf8, false),
            ]));
            let direct = serde_json::to_string(&json!({
                "activity_name": "Launch",
                "class_uid": 1007,
                "category_uid": 1,
                "time": 1710000000000_u64,
                "process": {"name": "powershell.exe", "pid": 123},
                "status": "Success"
            }))
            .unwrap();
            let camel = serde_json::to_string(&json!({
                "semantic_class": "process",
                "ocsf": {
                    "activity_id": 99,
                    "category_uid": 1,
                    "class_uid": 1007,
                    "severity_id": 1,
                    "time": 1534762063000_u64,
                    "type_uid": 100799,
                    "unmapped": {
                        "action": "added",
                        "calendarTime": "Mon Aug 20 10:47:43 2018 UTC",
                        "hostIdentifier": "host.example",
                        "unixTime": 1534762063,
                        "columns": {"path": "/bin/gawk", "status": ""}
                    }
                }
            }))
            .unwrap();
            let typed = RecordBatch::try_new(
                Arc::clone(&typed_schema),
                vec![
                    Arc::new(StringArray::from(vec!["e1", "e2", "e3", "e4"])) as ArrayRef,
                    Arc::new(StringArray::from(vec![
                        direct.as_str(),
                        camel.as_str(),
                        "{",
                        direct.as_str(),
                    ])) as ArrayRef,
                    Arc::new(StringArray::from(vec!["s1", "s2", "s3", "s4"])) as ArrayRef,
                ],
            )
            .unwrap();
            write_census_parquet(
                &root.path().join("semantic/ocsf_process_activity.parquet"),
                typed_schema,
                typed,
                1,
            );

            let relation_rows = BTreeMap::from([
                ("events", 4_u64),
                ("event_facets", 0),
                ("entities", 0),
                ("observables", 0),
                ("participants", 0),
                ("event_observables", 0),
                ("relationships", 0),
                ("ocsf_process_activity", 4),
            ]);
            let objects = relation_rows
                .iter()
                .map(|(relation, rows)| {
                    let relative = format!("semantic/{relation}.parquet");
                    json!({
                        "relation": relation,
                        "path": relative,
                        "rows": rows,
                        "sha256": sha256_bytes(&fs::read(root.path().join(&relative)).unwrap()),
                        "logical_sha256": "8".repeat(64)
                    })
                })
                .collect::<Vec<_>>();
            let receipt = json!({
                "schema_version": 1,
                "snapshot_manifest": {
                    "schema_version": 1,
                    "dataset_sha256": "d".repeat(64),
                    "source_inventory_sha256": "1".repeat(64),
                    "field_inventory_sha256": "2".repeat(64),
                    "ocsf_schema_sha256": "3".repeat(64),
                    "extension_pack_sha256": "4".repeat(64),
                    "mapping_pack_sha256": "b".repeat(64),
                    "relation_contract_sha256": "5".repeat(64),
                    "normalizer_sha256": "6".repeat(64),
                    "objects": objects,
                    "logical_sha256": "a".repeat(64)
                },
                "output_logical_sha256": "a".repeat(64),
                "runnable_snapshot": {
                    "component": {"id":"fixture.snapshot","version":"1","sha256":"a".repeat(64)},
                    "dataset_sha256":"d".repeat(64),
                    "mapping_pack":{"id":"fixture.mapping","version":"1","sha256":"b".repeat(64)},
                    "relation_contract":{"id":"fixture.relations","version":"1","sha256":"5".repeat(64)},
                    "normalized_events":4,
                    "source_rows":4
                },
                "closure": {
                    "input_rows":4,"mapped_source_records":4,"mapped_events":4,"event_rows":4,
                    "rejected_malformed_records":0,"unsupported_records":0,
                    "unresolved_provenance_fields":0,"provenance_digest_mismatches":0
                },
                "completeness_receipt": {
                    "dataset_sha256":"d".repeat(64),
                    "mapping_pack_sha256":"b".repeat(64),
                    "normalized_snapshot_sha256":"a".repeat(64),
                    "relation_contract_sha256":"5".repeat(64),
                    "metrics":{
                        "source_rows":4,"mapped_source_records":4,
                        "rejected_malformed_records":0,"normalized_events":4
                    }
                }
            });
            fs::write(
                root.path().join("build-receipt.json"),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
            Self { root }
        }
    }

    struct BenchmarkPreparationFixture {
        root: tempfile::TempDir,
    }

    impl BenchmarkPreparationFixture {
        fn write(document_count: usize, row_group_rows: usize) -> Self {
            let root = tempfile::tempdir().unwrap();
            fs::create_dir(root.path().join("semantic")).unwrap();

            let core_schema = Arc::new(Schema::new(vec![Field::new(
                "support_ref",
                DataType::Utf8,
                false,
            )]));
            for relation in [
                "event_facets",
                "entities",
                "observables",
                "participants",
                "event_observables",
                "relationships",
            ] {
                let batch = RecordBatch::try_new(
                    Arc::clone(&core_schema),
                    vec![Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef],
                )
                .unwrap();
                write_census_parquet(
                    &root.path().join(format!("semantic/{relation}.parquet")),
                    Arc::clone(&core_schema),
                    batch,
                    1,
                );
            }

            let event_ids = (0..document_count)
                .map(|ordinal| format!("event-{ordinal:05}"))
                .collect::<Vec<_>>();
            let support_refs = (0..document_count)
                .map(|ordinal| format!("support-{ordinal:05}"))
                .collect::<Vec<_>>();
            let event_times = (0..document_count)
                .map(|ordinal| 1_710_000_000_000_u64 + ordinal as u64)
                .collect::<Vec<_>>();
            let typed_json = (0..document_count)
                .map(|ordinal| {
                    let details = if ordinal % 257 == 0 {
                        "PowerShell logging bypass ".repeat(192)
                    } else if ordinal % 17 == 0 {
                        "encoded command ".repeat(12)
                    } else {
                        "normal launch".to_owned()
                    };
                    serde_json::to_string(&json!({
                        "activity_name": "Launch",
                        "class_uid": 1007,
                        "category_uid": 1,
                        "time": 1_710_000_000_000_u64 + ordinal as u64,
                        "process": {
                            "name": format!("benchmark-tool-{ordinal:05}"),
                            "pid": ordinal + 1,
                            "command_line": details
                        },
                        "status": "Success"
                    }))
                    .unwrap()
                })
                .collect::<Vec<_>>();

            let events_schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Utf8, false),
                Field::new("event_time_ms", DataType::UInt64, false),
                Field::new("support_ref", DataType::Utf8, false),
            ]));
            let events = RecordBatch::try_new(
                Arc::clone(&events_schema),
                vec![
                    Arc::new(StringArray::from(event_ids.clone())) as ArrayRef,
                    Arc::new(UInt64Array::from(event_times)) as ArrayRef,
                    Arc::new(StringArray::from(support_refs.clone())) as ArrayRef,
                ],
            )
            .unwrap();
            write_census_parquet(
                &root.path().join("semantic/events.parquet"),
                events_schema,
                events,
                row_group_rows,
            );

            let typed_schema = Arc::new(Schema::new(vec![
                Field::new("event_id", DataType::Utf8, false),
                Field::new("typed_event_json", DataType::Utf8, false),
                Field::new("support_ref", DataType::Utf8, false),
            ]));
            let typed = RecordBatch::try_new(
                Arc::clone(&typed_schema),
                vec![
                    Arc::new(StringArray::from(event_ids)) as ArrayRef,
                    Arc::new(StringArray::from(typed_json)) as ArrayRef,
                    Arc::new(StringArray::from(support_refs)) as ArrayRef,
                ],
            )
            .unwrap();
            write_census_parquet(
                &root.path().join("semantic/ocsf_process_activity.parquet"),
                typed_schema,
                typed,
                row_group_rows,
            );

            let rows = u64::try_from(document_count).unwrap();
            let relation_rows = BTreeMap::from([
                ("events", rows),
                ("event_facets", 0),
                ("entities", 0),
                ("observables", 0),
                ("participants", 0),
                ("event_observables", 0),
                ("relationships", 0),
                ("ocsf_process_activity", rows),
            ]);
            let objects = relation_rows
                .iter()
                .map(|(relation, rows)| {
                    let relative = format!("semantic/{relation}.parquet");
                    json!({
                        "relation": relation,
                        "path": relative,
                        "rows": rows,
                        "sha256": sha256_bytes(&fs::read(root.path().join(&relative)).unwrap()),
                        "logical_sha256": "8".repeat(64)
                    })
                })
                .collect::<Vec<_>>();
            let receipt = json!({
                "schema_version": 1,
                "snapshot_manifest": {
                    "schema_version": 1,
                    "dataset_sha256": "d".repeat(64),
                    "source_inventory_sha256": "1".repeat(64),
                    "field_inventory_sha256": "2".repeat(64),
                    "ocsf_schema_sha256": "3".repeat(64),
                    "extension_pack_sha256": "4".repeat(64),
                    "mapping_pack_sha256": "b".repeat(64),
                    "relation_contract_sha256": "5".repeat(64),
                    "normalizer_sha256": "6".repeat(64),
                    "objects": objects,
                    "logical_sha256": "a".repeat(64)
                },
                "output_logical_sha256": "a".repeat(64),
                "runnable_snapshot": {
                    "component": {"id":"fixture.snapshot","version":"1","sha256":"a".repeat(64)},
                    "dataset_sha256":"d".repeat(64),
                    "mapping_pack":{"id":"fixture.mapping","version":"1","sha256":"b".repeat(64)},
                    "relation_contract":{"id":"fixture.relations","version":"1","sha256":"5".repeat(64)},
                    "normalized_events":rows,
                    "source_rows":rows
                },
                "closure": {
                    "input_rows":rows,"mapped_source_records":rows,"mapped_events":rows,
                    "event_rows":rows,"rejected_malformed_records":0,"unsupported_records":0,
                    "unresolved_provenance_fields":0,"provenance_digest_mismatches":0
                },
                "completeness_receipt": {
                    "dataset_sha256":"d".repeat(64),
                    "mapping_pack_sha256":"b".repeat(64),
                    "normalized_snapshot_sha256":"a".repeat(64),
                    "relation_contract_sha256":"5".repeat(64),
                    "metrics":{
                        "source_rows":rows,"mapped_source_records":rows,
                        "rejected_malformed_records":0,"normalized_events":rows
                    }
                }
            });
            fs::write(
                root.path().join("build-receipt.json"),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
            Self { root }
        }
    }

    fn write_census_parquet(
        path: &Path,
        schema: Arc<Schema>,
        batch: RecordBatch,
        row_group_rows: usize,
    ) {
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            schema,
            Some(
                WriterProperties::builder()
                    .set_max_row_group_row_count(Some(row_group_rows))
                    .build(),
            ),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn parallel_census_uses_independent_row_group_readers_and_matches_serial_report_bytes() {
        let fixture = CensusFixture::write();
        let options = |workers| CensusOptions {
            snapshot: fixture.root.path().to_path_buf(),
            relations: vec!["ocsf_process_activity".into()],
            out: None,
            workers,
        };
        let reader = LocalSnapshotReader::open(fixture.root.path()).unwrap();
        let relation = reader.typed_relations().next().unwrap();
        let admitted = reader.admit_object(relation).unwrap();
        assert_eq!(admitted.row_groups().len(), 4);

        let serial = build_corpus_census_report(&options(1)).unwrap();
        let parallel = build_corpus_census_report(&options(3)).unwrap();
        assert_eq!(serial.component_sha256, parallel.component_sha256);
        assert_eq!(
            canonical_json_bytes(&serial).unwrap(),
            canonical_json_bytes(&parallel).unwrap()
        );
        assert_eq!(parallel.source_rows, 4);
        assert_eq!(parallel.semantic_occurrences, 3);
        assert_eq!(parallel.structured_only_occurrences, 1);
        assert_eq!(parallel.distinct_documents, 2);
    }

    #[test]
    fn ordinary_preparation_is_byte_identical_across_worker_counts() {
        let fixture = CensusFixture::write();
        let reader = LocalSnapshotReader::open(fixture.root.path()).unwrap();
        let relation = reader.typed_relations().next().unwrap();
        let admitted = reader.admit_object(relation).unwrap();
        assert_eq!(admitted.row_groups().len(), 4);

        let outputs = tempfile::tempdir().unwrap();
        let serial = outputs.path().join("serial");
        let parallel = outputs.path().join("parallel");
        let options = |out, workers| PrepareOptions {
            snapshot: fixture.root.path().to_path_buf(),
            dataset_id: "ordinary-worker-parity".into(),
            dataset_version: "1".into(),
            relations: vec!["ocsf_process_activity".into()],
            out,
            document_shard_rows: 1,
            workers,
        };
        prepare_with_document_run_rows(
            options(serial.clone(), 1),
            MAX_DOCUMENT_RUN_ROWS,
            PreparationProjection::Generic,
        )
        .unwrap();
        prepare_with_document_run_rows(
            options(parallel.clone(), 4),
            1,
            PreparationProjection::Generic,
        )
        .unwrap();

        let serial_tree = artifact_tree(&serial);
        let parallel_tree = artifact_tree(&parallel);
        assert_eq!(
            serial_tree.keys().collect::<Vec<_>>(),
            parallel_tree.keys().collect::<Vec<_>>()
        );
        assert_eq!(serial_tree, parallel_tree);

        let manifest: PreparedCorpusManifest = read_json(&serial.join("manifest.json")).unwrap();
        assert_eq!(manifest.document_count, 2);
        assert_eq!(manifest.occurrence_count, 3);
    }

    fn run_accumulator(document_id: &str, semantic_text: &str) -> DocumentAccumulator {
        let document_sha256 = sha256_bytes(semantic_text.as_bytes());
        DocumentAccumulator {
            document: FastDocument {
                document_id: document_id.into(),
                document_sha256,
                document_kind: "activity".into(),
                semantic_text: semantic_text.into(),
                facets_json: "{}".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: 0,
            },
            primary_relation: "relation-a".into(),
            relations: BTreeSet::from(["relation-a".into()]),
        }
    }

    #[test]
    fn sorted_document_runs_merge_duplicates_in_stable_order_with_a_hard_buffer_bound() {
        let root = tempfile::tempdir().unwrap();
        let run_root = root.path().join("runs");
        let mut runs = SortedDocumentRuns::new(run_root.clone(), 2).unwrap();
        runs.add("doc-c".into(), run_accumulator("doc-c", "c"))
            .unwrap();
        runs.add("doc-a".into(), run_accumulator("doc-a", "a"))
            .unwrap();
        runs.add("doc-b".into(), run_accumulator("doc-b", "b"))
            .unwrap();
        runs.add("doc-a".into(), run_accumulator("doc-a", "a"))
            .unwrap();
        runs.add("doc-d".into(), run_accumulator("doc-d", "d"))
            .unwrap();

        let mut merged = Vec::new();
        let stats = runs
            .merge(|document| {
                merged.push((
                    document.document.document_id,
                    document.document.occurrence_count,
                ));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            merged,
            vec![
                ("doc-a".into(), 2),
                ("doc-b".into(), 1),
                ("doc-c".into(), 1),
                ("doc-d".into(), 1),
            ]
        );
        assert_eq!(stats.run_count, 3);
        assert_eq!(stats.maximum_buffered_documents, 2);
        assert!(!run_root.exists());
    }

    #[test]
    fn sorted_document_runs_reject_content_collisions_and_clean_up_for_restart() {
        let root = tempfile::tempdir().unwrap();
        let run_root = root.path().join("runs");
        {
            let mut runs = SortedDocumentRuns::new(run_root.clone(), 1).unwrap();
            runs.add("doc-a".into(), run_accumulator("doc-a", "first"))
                .unwrap();
            runs.add("doc-b".into(), run_accumulator("doc-b", "other"))
                .unwrap();
            runs.add("doc-a".into(), run_accumulator("doc-a", "changed"))
                .unwrap();
            let error = runs.merge(|_| Ok(())).unwrap_err();
            assert!(matches!(error, Error::InconsistentDocument(id) if id == "doc-a"));
        }
        assert!(!run_root.exists());

        let mut restarted = SortedDocumentRuns::new(run_root.clone(), 1).unwrap();
        restarted
            .add("doc-a".into(), run_accumulator("doc-a", "first"))
            .unwrap();
        let mut count = 0;
        restarted
            .merge(|_| {
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 1);
        assert!(!run_root.exists());
    }

    #[test]
    fn dropping_unfinished_document_runs_removes_temporary_files() {
        let root = tempfile::tempdir().unwrap();
        let run_root = root.path().join("runs");
        {
            let mut runs = SortedDocumentRuns::new(run_root.clone(), 1).unwrap();
            runs.add("doc-a".into(), run_accumulator("doc-a", "a"))
                .unwrap();
            runs.add("doc-b".into(), run_accumulator("doc-b", "b"))
                .unwrap();
            assert!(run_root.exists());
        }
        assert!(!run_root.exists());
        SortedDocumentRuns::new(run_root.clone(), 1).unwrap();
        assert!(!run_root.exists());
    }

    #[test]
    #[ignore = "generated 750,000-document preparation scale acceptance test"]
    fn sorted_document_runs_accept_more_documents_than_the_removed_memory_ceiling() {
        const DOCUMENTS: usize = 750_000;
        let root = tempfile::tempdir().unwrap();
        let mut runs = SortedDocumentRuns::new(root.path().join("runs"), 10_000).unwrap();
        for ordinal in (0..DOCUMENTS).rev() {
            let document_id = format!("doc-{ordinal:06}");
            runs.add(
                document_id.clone(),
                run_accumulator(&document_id, "generated scale document"),
            )
            .unwrap();
        }
        let mut count = 0_usize;
        let mut previous = None;
        let stats = runs
            .merge(|document| {
                assert!(
                    previous
                        .as_ref()
                        .is_none_or(|value| value < &document.document.document_id)
                );
                previous = Some(document.document.document_id);
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, DOCUMENTS);
        assert_eq!(stats.run_count, 75);
        assert_eq!(stats.maximum_buffered_documents, 10_000);
    }

    #[test]
    fn one_row_group_many_batches_match_with_one_and_eight_workers() {
        const ROWS: usize = 2_049;
        let fixture = BenchmarkPreparationFixture::write(ROWS, ROWS);
        let reader = LocalSnapshotReader::open_with_batch_size(fixture.root.path(), 257).unwrap();
        let relation = reader
            .typed_relations()
            .find(|relation| relation.name == "ocsf_process_activity")
            .unwrap();
        let admitted = reader.admit_object(relation).unwrap();
        assert_eq!(admitted.row_groups().len(), 1);
        assert_eq!(admitted.row_groups()[0].rows, ROWS as u64);
        let batch_count = admitted
            .scan_row_group(0, &["typed_event_json"])
            .unwrap()
            .try_fold(0_usize, |count, batch| {
                batch.unwrap();
                count.checked_add(1)
            })
            .unwrap();
        assert_eq!(batch_count, 8);
        let context = projection_context(reader.identity());
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        let serial_census = census_row_group(&admitted, 0, &context, &serial_pool, 1).unwrap();
        let parallel_census = census_row_group(&admitted, 0, &context, &parallel_pool, 8).unwrap();
        assert_eq!(parallel_census, serial_census);

        let serial_prepared = project_prepared_row_group(
            &admitted,
            0,
            0,
            &relation.name,
            &context,
            PreparedProjectionExecution {
                batch: BatchProjectionExecution {
                    worker_pool: &serial_pool,
                    workers: 1,
                },
                projection: PreparationProjection::Generic,
            },
        )
        .unwrap();
        let parallel_prepared = project_prepared_row_group(
            &admitted,
            0,
            0,
            &relation.name,
            &context,
            PreparedProjectionExecution {
                batch: BatchProjectionExecution {
                    worker_pool: &parallel_pool,
                    workers: 8,
                },
                projection: PreparationProjection::Generic,
            },
        )
        .unwrap();
        assert_eq!(parallel_prepared, serial_prepared);
        assert!(
            parallel_prepared
                .occurrences
                .iter()
                .enumerate()
                .all(|(ordinal, row)| row.source_row_ordinal == ordinal as u64)
        );

        let serial_candidates =
            benchmark_candidate_row_group(&admitted, 0, &relation.name, &context, &serial_pool, 1)
                .unwrap();
        let parallel_candidates = benchmark_candidate_row_group(
            &admitted,
            0,
            &relation.name,
            &context,
            &parallel_pool,
            8,
        )
        .unwrap();
        assert_eq!(parallel_candidates, serial_candidates);
        let selected_ids = serial_candidates
            .candidates
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let selected_ranks = selected_ids
            .iter()
            .enumerate()
            .map(|(ordinal, id)| (id.as_str(), ordinal as u64))
            .collect::<BTreeMap<_, _>>();
        let serial_occurrences = benchmark_occurrence_row_group(
            &admitted,
            0,
            0,
            &relation.name,
            &context,
            &selected_ranks,
            BatchProjectionExecution {
                worker_pool: &serial_pool,
                workers: 1,
            },
        )
        .unwrap();
        let parallel_occurrences = benchmark_occurrence_row_group(
            &admitted,
            0,
            0,
            &relation.name,
            &context,
            &selected_ranks,
            BatchProjectionExecution {
                worker_pool: &parallel_pool,
                workers: 8,
            },
        )
        .unwrap();
        assert_eq!(parallel_occurrences, serial_occurrences);

        let census_options = |workers| CensusOptions {
            snapshot: fixture.root.path().to_path_buf(),
            relations: vec![relation.name.clone()],
            out: None,
            workers,
        };
        let serial_report = build_corpus_census_report(&census_options(1)).unwrap();
        let parallel_report = build_corpus_census_report(&census_options(8)).unwrap();
        assert_eq!(
            canonical_json_bytes(&parallel_report).unwrap(),
            canonical_json_bytes(&serial_report).unwrap()
        );

        let outputs = tempfile::tempdir().unwrap();
        let serial_output = outputs.path().join("serial");
        let parallel_output = outputs.path().join("parallel");
        let prepare_options = |out, workers| PrepareOptions {
            snapshot: fixture.root.path().to_path_buf(),
            dataset_id: "single-group-batch-parity".into(),
            dataset_version: "1".into(),
            relations: vec![relation.name.clone()],
            out,
            document_shard_rows: 4_096,
            workers,
        };
        prepare(prepare_options(serial_output.clone(), 1)).unwrap();
        prepare(prepare_options(parallel_output.clone(), 8)).unwrap();
        assert_eq!(
            artifact_tree(&parallel_output),
            artifact_tree(&serial_output)
        );
    }

    #[test]
    fn batch_ranges_use_the_bounded_pool_and_merge_in_source_order() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let active = AtomicUsize::new(0);
        let maximum_active = AtomicUsize::new(0);
        let mut merged = Vec::new();
        pool.install(|| {
            map_batch_ranges_in_source_order(
                &pool,
                8_192,
                8,
                |ordinal, range| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum_active.fetch_max(now, Ordering::SeqCst);
                    if ordinal < 8 {
                        barrier.wait();
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(range)
                },
                |ordinal, expected, actual| {
                    assert_eq!(actual, expected);
                    merged.push((ordinal, actual));
                    Ok(())
                },
            )
        })
        .unwrap();
        assert!(maximum_active.load(Ordering::SeqCst) > 1);
        assert_eq!(merged.len(), 16);
        assert_eq!(merged.first().unwrap().1.start, 0);
        assert_eq!(merged.last().unwrap().1.end, 8_192);
        assert!(
            merged
                .windows(2)
                .all(|pair| pair[0].1.end == pair[1].1.start)
        );
    }

    #[test]
    #[ignore = "manual timed synthetic one-row-group projection benchmark"]
    fn benchmark_single_row_group_batch_projection() {
        const ROWS: usize = 24_593;
        let fixture = BenchmarkPreparationFixture::write(ROWS, ROWS);
        let reader = LocalSnapshotReader::open(fixture.root.path()).unwrap();
        let relation = reader
            .typed_relations()
            .find(|relation| relation.name == "ocsf_process_activity")
            .unwrap();
        let admitted = reader.admit_object(relation).unwrap();
        let context = projection_context(reader.identity());
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut serial_micros = Vec::new();
        let mut four_worker_micros = Vec::new();
        let mut parallel_micros = Vec::new();
        let mut expected = None;
        for _ in 0..3 {
            let started = Instant::now();
            let serial = census_row_group(&admitted, 0, &context, &serial_pool, 1).unwrap();
            serial_micros.push(started.elapsed().as_micros());
            let started = Instant::now();
            let four_worker =
                census_row_group(&admitted, 0, &context, &four_worker_pool, 4).unwrap();
            four_worker_micros.push(started.elapsed().as_micros());
            let started = Instant::now();
            let parallel = census_row_group(&admitted, 0, &context, &parallel_pool, 8).unwrap();
            parallel_micros.push(started.elapsed().as_micros());
            assert_eq!(four_worker, serial);
            assert_eq!(parallel, serial);
            expected = Some(serial);
        }
        assert_eq!(expected.unwrap().source_rows, ROWS as u64);
        serial_micros.sort_unstable();
        four_worker_micros.sort_unstable();
        parallel_micros.sort_unstable();
        let serial_median = serial_micros[1];
        let four_worker_median = four_worker_micros[1];
        let parallel_median = parallel_micros[1];
        eprintln!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema_version":"livefire.rag.synthetic-batch-projection-benchmark/1",
                "rows":ROWS,
                "row_groups":1,
                "record_batch_rows":8_192,
                "serial_workers":1,
                "middle_workers":4,
                "parallel_workers":8,
                "serial_median_micros":serial_median,
                "four_worker_median_micros":four_worker_median,
                "parallel_median_micros":parallel_median,
                "four_worker_speedup":serial_median as f64 / four_worker_median as f64,
                "eight_worker_speedup":serial_median as f64 / parallel_median as f64
            }))
            .unwrap()
        );
    }

    fn artifact_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
        let mut files = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                let file_type = entry.file_type().unwrap();
                if file_type.is_dir() {
                    pending.push(entry.path());
                } else {
                    assert!(file_type.is_file());
                    let relative = entry
                        .path()
                        .strip_prefix(root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned();
                    files.insert(relative, fs::read(entry.path()).unwrap());
                }
            }
        }
        files
    }

    #[test]
    #[ignore = "manual full 10,000-document artifact parity check"]
    fn benchmark_preparation_is_byte_identical_across_worker_counts() {
        let fixture = BenchmarkPreparationFixture::write(10_001, 10_001);
        let reader = LocalSnapshotReader::open(fixture.root.path()).unwrap();
        let relation = reader.typed_relations().next().unwrap();
        let admitted = reader.admit_object(relation).unwrap();
        assert_eq!(admitted.row_groups().len(), 1);

        let outputs = tempfile::tempdir().unwrap();
        let serial = outputs.path().join("serial");
        let parallel = outputs.path().join("parallel");
        let options = |out, workers| PrepareBenchmarkOptions {
            snapshot: fixture.root.path().to_path_buf(),
            dataset_id: "benchmark-worker-parity".into(),
            dataset_version: "1".into(),
            relations: vec!["ocsf_process_activity".into()],
            out,
            document_shard_rows: 777,
            selection_seed: "worker-parity-seed".into(),
            workers,
        };
        prepare_benchmark(options(serial.clone(), 1)).unwrap();
        prepare_benchmark(options(parallel.clone(), 8)).unwrap();

        let serial_tree = artifact_tree(&serial);
        let parallel_tree = artifact_tree(&parallel);
        assert_eq!(
            serial_tree.keys().collect::<Vec<_>>(),
            parallel_tree.keys().collect::<Vec<_>>()
        );
        assert_eq!(serial_tree, parallel_tree);
        assert!(serial_tree.contains_key("selection-manifest.json"));
        for size in STANDARD_BENCHMARK_SIZES {
            assert!(serial_tree.contains_key(&format!("prepared-{size:05}/manifest.json")));
        }
    }

    #[test]
    fn benchmark_worker_count_is_bounded_before_opening_the_snapshot() {
        assert!((1..=8).contains(&default_prepare_workers()));
        for workers in [0, MAX_PREPARE_WORKERS + 1] {
            let error = prepare_benchmark(PrepareBenchmarkOptions {
                snapshot: PathBuf::from("not-opened-for-invalid-worker-count"),
                dataset_id: "invalid-workers".into(),
                dataset_version: "1".into(),
                relations: vec!["ocsf_process_activity".into()],
                out: PathBuf::from("not-created-for-invalid-worker-count"),
                document_shard_rows: 1,
                selection_seed: "invalid-workers".into(),
                workers,
            })
            .unwrap_err();
            assert!(matches!(error, Error::AccountingClosure(_)));
        }
    }

    #[test]
    fn ordinary_prepare_worker_count_is_bounded_before_opening_the_snapshot() {
        for workers in [0, MAX_PREPARE_WORKERS + 1] {
            let error = prepare(PrepareOptions {
                snapshot: PathBuf::from("not-opened-for-invalid-worker-count"),
                dataset_id: "invalid-workers".into(),
                dataset_version: "1".into(),
                relations: vec!["ocsf_process_activity".into()],
                out: PathBuf::from("not-created-for-invalid-worker-count"),
                document_shard_rows: 1,
                workers,
            })
            .unwrap_err();
            assert!(matches!(error, Error::AccountingClosure(_)));
        }
    }

    #[test]
    fn census_worker_count_has_a_clear_memory_bound() {
        assert!((1..=8).contains(&default_census_workers()));
        for workers in [0, MAX_CENSUS_WORKERS + 1] {
            let error = build_corpus_census_report(&CensusOptions {
                snapshot: PathBuf::from("not-opened-for-invalid-worker-count"),
                relations: vec![],
                out: None,
                workers,
            })
            .unwrap_err();
            assert!(matches!(error, Error::AccountingClosure(_)));
        }
    }

    fn write_tokenizer_verification_fixture(
        root: &Path,
        direct_ids: &[u32],
    ) -> VerifyTokenizerOptions {
        let tokenizer_path = root.join("tokenizer.json");
        fs::write(&tokenizer_path, TEST_TOKENIZER_JSON).unwrap();
        let tokenizer_sha256 = digest_bytes(TEST_TOKENIZER_JSON.as_bytes());
        let reference = ExecutableTokenizerRef {
            artifact: ComponentRef {
                id: "fixture.executable-tokenizer".into(),
                version: "model-revision-nonfc-v1".into(),
                sha256: tokenizer_sha256.clone(),
            },
            format: rag_pipeline::TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: "model-revision".into(),
            target_tokenizer: ComponentRef {
                id: "fixture.logical-tokenizer".into(),
                version: "model-revision".into(),
                sha256: Digest::new("c".repeat(64)).unwrap(),
            },
            add_special_tokens: false,
            maximum_input_bytes: 4,
        };
        let reference_path = root.join("tokenizer.ref.json");
        write_canonical_json(&reference_path, &reference).unwrap();
        let generated_ids = [4_u32];
        let generated_bytes = generated_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let fixture = json!({
            "schema_version": TOKENIZER_PARITY_FIXTURE_SCHEMA,
            "source": {
                "runtime": "captured test runtime",
                "model_file": "fixture.gguf",
                "model_revision": reference.model_revision,
                "source_tokenizer_json_revision": "source-revision",
                "source_tokenizer_json_sha256": tokenizer_sha256,
                "executable_tokenizer_json_sha256": reference.artifact.sha256,
                "add_special_tokens": false
            },
            "cases": [{
                "name": "direct",
                "input": "a b",
                "token_ids": direct_ids
            }],
            "generated_cases": [{
                "name": "maximum_input_bytes_ascii",
                "repeat": "a",
                "count": 4,
                "token_count": 1,
                "token_ids_u32le_sha256": digest_bytes(&generated_bytes)
            }]
        });
        let fixture_path = root.join("fixture.json");
        write_canonical_json(&fixture_path, &fixture).unwrap();
        VerifyTokenizerOptions {
            tokenizer_json: tokenizer_path,
            tokenizer_ref: reference_path,
            fixture: fixture_path,
        }
    }

    #[test]
    fn offline_tokenizer_verification_checks_direct_and_boundary_cases() {
        let root = tempfile::tempdir().unwrap();
        let options = write_tokenizer_verification_fixture(root.path(), &[0, 1]);
        let report = build_tokenizer_verification_report(&options).unwrap();
        assert_eq!(report.status, "passed");
        assert_eq!(report.direct_cases, 1);
        assert_eq!(report.generated_cases, 1);
        assert_eq!(report.maximum_input_boundary_cases, 1);
        assert_eq!(report.verified_inputs, 2);
        assert_eq!(report.verified_tokens, 3);
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.contains("a b"));
        assert!(!encoded.contains("captured test runtime"));
        assert!(!encoded.contains("fixture.gguf"));
        assert_eq!(report.component_sha256, component_digest(&report).unwrap());
    }

    #[test]
    fn offline_tokenizer_verification_fails_closed_on_token_drift() {
        let root = tempfile::tempdir().unwrap();
        let options = write_tokenizer_verification_fixture(root.path(), &[1, 0]);
        let error = build_tokenizer_verification_report(&options).unwrap_err();
        assert!(matches!(error, Error::AccountingClosure(_)));
    }

    #[test]
    fn offline_tokenizer_verification_fails_closed_on_source_reference_drift() {
        let root = tempfile::tempdir().unwrap();
        let options = write_tokenizer_verification_fixture(root.path(), &[0, 1]);
        let mut fixture: Value = read_json(&options.fixture).unwrap();
        fixture["source"]["executable_tokenizer_json_sha256"] = json!("f".repeat(64));
        write_canonical_json(&options.fixture, &fixture).unwrap();
        let error = build_tokenizer_verification_report(&options).unwrap_err();
        assert!(matches!(error, Error::AccountingClosure(_)));
    }

    fn benchmark_dataset(relations: &[&str]) -> DatasetIdentity {
        DatasetIdentity {
            id: "benchmark-test".into(),
            version: "1".into(),
            source_snapshot: benchmark_component("snapshot"),
            mapping: benchmark_component("mapping"),
            source_admission: vec![],
            included_relations: relations
                .iter()
                .map(|relation| (*relation).into())
                .collect(),
            excluded_relations: vec![],
            structured_only_relations: vec![],
        }
    }

    fn benchmark_candidate(
        ordinal: usize,
        relation: &str,
        semantic_text_utf8_bytes: u64,
    ) -> BenchmarkSelectionCandidate {
        let document_id = format!("{relation}-{ordinal:05}");
        BenchmarkSelectionCandidate {
            document_sha256: digest_bytes(format!("document:{document_id}").as_bytes()),
            semantic_text_sha256: digest_bytes(format!("text:{document_id}").as_bytes()),
            document_id,
            semantic_text_utf8_bytes,
            primary_relation: relation.into(),
        }
    }

    #[test]
    fn benchmark_length_strata_keep_the_observed_maximum_separate() {
        let candidates = (0..10_000)
            .map(|ordinal| {
                benchmark_candidate(
                    ordinal,
                    "relation_a",
                    if ordinal == 9_999 { 100 } else { 10 },
                )
            })
            .collect::<Vec<_>>();
        let strata = benchmark_length_strata(&candidates).unwrap();
        assert_eq!(strata.last().unwrap().minimum_utf8_bytes, 100);
        assert_eq!(strata.last().unwrap().maximum_utf8_bytes, None);
        for pair in strata.windows(2) {
            assert_eq!(
                pair[0].maximum_utf8_bytes.unwrap() + 1,
                pair[1].minimum_utf8_bytes
            );
        }
    }

    #[test]
    fn benchmark_quotas_are_nested_and_empty_cells_stay_zero() {
        let dataset = benchmark_dataset(&["relation_a", "relation_b", "relation_empty"]);
        let candidates = (0..12_000)
            .map(|ordinal| {
                let relation = if ordinal % 2 == 0 {
                    "relation_a"
                } else {
                    "relation_b"
                };
                let length = if ordinal % 4 < 2 { 10 } else { 100 };
                benchmark_candidate(ordinal, relation, length)
            })
            .collect::<Vec<_>>();
        let strata = benchmark_length_strata(&candidates).unwrap();
        let mut policy =
            benchmark_selection_policy(&dataset, &strata, &candidates, "test-seed".into()).unwrap();
        policy.seal(&dataset).unwrap();

        for (target_index, target) in policy.targets.iter().enumerate() {
            assert_eq!(
                target
                    .quotas
                    .iter()
                    .map(|quota| quota.documents)
                    .sum::<u64>(),
                STANDARD_BENCHMARK_SIZES[target_index]
            );
            assert!(
                target
                    .quotas
                    .iter()
                    .filter(|quota| quota.relation == "relation_empty")
                    .all(|quota| quota.documents == 0)
            );
            if target_index > 0 {
                assert!(
                    target
                        .quotas
                        .iter()
                        .zip(&policy.targets[target_index - 1].quotas)
                        .all(|(current, previous)| current.documents >= previous.documents)
                );
            }
        }
        let (_, _, selections) = select_benchmark_documents(
            &dataset,
            &benchmark_component("projection"),
            &policy,
            &candidates,
        )
        .unwrap();
        assert_eq!(selections.len(), 10_000);
    }

    fn occurrence(sequence: usize) -> PreparedOccurrenceRow {
        PreparedOccurrenceRow {
            occurrence_id: format!("occ-{sequence}"),
            document_id: "doc-a".into(),
            event_time_ms: Some(sequence as u64),
            relation: "events".into(),
            source_row_ordinal: sequence as u64,
            exact_attributes_json: "{}".into(),
            snapshot_sha256: Digest::new("a".repeat(64)).unwrap(),
            mapping_sha256: Digest::new("b".repeat(64)).unwrap(),
            event_id: format!("event-{sequence}"),
            support_ref: format!("support-{sequence}"),
        }
    }

    fn stage_validation_fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let documents_relative = "documents/part.parquet";
        let occurrences_relative = "occurrences/events/part.parquet";
        let documents_path = root.path().join(documents_relative);
        let occurrences_path = root.path().join(occurrences_relative);
        fs::create_dir_all(documents_path.parent().unwrap()).unwrap();
        fs::create_dir_all(occurrences_path.parent().unwrap()).unwrap();

        let document = PreparedDocumentRow {
            document_ordinal: 0,
            document_id: "doc-a".into(),
            document_sha256: Digest::new("d".repeat(64)).unwrap(),
            semantic_text_sha256: digest_bytes(b"a"),
            semantic_text: "a".into(),
            document_kind: DocumentKind::Activity,
            primary_relation: "events".into(),
            facets_json: "{}".into(),
            relations_json: "[]".into(),
            occurrence_count: 1,
        };
        let occurrence = occurrence(0);
        write_prepared_documents(&documents_path, std::slice::from_ref(&document)).unwrap();
        write_prepared_occurrences(&occurrences_path, std::slice::from_ref(&occurrence)).unwrap();

        let component = |id: &str, byte: char| ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        let document_object = PreparedDocumentObject {
            object: object_entry(
                documents_relative,
                &documents_path,
                1,
                canonical_digest(&vec![document.clone()]).unwrap(),
            )
            .unwrap(),
            ordinal: 0,
            first_document_id: document.document_id.clone(),
            last_document_id: document.document_id.clone(),
            embedding_input_order_sha256: embedding_input_order_digest([&document]),
        };
        let occurrence_object = PreparedOccurrenceObject {
            object: object_entry(
                occurrences_relative,
                &occurrences_path,
                1,
                canonical_digest(&vec![occurrence.clone()]).unwrap(),
            )
            .unwrap(),
            ordinal: 0,
            relation: "events".into(),
        };
        let mut manifest = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: zero_digest().unwrap(),
            dataset: DatasetIdentity {
                id: "stage-validation".into(),
                version: "1".into(),
                source_snapshot: component("snapshot", 'a'),
                mapping: component("mapping", 'b'),
                source_admission: vec![],
                included_relations: vec!["events".into()],
                excluded_relations: vec![],
                structured_only_relations: vec![],
            },
            projection_policy: component("projection", 'c'),
            document_schema: component("documents", 'e'),
            occurrence_schema: component("occurrences", 'f'),
            preparation_implementation: component("implementation", '1'),
            document_count: 1,
            occurrence_count: 1,
            document_order_sha256: document_order_digest([document.document_id.as_str()]),
            embedding_input_order_sha256: embedding_input_order_digest([&document]),
            documents: vec![document_object],
            occurrences: vec![occurrence_object],
            relation_accounting: BTreeMap::from([(
                "events".into(),
                RelationAccounting {
                    source_rows: 1,
                    searchable_occurrences: 1,
                    selected_occurrences: 1,
                    excluded_rows: 0,
                },
            )]),
        };
        manifest.seal().unwrap();
        write_canonical_json(&root.path().join(MANIFEST_FILE), &manifest).unwrap();
        root
    }

    fn fixture_embedding_task(prepared: &PreparedCorpusManifest) -> EmbeddingTaskV2 {
        let document_object = &prepared.documents[0];
        EmbeddingTaskV2 {
            task_id: "document-only-admission".into(),
            ordinal_start: 0,
            ordinal_end: 1,
            input_slices: vec![EmbeddingInputSliceV2 {
                path: document_object.object.path.clone(),
                object_sha256: document_object.object.sha256.clone(),
                row_offset: 0,
                rows: 1,
                embedding_input_order_sha256: document_object.embedding_input_order_sha256.clone(),
                token_count: 1,
                maximum_document_tokens: 1,
                document_token_counts_sha256: Digest::new("2".repeat(64)).unwrap(),
            }],
            embedding_input_order_sha256: prepared.embedding_input_order_sha256.clone(),
            token_count: 1,
            maximum_document_tokens: 1,
            document_token_counts_sha256: Digest::new("2".repeat(64)).unwrap(),
            result_path: SafeRelativePath::new("parts/document-only-admission.f32").unwrap(),
            receipt_path: SafeRelativePath::new("receipts/document-only-admission.json").unwrap(),
        }
    }

    #[test]
    fn planning_and_embedding_admission_do_not_open_occurrence_shards() {
        let fixture = stage_validation_fixture();
        assert_eq!(load_prepared(fixture.path()).unwrap().document_count, 1);
        fs::remove_file(fixture.path().join("occurrences/events/part.parquet")).unwrap();

        let prepared = load_prepared_documents_only(fixture.path()).unwrap();
        assert_eq!(
            load_all_prepared_documents(fixture.path(), &prepared)
                .unwrap()
                .len(),
            1
        );
        let mut loader = TaskDocumentLoaderV2::new(fixture.path(), &prepared);
        assert_eq!(
            loader
                .load(&fixture_embedding_task(&prepared))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn full_verification_and_assembly_reject_missing_or_corrupt_occurrences() {
        for corrupt in [false, true] {
            let fixture = stage_validation_fixture();
            let occurrence_path = fixture.path().join("occurrences/events/part.parquet");
            if corrupt {
                fs::write(&occurrence_path, b"corrupt occurrence shard").unwrap();
            } else {
                fs::remove_file(&occurrence_path).unwrap();
            }

            assert!(verify_prepared(fixture.path()).is_err());
            assert!(
                assemble(AssembleOptions {
                    prepared: fixture.path().to_path_buf(),
                    plan: fixture.path().join("unused-plan"),
                    embeddings: fixture.path().join("unused-embeddings"),
                    embedding_profile: fixture.path().join("unused-profile"),
                    out: fixture.path().join("unused-index"),
                    index_format: IndexFormat::SqliteV3,
                })
                .is_err()
            );
        }
    }

    fn test_embedding_task() -> EmbeddingTaskV2 {
        let count_digest = rag_pipeline::document_token_counts_digest(0, &[7]).unwrap();
        EmbeddingTaskV2 {
            task_id: "task-a".into(),
            ordinal_start: 0,
            ordinal_end: 1,
            input_slices: vec![EmbeddingInputSliceV2 {
                path: SafeRelativePath::new("documents/part.parquet").unwrap(),
                object_sha256: Digest::new("1".repeat(64)).unwrap(),
                row_offset: 0,
                rows: 1,
                embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
                token_count: 7,
                maximum_document_tokens: 7,
                document_token_counts_sha256: count_digest.clone(),
            }],
            embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
            token_count: 7,
            maximum_document_tokens: 7,
            document_token_counts_sha256: count_digest,
            result_path: SafeRelativePath::new("parts/task-a.f32").unwrap(),
            receipt_path: SafeRelativePath::new("receipts/task-a.json").unwrap(),
        }
    }

    fn test_embedding_plan() -> EmbeddingPlanV2 {
        let component = |id: &str, version: &str, byte: char| ComponentRef {
            id: id.into(),
            version: version.into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        let counts = rag_pipeline::encode_document_token_counts(&[7]);
        let count_digest = rag_pipeline::document_token_counts_digest(0, &[7]).unwrap();
        EmbeddingPlanV2 {
            schema_version: EMBEDDING_PLAN_V2_SCHEMA.into(),
            component_sha256: Digest::new("a".repeat(64)).unwrap(),
            prepared_corpus_sha256: Digest::new("b".repeat(64)).unwrap(),
            dataset: DatasetIdentity {
                id: "dataset".into(),
                version: "1".into(),
                source_snapshot: component("snapshot", "1", 'c'),
                mapping: component("mapping", "1", 'd'),
                source_admission: vec![],
                included_relations: vec!["events".into()],
                excluded_relations: vec![],
                structured_only_relations: vec![],
            },
            embedding_profile: EmbeddingProfileRef {
                component: component("profile", "1", 'e'),
                model_artifact: component("model", "model-revision", 'f'),
                tokenizer: component("logical-tokenizer", "1", '1'),
                maximum_input_tokens: 32,
                pooling: "last_token".into(),
                normalization: "l2".into(),
                dimensions: 4,
                dtype: "f32le".into(),
                document_format: "{semantic_text}".into(),
            },
            executable_tokenizer: ExecutableTokenizerRef {
                artifact: component("tokenizer-json", "source-revision", '2'),
                target_tokenizer: component("logical-tokenizer", "1", '1'),
                format: rag_pipeline::TokenizerArtifactFormat::HuggingFaceTokenizerJson,
                model_revision: "model-revision".into(),
                add_special_tokens: true,
                maximum_input_bytes: 16_384,
            },
            document_count: 1,
            document_order_sha256: Digest::new("3".repeat(64)).unwrap(),
            document_token_counts_sha256: count_digest.clone(),
            document_token_counts_object: rag_pipeline::DocumentTokenCountsObject {
                path: SafeRelativePath::new(rag_pipeline::DOCUMENT_TOKEN_COUNTS_PATH).unwrap(),
                rows: 1,
                bytes: counts.len() as u64,
                sha256: digest_bytes(&counts),
                document_token_counts_sha256: count_digest,
            },
            token_statistics: rag_pipeline::token_statistics(&[7]).unwrap(),
            maximum_task_tokens: 32,
            maximum_task_documents: 8,
            tasks: vec![test_embedding_task()],
        }
    }

    fn test_receipt() -> VectorResultReceipt {
        VectorResultReceipt {
            schema_version: VECTOR_RECEIPT_SCHEMA.into(),
            component_sha256: Digest::new("5".repeat(64)).unwrap(),
            plan_sha256: Digest::new("a".repeat(64)).unwrap(),
            prepared_corpus_sha256: Digest::new("b".repeat(64)).unwrap(),
            embedding_profile_sha256: Digest::new("e".repeat(64)).unwrap(),
            task_id: "task-a".into(),
            ordinal_start: 0,
            ordinal_end: 1,
            embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
            vector: VectorObject {
                path: SafeRelativePath::new("parts/task-a.f32").unwrap(),
                rows: 1,
                bytes: 80,
                sha256: Digest::new("6".repeat(64)).unwrap(),
                dimensions: 4,
                dtype: "f32le".into(),
                embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
            },
            executor: ExecutorReceipt {
                implementation: ComponentRef {
                    id: "executor".into(),
                    version: "1".into(),
                    sha256: Digest::new("7".repeat(64)).unwrap(),
                },
                runtime: ComponentRef {
                    id: "runtime".into(),
                    version: "1".into(),
                    sha256: Digest::new("8".repeat(64)).unwrap(),
                },
                returned_model: "model".into(),
                requests: 1,
                retries: 0,
                input_bytes_upper_bound: 20,
                elapsed_ms: 1_000,
                conformance_passed: true,
            },
            derivation: None,
            finite_values_validated: true,
            normalization_validated: true,
        }
    }

    fn valid_test_embedding_plan() -> EmbeddingPlanV2 {
        let mut plan = test_embedding_plan();
        let task = &mut plan.tasks[0];
        let task_id = canonical_digest(&json!({
            "schema_version": "livefire.rag.embedding-task/2",
            "prepared_corpus_sha256": plan.prepared_corpus_sha256,
            "embedding_profile_sha256": plan.embedding_profile.component.sha256,
            "tokenizer_sha256": plan.executable_tokenizer.artifact.sha256,
            "ordinal_start": task.ordinal_start,
            "ordinal_end": task.ordinal_end,
            "embedding_input_order_sha256": task.embedding_input_order_sha256,
            "document_token_counts_sha256": task.document_token_counts_sha256,
            "token_count": task.token_count,
        }))
        .unwrap()
        .to_string();
        task.task_id = task_id.clone();
        task.result_path = SafeRelativePath::new(format!("parts/{task_id}.f32")).unwrap();
        task.receipt_path = SafeRelativePath::new(format!("receipts/{task_id}.json")).unwrap();
        plan.component_sha256 = Digest::new("0".repeat(64)).unwrap();
        plan.seal().unwrap();
        plan
    }

    fn test_receipt_for_plan(plan: &EmbeddingPlanV2) -> VectorResultReceipt {
        let task = &plan.tasks[0];
        let mut receipt = test_receipt();
        receipt.plan_sha256 = plan.component_sha256.clone();
        receipt.prepared_corpus_sha256 = plan.prepared_corpus_sha256.clone();
        receipt.embedding_profile_sha256 = plan.embedding_profile.component.sha256.clone();
        receipt.task_id = task.task_id.clone();
        receipt.ordinal_start = task.ordinal_start;
        receipt.ordinal_end = task.ordinal_end;
        receipt.embedding_input_order_sha256 = task.embedding_input_order_sha256.clone();
        receipt.vector.path = task.result_path.clone();
        receipt.vector.rows = task.row_count();
        receipt.vector.embedding_input_order_sha256 = task.embedding_input_order_sha256.clone();
        receipt.seal().unwrap();
        receipt
    }

    fn test_compact_profile() -> rag_embedding::EmbeddingProfile {
        rag_embedding::EmbeddingProfile {
            id: "profile".into(),
            version: "1".into(),
            sha256: "e".repeat(64),
            model: "model".into(),
            dimensions: 4,
            normalization: "l2".into(),
            vector_derivation: None,
            query_instruction: None,
            query_composition: None,
        }
    }

    fn test_prepared_for_plan(plan: &EmbeddingPlanV2) -> PreparedCorpusManifest {
        let component = |id: &str, byte: char| ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: plan.prepared_corpus_sha256.clone(),
            dataset: plan.dataset.clone(),
            projection_policy: component("projection", '4'),
            document_schema: component("documents", '5'),
            occurrence_schema: component("occurrences", '6'),
            preparation_implementation: component("implementation", '7'),
            document_count: plan.document_count,
            occurrence_count: 1,
            document_order_sha256: plan.document_order_sha256.clone(),
            embedding_input_order_sha256: plan.tasks[0].embedding_input_order_sha256.clone(),
            documents: vec![],
            occurrences: vec![],
            relation_accounting: BTreeMap::from([(
                "events".into(),
                RelationAccounting {
                    source_rows: 1,
                    searchable_occurrences: 1,
                    selected_occurrences: 1,
                    excluded_rows: 0,
                },
            )]),
        }
    }

    fn empty_run_artifact_sizes() -> RunArtifactSizes {
        RunArtifactSizes {
            status: ObservationStatus::NotMeasured,
            prepared_corpus_bytes: None,
            embedding_plan_bytes: None,
            embedding_profile_bytes: None,
            vector_shards_bytes: None,
            receipts_bytes: None,
            task_reports_bytes: None,
        }
    }

    fn test_builder_task_report(
        started_unix_ms: u64,
        finished_unix_ms: u64,
    ) -> BuilderEmbeddingTaskReport {
        let context = LocalRunContext::observe();
        BuilderEmbeddingTaskReport {
            schema_version: "livefire.rag.embedding-task-run-report/1".into(),
            plan_sha256: Digest::new("a".repeat(64)).unwrap(),
            source_snapshot_sha256: Digest::new("b".repeat(64)).unwrap(),
            prepared_corpus_sha256: Digest::new("c".repeat(64)).unwrap(),
            embedding_profile_sha256: Digest::new("d".repeat(64)).unwrap(),
            tokenizer_sha256: Digest::new("e".repeat(64)).unwrap(),
            task_id: "task".into(),
            task_index: 0,
            ordinal_start: 0,
            ordinal_end: 1,
            document_count: 1,
            token_count: 7,
            receipt_sha256: Digest::new("f".repeat(64)).unwrap(),
            outcome: TaskRunOutcome::Executed,
            started_unix_ms: Some(started_unix_ms),
            finished_unix_ms: Some(finished_unix_ms),
            git: context.git,
            machine: context.machine,
            lm_studio: LmStudioContext::embedding("model", "model", 16, 1),
            transport_bytes: TransportByteAccounting {
                status: ObservationStatus::Partial,
                request_body_bytes: None,
                response_body_bytes: None,
                submitted_text_bytes: Some(20),
                decoded_vector_bytes: Some(16),
            },
            resource_usage: context.resources,
            artifact_sizes: TaskArtifactSizes {
                status: ObservationStatus::Partial,
                vector_shard_bytes: Some(80),
                receipt_bytes: Some(400),
                task_report_bytes: None,
            },
            execution: Some(EmbeddingTaskReport {
                rows: 1,
                batches: 1,
                attempts: 1,
                retries: 0,
                unique_input_text_bytes: 20,
                sent_input_text_bytes: 20,
                vector_bytes: 16,
                shard_bytes: 80,
                elapsed_micros: finished_unix_ms.saturating_sub(started_unix_ms) * 1_000,
                request_elapsed_micros: 1,
                retry_backoff_micros: 0,
                peak_in_flight: 1,
                batch_reports: vec![],
            }),
        }
    }

    fn test_builder_task_report_v2(
        started_unix_ms: u64,
        finished_unix_ms: u64,
    ) -> BuilderEmbeddingTaskReportV2 {
        let v1 = test_builder_task_report(started_unix_ms, finished_unix_ms);
        let component = |id: &str, byte: char| ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        BuilderEmbeddingTaskReportV2 {
            schema_version: "livefire.rag.embedding-task-run-report/2".into(),
            plan_sha256: v1.plan_sha256,
            source_snapshot_sha256: v1.source_snapshot_sha256,
            prepared_corpus_sha256: v1.prepared_corpus_sha256,
            embedding_profile_sha256: v1.embedding_profile_sha256,
            tokenizer_sha256: v1.tokenizer_sha256,
            task_id: v1.task_id,
            task_index: v1.task_index,
            ordinal_start: v1.ordinal_start,
            ordinal_end: v1.ordinal_end,
            document_count: v1.document_count,
            token_count: v1.token_count,
            receipt_sha256: v1.receipt_sha256,
            outcome: v1.outcome,
            started_unix_ms: v1.started_unix_ms,
            finished_unix_ms: v1.finished_unix_ms,
            execution_identity: EmbeddingExecutionIdentityV2 {
                backend_kind: "tei".into(),
                executor_image: component("image", '1'),
                executor_image_build: component("image-build", '6'),
                runtime: component("runtime", '2'),
                worker_binary: component("worker", '3'),
                model_artifact: component("model", '4'),
                embedding_profile: component("profile", '5'),
                returned_model: "model".into(),
                accelerator: EmbeddingAcceleratorPolicyV2 {
                    provider: "runpod".into(),
                    model: "NVIDIA A40".into(),
                    architecture: "ampere".into(),
                    compute_capability: "8.6".into(),
                    count: 1,
                },
            },
            git: v1.git,
            machine: v1.machine,
            accelerator: AcceleratorContextV2 {
                status: ObservationStatus::Observed,
                provider: Some("runpod".into()),
                machine_id: Some("machine-a".into()),
                model: Some("NVIDIA A40".into()),
                architecture: Some("ampere".into()),
                compute_capability: Some("8.6".into()),
                count: Some(1),
            },
            backend: EmbeddingBackendContextV2 {
                status: ObservationStatus::Observed,
                kind: "tei".into(),
                version: Some("1.9.3".into()),
                endpoint_kind: "worker_loopback".into(),
                batch_size: 16,
                requests_in_flight: 4,
                cold_load_micros: Some(1),
            },
            transport_bytes: v1.transport_bytes,
            resource_usage: ResourceUsageV2 {
                status: ObservationStatus::Partial,
                worker_peak_rss_bytes: Some(1),
                backend_peak_rss_bytes: Some(2),
            },
            artifact_sizes: v1.artifact_sizes,
            execution: v1.execution,
        }
    }

    fn write_test_part(path: &Path, task: &EmbeddingTaskV2) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let expected = task_shard_expectation_v2(task, 4).unwrap();
        File::create(path).unwrap();
        let mut writer =
            rag_embedding::EmbeddingShardWriter::create(path, expected.into()).unwrap();
        writer.write_vector(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn deterministic_test_vectors_are_stable_ordered_4096d_units() {
        let row = PreparedDocumentRow {
            document_ordinal: 7,
            document_id: "doc-seven".into(),
            document_sha256: digest_bytes(b"document-seven"),
            semantic_text_sha256: digest_bytes(b"semantic text"),
            semantic_text: "semantic text".into(),
            document_kind: DocumentKind::Activity,
            primary_relation: "events".into(),
            facets_json: "{}".into(),
            relations_json: "[\"events\"]".into(),
            occurrence_count: 1,
        };
        let first = deterministic_test_vector(&row).unwrap();
        let second = deterministic_test_vector(&row).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 4_096);
        assert_eq!(first.iter().filter(|value| **value != 0.0).count(), 16);
        let norm = first.iter().map(|value| value * value).sum::<f32>();
        assert_eq!(norm, 1.0);

        let mut next = row;
        next.document_ordinal += 1;
        assert_ne!(first, deterministic_test_vector(&next).unwrap());
    }

    #[test]
    fn test_vectors_finalize_assemble_deterministically_and_normal_open_refuses_them() {
        const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"<unk>":1},"unk_token":"<unk>"}}"#;
        let fixture = stage_validation_fixture();
        let prepared = load_prepared(fixture.path()).unwrap();
        let documents = load_all_prepared_documents(fixture.path(), &prepared).unwrap();
        let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json");
        let profile_bytes = fs::read(&profile_path).unwrap();
        let compact = parse_embedding_profile(&profile_bytes).unwrap();
        let planned_profile = profile_ref(&profile_bytes, &compact).unwrap();
        let tokenizer = ExecutableTokenizerRef {
            artifact: ComponentRef {
                id: "test.tokenizer-json".into(),
                version: "1".into(),
                sha256: digest_bytes(TOKENIZER.as_bytes()),
            },
            format: rag_pipeline::TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: planned_profile.model_artifact.version.clone(),
            target_tokenizer: planned_profile.tokenizer.clone(),
            add_special_tokens: false,
            maximum_input_bytes: 1_024,
        };
        let (plan, counts) = build_token_balanced_plan_with_counts(
            &prepared,
            &documents,
            planned_profile,
            tokenizer,
            TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 32,
                maximum_task_documents: 8,
            },
        )
        .unwrap();
        let plan_root = fixture.path().join("plan");
        fs::create_dir(&plan_root).unwrap();
        plan.write_document_token_counts(&plan_root, &counts)
            .unwrap();
        write_canonical_json(&plan_root.join("plan.json"), &plan).unwrap();

        let generated = tempfile::tempdir().unwrap();
        let first = generated.path().join("test-embeddings-a");
        let second = generated.path().join("test-embeddings-b");
        for out in [&first, &second] {
            test_embed(TestEmbedOptions {
                prepared: fixture.path().to_path_buf(),
                plan: plan_root.clone(),
                embedding_profile: profile_path.clone(),
                out: out.clone(),
            })
            .unwrap();
        }
        for relative in [
            "embedding-profile.json".to_owned(),
            "manifest.json".to_owned(),
            "summary.json".to_owned(),
            plan.tasks[0].result_path.as_str().to_owned(),
            plan.tasks[0].receipt_path.as_str().to_owned(),
            format!("reports/{}.json", plan.tasks[0].task_id),
        ] {
            assert_eq!(
                fs::read(first.join(&relative)).unwrap(),
                fs::read(second.join(&relative)).unwrap()
            );
        }
        let result: EmbeddingResultSetManifest = read_json(&first.join("manifest.json")).unwrap();
        assert!(result.test_only);
        assert_eq!(result.schema_version, TEST_RESULT_SET_SCHEMA);

        let index = generated.path().join("test-index");
        assemble(AssembleOptions {
            prepared: fixture.path().to_path_buf(),
            plan: plan_root,
            embeddings: first,
            embedding_profile: profile_path,
            out: index.clone(),
            index_format: IndexFormat::SqliteV3,
        })
        .unwrap();
        assert!(rag_index::FastIndex::open(&index).is_err());
        let diagnostic = rag_index::FastIndex::open_allow_test_only(&index).unwrap();
        assert!(diagnostic.manifest.test_only);
        assert_eq!(diagnostic.manifest.vectors.dimensions, 4_096);
        assert_eq!(diagnostic.manifest.vectors.count, 1);
    }

    #[test]
    fn projection_policy_uses_canonical_material_identity() {
        assert_eq!(
            projection_policy_component().unwrap().sha256.as_str(),
            "426a3543df1c11990bfdf32f269da25808fda65586ecd855d07192a60f375acd"
        );
    }

    #[test]
    fn occurrence_preparation_flushes_fixed_bounded_parts_and_remainder() {
        let directory = tempfile::tempdir().unwrap();
        let mut buffer = (0..OCCURRENCE_SHARD_ROWS + 3)
            .map(occurrence)
            .collect::<Vec<_>>();
        let mut part = 0;
        let mut objects = Vec::new();
        flush_occurrence_shards(
            directory.path(),
            "events",
            &mut buffer,
            &mut part,
            &mut objects,
            false,
        )
        .unwrap();
        assert_eq!(buffer.len(), 3);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].object.rows, OCCURRENCE_SHARD_ROWS as u64);
        flush_occurrence_shards(
            directory.path(),
            "events",
            &mut buffer,
            &mut part,
            &mut objects,
            true,
        )
        .unwrap();
        assert!(buffer.is_empty());
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[1].object.rows, 3);
        assert_eq!(
            read_prepared_occurrences(&objects[1].object.path.join_to(directory.path()))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn dataset_accounting_includes_nonsearchable_rows_from_included_relations() {
        let component = |id: &str, byte: char| ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: Digest::new(byte.to_string().repeat(64)).unwrap(),
        };
        let prepared = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: Digest::new("0".repeat(64)).unwrap(),
            dataset: DatasetIdentity {
                id: "mixed-events".into(),
                version: "1".into(),
                source_snapshot: component("snapshot", 'a'),
                mapping: component("mapping", 'b'),
                source_admission: vec![],
                included_relations: vec!["events".into()],
                excluded_relations: vec!["network".into()],
                structured_only_relations: vec!["metrics".into()],
            },
            projection_policy: component("projection", 'c'),
            document_schema: component("documents", 'd'),
            occurrence_schema: component("occurrences", 'e'),
            preparation_implementation: component("implementation", 'f'),
            document_count: 1,
            occurrence_count: 1,
            document_order_sha256: Digest::new("1".repeat(64)).unwrap(),
            embedding_input_order_sha256: Digest::new("2".repeat(64)).unwrap(),
            documents: vec![],
            occurrences: vec![],
            relation_accounting: BTreeMap::from([
                (
                    "events".into(),
                    RelationAccounting {
                        source_rows: 2,
                        searchable_occurrences: 1,
                        selected_occurrences: 1,
                        excluded_rows: 1,
                    },
                ),
                (
                    "metrics".into(),
                    RelationAccounting {
                        source_rows: 3,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 3,
                    },
                ),
                (
                    "network".into(),
                    RelationAccounting {
                        source_rows: 4,
                        searchable_occurrences: 0,
                        selected_occurrences: 0,
                        excluded_rows: 4,
                    },
                ),
            ]),
        };
        let accounting = portable_dataset_accounting(&prepared, 1).unwrap();
        assert_eq!(accounting["source_records"], 9);
        assert_eq!(accounting["indexed_occurrences"], 1);
        assert_eq!(accounting["structured_only_occurrences"], 4);
        assert_eq!(accounting["structured_only_by_relation"]["events"], 1);
        assert_eq!(accounting["structured_only_by_relation"]["metrics"], 3);
        assert_eq!(accounting["excluded_by_scope_occurrences"], 4);
    }

    #[test]
    fn task_range_is_strict_and_checked_against_the_plan_length() {
        assert_eq!(parse_task_selection(None).unwrap(), TaskSelection::All);
        assert_eq!(
            parse_task_selection(Some("2..5")).unwrap(),
            TaskSelection::Range { start: 2, end: 5 }
        );
        for invalid in ["", "1", "1...2", "2..2", "3..2", "a..2"] {
            assert!(parse_task_selection(Some(invalid)).is_err(), "{invalid}");
        }
        assert_eq!(
            parse_task_selection(Some("2..5"))
                .unwrap()
                .resolve(5)
                .unwrap(),
            2..5
        );
        assert!(
            parse_task_selection(Some("2..6"))
                .unwrap()
                .resolve(5)
                .is_err()
        );
    }

    #[test]
    fn restart_keeps_the_original_sanitized_execution_report() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("reports")).unwrap();
        let plan = valid_test_embedding_plan();
        let task = &plan.tasks[0];
        let receipt = test_receipt();
        let prepared = test_prepared_for_plan(&plan);
        let profile = test_compact_profile();
        let path = task_report_path(root.path(), task).unwrap();
        let vector_path = task.result_path.join_to(root.path());
        let receipt_path = task.receipt_path.join_to(root.path());
        fs::create_dir_all(vector_path.parent().unwrap()).unwrap();
        fs::create_dir_all(receipt_path.parent().unwrap()).unwrap();
        fs::write(&vector_path, vec![0; 80]).unwrap();
        fs::write(&receipt_path, b"{}").unwrap();
        let run_context = LocalRunContext::observe();
        let execution = EmbeddingTaskReport {
            rows: 1,
            batches: 1,
            attempts: 1,
            retries: 0,
            unique_input_text_bytes: 20,
            sent_input_text_bytes: 20,
            vector_bytes: 16,
            shard_bytes: 80,
            elapsed_micros: 1_000_000,
            request_elapsed_micros: 900_000,
            retry_backoff_micros: 0,
            peak_in_flight: 1,
            batch_reports: vec![],
        };
        ensure_task_report(
            &path,
            0,
            task,
            &plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &profile,
                receipt: &receipt,
                vector_path: &vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: 16,
                requests_in_flight: 1,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::Executed,
                started_unix_ms: Some(1_000),
                finished_unix_ms: Some(2_000),
                execution: Some(execution),
            },
        )
        .unwrap();
        let before = fs::read(&path).unwrap();
        ensure_task_report(
            &path,
            0,
            task,
            &plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &profile,
                receipt: &receipt,
                vector_path: &vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: 16,
                requests_in_flight: 1,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::Reused,
                started_unix_ms: None,
                finished_unix_ms: None,
                execution: None,
            },
        )
        .unwrap();
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[test]
    fn finalizer_requires_exact_complete_artifact_coverage() {
        let root = tempfile::tempdir().unwrap();
        let plan = test_embedding_plan();
        for path in [
            "parts/task-a.f32",
            "receipts/task-a.json",
            "reports/task-a.json",
            "manifest.json",
            "summary.json",
            EMBEDDING_PROFILE_FILE,
        ] {
            let path = root.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"fixture").unwrap();
        }
        validate_embedding_artifact_coverage_v2(root.path(), &plan, true).unwrap();
        fs::write(root.path().join("unexpected.json"), b"fixture").unwrap();
        assert!(validate_embedding_artifact_coverage_v2(root.path(), &plan, true).is_err());
    }

    #[test]
    fn finalizer_cleans_dead_internal_stages_and_preserves_live_ones() {
        let root = tempfile::tempdir().unwrap();
        let plan = test_embedding_plan();
        for path in [
            "parts/task-a.f32",
            "receipts/task-a.json",
            "reports/task-a.json",
            "manifest.json",
            "summary.json",
            EMBEDDING_PROFILE_FILE,
        ] {
            let path = root.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"fixture").unwrap();
        }
        let dead_process = u32::MAX;
        let dead_stages = [
            format!("parts/.task-a.f32.{dead_process}.0.partial"),
            format!("receipts/.livefire-rag-atomic-{dead_process}-0-dead.partial"),
            format!("reports/.livefire-rag-atomic-{dead_process}-1-dead.partial"),
        ];
        for relative in &dead_stages {
            fs::write(root.path().join(relative), b"incomplete").unwrap();
        }
        let live_process = std::process::id();
        let live_stages = [
            format!("parts/.task-a.f32.{live_process}.0.partial"),
            format!("receipts/.livefire-rag-atomic-{live_process}-0-live.partial"),
            format!("reports/.livefire-rag-atomic-{live_process}-1-live.partial"),
        ];
        for relative in &live_stages {
            fs::write(root.path().join(relative), b"in-progress").unwrap();
        }

        validate_embedding_artifact_coverage_v2(root.path(), &plan, true).unwrap();
        for relative in dead_stages {
            assert!(!root.path().join(relative).exists());
        }
        for relative in live_stages {
            assert!(root.path().join(relative).exists());
        }

        fs::write(root.path().join("receipts/.unowned.partial"), b"unexpected").unwrap();
        assert!(validate_embedding_artifact_coverage_v2(root.path(), &plan, true).is_err());
    }

    #[test]
    fn summary_reports_throughput_and_nearest_rank_latency_without_content() {
        let plan = valid_test_embedding_plan();
        let receipt = test_receipt_for_plan(&plan);
        let prepared = test_prepared_for_plan(&plan);
        let execution = EmbeddingTaskReport {
            rows: 1,
            batches: 1,
            attempts: 1,
            retries: 0,
            unique_input_text_bytes: 20,
            sent_input_text_bytes: 20,
            vector_bytes: 16,
            shard_bytes: 80,
            elapsed_micros: 1_000_000,
            request_elapsed_micros: 900_000,
            retry_backoff_micros: 0,
            peak_in_flight: 1,
            batch_reports: vec![rag_embedding::EmbeddingBatchReport {
                batch_ordinal: 0,
                row_start: 0,
                row_end: 1,
                input_text_bytes: 20,
                vector_bytes: 16,
                elapsed_micros: 900_000,
                backoff_micros: 0,
                attempts: vec![rag_embedding::EmbeddingAttemptReport {
                    attempt: 1,
                    input_rows: 1,
                    input_text_bytes: 20,
                    vector_bytes: 16,
                    elapsed_micros: 900_000,
                    backoff_micros: 0,
                    outcome: rag_embedding::EmbeddingAttemptOutcome::Success,
                }],
            }],
        };
        let root = tempfile::tempdir().unwrap();
        for directory in ["parts", "receipts", "reports"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let task = &plan.tasks[0];
        let vector_path = task.result_path.join_to(root.path());
        let receipt_path = task.receipt_path.join_to(root.path());
        fs::write(&vector_path, vec![0; 80]).unwrap();
        fs::write(&receipt_path, b"{}").unwrap();
        let report_path = task_report_path(root.path(), task).unwrap();
        let profile = test_compact_profile();
        let run_context = LocalRunContext::observe();
        ensure_task_report(
            &report_path,
            0,
            task,
            &plan,
            TaskReportBindings {
                prepared: &prepared,
                profile: &profile,
                receipt: &receipt,
                vector_path: &vector_path,
                receipt_path: &receipt_path,
                run_context: &run_context,
                batch_size: 16,
                requests_in_flight: 1,
            },
            TaskRunDetails {
                outcome: TaskRunOutcome::Executed,
                started_unix_ms: Some(1_000),
                finished_unix_ms: Some(2_000),
                execution: Some(execution),
            },
        )
        .unwrap();
        let report: BuilderEmbeddingTaskReport = read_json(&report_path).unwrap();
        let summary = embedding_run_summary_v1(
            &plan,
            &prepared,
            &[receipt],
            &[report],
            &[7],
            empty_run_artifact_sizes(),
        )
        .unwrap();
        assert_eq!(summary.documents, 1);
        assert_eq!(summary.tokens, 7);
        assert_eq!(summary.calendar_span_micros, Some(1_000_000));
        assert_eq!(summary.wall_time_micros, Some(1_000_000));
        assert_eq!(summary.documents_per_second, Some(1.0));
        assert_eq!(summary.tokens_per_second, Some(7.0));
        assert_eq!(summary.request_latency_micros.p50, Some(900_000));
        assert_eq!(summary.length_bucket_throughput[0].documents, 1);
        let encoded = serde_json::to_string(&summary).unwrap();
        assert!(!encoded.contains("http"));
        assert!(!encoded.contains("semantic_text"));

        let reject = |tampered: &EmbeddingRunSummary| {
            assert_ne!(tampered, &summary);
        };
        let mut tampered = summary.clone();
        tampered.requests += 1;
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.retries += 1;
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.tokens += 1;
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.calendar_span_micros = Some(2_000_000);
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.documents_per_second = Some(99.0);
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.request_latency_micros.p50 = Some(1);
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.machine.cpu_model = Some("tampered".into());
        reject(&tampered);
        let mut tampered = summary.clone();
        tampered.artifact_sizes.vector_shards_bytes = Some(1);
        reject(&tampered);
    }

    #[test]
    fn summary_time_separates_idle_gaps_and_merges_overlapping_workers() {
        let gap_reports = vec![
            test_builder_task_report(1_000, 3_000),
            test_builder_task_report(5_000, 6_000),
        ];
        assert_eq!(
            execution_time_bounds(&gap_reports, Some(3_000_000)).unwrap(),
            (Some(5_000_000), Some(3_000_000))
        );

        let overlap_reports = vec![
            test_builder_task_report(1_000, 3_000),
            test_builder_task_report(2_000, 4_000),
        ];
        assert_eq!(
            execution_time_bounds(&overlap_reports, Some(4_000_000)).unwrap(),
            (Some(3_000_000), Some(3_000_000))
        );
    }

    #[test]
    fn summary_rejects_mixed_execution_provenance() {
        let first = test_builder_task_report(1_000, 2_000);
        let mut second = test_builder_task_report(2_000, 3_000);
        second.git.working_tree_dirty = Some(!first.git.working_tree_dirty.unwrap_or(false));
        assert!(homogeneous_report_context(&[first, second]).is_err());
    }

    #[test]
    fn v2_accepts_mixed_machines_with_one_sealed_execution() {
        let first = test_builder_task_report_v2(1_000, 2_000);
        let mut second = test_builder_task_report_v2(2_000, 3_000);
        second.machine.cpu_model = Some("different host CPU".into());
        second.accelerator.machine_id = Some("machine-b".into());
        assert_eq!(
            homogeneous_execution_identity_v2(&[first.clone(), second])
                .unwrap()
                .clone(),
            first.execution_identity
        );
    }

    #[test]
    fn v2_rejects_mixed_accelerator_class() {
        let first = test_builder_task_report_v2(1_000, 2_000);
        let mut second = test_builder_task_report_v2(2_000, 3_000);
        second.accelerator.model = Some("NVIDIA A10".into());
        assert!(homogeneous_execution_identity_v2(&[first, second]).is_err());
    }

    #[test]
    fn v2_rejects_mixed_execution_identity() {
        let first = test_builder_task_report_v2(1_000, 2_000);
        let mut second = test_builder_task_report_v2(2_000, 3_000);
        second.execution_identity.worker_binary.sha256 = Digest::new("9".repeat(64)).unwrap();
        assert!(homogeneous_execution_identity_v2(&[first, second]).is_err());
    }

    #[test]
    fn tei_reuse_requires_a_v2_report_with_the_same_sealed_execution() {
        let report = test_builder_task_report_v2(1_000, 2_000);
        let identity = report.execution_identity.clone();
        assert!(reusable_tei_report(
            &ValidatedEmbeddingTaskReport::V2(Box::new(report.clone())),
            &identity,
        ));
        let mut other = identity.clone();
        other.worker_binary.sha256 = Digest::new("9".repeat(64)).unwrap();
        assert!(!reusable_tei_report(
            &ValidatedEmbeddingTaskReport::V2(Box::new(report)),
            &other,
        ));
        assert!(!reusable_tei_report(
            &ValidatedEmbeddingTaskReport::V1(Box::new(test_builder_task_report(1_000, 2_000))),
            &identity,
        ));
    }

    #[test]
    fn v2_task_reports_finalize_to_a_v2_summary() {
        let plan = valid_test_embedding_plan();
        let receipt = test_receipt_for_plan(&plan);
        let prepared = test_prepared_for_plan(&plan);
        let mut report = test_builder_task_report_v2(1_000, 2_000);
        report.task_id = plan.tasks[0].task_id.clone();
        let summary = embedding_run_summary(
            &plan,
            &prepared,
            &[receipt],
            &[ValidatedEmbeddingTaskReport::V2(Box::new(report))],
            &[7],
            empty_run_artifact_sizes(),
        )
        .unwrap();
        let EmbeddingRunSummaryContract::V2(summary) = summary else {
            panic!("v2 reports produced a v1 summary");
        };
        assert_eq!(
            summary.schema_version,
            "livefire.rag.embedding-run-summary/2"
        );
        assert_eq!(summary.aggregate.tasks, 1);
        assert_eq!(summary.workers.len(), 1);
    }

    #[test]
    fn v1_task_report_wire_contract_still_decodes_unchanged() {
        let report = test_builder_task_report(1_000, 2_000);
        let original = serde_json::to_value(&report).unwrap();
        let decoded = decode_task_report(original.clone()).unwrap();
        let ValidatedEmbeddingTaskReport::V1(decoded) = decoded else {
            panic!("v1 report decoded as another version");
        };
        assert_eq!(serde_json::to_value(decoded).unwrap(), original);
    }

    #[test]
    fn tei_profile_bytes_cannot_be_relabelled_with_another_profile_digest() {
        let bytes = br#"{"schema_version":"livefire.rag.embedding-policy/3"}"#;
        let wrong_profile = "a".repeat(64);
        assert_ne!(sha256_bytes(bytes), wrong_profile);
        assert!(parse_bound_portable_profile(bytes, &wrong_profile).is_err());
    }

    #[test]
    fn recovery_quarantines_and_explicitly_restores_a_valid_orphan_part() {
        let root = tempfile::tempdir().unwrap();
        for directory in ["parts", "receipts", "reports"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let plan = test_embedding_plan();
        let task = &plan.tasks[0];
        let part_path = task.result_path.join_to(root.path());
        write_test_part(&part_path, task);
        let runtime = test_receipt_for_plan(&plan).executor.runtime;
        let inspection = inspect_embedding_task_artifacts(
            root.path(),
            &plan,
            0,
            task,
            &test_compact_profile(),
            &runtime,
        )
        .unwrap();
        assert_eq!(inspection.part, RecoveryArtifactState::Orphan);
        assert_eq!(inspection.receipt, RecoveryArtifactState::Absent);
        let mut changed = Vec::new();
        quarantine_embedding_task_artifacts(
            root.path(),
            task,
            &test_compact_profile(),
            &inspection,
            &mut changed,
        )
        .unwrap();
        assert!(!part_path.exists());
        assert!(quarantine_path(&part_path).unwrap().exists());
        restore_embedding_task_artifacts(root.path(), task, &mut changed).unwrap();
        assert!(part_path.exists());
        assert!(!quarantine_path(&part_path).unwrap().exists());
    }

    #[test]
    fn recovery_preserves_corrupt_and_wrong_plan_artifacts_in_quarantine() {
        let root = tempfile::tempdir().unwrap();
        for directory in ["parts", "receipts", "reports"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let plan = valid_test_embedding_plan();
        let task = &plan.tasks[0];
        let part_path = task.result_path.join_to(root.path());
        fs::write(&part_path, b"corrupt vector bytes").unwrap();
        let mut wrong_plan_receipt = test_receipt_for_plan(&plan);
        wrong_plan_receipt.plan_sha256 = Digest::new("9".repeat(64)).unwrap();
        wrong_plan_receipt.seal().unwrap();
        let receipt_path = task.receipt_path.join_to(root.path());
        write_canonical_json(&receipt_path, &wrong_plan_receipt).unwrap();
        let report_path = task_report_path(root.path(), task).unwrap();
        fs::write(&report_path, b"corrupt report bytes").unwrap();
        let runtime = wrong_plan_receipt.executor.runtime.clone();
        let inspection = inspect_embedding_task_artifacts(
            root.path(),
            &plan,
            0,
            task,
            &test_compact_profile(),
            &runtime,
        )
        .unwrap();
        assert_eq!(inspection.part, RecoveryArtifactState::Invalid);
        assert_eq!(inspection.receipt, RecoveryArtifactState::Invalid);
        assert_eq!(inspection.report, RecoveryArtifactState::Orphan);
        let mut changed = Vec::new();
        quarantine_embedding_task_artifacts(
            root.path(),
            task,
            &test_compact_profile(),
            &inspection,
            &mut changed,
        )
        .unwrap();
        for path in [&part_path, &receipt_path, &report_path] {
            assert!(!path.exists());
            assert!(quarantine_path(path).unwrap().exists());
        }
    }

    #[test]
    fn crash_recovery_removes_quarantine_only_after_replacements_exist() {
        let root = tempfile::tempdir().unwrap();
        let receipt_path = root.path().join("receipt.json");
        fs::write(&receipt_path, b"invalid old receipt").unwrap();
        quarantine_regular_file(&receipt_path).unwrap();
        assert!(complete_regular_file_recovery(&receipt_path).is_err());
        assert!(quarantine_path(&receipt_path).unwrap().exists());
        fs::write(&receipt_path, b"validated replacement receipt").unwrap();
        complete_regular_file_recovery(&receipt_path).unwrap();
        assert_eq!(
            fs::read(&receipt_path).unwrap(),
            b"validated replacement receipt"
        );
        assert!(!quarantine_path(&receipt_path).unwrap().exists());
    }
}
