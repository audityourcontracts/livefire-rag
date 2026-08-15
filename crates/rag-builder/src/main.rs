use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use arrow_array::{Array, RecordBatch, StringArray};
use chrono::DateTime;
use clap::{Parser, Subcommand, ValueEnum};
use rag_embedding::{
    EmbeddingCache, EmbeddingInput, LmStudioEmbedder, adapt_model_vector, ensure_cached,
    parse_embedding_profile, try_compose_query,
};
use rag_index::{
    BuildScope, FastDocument, FastIndex, FastOccurrence, IndexError, SearchFilters, SearchMode,
    SourceBinding, document_order_sha256, write_fast_index_streaming,
};
use rag_ocsf::{LocalSnapshotReader, SnapshotReader};
use rag_projection::{
    ComponentRef, EventTimeAvailability, ProjectedDocument, ProjectionContext, ProjectionInput,
    project, project_document_summary, project_event_time,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod catalogue;
mod portable;
mod report;

use report::{
    GitState, LmStudioContext, LocalRunContext, MachineContext, ObservationStatus,
    QueryArtifactSizes, ResourceUsage, TransportByteAccounting, query_artifact_sizes,
};

#[derive(Parser)]
#[command(name = "rag", about = "Fast experimental RAG builder and query CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build, validate, or search a catalogue of completed dataset indexes.
    Catalogue {
        #[command(subcommand)]
        command: CatalogueCommand,
    },
    /// Count current Rust projection documents and dispositions without
    /// embedding or assembling an index.
    Census {
        #[arg(long)]
        snapshot: PathBuf,
        /// Count only these typed relations. Repeat the option as needed. An
        /// empty list counts every admitted typed relation.
        #[arg(long)]
        relation: Vec<String>,
        /// Optionally write the canonical report to this path. The report is
        /// always printed to stdout as well.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Maximum number of Parquet row groups processed at once. The
        /// default uses no more than eight available CPU threads.
        #[arg(long, default_value_t = portable::default_census_workers())]
        workers: usize,
    },
    /// Verify a frozen tokenizer against token IDs captured from the exact
    /// llama.cpp GGUF model. This does not contact a model server.
    VerifyTokenizer {
        #[arg(long)]
        tokenizer_json: PathBuf,
        #[arg(long)]
        tokenizer_ref: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
    },
    /// Project one dataset into reusable, model-independent Parquet shards.
    Prepare {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        dataset_id: String,
        #[arg(long, default_value = "1")]
        dataset_version: String,
        /// Include exactly these typed OCSF relations. Repeat the option for
        /// additional relations.
        #[arg(long, required = true)]
        relation: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 2_048)]
        document_shard_rows: usize,
        /// Maximum number of admitted Parquet row groups projected at once.
        /// The default uses no more than eight available CPU threads.
        #[arg(long, default_value_t = portable::default_prepare_workers())]
        workers: usize,
    },
    /// Re-open a prepared dataset and verify every document and occurrence
    /// object without changing it.
    VerifyPrepared {
        #[arg(long)]
        prepared: PathBuf,
    },
    /// Build the fixed nested 512, 2,000, and 10,000 document performance
    /// corpora directly from an admitted snapshot.
    PrepareBenchmark {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        dataset_id: String,
        #[arg(long, default_value = "1")]
        dataset_version: String,
        /// Include exactly these typed OCSF relations. Repeat the option for
        /// additional relations.
        #[arg(long, required = true)]
        relation: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 2_048)]
        document_shard_rows: usize,
        /// Public, non-secret text included in the selection identity. Keep
        /// this unchanged when comparing embedding backends.
        #[arg(long, default_value = "local-scale-benchmark-v1")]
        selection_seed: String,
        /// Maximum number of admitted Parquet row groups projected at once.
        /// The default uses no more than eight available CPU threads.
        #[arg(long, default_value_t = portable::default_prepare_workers())]
        workers: usize,
    },
    /// Freeze model-specific embedding tasks over a prepared dataset.
    PlanEmbeddings {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        /// Frozen Hugging Face tokenizer.json used to count the exact model
        /// inputs before any embedding requests are made.
        #[arg(long)]
        tokenizer_json: PathBuf,
        /// Tracked JSON reference that pins the tokenizer identity, source and
        /// model revisions, byte digest, special-token behavior, and byte cap.
        #[arg(long)]
        tokenizer_ref: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        maximum_task_tokens: u64,
        #[arg(long)]
        maximum_task_documents: u32,
    },
    /// Execute unfinished embedding tasks against local LM Studio.
    Embed {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        requests_in_flight: usize,
        /// Execute only this start-inclusive, end-exclusive task range, for
        /// example `0..8`. Omit it to execute every task.
        #[arg(long)]
        task_range: Option<String>,
    },
    /// Write deterministic 4096-dimensional unit vectors for artifact-chain
    /// tests without contacting LM Studio. Every result and assembled index
    /// is marked test-only and refused by normal query/provider paths.
    TestEmbed {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Validate every planned embedding part and receipt, then publish the
    /// complete result-set manifest required by assembly.
    FinalizeEmbeddings {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
    },
    /// Create a separately identified 2,048- or 1,024-value vector set from
    /// completed 4,096-value model output without contacting the model.
    DeriveEmbeddings {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
        #[arg(long)]
        dimensions: u32,
        /// Output bundle containing `plan`, `results`, and the new profile.
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify, quarantine, or explicitly restore one planned embedding task's
    /// local artifacts without contacting an embedding model.
    RecoverEmbeddingTask {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
        #[arg(long)]
        task_id: String,
        #[arg(long, value_enum)]
        action: RecoveryAction,
    },
    /// Assemble one prepared dataset and complete embedding set into an index.
    Assemble {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        embeddings: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Lexical storage format. Version 2 remains the default for existing
        /// packaged-tool workflows; choose `sqlite-v3` for scalable indexes.
        #[arg(long, value_enum, default_value_t = IndexFormat::LegacyJsonV2)]
        index_format: IndexFormat,
    },
    Build {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        embedding_profile: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        resume: PathBuf,
        #[arg(long, default_value_t = 16)]
        embedding_batch_size: usize,
        /// Build the fixed scenario-blind representative sample: census at
        /// <=1000, then a 2000-document snapshot-bound hash-min cap per larger
        /// searchable relation (effective full retention through 2000).
        #[arg(long)]
        representative_sample: bool,
    },
    Query {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long,value_enum,default_value_t=Mode::Fused)]
        mode: Mode,
        #[arg(long, default_value_t = 20)]
        top_n: usize,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        relation: Vec<String>,
    },
    /// Measure query embedding separately from index-only dense, fused, and
    /// lexical search. The report never includes the query text.
    BenchmarkQuery {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long)]
        query_id: String,
        #[arg(long, default_value_t = 20)]
        top_n: usize,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        relation: Vec<String>,
        #[arg(long, default_value_t = 3)]
        warmups: usize,
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Untimed model-only query embedding calls before measurement.
        #[arg(long, default_value_t = 1)]
        embedding_warmups: usize,
        /// Timed model-only query embedding calls. One returned vector is then
        /// reused for every index-only measurement.
        #[arg(long, default_value_t = 5)]
        embedding_repeats: usize,
        /// Also time this many complete compose, embed, bind, and fused-search
        /// calls. Zero avoids additional model requests.
        #[arg(long, default_value_t = 0)]
        end_to_end_repeats: usize,
    },
    /// Compare the top results from a 4,096-value index and an index derived
    /// from it. The query is embedded once, then transformed locally.
    CompareIndexOverlap {
        #[arg(long)]
        full_index: PathBuf,
        #[arg(long)]
        reduced_index: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 20)]
        top_n: usize,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        relation: Vec<String>,
    },
    /// Execute a frozen JSONL query plan while opening and hashing the index
    /// only once. Results are emitted as JSONL in request order.
    BatchQuery {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        requests: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
    },
    Inspect {
        #[arg(long)]
        index: PathBuf,
        /// Permit opening a deterministic test-vector index for diagnostics.
        #[arg(long)]
        allow_test_only: bool,
    },
}

#[derive(Subcommand)]
enum CatalogueCommand {
    /// Build a sealed catalogue from completed dataset artifact chains.
    Build {
        /// Supply PREPARED, PLAN, RESULTS, and INDEX paths for each dataset.
        /// Repeat this option to add another dataset.
        #[arg(long, num_args = 4, value_names = ["PREPARED", "PLAN", "RESULTS", "INDEX"], required = true)]
        dataset: Vec<PathBuf>,
        /// Permit one relation overlap as RELATION=REASON. Repeat as needed.
        #[arg(long)]
        allow_relation_overlap: Vec<String>,
        /// Mark every entry and the catalogue as synthetic test data.
        #[arg(long)]
        test_only: bool,
        /// Catalogue JSON path. All supplied artifacts must be below its
        /// parent directory so stored paths remain safe and portable.
        #[arg(long)]
        out: PathBuf,
    },
    /// Re-open every referenced artifact and verify the complete bindings.
    Validate {
        #[arg(long)]
        catalogue: PathBuf,
    },
    /// Search every compatible dataset index and merge stable per-index ranks.
    Search {
        #[arg(long)]
        catalogue: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, value_enum, default_value_t = Mode::Fused)]
        mode: Mode,
        #[arg(long, default_value_t = 20)]
        top_n: usize,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        #[arg(long)]
        relation: Vec<String>,
        /// Maximum number of indexes searched at once.
        #[arg(long, default_value_t = portable::default_prepare_workers())]
        workers: usize,
        /// Test-only catalogues require this explicit acknowledgement.
        #[arg(long)]
        allow_test_only: bool,
    },
    /// Execute a frozen JSONL query plan and atomically publish a complete run
    /// directory after every request succeeds.
    BatchSearch {
        #[arg(long)]
        catalogue: PathBuf,
        #[arg(long)]
        requests: PathBuf,
        /// New run directory containing requests.jsonl, results.jsonl, and
        /// manifest.json. Existing paths are refused.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "http://127.0.0.1:1234")]
        embedding_endpoint: String,
        /// Maximum number of indexes searched at once for each request.
        #[arg(long, default_value_t = portable::default_prepare_workers())]
        workers: usize,
        /// Test-only catalogues require this explicit acknowledgement.
        #[arg(long)]
        allow_test_only: bool,
    },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Dense,
    Lexical,
    #[default]
    Fused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RecoveryAction {
    Verify,
    Quarantine,
    Restore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IndexFormat {
    LegacyJsonV2,
    SqliteV3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchQueryRequest {
    query_id: String,
    query: String,
    mode: Mode,
    top_n: usize,
    relations: Vec<String>,
}

struct BatchQueryPlan {
    requests: Vec<BatchQueryRequest>,
    source_sha256: rag_pipeline::Digest,
    source_bytes: u64,
}

const MAX_BATCH_QUERY_REQUESTS: usize = 10_000;
const MAX_BATCH_QUERY_ID_BYTES: usize = 128;
const MAX_BATCH_QUERY_BYTES: usize = 8_192;
const MAX_BATCH_RELATIONS: usize = 64;
const MAX_BATCH_RELATION_BYTES: usize = 256;

impl BatchQueryRequest {
    fn validate(&self) -> Result<()> {
        if self.query_id.trim().is_empty()
            || self.query_id.len() > MAX_BATCH_QUERY_ID_BYTES
            || self.query.trim().is_empty()
            || self.query.len() > MAX_BATCH_QUERY_BYTES
            || self.top_n == 0
            || self.top_n > 100
            || self.relations.len() > MAX_BATCH_RELATIONS
            || self.relations.iter().any(|relation| {
                relation.trim().is_empty() || relation.len() > MAX_BATCH_RELATION_BYTES
            })
            || !self
                .relations
                .windows(2)
                .all(|pair| pair[0].as_str() < pair[1].as_str())
        {
            return Err(Error::InvalidBatchQuery);
        }
        Ok(())
    }
}
impl From<Mode> for SearchMode {
    fn from(value: Mode) -> Self {
        match value {
            Mode::Dense => Self::Dense,
            Mode::Lexical => Self::Lexical,
            Mode::Fused => Self::Fused,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildSelection {
    Full,
    Representative,
}

impl BuildSelection {
    fn from_flag(enabled: bool) -> Self {
        if enabled {
            Self::Representative
        } else {
            Self::Full
        }
    }

    fn scope(self) -> BuildScope {
        match self {
            Self::Full => BuildScope::Full,
            Self::Representative => BuildScope::Sample,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
struct TimestampAccounting {
    available: u64,
    indexed_milliseconds: u64,
    before_unix_epoch: u64,
    missing: u64,
    present_unparsed: u64,
}

impl TimestampAccounting {
    fn observe(&mut self, availability: EventTimeAvailability, event_time_ms: Option<u64>) {
        match availability {
            EventTimeAvailability::Available => {
                self.available += 1;
                if event_time_ms.is_some() {
                    self.indexed_milliseconds += 1;
                } else {
                    self.before_unix_epoch += 1;
                }
            }
            EventTimeAvailability::Missing => self.missing += 1,
            EventTimeAvailability::PresentUnparsed => self.present_unparsed += 1,
        }
    }

    fn total(&self) -> u64 {
        self.available + self.missing + self.present_unparsed
    }
}

#[derive(Debug, Serialize)]
struct SamplingAccounting {
    policy: &'static str,
    declared_census_at_or_below: Option<usize>,
    hash_min_cap_above_census_threshold: Option<usize>,
    effective_full_retention_at_or_below: Option<usize>,
    max_documents_per_searchable_relation: Option<usize>,
    source_documents_by_relation: BTreeMap<String, usize>,
    relation_budgets: BTreeMap<String, usize>,
    selected_by_relation: BTreeMap<String, usize>,
    selected_documents: usize,
    selected_occurrences: usize,
    selected_document_order_sha256: String,
}

#[derive(Debug, Serialize)]
struct BuildAccounting {
    coverage_semantics: &'static str,
    semantic_source_coverage_complete: bool,
    source_rows_scanned: u64,
    source_rows_by_relation: BTreeMap<String, u64>,
    materialization_rows_scanned: u64,
    projected_semantic_occurrences: u64,
    structured_only_occurrences: u64,
    structured_only_by_relation: BTreeMap<String, u64>,
    source_timestamps: TimestampAccounting,
    indexed_timestamps: TimestampAccounting,
    sampling: SamplingAccounting,
}

#[derive(Debug)]
struct SelectedDocument {
    document: FastDocument,
    primary_relation: String,
    relations: BTreeSet<String>,
}

/// First-pass document census and relation-stratified hash-min sampler. A
/// document's opaque identity receives a snapshot-bound SHA-256 priority; no
/// semantic value, query, label, or qrel is inspected. The second source scan
/// materializes occurrences only for the final selected document set, so
/// high-fanout membership is never accumulated during sampling.
#[derive(Debug)]
struct DocumentCollector {
    selection: BuildSelection,
    sample_seed: String,
    selected: BTreeMap<String, SelectedDocument>,
    /// Ordered by `(priority SHA-256, document ID)` so the greatest item is
    /// the deterministic eviction candidate.
    priorities: BTreeMap<String, BTreeSet<(String, String)>>,
    seen_by_relation: BTreeMap<String, BTreeSet<String>>,
    source_rows_scanned: u64,
    source_rows_by_relation: BTreeMap<String, u64>,
    structured_only_by_relation: BTreeMap<String, u64>,
    projected_semantic_occurrences: u64,
    source_timestamps: TimestampAccounting,
}

impl DocumentCollector {
    fn new(selection: BuildSelection, snapshot_sha256: &str, relations: &[String]) -> Self {
        Self {
            selection,
            sample_seed: snapshot_sha256.to_owned(),
            selected: BTreeMap::new(),
            priorities: BTreeMap::new(),
            seen_by_relation: relations
                .iter()
                .map(|relation| (relation.clone(), BTreeSet::new()))
                .collect(),
            source_rows_scanned: 0,
            source_rows_by_relation: BTreeMap::new(),
            structured_only_by_relation: BTreeMap::new(),
            projected_semantic_occurrences: 0,
            source_timestamps: TimestampAccounting::default(),
        }
    }

    fn observe_source_row(
        &mut self,
        relation: &str,
        has_semantic_document: bool,
        availability: EventTimeAvailability,
        event_time_ms: Option<u64>,
    ) {
        self.source_rows_scanned += 1;
        *self
            .source_rows_by_relation
            .entry(relation.to_owned())
            .or_default() += 1;
        if !has_semantic_document {
            *self
                .structured_only_by_relation
                .entry(relation.to_owned())
                .or_default() += 1;
        }
        self.source_timestamps.observe(availability, event_time_ms);
    }

    fn retain(&mut self, document: FastDocument, relation: &str) -> Result<()> {
        self.projected_semantic_occurrences += 1;
        self.seen_by_relation
            .entry(relation.to_owned())
            .or_default()
            .insert(document.document_id.clone());
        if let Some(selected) = self.selected.get_mut(&document.document_id) {
            if selected.primary_relation != relation {
                return Err(Error::InconsistentDocument(document.document_id));
            }
            if selected.document.document_sha256 != document.document_sha256
                || selected.document.semantic_text != document.semantic_text
                || selected.document.document_kind != document.document_kind
                || selected.document.facets_json != document.facets_json
            {
                return Err(Error::InconsistentDocument(document.document_id));
            }
            selected.relations.insert(relation.to_owned());
            return Ok(());
        }

        let priority = sample_priority(&self.sample_seed, &document.document_id);
        if let BuildSelection::Representative = self.selection {
            let priorities = self.priorities.entry(relation.to_owned()).or_default();
            if priorities.len() == REPRESENTATIVE_RELATION_CAP {
                let worst = priorities
                    .last()
                    .expect("a full relation sample has a priority")
                    .clone();
                let candidate = (priority.clone(), document.document_id.clone());
                if candidate >= worst {
                    return Ok(());
                }
                priorities.remove(&worst);
                self.selected.remove(&worst.1);
            }
            priorities.insert((priority, document.document_id.clone()));
        }

        let mut relations = BTreeSet::new();
        relations.insert(relation.to_owned());
        self.selected.insert(
            document.document_id.clone(),
            SelectedDocument {
                document,
                primary_relation: relation.to_owned(),
                relations,
            },
        );
        Ok(())
    }

    fn finish(self) -> Result<(Vec<FastDocument>, FirstPassAccounting)> {
        let mut documents = Vec::with_capacity(self.selected.len());
        let mut selected_by_relation = BTreeMap::<String, usize>::new();
        for mut selected in self.selected.into_values() {
            *selected_by_relation
                .entry(selected.primary_relation.clone())
                .or_default() += 1;
            selected.document.relations_json = serde_json::to_string(&selected.relations)?;
            documents.push(selected.document);
        }
        documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        for (ordinal, document) in documents.iter_mut().enumerate() {
            document.vector_ordinal = ordinal as u64;
        }
        let source_documents_by_relation = self
            .seen_by_relation
            .into_iter()
            .map(|(relation, documents)| (relation, documents.len()))
            .collect::<BTreeMap<_, _>>();
        let relation_budgets = source_documents_by_relation
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(relation, count)| {
                let budget = match self.selection {
                    BuildSelection::Full => *count,
                    BuildSelection::Representative => (*count).min(REPRESENTATIVE_RELATION_CAP),
                };
                (relation.clone(), budget)
            })
            .collect::<BTreeMap<_, _>>();
        if selected_by_relation != relation_budgets {
            return Err(Error::AccountingClosure(
                "selected relation counts do not match relation budgets",
            ));
        }
        let structured_only_occurrences = self
            .structured_only_by_relation
            .values()
            .copied()
            .sum::<u64>();
        if self.source_rows_scanned
            != self
                .projected_semantic_occurrences
                .checked_add(structured_only_occurrences)
                .ok_or(Error::CountOverflow)?
            || self.source_rows_scanned
                != self.source_rows_by_relation.values().copied().sum::<u64>()
        {
            return Err(Error::AccountingClosure(
                "source rows do not reconcile to semantic and structured-only rows",
            ));
        }
        let accounting = FirstPassAccounting {
            source_rows_scanned: self.source_rows_scanned,
            source_rows_by_relation: self.source_rows_by_relation,
            structured_only_by_relation: self.structured_only_by_relation,
            projected_semantic_occurrences: self.projected_semantic_occurrences,
            source_timestamps: self.source_timestamps,
            source_documents_by_relation,
            relation_budgets,
            selected_by_relation,
        };
        debug_assert_eq!(
            accounting.source_timestamps.total(),
            self.source_rows_scanned
        );
        Ok((documents, accounting))
    }
}

const REPRESENTATIVE_RELATION_CENSUS_THRESHOLD: usize = 1_000;
const REPRESENTATIVE_RELATION_CAP: usize = 2_000;

#[derive(Debug)]
struct FirstPassAccounting {
    source_rows_scanned: u64,
    source_rows_by_relation: BTreeMap<String, u64>,
    structured_only_by_relation: BTreeMap<String, u64>,
    projected_semantic_occurrences: u64,
    source_timestamps: TimestampAccounting,
    source_documents_by_relation: BTreeMap<String, usize>,
    relation_budgets: BTreeMap<String, usize>,
    selected_by_relation: BTreeMap<String, usize>,
}

fn sample_priority(snapshot_sha256: &str, document_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"livefire-rag-scenario-blind-document-sample-v1\0");
    hasher.update(snapshot_sha256.as_bytes());
    hasher.update([0]);
    hasher.update(document_id.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    Ocsf(#[from] rag_ocsf::OcsfError),
    #[error(transparent)]
    Projection(#[from] rag_projection::ProjectionError),
    #[error(transparent)]
    Embedding(#[from] rag_embedding::EmbeddingError),
    #[error(transparent)]
    Index(#[from] rag_index::IndexError),
    #[error(transparent)]
    Pipeline(#[from] rag_pipeline::PipelineError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid typed relation column: {0}")]
    Column(&'static str),
    #[error("invalid event timestamp emitted by projection: {0}")]
    Timestamp(String),
    #[error("document {0} changed content within one source snapshot")]
    InconsistentDocument(String),
    #[error("selected occurrence materialization did not close")]
    OccurrenceClosure,
    #[error("build count overflow")]
    CountOverflow,
    #[error("build accounting closure failed: {0}")]
    AccountingClosure(&'static str),
    #[error("{0}")]
    UnsupportedPlanVersion(String),
    #[error("invalid task range; use START..END with START < END")]
    InvalidTaskRange,
    #[error("batch query plan contains an invalid row")]
    InvalidBatchQuery,
    #[error("query embedding response does not match the index embedding profile")]
    InvalidQueryEmbeddingResponse,
    #[error("query benchmark options, model response, or repeated results are invalid")]
    InvalidQueryBenchmark,
    #[error("embedding task ID is not present in the selected plan")]
    UnknownEmbeddingTask,
    #[error("embedding task artifacts are incomplete or invalid")]
    EmbeddingTaskIncomplete,
}
type Result<T> = std::result::Result<T, Error>;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("rag: {error}");
        std::process::exit(1)
    }
}
async fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Catalogue { command } => match command {
            CatalogueCommand::Build {
                dataset,
                allow_relation_overlap,
                test_only,
                out,
            } => catalogue::build_catalogue(catalogue::CatalogueBuildOptions {
                dataset_paths: dataset,
                overlap_allowances: allow_relation_overlap,
                test_only,
                out,
            }),
            CatalogueCommand::Validate { catalogue } => catalogue::validate_catalogue(&catalogue),
            CatalogueCommand::Search {
                catalogue,
                query,
                mode,
                top_n,
                embedding_endpoint,
                relation,
                workers,
                allow_test_only,
            } => {
                catalogue::search_catalogue(catalogue::CatalogueSearchOptions {
                    catalogue: &catalogue,
                    query: &query,
                    mode,
                    top_n,
                    endpoint: &embedding_endpoint,
                    relations: relation,
                    workers,
                    allow_test_only,
                })
                .await
            }
            CatalogueCommand::BatchSearch {
                catalogue,
                requests,
                out,
                embedding_endpoint,
                workers,
                allow_test_only,
            } => {
                catalogue::batch_search_catalogue(catalogue::CatalogueBatchSearchOptions {
                    catalogue: &catalogue,
                    requests: &requests,
                    out: &out,
                    endpoint: &embedding_endpoint,
                    workers,
                    allow_test_only,
                })
                .await
            }
        },
        Command::Census {
            snapshot,
            relation,
            out,
            workers,
        } => portable::census(portable::CensusOptions {
            snapshot,
            relations: relation,
            out,
            workers,
        }),
        Command::VerifyTokenizer {
            tokenizer_json,
            tokenizer_ref,
            fixture,
        } => portable::verify_tokenizer(portable::VerifyTokenizerOptions {
            tokenizer_json,
            tokenizer_ref,
            fixture,
        }),
        Command::Prepare {
            snapshot,
            dataset_id,
            dataset_version,
            relation,
            out,
            document_shard_rows,
            workers,
        } => portable::prepare(portable::PrepareOptions {
            snapshot,
            dataset_id,
            dataset_version,
            relations: relation,
            out,
            document_shard_rows,
            workers,
        }),
        Command::PrepareBenchmark {
            snapshot,
            dataset_id,
            dataset_version,
            relation,
            out,
            document_shard_rows,
            selection_seed,
            workers,
        } => portable::prepare_benchmark(portable::PrepareBenchmarkOptions {
            snapshot,
            dataset_id,
            dataset_version,
            relations: relation,
            out,
            document_shard_rows,
            selection_seed,
            workers,
        }),
        Command::VerifyPrepared { prepared } => portable::verify_prepared(&prepared),
        Command::PlanEmbeddings {
            prepared,
            embedding_profile,
            tokenizer_json,
            tokenizer_ref,
            out,
            maximum_task_tokens,
            maximum_task_documents,
        } => portable::plan_embeddings(portable::PlanOptions {
            prepared,
            embedding_profile,
            tokenizer_json,
            tokenizer_ref,
            out,
            maximum_task_tokens,
            maximum_task_documents,
        }),
        Command::Embed {
            prepared,
            plan,
            embedding_profile,
            embedding_endpoint,
            out,
            batch_size,
            requests_in_flight,
            task_range,
        } => {
            portable::embed(portable::EmbedOptions {
                prepared,
                plan,
                embedding_profile,
                embedding_endpoint,
                out,
                batch_size,
                requests_in_flight,
                task_range,
            })
            .await
        }
        Command::TestEmbed {
            prepared,
            plan,
            embedding_profile,
            out,
        } => portable::test_embed(portable::TestEmbedOptions {
            prepared,
            plan,
            embedding_profile,
            out,
        }),
        Command::FinalizeEmbeddings {
            prepared,
            plan,
            embedding_profile,
            embeddings,
        } => portable::finalize_embeddings(portable::FinalizeOptions {
            prepared,
            plan,
            embedding_profile,
            embeddings,
        }),
        Command::DeriveEmbeddings {
            prepared,
            plan,
            embedding_profile,
            embeddings,
            dimensions,
            out,
        } => portable::derive_embeddings(portable::DeriveEmbeddingsOptions {
            prepared,
            plan,
            embedding_profile,
            embeddings,
            dimensions,
            out,
        }),
        Command::RecoverEmbeddingTask {
            plan,
            embedding_profile,
            embeddings,
            task_id,
            action,
        } => portable::recover_embedding_task(portable::RecoveryOptions {
            plan,
            embedding_profile,
            embeddings,
            task_id,
            action,
        }),
        Command::Assemble {
            prepared,
            plan,
            embeddings,
            embedding_profile,
            out,
            index_format,
        } => portable::assemble(portable::AssembleOptions {
            prepared,
            plan,
            embeddings,
            embedding_profile,
            out,
            index_format,
        }),
        Command::Build {
            snapshot,
            out,
            embedding_profile,
            embedding_endpoint,
            resume,
            embedding_batch_size,
            representative_sample,
        } => {
            build(
                &snapshot,
                &out,
                &embedding_profile,
                &embedding_endpoint,
                &resume,
                embedding_batch_size,
                BuildSelection::from_flag(representative_sample),
            )
            .await
        }
        Command::Query {
            index,
            query: query_text,
            mode,
            top_n,
            embedding_endpoint,
            relation,
        } => {
            query(
                &index,
                &query_text,
                mode,
                top_n,
                &embedding_endpoint,
                relation,
            )
            .await
        }
        Command::BenchmarkQuery {
            index,
            query,
            query_id,
            top_n,
            embedding_endpoint,
            relation,
            warmups,
            repeats,
            embedding_warmups,
            embedding_repeats,
            end_to_end_repeats,
        } => {
            benchmark_query(BenchmarkQueryOptions {
                index: &index,
                query: &query,
                query_id: &query_id,
                top_n,
                endpoint: &embedding_endpoint,
                relations: relation,
                warmups,
                repeats,
                embedding_warmups,
                embedding_repeats,
                end_to_end_repeats,
            })
            .await
        }
        Command::CompareIndexOverlap {
            full_index,
            reduced_index,
            query,
            top_n,
            embedding_endpoint,
            relation,
        } => {
            compare_index_overlap(
                &full_index,
                &reduced_index,
                &query,
                top_n,
                &embedding_endpoint,
                relation,
            )
            .await
        }
        Command::BatchQuery {
            index,
            requests,
            embedding_endpoint,
        } => batch_query(&index, &requests, &embedding_endpoint).await,
        Command::Inspect {
            index,
            allow_test_only,
        } => {
            let opened = if allow_test_only {
                FastIndex::open_allow_test_only(&index)?
            } else {
                FastIndex::open(&index)?
            };
            println!("{}", serde_json::to_string_pretty(&opened.manifest)?);
            Ok(())
        }
    }
}

async fn build(
    snapshot: &Path,
    out: &Path,
    profile_path: &Path,
    endpoint: &str,
    resume: &Path,
    batch_size: usize,
    selection: BuildSelection,
) -> Result<()> {
    let reader = LocalSnapshotReader::open(snapshot)?;
    let identity = reader.identity();
    let context = ProjectionContext {
        snapshot: ComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };
    let mut relations = reader
        .typed_relations()
        .map(|relation| relation.name.clone())
        .collect::<Vec<_>>();
    relations.sort();
    let mut collector = DocumentCollector::new(selection, &context.snapshot.sha256, &relations);
    for relation in reader.typed_relations() {
        for batch in reader.scan(relation)? {
            project_batch(&batch?, &relation.name, &context, &mut collector)?;
        }
    }
    let (mut documents, first_pass) = collector.finish()?;
    let spill_parent = out.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(spill_parent)?;
    let spill = tempfile::Builder::new()
        .prefix(".rag-selected-occurrences-")
        .tempfile_in(spill_parent)?;
    let (selected_occurrences, indexed_timestamps, materialization_rows_scanned) =
        materialize_selected_occurrences(
            &reader,
            &context,
            &mut documents,
            spill.path(),
            selection,
        )?;
    let structured_only_occurrences = first_pass.structured_only_by_relation.values().sum();
    let accounting = BuildAccounting {
        coverage_semantics: "searchable_projection_only_not_source_row_coverage",
        semantic_source_coverage_complete: matches!(selection, BuildSelection::Full)
            && structured_only_occurrences == 0,
        source_rows_scanned: first_pass.source_rows_scanned,
        source_rows_by_relation: first_pass.source_rows_by_relation,
        projected_semantic_occurrences: first_pass.projected_semantic_occurrences,
        structured_only_occurrences,
        structured_only_by_relation: first_pass.structured_only_by_relation,
        source_timestamps: first_pass.source_timestamps,
        indexed_timestamps,
        materialization_rows_scanned,
        sampling: SamplingAccounting {
            policy: match selection {
                BuildSelection::Full => "full",
                BuildSelection::Representative => {
                    "relation_census_1000_snapshot_bound_sha256_hash_min_cap_2000_v1"
                }
            },
            declared_census_at_or_below: matches!(selection, BuildSelection::Representative)
                .then_some(REPRESENTATIVE_RELATION_CENSUS_THRESHOLD),
            hash_min_cap_above_census_threshold: matches!(
                selection,
                BuildSelection::Representative
            )
            .then_some(REPRESENTATIVE_RELATION_CAP),
            effective_full_retention_at_or_below: matches!(
                selection,
                BuildSelection::Representative
            )
            .then_some(REPRESENTATIVE_RELATION_CAP),
            max_documents_per_searchable_relation: matches!(
                selection,
                BuildSelection::Representative
            )
            .then_some(REPRESENTATIVE_RELATION_CAP),
            source_documents_by_relation: first_pass.source_documents_by_relation,
            relation_budgets: first_pass.relation_budgets,
            selected_by_relation: first_pass.selected_by_relation,
            selected_documents: documents.len(),
            selected_occurrences,
            selected_document_order_sha256: document_order_sha256(&documents),
        },
    };
    let profile = parse_embedding_profile(&fs::read(profile_path)?)?;
    if profile.vector_derivation.is_some() {
        return Err(Error::AccountingClosure(
            "derived profiles must use derive-embeddings, not model embedding",
        ));
    }
    let inputs = documents
        .iter()
        .map(|document| EmbeddingInput {
            document_id: document.document_id.clone(),
            document_sha256: document.document_sha256.clone(),
            text: document.semantic_text.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(parent) = resume.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut cache = EmbeddingCache::open(resume)?;
    let embedder = LmStudioEmbedder::new(endpoint, &profile.model);
    let cache_stats = ensure_cached(&embedder, &mut cache, &profile, &inputs, batch_size).await?;
    let occurrence_file = File::open(spill.path())?;
    let occurrence_rows = BufReader::new(occurrence_file).lines().map(|line| {
        let line = line.map_err(IndexError::Io)?;
        serde_json::from_str::<FastOccurrence>(&line)
            .map_err(|_| IndexError::Invalid("selected occurrence spill"))
    });
    let vector_rows = inputs.iter().map(|input| {
        cache
            .vector(&profile, input)
            .map_err(|_| IndexError::Invalid("embedding cache vector"))
    });
    let manifest = write_fast_index_streaming(
        out,
        SourceBinding {
            snapshot_sha256: identity.snapshot_sha256.to_string(),
            mapping_sha256: identity.mapping_sha256.to_string(),
        },
        selection.scope(),
        &documents,
        occurrence_rows,
        vector_rows,
        profile.clone(),
    )?;
    let report = serde_json::json!({
        "schema_version": "livefire.rag.fast-build-report/1",
        "source": manifest.source.clone(),
        "build_scope": manifest.build_scope.clone(),
        "complete": manifest.complete,
        "document_count": manifest.documents.rows,
        "occurrence_count": manifest.occurrences.rows,
        "vector_count": manifest.vectors.count,
        "embedding_profile_sha256": manifest.embedding_profile.sha256.clone(),
        "accounting": &accounting,
        "cache_hits": cache_stats.cache_hits,
        "embedded": cache_stats.embedded
    });
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    let report_staging = out.join(format!(".build-report.json.tmp-{}", std::process::id()));
    fs::write(&report_staging, report_bytes)?;
    fs::rename(report_staging, out.join("build-report.json"))?;
    let output = serde_json::json!({
        "schema_version": "livefire.rag.fast-build-output/1",
        "index": manifest,
        "accounting": accounting,
        "cache_hits": cache_stats.cache_hits,
        "embedded": cache_stats.embedded
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn project_batch(
    batch: &RecordBatch,
    relation: &str,
    context: &ProjectionContext,
    collector: &mut DocumentCollector,
) -> Result<()> {
    let json = strings(batch, "typed_event_json")?;
    if relation == "ocsf_ext_livefire_system_metric" {
        for row in 0..batch.num_rows() {
            let (event_time, availability) = project_event_time(json.value(row));
            let event_time_ms = parse_event_time_ms(event_time.as_deref(), availability)?;
            collector.observe_source_row(relation, false, availability, event_time_ms);
        }
        return Ok(());
    }
    for row in 0..batch.num_rows() {
        let output = project_document_summary(relation, json.value(row), context)?;
        let event_time_ms =
            parse_event_time_ms(output.event_time.as_deref(), output.event_time_availability)?;
        collector.observe_source_row(
            relation,
            output.document.is_some(),
            output.event_time_availability,
            event_time_ms,
        );
        if let Some(document) = output.document {
            let document_bytes = serde_json::to_vec(&document)?;
            let document_sha256 = format!("{:x}", Sha256::digest(document_bytes));
            let fast_document = fast_document(document, document_sha256)?;
            collector.retain(fast_document, relation)?;
        }
    }
    Ok(())
}

fn materialize_selected_occurrences(
    reader: &LocalSnapshotReader,
    context: &ProjectionContext,
    documents: &mut [FastDocument],
    spill_path: &Path,
    selection: BuildSelection,
) -> Result<(usize, TimestampAccounting, u64)> {
    let selected = documents
        .iter()
        .enumerate()
        .map(|(index, document)| (document.document_id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut writer = BufWriter::new(File::create(spill_path)?);
    let mut timestamps = TimestampAccounting::default();
    let mut occurrence_count = 0_usize;
    let mut rows_scanned = 0_u64;
    for relation in reader.typed_relations() {
        if relation.name == "ocsf_ext_livefire_system_metric" {
            continue;
        }
        for batch in reader.scan(relation)? {
            let batch = batch?;
            rows_scanned = rows_scanned
                .checked_add(batch.num_rows() as u64)
                .ok_or(Error::CountOverflow)?;
            let event_ids = strings(&batch, "event_id")?;
            let json = strings(&batch, "typed_event_json")?;
            let support = strings(&batch, "support_ref")?;
            for row in 0..batch.num_rows() {
                if matches!(selection, BuildSelection::Representative) {
                    let summary =
                        project_document_summary(&relation.name, json.value(row), context)?;
                    let Some(document) = summary.document else {
                        continue;
                    };
                    if !selected.contains_key(&document.document_id) {
                        continue;
                    }
                }
                let output = project(ProjectionInput {
                    relation_name: &relation.name,
                    event_id: event_ids.value(row),
                    typed_event_json: json.value(row),
                    support_ref: support.value(row),
                    context,
                })?;
                let Some(projected_document) = output.document else {
                    continue;
                };
                let Some(index) = selected.get(&projected_document.document_id).copied() else {
                    continue;
                };
                let document_bytes = serde_json::to_vec(&projected_document)?;
                let document_sha256 = format!("{:x}", Sha256::digest(document_bytes));
                if documents[index].document_sha256 != document_sha256 {
                    return Err(Error::InconsistentDocument(projected_document.document_id));
                }
                let event_time_ms = parse_event_time_ms(
                    output.occurrence.event_time.as_deref(),
                    output.occurrence.event_time_availability,
                )?;
                timestamps.observe(output.occurrence.event_time_availability, event_time_ms);
                let mut hasher = Sha256::new();
                hasher.update(context.snapshot.sha256.as_bytes());
                hasher.update([0]);
                hasher.update(relation.name.as_bytes());
                hasher.update([0]);
                hasher.update(event_ids.value(row).as_bytes());
                let occurrence = FastOccurrence {
                    occurrence_id: format!("occ-{:x}", hasher.finalize()),
                    document_id: projected_document.document_id,
                    event_time_ms,
                    relation: relation.name.clone(),
                    exact_attributes_json: serde_json::to_string(
                        &output.occurrence.exact_attributes,
                    )?,
                    snapshot_sha256: context.snapshot.sha256.clone(),
                    mapping_sha256: context.mapping_pack.sha256.clone(),
                    event_id: event_ids.value(row).into(),
                    support_ref: support.value(row).into(),
                };
                serde_json::to_writer(&mut writer, &occurrence)?;
                writer.write_all(b"\n")?;
                documents[index].occurrence_count = documents[index]
                    .occurrence_count
                    .checked_add(1)
                    .ok_or(Error::CountOverflow)?;
                occurrence_count = occurrence_count
                    .checked_add(1)
                    .ok_or(Error::CountOverflow)?;
            }
        }
    }
    writer.flush()?;
    if documents
        .iter()
        .any(|document| document.occurrence_count == 0)
        || timestamps.total() != occurrence_count as u64
    {
        return Err(Error::OccurrenceClosure);
    }
    Ok((occurrence_count, timestamps, rows_scanned))
}
fn strings<'a>(batch: &'a RecordBatch, name: &'static str) -> Result<&'a StringArray> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| Error::Column(name))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(Error::Column(name))
}
fn fast_document(document: ProjectedDocument, document_sha256: String) -> Result<FastDocument> {
    Ok(FastDocument {
        document_id: document.document_id,
        document_sha256,
        document_kind: format!("{:?}", document.document_kind).to_lowercase(),
        semantic_text: document.semantic_text,
        facets_json: serde_json::to_string(&document.facets)?,
        relations_json: "[]".to_owned(),
        occurrence_count: 0,
        vector_ordinal: 0,
    })
}

fn parse_event_time_ms(
    value: Option<&str>,
    availability: EventTimeAvailability,
) -> Result<Option<u64>> {
    match availability {
        EventTimeAvailability::Missing | EventTimeAvailability::PresentUnparsed => Ok(None),
        EventTimeAvailability::Available => {
            let value = value.ok_or_else(|| Error::Timestamp("available time is absent".into()))?;
            let parsed = DateTime::parse_from_rfc3339(value)
                .map_err(|_| Error::Timestamp(value.to_owned()))?;
            let milliseconds = parsed.timestamp_millis();
            Ok(u64::try_from(milliseconds).ok())
        }
    }
}

async fn query(
    index_path: &Path,
    text: &str,
    mode: Mode,
    top_n: usize,
    endpoint: &str,
    relations: Vec<String>,
) -> Result<()> {
    let index = FastIndex::open(index_path)?;
    let embedder = LmStudioEmbedder::new(endpoint, &index.manifest.embedding_profile.model);
    let hits = search_index(&index, &embedder, text, mode, top_n, relations).await?;
    #[derive(Serialize)]
    struct Output<'a> {
        schema_version: &'static str,
        query: &'a str,
        hits: &'a [rag_index::SearchHit],
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Output {
            schema_version: "livefire.rag.fast-search-result/1",
            query: text,
            hits: &hits
        })?
    );
    Ok(())
}

struct BenchmarkQueryOptions<'a> {
    index: &'a Path,
    query: &'a str,
    query_id: &'a str,
    top_n: usize,
    endpoint: &'a str,
    relations: Vec<String>,
    warmups: usize,
    repeats: usize,
    embedding_warmups: usize,
    embedding_repeats: usize,
    end_to_end_repeats: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LatencySummary {
    warmups: usize,
    samples: usize,
    min_micros: u64,
    p50_micros: u64,
    p95_micros: u64,
    max_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexBenchmarkStage {
    latency: LatencySummary,
    hits: usize,
    result_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryEmbeddingBenchmark {
    calls: usize,
    latency: LatencySummary,
    returned_model: String,
    vector_dimensions: usize,
    vector_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryBenchmarkReport {
    schema_version: String,
    query_id: String,
    query_sha256: String,
    source_snapshot_sha256: String,
    index_component_sha256: String,
    index_schema_version: String,
    embedding_profile_id: String,
    embedding_profile_version: String,
    embedding_profile_sha256: String,
    configured_model: String,
    returned_model: String,
    git: GitState,
    machine: MachineContext,
    lm_studio: LmStudioContext,
    resource_usage: ResourceUsage,
    artifact_sizes: QueryArtifactSizes,
    transport_bytes: TransportByteAccounting,
    top_n: usize,
    relation_filters: Vec<String>,
    warmups: usize,
    repeats: usize,
    embedding_warmups: usize,
    embedding_repeats: usize,
    query_embedding: QueryEmbeddingBenchmark,
    dense_index_only: IndexBenchmarkStage,
    fused_index_only: IndexBenchmarkStage,
    lexical_index_only: IndexBenchmarkStage,
    end_to_end_fused: Option<IndexBenchmarkStage>,
    total_model_calls: usize,
}

async fn benchmark_query(options: BenchmarkQueryOptions<'_>) -> Result<()> {
    let index = FastIndex::open(options.index)?;
    let embedder = LmStudioEmbedder::new(options.endpoint, &index.manifest.embedding_profile.model);
    let mut report = benchmark_query_with_embedder(
        &index,
        &embedder,
        options.query,
        options.query_id,
        options.top_n,
        options.relations,
        options.warmups,
        options.repeats,
        options.embedding_warmups,
        options.embedding_repeats,
        options.end_to_end_repeats,
    )
    .await?;
    report.artifact_sizes = query_artifact_sizes(options.index);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn benchmark_query_with_embedder<E: rag_embedding::IdentifiedEmbedder>(
    index: &FastIndex,
    embedder: &E,
    query_text: &str,
    query_id: &str,
    top_n: usize,
    mut relations: Vec<String>,
    warmups: usize,
    repeats: usize,
    embedding_warmups: usize,
    embedding_repeats: usize,
    end_to_end_repeats: usize,
) -> Result<QueryBenchmarkReport> {
    if query_text.is_empty()
        || query_id.is_empty()
        || top_n == 0
        || top_n > 100
        || warmups > 100
        || repeats == 0
        || repeats > 10_000
        || embedding_warmups > 20
        || embedding_repeats == 0
        || embedding_repeats > 100
        || end_to_end_repeats > 100
        || relations.iter().any(String::is_empty)
    {
        return Err(Error::InvalidQueryBenchmark);
    }
    relations.sort();
    relations.dedup();
    let filters = SearchFilters {
        relations: relations.iter().cloned().collect(),
        ..Default::default()
    };
    let composed = try_compose_query(&index.manifest.embedding_profile, query_text)?;
    for _ in 0..embedding_warmups {
        validate_query_embedding_response(
            index,
            rag_embedding::IdentifiedEmbedder::embed_identified(
                embedder,
                std::slice::from_ref(&composed),
            )
            .await?,
        )?;
    }
    let mut embedding_latencies = Vec::with_capacity(embedding_repeats);
    let mut query_vector = None;
    let mut query_vector_sha256 = None;
    let mut returned_model = None;
    for _ in 0..embedding_repeats {
        let embedding_started = Instant::now();
        let mut batch = validate_query_embedding_response(
            index,
            rag_embedding::IdentifiedEmbedder::embed_identified(
                embedder,
                std::slice::from_ref(&composed),
            )
            .await?,
        )?;
        embedding_latencies.push(duration_micros(embedding_started));
        let vector_sha256 = query_vector_digest(&batch.vectors[0]);
        if query_vector_sha256
            .as_ref()
            .is_some_and(|expected| expected != &vector_sha256)
        {
            return Err(Error::InvalidQueryBenchmark);
        }
        query_vector_sha256 = Some(vector_sha256);
        returned_model = Some(batch.returned_model);
        query_vector = batch.vectors.pop();
    }
    let query_vector = query_vector.ok_or(Error::InvalidQueryBenchmark)?;
    let bound = index.validate_query_vector(&query_vector)?;

    // Check that the optimized bound-vector calls return exactly what the
    // normal query path returns before measuring them.
    let legacy_dense = index.search(
        SearchMode::Dense,
        query_text,
        Some(bound.values()),
        &filters,
        top_n,
    )?;
    let bound_dense = index.search_dense_with_vector(&bound, &filters, top_n)?;
    let legacy_fused = index.search(
        SearchMode::Fused,
        query_text,
        Some(bound.values()),
        &filters,
        top_n,
    )?;
    let bound_fused = index.search_fused_with_vector(query_text, &bound, &filters, top_n)?;
    if legacy_dense != bound_dense || legacy_fused != bound_fused {
        return Err(Error::InvalidQueryBenchmark);
    }

    let dense_index_only = measure_index_stage(warmups, repeats, || {
        Ok(index.search_dense_with_vector(&bound, &filters, top_n)?)
    })?;
    let fused_index_only = measure_index_stage(warmups, repeats, || {
        Ok(index.search_fused_with_vector(query_text, &bound, &filters, top_n)?)
    })?;
    let lexical_index_only = measure_index_stage(warmups, repeats, || {
        Ok(index.search(SearchMode::Lexical, query_text, None, &filters, top_n)?)
    })?;

    let end_to_end_fused = if end_to_end_repeats == 0 {
        None
    } else {
        let mut latencies = Vec::with_capacity(end_to_end_repeats);
        let mut expected_digest = None;
        let mut hit_count = 0;
        for _ in 0..end_to_end_repeats {
            let started = Instant::now();
            let composed = try_compose_query(&index.manifest.embedding_profile, query_text)?;
            let mut response =
                rag_embedding::IdentifiedEmbedder::embed_identified(embedder, &[composed]).await?;
            if response.returned_model != index.manifest.embedding_profile.model
                || response.vectors.len() != 1
            {
                return Err(Error::InvalidQueryBenchmark);
            }
            let vector = adapt_model_vector(
                &index.manifest.embedding_profile,
                response.vectors.remove(0),
            )?;
            let bound = index.validate_query_vector(&vector)?;
            let hits = index.search_fused_with_vector(query_text, &bound, &filters, top_n)?;
            latencies.push(duration_micros(started));
            let digest = search_hits_digest(&hits)?;
            if expected_digest
                .as_ref()
                .is_some_and(|expected| expected != &digest)
            {
                return Err(Error::InvalidQueryBenchmark);
            }
            expected_digest = Some(digest);
            hit_count = hits.len();
        }
        Some(IndexBenchmarkStage {
            latency: summarize_latencies(0, latencies)?,
            hits: hit_count,
            result_sha256: expected_digest.ok_or(Error::InvalidQueryBenchmark)?,
        })
    };
    let returned_model = returned_model.ok_or(Error::InvalidQueryBenchmark)?;
    let total_model_calls = embedding_warmups + embedding_repeats + end_to_end_repeats;
    let submitted_text_bytes = u64::try_from(composed.len())
        .ok()
        .and_then(|bytes| bytes.checked_mul(u64::try_from(total_model_calls).ok()?));
    let decoded_vector_bytes = u64::try_from(bound.values().len())
        .ok()
        .and_then(|dimensions| dimensions.checked_mul(4))
        .and_then(|bytes| bytes.checked_mul(u64::try_from(total_model_calls).ok()?));
    let run_context = LocalRunContext::observe();
    Ok(QueryBenchmarkReport {
        schema_version: "livefire.rag.query-benchmark/1".into(),
        query_id: query_id.to_owned(),
        query_sha256: format!("{:x}", Sha256::digest(query_text.as_bytes())),
        source_snapshot_sha256: index.manifest.source.snapshot_sha256.clone(),
        index_component_sha256: index.manifest.component_sha256.clone(),
        index_schema_version: index.manifest.schema_version.clone(),
        embedding_profile_id: index.manifest.embedding_profile.id.clone(),
        embedding_profile_version: index.manifest.embedding_profile.version.clone(),
        embedding_profile_sha256: bound.embedding_profile_sha256().to_owned(),
        configured_model: index.manifest.embedding_profile.model.clone(),
        returned_model: returned_model.clone(),
        git: run_context.git,
        machine: run_context.machine,
        lm_studio: LmStudioContext::query(&index.manifest.embedding_profile.model, &returned_model),
        resource_usage: run_context.resources,
        artifact_sizes: QueryArtifactSizes {
            status: ObservationStatus::NotMeasured,
            index_bytes: None,
        },
        transport_bytes: TransportByteAccounting {
            status: ObservationStatus::Partial,
            request_body_bytes: None,
            response_body_bytes: None,
            submitted_text_bytes,
            decoded_vector_bytes,
        },
        top_n,
        relation_filters: relations,
        warmups,
        repeats,
        embedding_warmups,
        embedding_repeats,
        query_embedding: QueryEmbeddingBenchmark {
            calls: embedding_warmups + embedding_repeats,
            latency: summarize_latencies(embedding_warmups, embedding_latencies)?,
            returned_model,
            vector_dimensions: bound.values().len(),
            vector_sha256: query_vector_sha256.ok_or(Error::InvalidQueryBenchmark)?,
        },
        dense_index_only,
        fused_index_only,
        lexical_index_only,
        end_to_end_fused,
        total_model_calls,
    })
}

fn validate_query_embedding_response(
    index: &FastIndex,
    mut batch: rag_embedding::IdentifiedEmbeddingBatch,
) -> Result<rag_embedding::IdentifiedEmbeddingBatch> {
    if batch.returned_model != index.manifest.embedding_profile.model || batch.vectors.len() != 1 {
        return Err(Error::InvalidQueryBenchmark);
    }
    batch.vectors[0] = adapt_model_vector(
        &index.manifest.embedding_profile,
        std::mem::take(&mut batch.vectors[0]),
    )?;
    index.validate_query_vector(&batch.vectors[0])?;
    Ok(batch)
}

fn query_vector_digest(vector: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"livefire.rag.query-vector/1\0");
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Serialize)]
struct IndexOverlapReport {
    schema_version: &'static str,
    full_index_sha256: String,
    reduced_index_sha256: String,
    full_profile_sha256: String,
    reduced_profile_sha256: String,
    reduced_dimensions: u32,
    top_n: usize,
    dense: TopKOverlap,
    fused: TopKOverlap,
}

#[derive(Debug, Serialize)]
struct TopKOverlap {
    full_hits: usize,
    reduced_hits: usize,
    shared_hits: usize,
    overlap_fraction_of_full: f64,
    jaccard: f64,
    full_document_ids: Vec<String>,
    reduced_document_ids: Vec<String>,
}

fn top_k_overlap(
    full: Vec<rag_index::SearchHit>,
    reduced: Vec<rag_index::SearchHit>,
) -> TopKOverlap {
    let full_document_ids = full
        .into_iter()
        .map(|hit| hit.document_id)
        .collect::<Vec<_>>();
    let reduced_document_ids = reduced
        .into_iter()
        .map(|hit| hit.document_id)
        .collect::<Vec<_>>();
    top_k_overlap_ids(full_document_ids, reduced_document_ids)
}

fn top_k_overlap_ids(
    full_document_ids: Vec<String>,
    reduced_document_ids: Vec<String>,
) -> TopKOverlap {
    let full_set = full_document_ids.iter().collect::<BTreeSet<_>>();
    let reduced_set = reduced_document_ids.iter().collect::<BTreeSet<_>>();
    let shared_hits = full_set.intersection(&reduced_set).count();
    let union = full_set.union(&reduced_set).count();
    TopKOverlap {
        full_hits: full_document_ids.len(),
        reduced_hits: reduced_document_ids.len(),
        shared_hits,
        overlap_fraction_of_full: if full_document_ids.is_empty() {
            1.0
        } else {
            shared_hits as f64 / full_document_ids.len() as f64
        },
        jaccard: if union == 0 {
            1.0
        } else {
            shared_hits as f64 / union as f64
        },
        full_document_ids,
        reduced_document_ids,
    }
}

async fn compare_index_overlap(
    full_path: &Path,
    reduced_path: &Path,
    query: &str,
    top_n: usize,
    endpoint: &str,
    relations: Vec<String>,
) -> Result<()> {
    let full = FastIndex::open(full_path)?;
    let reduced = FastIndex::open(reduced_path)?;
    validate_reduced_profile_pair(
        &full.manifest.embedding_profile,
        &reduced.manifest.embedding_profile,
    )?;
    if query.is_empty() || top_n == 0 || top_n > 100 {
        return Err(Error::AccountingClosure("invalid index overlap options"));
    }
    let embedder = LmStudioEmbedder::new(endpoint, &full.manifest.embedding_profile.model);
    let composed = try_compose_query(&full.manifest.embedding_profile, query)?;
    let response =
        rag_embedding::IdentifiedEmbedder::embed_identified(&embedder, &[composed]).await?;
    if response.returned_model != full.manifest.embedding_profile.model
        || response.vectors.len() != 1
    {
        return Err(Error::InvalidQueryBenchmark);
    }
    let raw = response
        .vectors
        .into_iter()
        .next()
        .ok_or(Error::InvalidQueryBenchmark)?;
    let full_vector = adapt_model_vector(&full.manifest.embedding_profile, raw.clone())?;
    let reduced_vector = adapt_model_vector(&reduced.manifest.embedding_profile, raw)?;
    let full_bound = full.validate_query_vector(&full_vector)?;
    let reduced_bound = reduced.validate_query_vector(&reduced_vector)?;
    let filters = SearchFilters {
        relations: relations.into_iter().collect(),
        ..Default::default()
    };
    let dense = top_k_overlap(
        full.search_dense_with_vector(&full_bound, &filters, top_n)?,
        reduced.search_dense_with_vector(&reduced_bound, &filters, top_n)?,
    );
    let fused = top_k_overlap(
        full.search_fused_with_vector(query, &full_bound, &filters, top_n)?,
        reduced.search_fused_with_vector(query, &reduced_bound, &filters, top_n)?,
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&IndexOverlapReport {
            schema_version: "livefire.rag.index-overlap/1",
            full_index_sha256: full.manifest.component_sha256.clone(),
            reduced_index_sha256: reduced.manifest.component_sha256.clone(),
            full_profile_sha256: full.manifest.embedding_profile.sha256.clone(),
            reduced_profile_sha256: reduced.manifest.embedding_profile.sha256.clone(),
            reduced_dimensions: reduced.manifest.embedding_profile.dimensions,
            top_n,
            dense,
            fused,
        })?
    );
    Ok(())
}

fn validate_reduced_profile_pair(
    full: &rag_embedding::EmbeddingProfile,
    reduced: &rag_embedding::EmbeddingProfile,
) -> Result<()> {
    let derivation = reduced
        .vector_derivation
        .as_ref()
        .ok_or(Error::AccountingClosure(
            "reduced index profile has no vector derivation",
        ))?;
    if derivation.parent_embedding_profile_sha256 != full.sha256
        || derivation.parent_dimensions != full.dimensions
        || full.vector_derivation.is_some()
        || full.model != reduced.model
        || full.query_instruction != reduced.query_instruction
        || full.query_composition != reduced.query_composition
    {
        return Err(Error::AccountingClosure(
            "full and reduced index profiles are not a matching pair",
        ));
    }
    Ok(())
}

fn measure_index_stage(
    warmups: usize,
    repeats: usize,
    mut search: impl FnMut() -> Result<Vec<rag_index::SearchHit>>,
) -> Result<IndexBenchmarkStage> {
    for _ in 0..warmups {
        search()?;
    }
    let mut latencies = Vec::with_capacity(repeats);
    let mut expected_digest = None;
    let mut hit_count = 0;
    for _ in 0..repeats {
        let started = Instant::now();
        let hits = search()?;
        latencies.push(duration_micros(started));
        let digest = search_hits_digest(&hits)?;
        if expected_digest
            .as_ref()
            .is_some_and(|expected| expected != &digest)
        {
            return Err(Error::InvalidQueryBenchmark);
        }
        expected_digest = Some(digest);
        hit_count = hits.len();
    }
    Ok(IndexBenchmarkStage {
        latency: summarize_latencies(warmups, latencies)?,
        hits: hit_count,
        result_sha256: expected_digest.ok_or(Error::InvalidQueryBenchmark)?,
    })
}

fn summarize_latencies(warmups: usize, mut values: Vec<u64>) -> Result<LatencySummary> {
    if values.is_empty() {
        return Err(Error::InvalidQueryBenchmark);
    }
    values.sort_unstable();
    Ok(LatencySummary {
        warmups,
        samples: values.len(),
        min_micros: values[0],
        p50_micros: latency_percentile(&values, 50),
        p95_micros: latency_percentile(&values, 95),
        max_micros: values[values.len() - 1],
    })
}

fn latency_percentile(sorted: &[u64], percentile: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn duration_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().try_into().unwrap_or(u64::MAX)
}

fn search_hits_digest(hits: &[rag_index::SearchHit]) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(hits)?)))
}

async fn search_index<E: rag_embedding::IdentifiedEmbedder>(
    index: &FastIndex,
    embedder: &E,
    text: &str,
    mode: Mode,
    top_n: usize,
    relations: Vec<String>,
) -> Result<Vec<rag_index::SearchHit>> {
    let search_mode: SearchMode = mode.into();
    let query_vector = if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) {
        let composed = try_compose_query(&index.manifest.embedding_profile, text)?;
        let mut response = rag_embedding::IdentifiedEmbedder::embed_identified(
            embedder,
            std::slice::from_ref(&composed),
        )
        .await?;
        if response.returned_model != index.manifest.embedding_profile.model
            || response.vectors.len() != 1
        {
            return Err(Error::InvalidQueryEmbeddingResponse);
        }
        let vector = adapt_model_vector(
            &index.manifest.embedding_profile,
            response
                .vectors
                .pop()
                .ok_or(Error::InvalidQueryEmbeddingResponse)?,
        )?;
        index.validate_query_vector(&vector)?;
        Some(vector)
    } else {
        None
    };
    let filters = SearchFilters {
        relations: relations.into_iter().collect(),
        ..Default::default()
    };
    Ok(index.search(search_mode, text, query_vector.as_deref(), &filters, top_n)?)
}

async fn batch_query(index_path: &Path, requests_path: &Path, endpoint: &str) -> Result<()> {
    let requests = read_batch_query_requests(requests_path)?;
    let index = FastIndex::open(index_path)?;
    let embedder = LmStudioEmbedder::new(endpoint, &index.manifest.embedding_profile.model);
    let stdout = std::io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    for request in requests {
        let hits = search_index(
            &index,
            &embedder,
            &request.query,
            request.mode,
            request.top_n,
            request.relations,
        )
        .await?;
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "schema_version":"livefire.rag.fast-batch-query-result/1",
                "query_id":request.query_id,
                "query":request.query,
                "mode":request.mode,
                "hits":hits
            }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn read_batch_query_requests(requests_path: &Path) -> Result<Vec<BatchQueryRequest>> {
    Ok(read_batch_query_plan(requests_path)?.requests)
}

fn read_batch_query_plan(requests_path: &Path) -> Result<BatchQueryPlan> {
    let bytes = fs::read(requests_path)?;
    if bytes.last() != Some(&b'\n') {
        return Err(Error::InvalidBatchQuery);
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| Error::InvalidBatchQuery)?;
    let mut requests = Vec::new();
    for line in text.lines() {
        let request = serde_json::from_str::<BatchQueryRequest>(line)?;
        request.validate()?;
        requests.push(request);
        if requests.len() > MAX_BATCH_QUERY_REQUESTS {
            return Err(Error::InvalidBatchQuery);
        }
    }
    if requests.is_empty() {
        return Err(Error::InvalidBatchQuery);
    }
    let mut surfaces = BTreeSet::new();
    let mut query_id_contract = BTreeMap::<&str, (&str, usize, &[String])>::new();
    for request in &requests {
        if !surfaces.insert((request.query_id.as_str(), request.mode)) {
            return Err(Error::InvalidBatchQuery);
        }
        if let Some((query, top_n, relations)) = query_id_contract.get(request.query_id.as_str()) {
            if *query != request.query
                || *top_n != request.top_n
                || *relations != request.relations.as_slice()
            {
                return Err(Error::InvalidBatchQuery);
            }
        } else {
            query_id_contract.insert(
                &request.query_id,
                (&request.query, request.top_n, &request.relations),
            );
        }
    }
    Ok(BatchQueryPlan {
        requests,
        source_sha256: rag_pipeline::Digest::new(format!("{:x}", Sha256::digest(&bytes)))?,
        source_bytes: u64::try_from(bytes.len()).map_err(|_| Error::CountOverflow)?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct FakeQueryEmbedder {
        calls: AtomicUsize,
    }

    impl rag_embedding::Embedder for FakeQueryEmbedder {
        async fn embed(&self, texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    impl rag_embedding::IdentifiedEmbedder for FakeQueryEmbedder {
        async fn embed_identified(
            &self,
            texts: &[String],
        ) -> rag_embedding::Result<rag_embedding::IdentifiedEmbeddingBatch> {
            Ok(rag_embedding::IdentifiedEmbeddingBatch {
                vectors: rag_embedding::Embedder::embed(self, texts).await?,
                returned_model: "fake-query-model".into(),
            })
        }
    }

    struct DriftingQueryEmbedder {
        calls: AtomicUsize,
    }

    impl rag_embedding::Embedder for DriftingQueryEmbedder {
        async fn embed(&self, texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            let vector = if call.is_multiple_of(2) {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
            Ok(texts.iter().map(|_| vector.clone()).collect())
        }
    }

    impl rag_embedding::IdentifiedEmbedder for DriftingQueryEmbedder {
        async fn embed_identified(
            &self,
            texts: &[String],
        ) -> rag_embedding::Result<rag_embedding::IdentifiedEmbeddingBatch> {
            Ok(rag_embedding::IdentifiedEmbeddingBatch {
                vectors: rag_embedding::Embedder::embed(self, texts).await?,
                returned_model: "fake-query-model".into(),
            })
        }
    }

    struct FixedIdentifiedQueryEmbedder {
        calls: AtomicUsize,
        returned_model: &'static str,
        vectors: Vec<Vec<f32>>,
    }

    impl rag_embedding::Embedder for FixedIdentifiedQueryEmbedder {
        async fn embed(&self, _texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            panic!("single-index query must use embed_identified")
        }
    }

    impl rag_embedding::IdentifiedEmbedder for FixedIdentifiedQueryEmbedder {
        async fn embed_identified(
            &self,
            _texts: &[String],
        ) -> rag_embedding::Result<rag_embedding::IdentifiedEmbeddingBatch> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(rag_embedding::IdentifiedEmbeddingBatch {
                vectors: self.vectors.clone(),
                returned_model: self.returned_model.into(),
            })
        }
    }

    fn benchmark_fixture_index(root: &Path) -> FastIndex {
        let path = root.join("index");
        let documents = vec![
            FastDocument {
                document_id: "doc-a".into(),
                document_sha256: "a".repeat(64),
                document_kind: "activity".into(),
                semantic_text: "alpha activity".into(),
                facets_json: "{}".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: 0,
            },
            FastDocument {
                document_id: "doc-b".into(),
                document_sha256: "b".repeat(64),
                document_kind: "activity".into(),
                semantic_text: "beta activity".into(),
                facets_json: "{}".into(),
                relations_json: "[]".into(),
                occurrence_count: 1,
                vector_ordinal: 1,
            },
        ];
        let occurrences = vec![
            FastOccurrence {
                occurrence_id: "occ-a".into(),
                document_id: "doc-a".into(),
                event_time_ms: Some(1),
                relation: "events".into(),
                exact_attributes_json: "{}".into(),
                snapshot_sha256: "c".repeat(64),
                mapping_sha256: "d".repeat(64),
                event_id: "event-a".into(),
                support_ref: "support-a".into(),
            },
            FastOccurrence {
                occurrence_id: "occ-b".into(),
                document_id: "doc-b".into(),
                event_time_ms: Some(2),
                relation: "events".into(),
                exact_attributes_json: "{}".into(),
                snapshot_sha256: "c".repeat(64),
                mapping_sha256: "d".repeat(64),
                event_id: "event-b".into(),
                support_ref: "support-b".into(),
            },
        ];
        rag_index::write_fast_index(
            &path,
            SourceBinding {
                snapshot_sha256: "c".repeat(64),
                mapping_sha256: "d".repeat(64),
            },
            BuildScope::Full,
            &documents,
            &occurrences,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            rag_embedding::EmbeddingProfile {
                id: "fake-profile".into(),
                version: "1".into(),
                sha256: "e".repeat(64),
                model: "fake-query-model".into(),
                dimensions: 2,
                normalization: "l2".into(),
                vector_derivation: None,
                query_instruction: None,
                query_composition: None,
            },
        )
        .unwrap();
        FastIndex::open(&path).unwrap()
    }

    fn document(id: &str) -> FastDocument {
        FastDocument {
            document_id: id.to_owned(),
            document_sha256: format!("sha-{id}"),
            document_kind: "activity".to_owned(),
            semantic_text: format!("activity {id}"),
            facets_json: "{}".to_owned(),
            relations_json: "[]".to_owned(),
            occurrence_count: 0,
            vector_ordinal: 0,
        }
    }

    fn collect(order: &[&str]) -> (Vec<FastDocument>, FirstPassAccounting) {
        let mut collector = DocumentCollector::new(
            BuildSelection::Representative,
            &"a".repeat(64),
            &["ocsf_process_activity".to_owned()],
        );
        for sequence in 0..2 {
            for id in order {
                collector.observe_source_row(
                    "ocsf_process_activity",
                    true,
                    EventTimeAvailability::Available,
                    Some(sequence),
                );
                collector
                    .retain(document(id), "ocsf_process_activity")
                    .expect("collect");
            }
        }
        collector.finish().expect("finish")
    }

    #[test]
    fn representative_sample_censuses_rare_relations_and_caps_large_relations() {
        let relations = [
            "ocsf_network_activity".to_owned(),
            "ocsf_process_activity".to_owned(),
        ];
        let mut collector =
            DocumentCollector::new(BuildSelection::Representative, &"a".repeat(64), &relations);
        for sequence in 0..2_500 {
            let id = format!("network-{sequence}");
            collector.observe_source_row(
                "ocsf_network_activity",
                true,
                EventTimeAvailability::Available,
                Some(sequence),
            );
            collector
                .retain(document(&id), "ocsf_network_activity")
                .unwrap();
        }
        for sequence in 0..3 {
            let id = format!("process-{sequence}");
            collector.observe_source_row(
                "ocsf_process_activity",
                true,
                EventTimeAvailability::Available,
                Some(sequence),
            );
            collector
                .retain(document(&id), "ocsf_process_activity")
                .unwrap();
        }
        let (documents, accounting) = collector.finish().unwrap();
        assert_eq!(documents.len(), 2_003);
        assert_eq!(
            accounting.selected_by_relation,
            BTreeMap::from([
                ("ocsf_network_activity".into(), 2_000),
                ("ocsf_process_activity".into(), 3),
            ])
        );
        assert_eq!(
            accounting.relation_budgets,
            BTreeMap::from([
                ("ocsf_network_activity".into(), 2_000),
                ("ocsf_process_activity".into(), 3),
            ])
        );
        assert_eq!(
            accounting.source_documents_by_relation["ocsf_network_activity"],
            2_500
        );
        assert_eq!(
            accounting.source_documents_by_relation["ocsf_process_activity"],
            3
        );
    }

    #[test]
    fn relation_hash_min_sample_is_order_independent() {
        let forward = (0..2_500).map(|i| format!("doc-{i}")).collect::<Vec<_>>();
        let mut reverse = forward.clone();
        reverse.reverse();
        let forward_refs = forward.iter().map(String::as_str).collect::<Vec<_>>();
        let reverse_refs = reverse.iter().map(String::as_str).collect::<Vec<_>>();
        let (documents, accounting) = collect(&forward_refs);
        let (reverse_documents, _) = collect(&reverse_refs);

        let ids = documents
            .iter()
            .map(|document| document.document_id.as_str())
            .collect::<Vec<_>>();
        let reverse_ids = reverse_documents
            .iter()
            .map(|document| document.document_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, reverse_ids);
        assert_eq!(documents.len(), 2_000);
        assert_eq!(accounting.source_rows_scanned, 5_000);
        assert_eq!(accounting.projected_semantic_occurrences, 5_000);
        assert_eq!(
            accounting.relation_budgets,
            BTreeMap::from([("ocsf_process_activity".to_owned(), 2_000)])
        );
    }

    #[test]
    fn timestamp_parser_indexes_normalized_rfc3339_and_accounts_edge_states() {
        assert_eq!(
            parse_event_time_ms(
                Some("2024-01-01T00:00:00.123Z"),
                EventTimeAvailability::Available,
            )
            .expect("timestamp"),
            Some(1_704_067_200_123)
        );
        assert_eq!(
            parse_event_time_ms(
                Some("2024-01-01T10:00:00+10:00"),
                EventTimeAvailability::Available,
            )
            .expect("offset timestamp"),
            Some(1_704_067_200_000)
        );
        assert_eq!(
            parse_event_time_ms(
                Some("1969-12-31T23:59:59Z"),
                EventTimeAvailability::Available,
            )
            .expect("pre-epoch timestamp"),
            None
        );
        assert_eq!(
            parse_event_time_ms(Some("unknown"), EventTimeAvailability::PresentUnparsed)
                .expect("unparsed is accounted, not indexed"),
            None
        );
        assert!(
            parse_event_time_ms(Some("not-rfc3339"), EventTimeAvailability::Available).is_err()
        );
    }

    #[test]
    fn structured_only_rows_are_terminally_accounted_by_relation() {
        let relation = "ocsf_ext_livefire_system_metric";
        let mut collector = DocumentCollector::new(
            BuildSelection::Representative,
            &"a".repeat(64),
            &[relation.to_owned()],
        );
        collector.observe_source_row(relation, false, EventTimeAvailability::Missing, None);
        let (documents, accounting) = collector.finish().expect("finish");
        assert!(documents.is_empty());
        assert_eq!(accounting.source_rows_by_relation[relation], 1);
        assert_eq!(accounting.structured_only_by_relation[relation], 1);
        assert_eq!(accounting.projected_semantic_occurrences, 0);
        assert!(accounting.relation_budgets.is_empty());
    }

    #[test]
    fn cli_exposes_only_the_fixed_representative_sample_policy() {
        let base = [
            "rag",
            "build",
            "--snapshot",
            "snapshot",
            "--out",
            "index",
            "--embedding-profile",
            "profile.json",
            "--resume",
            "cache.sqlite3",
        ];
        let full = Cli::try_parse_from(base).expect("full build CLI");
        assert!(matches!(
            full.command,
            Command::Build {
                representative_sample: false,
                ..
            }
        ));

        let mut sampled_args = base.to_vec();
        sampled_args.push("--representative-sample");
        let sampled = Cli::try_parse_from(sampled_args).expect("sample CLI");
        assert!(matches!(
            sampled.command,
            Command::Build {
                representative_sample: true,
                ..
            }
        ));

        let mut old_args = base.to_vec();
        old_args.extend(["--sample-documents", "20000"]);
        assert!(Cli::try_parse_from(old_args).is_err());
    }

    #[test]
    fn tokenizer_verification_cli_requires_all_offline_artifacts() {
        let parsed = Cli::try_parse_from([
            "rag",
            "verify-tokenizer",
            "--tokenizer-json",
            "tokenizer.json",
            "--tokenizer-ref",
            "tokenizer.ref.json",
            "--fixture",
            "fixture.json",
        ])
        .expect("offline tokenizer verification CLI");
        assert!(matches!(parsed.command, Command::VerifyTokenizer { .. }));
        assert!(
            Cli::try_parse_from([
                "rag",
                "verify-tokenizer",
                "--tokenizer-json",
                "tokenizer.json",
                "--tokenizer-ref",
                "tokenizer.ref.json",
            ])
            .is_err()
        );
    }

    #[test]
    fn modular_embedding_cli_requires_exact_plan_inputs_and_checked_range_text() {
        assert!(matches!(
            Cli::try_parse_from(["rag", "verify-prepared", "--prepared", "prepared"])
                .expect("prepared verification CLI")
                .command,
            Command::VerifyPrepared { .. }
        ));
        let planned = Cli::try_parse_from([
            "rag",
            "plan-embeddings",
            "--prepared",
            "prepared",
            "--embedding-profile",
            "profile.json",
            "--tokenizer-json",
            "tokenizer.json",
            "--tokenizer-ref",
            "tokenizer.ref.json",
            "--maximum-task-tokens",
            "262144",
            "--maximum-task-documents",
            "2048",
            "--out",
            "plan",
        ])
        .expect("v2 plan CLI");
        assert!(matches!(planned.command, Command::PlanEmbeddings { .. }));

        assert!(
            Cli::try_parse_from([
                "rag",
                "plan-embeddings",
                "--prepared",
                "prepared",
                "--embedding-profile",
                "profile.json",
                "--out",
                "plan",
            ])
            .is_err()
        );
        let ranged = Cli::try_parse_from([
            "rag",
            "embed",
            "--prepared",
            "prepared",
            "--plan",
            "plan",
            "--embedding-profile",
            "profile.json",
            "--out",
            "embeddings",
            "--task-range",
            "8..16",
        ])
        .expect("ranged embedding CLI");
        assert!(matches!(ranged.command, Command::Embed { .. }));
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "test-embed",
                "--prepared",
                "prepared",
                "--plan",
                "plan",
                "--embedding-profile",
                "profile.json",
                "--out",
                "test-embeddings",
            ])
            .expect("test-only embedding CLI")
            .command,
            Command::TestEmbed { .. }
        ));
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "finalize-embeddings",
                "--prepared",
                "prepared",
                "--plan",
                "plan",
                "--embedding-profile",
                "profile.json",
                "--embeddings",
                "embeddings",
            ])
            .expect("finalizer CLI")
            .command,
            Command::FinalizeEmbeddings { .. }
        ));
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "derive-embeddings",
                "--prepared",
                "prepared",
                "--plan",
                "plan",
                "--embedding-profile",
                "profile.json",
                "--embeddings",
                "embeddings",
                "--dimensions",
                "2048",
                "--out",
                "derived",
            ])
            .expect("local vector derivation CLI")
            .command,
            Command::DeriveEmbeddings {
                dimensions: 2_048,
                ..
            }
        ));
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "recover-embedding-task",
                "--plan",
                "plan",
                "--embedding-profile",
                "profile.json",
                "--embeddings",
                "embeddings",
                "--task-id",
                "task-sha256",
                "--action",
                "verify",
            ])
            .expect("task recovery CLI")
            .command,
            Command::RecoverEmbeddingTask {
                action: RecoveryAction::Verify,
                ..
            }
        ));
    }

    #[test]
    fn assembly_keeps_v2_default_and_requires_an_explicit_v3_choice() {
        let base = [
            "rag",
            "assemble",
            "--prepared",
            "prepared",
            "--plan",
            "plan",
            "--embeddings",
            "embeddings",
            "--embedding-profile",
            "profile.json",
            "--out",
            "index",
        ];
        let default = Cli::try_parse_from(base).unwrap();
        assert!(matches!(
            default.command,
            Command::Assemble {
                index_format: IndexFormat::LegacyJsonV2,
                ..
            }
        ));
        let mut scalable = base.to_vec();
        scalable.extend(["--index-format", "sqlite-v3"]);
        assert!(matches!(
            Cli::try_parse_from(scalable).unwrap().command,
            Command::Assemble {
                index_format: IndexFormat::SqliteV3,
                ..
            }
        ));
    }

    #[test]
    fn benchmark_preparation_cli_requires_an_explicit_source_scope() {
        let parsed = Cli::try_parse_from([
            "rag",
            "prepare-benchmark",
            "--snapshot",
            "snapshot",
            "--dataset-id",
            "local-benchmark",
            "--relation",
            "ocsf_process_activity",
            "--out",
            "benchmark",
        ])
        .expect("benchmark preparation CLI");
        assert!(matches!(
            parsed.command,
            Command::PrepareBenchmark { workers, .. } if (1..=8).contains(&workers)
        ));
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "prepare-benchmark",
                "--snapshot",
                "snapshot",
                "--dataset-id",
                "local-benchmark",
                "--relation",
                "ocsf_process_activity",
                "--out",
                "benchmark",
                "--workers",
                "64",
            ])
            .expect("explicit benchmark workers")
            .command,
            Command::PrepareBenchmark { workers: 64, .. }
        ));
        assert!(
            Cli::try_parse_from([
                "rag",
                "prepare-benchmark",
                "--snapshot",
                "snapshot",
                "--dataset-id",
                "local-benchmark",
                "--out",
                "benchmark",
            ])
            .is_err()
        );
    }

    #[test]
    fn ordinary_preparation_cli_exposes_the_bounded_worker_setting() {
        let base = [
            "rag",
            "prepare",
            "--snapshot",
            "snapshot",
            "--dataset-id",
            "local-dataset",
            "--relation",
            "ocsf_process_activity",
            "--out",
            "prepared",
        ];
        assert!(matches!(
            Cli::try_parse_from(base).expect("ordinary preparation CLI").command,
            Command::Prepare { workers, .. } if (1..=8).contains(&workers)
        ));

        let mut explicit = base.to_vec();
        explicit.extend(["--workers", "64"]);
        assert!(matches!(
            Cli::try_parse_from(explicit)
                .expect("explicit ordinary preparation workers")
                .command,
            Command::Prepare { workers: 64, .. }
        ));
    }

    #[test]
    fn catalogue_cli_groups_complete_dataset_artifact_chains() {
        let parsed = Cli::try_parse_from([
            "rag",
            "catalogue",
            "build",
            "--dataset",
            "a/prepared",
            "a/plan",
            "a/results",
            "a/index",
            "--dataset",
            "b/prepared",
            "b/plan",
            "b/results",
            "b/index",
            "--out",
            "catalogue.json",
        ])
        .expect("catalogue build CLI");
        assert!(matches!(
            parsed.command,
            Command::Catalogue {
                command: CatalogueCommand::Build { dataset, .. }
            } if dataset.len() == 8
        ));

        let parsed = Cli::try_parse_from([
            "rag",
            "catalogue",
            "search",
            "--catalogue",
            "catalogue.json",
            "--query",
            "encoded process",
        ])
        .expect("catalogue search CLI");
        assert!(matches!(
            parsed.command,
            Command::Catalogue {
                command: CatalogueCommand::Search { workers, .. }
            } if (1..=8).contains(&workers)
        ));

        let parsed = Cli::try_parse_from([
            "rag",
            "catalogue",
            "batch-search",
            "--catalogue",
            "catalogue.json",
            "--requests",
            "queries.jsonl",
            "--out",
            "run",
            "--workers",
            "3",
            "--allow-test-only",
        ])
        .expect("catalogue batch-search CLI");
        assert!(matches!(
            parsed.command,
            Command::Catalogue {
                command: CatalogueCommand::BatchSearch {
                    workers: 3,
                    allow_test_only: true,
                    ..
                }
            }
        ));
    }

    #[test]
    fn batch_query_plan_is_closed_and_bounded() {
        let request: BatchQueryRequest = serde_json::from_value(serde_json::json!({
            "query_id":"q-1",
            "query":"encoded interpreter activity",
            "mode":"fused",
            "top_n":20,
            "relations":["ocsf_process_activity"]
        }))
        .expect("batch row");
        request.validate().expect("valid batch row");
        assert!(
            serde_json::from_value::<BatchQueryRequest>(serde_json::json!({
                "query_id":"q-1","query":"x","mode":"lexical","top_n":1,"unknown":true
            }))
            .is_err()
        );
        let invalid: BatchQueryRequest = serde_json::from_value(serde_json::json!({
            "query_id":"q-1","query":"x","mode":"dense","top_n":101,"relations":[]
        }))
        .expect("shape-valid row");
        assert!(invalid.validate().is_err());
        assert!(matches!(
            Cli::try_parse_from([
                "rag",
                "batch-query",
                "--index",
                "index",
                "--requests",
                "queries.jsonl"
            ])
            .expect("batch CLI")
            .command,
            Command::BatchQuery { .. }
        ));

        let root = tempfile::tempdir().unwrap();
        let malformed = root.path().join("malformed.jsonl");
        fs::write(&malformed, "{not-json}\n").unwrap();
        assert!(read_batch_query_requests(&malformed).is_err());

        let missing_final_lf = root.path().join("missing-final-lf.jsonl");
        fs::write(
            &missing_final_lf,
            "{\"query_id\":\"q\",\"query\":\"x\",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[]}",
        )
        .unwrap();
        assert!(read_batch_query_requests(&missing_final_lf).is_err());

        let final_lf = root.path().join("final-lf.jsonl");
        fs::write(
            &final_lf,
            "{\"query_id\":\"q\",\"query\":\"x\",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[]}\n",
        )
        .unwrap();
        assert_eq!(read_batch_query_requests(&final_lf).unwrap().len(), 1);

        let duplicate = root.path().join("duplicate.jsonl");
        fs::write(
            &duplicate,
            concat!(
                "{\"query_id\":\"q-1\",\"query\":\"a\",\"mode\":\"fused\",\"top_n\":1,\"relations\":[]}\n",
                "{\"query_id\":\"q-1\",\"query\":\"b\",\"mode\":\"fused\",\"top_n\":1,\"relations\":[]}\n"
            ),
        )
        .unwrap();
        assert!(read_batch_query_requests(&duplicate).is_err());

        let mode_pair = root.path().join("mode-pair.jsonl");
        fs::write(
            &mode_pair,
            concat!(
                "{\"query_id\":\"q-1\",\"query\":\"a\",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[]}\n",
                "{\"query_id\":\"q-1\",\"query\":\"a\",\"mode\":\"fused\",\"top_n\":1,\"relations\":[]}\n"
            ),
        )
        .unwrap();
        assert_eq!(read_batch_query_requests(&mode_pair).unwrap().len(), 2);
        let unknown = root.path().join("unknown.jsonl");
        fs::write(
            &unknown,
            "{\"query_id\":\"q\",\"query\":\"x\",\"mode\":\"lexical\",\"top_n\":1,\"unknown\":true}\n",
        )
        .unwrap();
        assert!(read_batch_query_requests(&unknown).is_err());
        for (name, row) in [
            (
                "blank",
                "{\"query_id\":\"q\",\"query\":\"   \",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[]}\n",
            ),
            (
                "unsorted",
                "{\"query_id\":\"q\",\"query\":\"x\",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[\"b\",\"a\"]}\n",
            ),
            (
                "duplicate-relation",
                "{\"query_id\":\"q\",\"query\":\"x\",\"mode\":\"lexical\",\"top_n\":1,\"relations\":[\"a\",\"a\"]}\n",
            ),
            (
                "ambiguous-id",
                "{\"query_id\":\"q\",\"query\":\"one\",\"mode\":\"dense\",\"top_n\":1,\"relations\":[]}\n{\"query_id\":\"q\",\"query\":\"two\",\"mode\":\"fused\",\"top_n\":1,\"relations\":[]}\n",
            ),
        ] {
            let path = root.path().join(format!("{name}.jsonl"));
            fs::write(&path, row).unwrap();
            assert!(read_batch_query_requests(&path).is_err(), "{name}");
        }
    }

    #[tokio::test]
    async fn single_index_search_rejects_wrong_model_and_vector_count() {
        let root = tempfile::tempdir().unwrap();
        let index = benchmark_fixture_index(root.path());
        let wrong_model = FixedIdentifiedQueryEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "different-model",
            vectors: vec![vec![1.0, 0.0]],
        };
        assert!(matches!(
            search_index(
                &index,
                &wrong_model,
                "alpha activity",
                Mode::Dense,
                2,
                vec![]
            )
            .await,
            Err(Error::InvalidQueryEmbeddingResponse)
        ));
        assert_eq!(wrong_model.calls.load(Ordering::Relaxed), 1);

        let wrong_count = FixedIdentifiedQueryEmbedder {
            calls: AtomicUsize::new(0),
            returned_model: "fake-query-model",
            vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert!(matches!(
            search_index(
                &index,
                &wrong_count,
                "alpha activity",
                Mode::Fused,
                2,
                vec![]
            )
            .await,
            Err(Error::InvalidQueryEmbeddingResponse)
        ));
        assert_eq!(wrong_count.calls.load(Ordering::Relaxed), 1);

        let hits = search_index(
            &index,
            &wrong_model,
            "alpha activity",
            Mode::Lexical,
            2,
            vec![],
        )
        .await
        .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(wrong_model.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn benchmark_cli_has_explicit_timing_controls() {
        let parsed = Cli::try_parse_from([
            "rag",
            "benchmark-query",
            "--index",
            "index",
            "--query",
            "alpha activity",
            "--query-id",
            "q-alpha",
            "--warmups",
            "2",
            "--repeats",
            "10",
            "--end-to-end-repeats",
            "3",
        ])
        .expect("benchmark CLI");
        assert!(matches!(parsed.command, Command::BenchmarkQuery { .. }));
        assert!(
            Cli::try_parse_from([
                "rag",
                "benchmark-query",
                "--index",
                "index",
                "--query",
                "alpha activity",
            ])
            .is_err()
        );
    }

    #[test]
    fn latency_summary_uses_nearest_rank_percentiles() {
        let summary = summarize_latencies(3, vec![100, 10, 50, 20, 90]).unwrap();
        assert_eq!(summary.warmups, 3);
        assert_eq!(summary.samples, 5);
        assert_eq!(summary.min_micros, 10);
        assert_eq!(summary.p50_micros, 50);
        assert_eq!(summary.p95_micros, 100);
        assert_eq!(summary.max_micros, 100);
    }

    #[tokio::test]
    async fn benchmark_separates_repeated_fake_embedding_from_index_only_search() {
        let root = tempfile::tempdir().unwrap();
        let index = benchmark_fixture_index(root.path());
        let embedder = FakeQueryEmbedder {
            calls: AtomicUsize::new(0),
        };
        let report = benchmark_query_with_embedder(
            &index,
            &embedder,
            "alpha activity",
            "q-alpha",
            2,
            vec![],
            1,
            3,
            0,
            3,
            0,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 3);
        assert_eq!(report.query_embedding.calls, 3);
        assert_eq!(report.query_embedding.latency.samples, 3);
        assert_eq!(report.total_model_calls, 3);
        assert_eq!(report.dense_index_only.latency.samples, 3);
        assert_eq!(report.fused_index_only.latency.samples, 3);
        assert_eq!(report.lexical_index_only.latency.samples, 3);
        assert!(report.end_to_end_fused.is_none());
        assert_eq!(
            report.query_sha256,
            format!("{:x}", Sha256::digest(b"alpha activity"))
        );
        assert_eq!(
            report.embedding_profile_sha256,
            index.manifest.embedding_profile.sha256
        );
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.contains("alpha activity"));
        assert!(encoded.contains("q-alpha"));
    }

    #[tokio::test]
    async fn optional_end_to_end_benchmark_counts_each_additional_model_call() {
        let root = tempfile::tempdir().unwrap();
        let index = benchmark_fixture_index(root.path());
        let embedder = FakeQueryEmbedder {
            calls: AtomicUsize::new(0),
        };
        let report = benchmark_query_with_embedder(
            &index,
            &embedder,
            "alpha activity",
            "q-alpha-e2e",
            2,
            vec![],
            0,
            1,
            1,
            2,
            2,
        )
        .await
        .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 5);
        assert_eq!(report.total_model_calls, 5);
        assert_eq!(report.end_to_end_fused.unwrap().latency.samples, 2);
    }

    #[tokio::test]
    async fn benchmark_rejects_drifting_timed_query_vectors() {
        let root = tempfile::tempdir().unwrap();
        let index = benchmark_fixture_index(root.path());
        let embedder = DriftingQueryEmbedder {
            calls: AtomicUsize::new(0),
        };
        assert!(
            benchmark_query_with_embedder(
                &index,
                &embedder,
                "alpha activity",
                "q-drift",
                2,
                vec![],
                0,
                1,
                0,
                2,
                0,
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn overlap_report_counts_shared_top_k_documents() {
        let overlap = top_k_overlap_ids(
            vec!["a".into(), "b".into(), "c".into()],
            vec!["b".into(), "c".into(), "d".into()],
        );
        assert_eq!(overlap.shared_hits, 2);
        assert_eq!(overlap.overlap_fraction_of_full, 2.0 / 3.0);
        assert_eq!(overlap.jaccard, 0.5);
    }

    #[test]
    fn overlap_comparison_refuses_a_reduced_profile_from_another_parent() {
        let mut full = benchmark_fixture_index(tempfile::tempdir().unwrap().path())
            .manifest
            .embedding_profile;
        full.dimensions = 4_096;
        let mut reduced = full.clone();
        reduced.sha256 = "f".repeat(64);
        reduced.dimensions = 1_024;
        reduced.vector_derivation = Some(rag_embedding::VectorDerivation {
            parent_embedding_profile_sha256: "0".repeat(64),
            parent_dimensions: 4_096,
            transformation: rag_embedding::PREFIX_L2_NORMALIZE_V1.into(),
        });
        assert!(validate_reduced_profile_pair(&full, &reduced).is_err());
        reduced
            .vector_derivation
            .as_mut()
            .unwrap()
            .parent_embedding_profile_sha256 = full.sha256.clone();
        validate_reduced_profile_pair(&full, &reduced).unwrap();
    }
}
