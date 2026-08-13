use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use arrow_array::{Array, RecordBatch, StringArray};
use chrono::DateTime;
use clap::{Parser, Subcommand, ValueEnum};
use rag_embedding::{
    EmbeddingCache, EmbeddingInput, LmStudioEmbedder, ensure_cached, parse_embedding_profile,
    try_compose_query,
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

#[derive(Parser)]
#[command(name = "rag", about = "Fast experimental RAG builder and query CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Dense,
    Lexical,
    #[default]
    Fused,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchQueryRequest {
    query_id: String,
    query: String,
    mode: Mode,
    top_n: usize,
    #[serde(default)]
    relations: Vec<String>,
}

impl BatchQueryRequest {
    fn validate(&self) -> Result<()> {
        if self.query_id.is_empty()
            || self.query.is_empty()
            || self.top_n == 0
            || self.top_n > 100
            || self.relations.iter().any(String::is_empty)
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
    #[error("batch query plan contains an invalid row")]
    InvalidBatchQuery,
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
        Command::BatchQuery {
            index,
            requests,
            embedding_endpoint,
        } => batch_query(&index, &requests, &embedding_endpoint).await,
        Command::Inspect { index } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&FastIndex::open(&index)?.manifest)?
            );
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

async fn search_index(
    index: &FastIndex,
    embedder: &LmStudioEmbedder,
    text: &str,
    mode: Mode,
    top_n: usize,
    relations: Vec<String>,
) -> Result<Vec<rag_index::SearchHit>> {
    let search_mode: SearchMode = mode.into();
    let query_vector = if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) {
        let composed = try_compose_query(&index.manifest.embedding_profile, text)?;
        Some(
            rag_embedding::Embedder::embed(embedder, &[composed])
                .await?
                .remove(0),
        )
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
    let requests = BufReader::new(File::open(requests_path)?)
        .lines()
        .map(|line| {
            let line = line?;
            let request = serde_json::from_str::<BatchQueryRequest>(&line)?;
            request.validate()?;
            Ok(request)
        })
        .collect::<Result<Vec<_>>>()?;
    if requests.is_empty() {
        return Err(Error::InvalidBatchQuery);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
            "query_id":"q-1","query":"x","mode":"dense","top_n":101
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
    }
}
