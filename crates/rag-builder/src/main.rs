use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use arrow_array::{Array, RecordBatch, StringArray};
use chrono::DateTime;
use clap::{Parser, Subcommand, ValueEnum};
use rag_embedding::{
    EmbeddingCache, EmbeddingInput, LmStudioEmbedder, embed_resumable, parse_embedding_profile,
    try_compose_query,
};
use rag_index::{
    BuildScope, FastDocument, FastIndex, FastOccurrence, SearchFilters, SearchMode, SourceBinding,
    document_order_sha256, write_fast_index,
};
use rag_ocsf::{LocalSnapshotReader, SnapshotReader};
use rag_projection::{
    ComponentRef, EventTimeAvailability, ProjectedDocument, ProjectionContext, ProjectionInput,
    project,
};
use serde::Serialize;
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
        /// Retain the deterministic hash-min sample of at most COUNT documents.
        /// Every occurrence of a selected document is retained.
        #[arg(long, value_name = "COUNT", value_parser = positive_usize)]
        sample_documents: Option<usize>,
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
    Inspect {
        #[arg(long)]
        index: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum, Default)]
enum Mode {
    Dense,
    Lexical,
    #[default]
    Fused,
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

fn positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "expected a positive integer".to_owned())?;
    if parsed == 0 {
        return Err("expected a positive integer".to_owned());
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildSelection {
    Full,
    SampleDocuments(usize),
}

impl BuildSelection {
    fn from_limit(limit: Option<usize>) -> Self {
        limit.map_or(Self::Full, Self::SampleDocuments)
    }

    fn scope(self) -> BuildScope {
        match self {
            Self::Full => BuildScope::Full,
            Self::SampleDocuments(_) => BuildScope::Sample,
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

    fn merge(&mut self, other: &Self) {
        self.available += other.available;
        self.indexed_milliseconds += other.indexed_milliseconds;
        self.before_unix_epoch += other.before_unix_epoch;
        self.missing += other.missing;
        self.present_unparsed += other.present_unparsed;
    }

    fn total(&self) -> u64 {
        self.available + self.missing + self.present_unparsed
    }
}

#[derive(Debug, Serialize)]
struct SamplingAccounting {
    policy: &'static str,
    document_limit: Option<usize>,
    relation_budgets: BTreeMap<String, usize>,
    selected_by_relation: BTreeMap<String, usize>,
    selected_documents: usize,
    selected_occurrences: usize,
    selected_document_order_sha256: String,
}

#[derive(Debug, Serialize)]
struct BuildAccounting {
    source_rows_scanned: u64,
    projected_semantic_occurrences: u64,
    structured_only_occurrences: u64,
    source_timestamps: TimestampAccounting,
    indexed_timestamps: TimestampAccounting,
    sampling: SamplingAccounting,
}

#[derive(Debug)]
struct SelectedDocument {
    document: FastDocument,
    primary_relation: String,
    relations: BTreeSet<String>,
    occurrences: Vec<FastOccurrence>,
    timestamps: TimestampAccounting,
}

/// One-pass document reservoir. A document's opaque identity receives a
/// snapshot-bound SHA-256 priority; no semantic value, query, label, or qrel
/// is inspected. Once full, the greatest retained priority can only decrease.
/// Therefore a rejected or evicted document can never re-enter later, while
/// every document that survives to the final sample has been retained since
/// its first occurrence. This is what permits occurrence closure without a
/// parent replay.
#[derive(Debug)]
struct DocumentCollector {
    selection: BuildSelection,
    sample_seed: String,
    selected: BTreeMap<String, SelectedDocument>,
    /// Ordered by `(priority SHA-256, document ID)` so the greatest item is
    /// the deterministic eviction candidate.
    priorities: BTreeMap<String, BTreeSet<(String, String)>>,
    relation_budgets: BTreeMap<String, usize>,
    source_rows_scanned: u64,
    projected_semantic_occurrences: u64,
    source_timestamps: TimestampAccounting,
}

impl DocumentCollector {
    fn new(selection: BuildSelection, snapshot_sha256: &str, _relations: &[String]) -> Self {
        let relation_budgets = match selection {
            BuildSelection::Full => BTreeMap::new(),
            BuildSelection::SampleDocuments(limit) => BTreeMap::from([("*".into(), limit)]),
        };
        Self {
            selection,
            sample_seed: snapshot_sha256.to_owned(),
            selected: BTreeMap::new(),
            priorities: BTreeMap::new(),
            relation_budgets,
            source_rows_scanned: 0,
            projected_semantic_occurrences: 0,
            source_timestamps: TimestampAccounting::default(),
        }
    }

    fn observe_source_row(
        &mut self,
        availability: EventTimeAvailability,
        event_time_ms: Option<u64>,
    ) {
        self.source_rows_scanned += 1;
        self.source_timestamps.observe(availability, event_time_ms);
    }

    fn retain(
        &mut self,
        document: FastDocument,
        relation: &str,
        occurrence: FastOccurrence,
        availability: EventTimeAvailability,
    ) -> Result<()> {
        self.projected_semantic_occurrences += 1;
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
            selected
                .timestamps
                .observe(availability, occurrence.event_time_ms);
            selected.occurrences.push(occurrence);
            return Ok(());
        }

        let priority = sample_priority(&self.sample_seed, &document.document_id);
        if let BuildSelection::SampleDocuments(limit) = self.selection {
            let priorities = self.priorities.entry("*".to_owned()).or_default();
            if priorities.len() == limit {
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
        let mut timestamps = TimestampAccounting::default();
        timestamps.observe(availability, occurrence.event_time_ms);
        self.selected.insert(
            document.document_id.clone(),
            SelectedDocument {
                document,
                primary_relation: relation.to_owned(),
                relations,
                occurrences: vec![occurrence],
                timestamps,
            },
        );
        Ok(())
    }

    fn finish(self) -> Result<(Vec<FastDocument>, Vec<FastOccurrence>, BuildAccounting)> {
        let mut documents = Vec::with_capacity(self.selected.len());
        let mut occurrences = Vec::new();
        let mut indexed_timestamps = TimestampAccounting::default();
        let mut selected_by_relation = BTreeMap::<String, usize>::new();
        for mut selected in self.selected.into_values() {
            *selected_by_relation
                .entry(selected.primary_relation.clone())
                .or_default() += 1;
            selected.document.occurrence_count = selected.occurrences.len() as u64;
            selected.document.relations_json = serde_json::to_string(&selected.relations)?;
            indexed_timestamps.merge(&selected.timestamps);
            documents.push(selected.document);
            occurrences.append(&mut selected.occurrences);
        }
        documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        for (ordinal, document) in documents.iter_mut().enumerate() {
            document.vector_ordinal = ordinal as u64;
        }
        occurrences.sort_by(|left, right| left.occurrence_id.cmp(&right.occurrence_id));
        let selected_document_order_sha256 = document_order_sha256(&documents);
        let document_limit = match self.selection {
            BuildSelection::Full => None,
            BuildSelection::SampleDocuments(limit) => Some(limit),
        };
        let accounting = BuildAccounting {
            source_rows_scanned: self.source_rows_scanned,
            projected_semantic_occurrences: self.projected_semantic_occurrences,
            structured_only_occurrences: self
                .source_rows_scanned
                .saturating_sub(self.projected_semantic_occurrences),
            source_timestamps: self.source_timestamps,
            indexed_timestamps,
            sampling: SamplingAccounting {
                policy: if document_limit.is_some() {
                    "global_snapshot_bound_sha256_hash_min_v1"
                } else {
                    "full"
                },
                document_limit,
                relation_budgets: self.relation_budgets,
                selected_by_relation,
                selected_documents: documents.len(),
                selected_occurrences: occurrences.len(),
                selected_document_order_sha256,
            },
        };
        debug_assert_eq!(
            accounting.source_timestamps.total(),
            self.source_rows_scanned
        );
        debug_assert_eq!(
            accounting.indexed_timestamps.total(),
            occurrences.len() as u64
        );
        Ok((documents, occurrences, accounting))
    }
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
            sample_documents,
        } => {
            build(
                &snapshot,
                &out,
                &embedding_profile,
                &embedding_endpoint,
                &resume,
                embedding_batch_size,
                BuildSelection::from_limit(sample_documents),
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
            id: "livefire-ocsf-snapshot".into(),
            version: "1".into(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: ComponentRef {
            id: "livefire-ocsf-mapping".into(),
            version: "1".into(),
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
    let (documents, occurrences, accounting) = collector.finish()?;
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
    let cache = EmbeddingCache::open(resume)?;
    let embedder = LmStudioEmbedder::new(endpoint, &profile.model);
    let embedded = embed_resumable(&embedder, &cache, &profile, &inputs, batch_size).await?;
    let cache_hits = embedded.iter().filter(|item| item.from_cache).count();
    let vectors = embedded
        .into_iter()
        .map(|item| item.vector)
        .collect::<Vec<_>>();
    let manifest = write_fast_index(
        out,
        SourceBinding {
            snapshot_sha256: identity.snapshot_sha256.to_string(),
            mapping_sha256: identity.mapping_sha256.to_string(),
        },
        selection.scope(),
        &documents,
        &occurrences,
        &vectors,
        profile,
    )?;
    let report = serde_json::json!({
        "schema_version": "livefire.rag.fast-build-report/1",
        "index":manifest,
        "accounting": accounting,
        "cache_hits":cache_hits,
        "embedded":documents.len()-cache_hits
    });
    let mut report_bytes = serde_json::to_vec_pretty(&report)?;
    report_bytes.push(b'\n');
    let report_staging = out.join(format!(".build-report.json.tmp-{}", std::process::id()));
    fs::write(&report_staging, report_bytes)?;
    fs::rename(report_staging, out.join("build-report.json"))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn project_batch(
    batch: &RecordBatch,
    relation: &str,
    context: &ProjectionContext,
    collector: &mut DocumentCollector,
) -> Result<()> {
    let event_ids = strings(batch, "event_id")?;
    let json = strings(batch, "typed_event_json")?;
    let support = strings(batch, "support_ref")?;
    for row in 0..batch.num_rows() {
        let output = project(ProjectionInput {
            relation_name: relation,
            event_id: event_ids.value(row),
            typed_event_json: json.value(row),
            support_ref: support.value(row),
            context,
        })?;
        let event_time_ms = parse_event_time_ms(
            output.occurrence.event_time.as_deref(),
            output.occurrence.event_time_availability,
        )?;
        collector.observe_source_row(output.occurrence.event_time_availability, event_time_ms);
        if let Some(document) = output.document {
            let document_bytes = serde_json::to_vec(&document)?;
            let document_sha256 = format!("{:x}", Sha256::digest(document_bytes));
            let mut hasher = Sha256::new();
            hasher.update(context.snapshot.sha256.as_bytes());
            hasher.update([0]);
            hasher.update(relation.as_bytes());
            hasher.update([0]);
            hasher.update(event_ids.value(row).as_bytes());
            let occurrence = FastOccurrence {
                occurrence_id: format!("occ-{:x}", hasher.finalize()),
                document_id: document.document_id.clone(),
                event_time_ms,
                relation: relation.into(),
                exact_attributes_json: serde_json::to_string(&output.occurrence.exact_attributes)?,
                snapshot_sha256: context.snapshot.sha256.clone(),
                mapping_sha256: context.mapping_pack.sha256.clone(),
                event_id: event_ids.value(row).into(),
                support_ref: support.value(row).into(),
            };
            let fast_document = fast_document(document, document_sha256)?;
            collector.retain(
                fast_document,
                relation,
                occurrence,
                output.occurrence.event_time_availability,
            )?;
        }
    }
    Ok(())
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
    let search_mode: SearchMode = mode.into();
    let query_vector = if matches!(search_mode, SearchMode::Dense | SearchMode::Fused) {
        let embedder = LmStudioEmbedder::new(endpoint, &index.manifest.embedding_profile.model);
        let composed = try_compose_query(&index.manifest.embedding_profile, text)?;
        Some(
            rag_embedding::Embedder::embed(&embedder, &[composed])
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
    let hits = index.search(search_mode, text, query_vector.as_deref(), &filters, top_n)?;
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

    fn occurrence(id: &str, sequence: usize) -> FastOccurrence {
        FastOccurrence {
            occurrence_id: format!("occ-{id}-{sequence}"),
            document_id: id.to_owned(),
            event_time_ms: Some(sequence as u64),
            relation: "ocsf_process_activity".to_owned(),
            exact_attributes_json: "[]".to_owned(),
            snapshot_sha256: "a".repeat(64),
            mapping_sha256: "b".repeat(64),
            event_id: format!("evt-{id}-{sequence}"),
            support_ref: format!("sup-{id}-{sequence}"),
        }
    }

    fn collect(
        order: &[&str],
        limit: usize,
    ) -> (Vec<FastDocument>, Vec<FastOccurrence>, BuildAccounting) {
        let mut collector = DocumentCollector::new(
            BuildSelection::SampleDocuments(limit),
            &"a".repeat(64),
            &["ocsf_process_activity".to_owned()],
        );
        for sequence in 0..2 {
            for id in order {
                collector.observe_source_row(EventTimeAvailability::Available, Some(sequence));
                collector
                    .retain(
                        document(id),
                        "ocsf_process_activity",
                        occurrence(id, sequence as usize),
                        EventTimeAvailability::Available,
                    )
                    .expect("collect");
            }
        }
        collector.finish().expect("finish")
    }

    #[test]
    fn global_sample_fills_budget_without_allocating_to_empty_relations() {
        let relations = [
            "ocsf_network_activity".to_owned(),
            "ocsf_process_activity".to_owned(),
        ];
        let mut collector = DocumentCollector::new(
            BuildSelection::SampleDocuments(2),
            &"a".repeat(64),
            &relations,
        );
        for sequence in 0..10 {
            let id = format!("network-{sequence}");
            collector.observe_source_row(EventTimeAvailability::Available, Some(sequence));
            collector
                .retain(
                    document(&id),
                    "ocsf_network_activity",
                    occurrence(&id, sequence as usize),
                    EventTimeAvailability::Available,
                )
                .unwrap();
        }
        collector.observe_source_row(EventTimeAvailability::Available, Some(100));
        collector
            .retain(
                document("process-one"),
                "ocsf_process_activity",
                occurrence("process-one", 100),
                EventTimeAvailability::Available,
            )
            .unwrap();
        let (documents, _, accounting) = collector.finish().unwrap();
        assert_eq!(documents.len(), 2);
        assert_eq!(
            accounting
                .sampling
                .selected_by_relation
                .values()
                .sum::<usize>(),
            2
        );
        assert_eq!(
            accounting.sampling.relation_budgets,
            BTreeMap::from([("*".into(), 2)])
        );
    }

    #[test]
    fn hash_min_sample_is_bounded_deterministic_and_occurrence_complete() {
        let forward = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"];
        let reverse = ["doc-e", "doc-d", "doc-c", "doc-b", "doc-a"];
        let (documents, occurrences, accounting) = collect(&forward, 2);
        let (reverse_documents, reverse_occurrences, _) = collect(&reverse, 2);

        let ids = documents
            .iter()
            .map(|document| document.document_id.as_str())
            .collect::<Vec<_>>();
        let reverse_ids = reverse_documents
            .iter()
            .map(|document| document.document_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, reverse_ids);
        assert_eq!(documents.len(), 2);
        assert_eq!(occurrences, reverse_occurrences);
        assert_eq!(occurrences.len(), 4);
        assert!(
            documents
                .iter()
                .all(|document| document.occurrence_count == 2)
        );
        for document in &documents {
            assert_eq!(
                occurrences
                    .iter()
                    .filter(|occurrence| occurrence.document_id == document.document_id)
                    .count(),
                2
            );
        }
        assert_eq!(accounting.source_rows_scanned, 10);
        assert_eq!(accounting.projected_semantic_occurrences, 10);
        assert_eq!(accounting.sampling.document_limit, Some(2));
        assert_eq!(accounting.sampling.selected_occurrences, 4);
        assert_eq!(
            accounting.sampling.relation_budgets,
            BTreeMap::from([("*".to_owned(), 2)])
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
    fn cli_requires_an_explicit_positive_sample_document_limit() {
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
                sample_documents: None,
                ..
            }
        ));

        let mut sampled_args = base.to_vec();
        sampled_args.extend(["--sample-documents", "37"]);
        let sampled = Cli::try_parse_from(sampled_args).expect("sample CLI");
        assert!(matches!(
            sampled.command,
            Command::Build {
                sample_documents: Some(37),
                ..
            }
        ));

        let mut zero_args = base.to_vec();
        zero_args.extend(["--sample-documents", "0"]);
        assert!(Cli::try_parse_from(zero_args).is_err());

        let mut old_args = base.to_vec();
        old_args.push("--sample");
        assert!(Cli::try_parse_from(old_args).is_err());
    }
}
