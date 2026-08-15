#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
};

use arrow_array::{Array, StringArray};
use clap::Parser;
use rag_ocsf::{LocalSnapshotReader, SnapshotReader};
use rag_pipeline::{
    ComponentRef, Digest, atomic_write, canonical_digest, component_digest, digest_bytes,
};
use rag_projection::{
    PROJECTION_POLICY_ID, PROJECTION_POLICY_VERSION, ProjectionContext, ProjectionInput, project,
    project_document_summary,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest as _, Sha256};

const REPORT_SCHEMA: &str = "livefire.rag.projection-parity-report/1";
const SAMPLE_ALGORITHM: &str = "snapshot_relation_ordinal_sha256_min_k_v1";
const SAMPLE_RANK_DOMAIN: &[u8] = b"livefire.rag.projection-parity-sample-rank/1\0";
const MISMATCH_FIELDS: [&str; 8] = [
    "searchable",
    "document_kind",
    "document_id_sha256",
    "semantic_group_id_sha256",
    "semantic_group_sha256_sha256",
    "semantic_text_sha256",
    "facets_sha256",
    "event_time_summary_sha256",
];

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Parser)]
#[command(about = "Compare Python and Rust projection on deterministic real snapshot rows")]
struct Args {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16)]
    samples_per_relation: u32,
    #[arg(long, default_value = ".")]
    repository: PathBuf,
    #[arg(long, default_value = "uv")]
    uv: PathBuf,
}

#[derive(Debug, Clone)]
struct SampleRow {
    relation: String,
    relation_rows: u64,
    row_ordinal: u64,
    sample_rank_sha256: String,
    event_id: String,
    typed_event_json: String,
    support_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionDigest {
    sample_id: String,
    searchable: bool,
    document_kind: String,
    document_id_sha256: String,
    semantic_group_id_sha256: String,
    semantic_group_sha256_sha256: String,
    semantic_text_sha256: String,
    facets_sha256: String,
    event_time_summary_sha256: String,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequest<'a> {
    sample_id: &'a str,
    relation: &'a str,
    event_id: &'a str,
    typed_event_json: &'a str,
    support_ref: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationParity {
    relation: String,
    source_rows: u64,
    sampled_rows: u64,
    matched_rows: u64,
    mismatched_rows: u64,
    sample_membership_sha256: Digest,
    rust_projection_sha256: Digest,
    python_projection_sha256: Digest,
    mismatch_field_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionParityReport {
    schema_version: String,
    component_sha256: Digest,
    source_snapshot: ComponentRef,
    mapping: ComponentRef,
    projection_policy: ComponentRef,
    sampler: ComponentRef,
    rust_implementation: ComponentRef,
    python_implementation: ComponentRef,
    samples_per_relation: u32,
    relation_count: u64,
    sampled_rows: u64,
    matched_rows: u64,
    mismatched_rows: u64,
    sample_membership_sha256: Digest,
    rust_projection_sha256: Digest,
    python_projection_sha256: Digest,
    relations: Vec<RelationParity>,
    parity: bool,
}

impl ProjectionParityReport {
    fn validate(&self) -> AnyResult<()> {
        if self.schema_version != REPORT_SCHEMA
            || self.samples_per_relation == 0
            || self.samples_per_relation > 256
            || self.relations.is_empty()
            || usize::try_from(self.relation_count).ok() != Some(self.relations.len())
        {
            return Err("invalid projection parity report header".into());
        }
        for component in [
            &self.source_snapshot,
            &self.mapping,
            &self.projection_policy,
            &self.sampler,
            &self.rust_implementation,
            &self.python_implementation,
        ] {
            component.validate()?;
        }
        let mut sampled = 0_u64;
        let mut matched = 0_u64;
        let mut mismatched = 0_u64;
        let mut previous: Option<&str> = None;
        let expected_fields = MISMATCH_FIELDS.into_iter().collect::<BTreeSet<_>>();
        for relation in &self.relations {
            if relation.relation.is_empty()
                || previous.is_some_and(|value| value >= relation.relation.as_str())
                || relation.sampled_rows == 0
                || relation.sampled_rows > u64::from(self.samples_per_relation)
                || relation.sampled_rows > relation.source_rows
                || relation.matched_rows + relation.mismatched_rows != relation.sampled_rows
                || relation
                    .mismatch_field_counts
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    != expected_fields
                || relation
                    .mismatch_field_counts
                    .values()
                    .any(|count| *count > relation.mismatched_rows)
            {
                return Err("invalid projection parity relation accounting".into());
            }
            sampled = sampled
                .checked_add(relation.sampled_rows)
                .ok_or("row total")?;
            matched = matched
                .checked_add(relation.matched_rows)
                .ok_or("row total")?;
            mismatched = mismatched
                .checked_add(relation.mismatched_rows)
                .ok_or("row total")?;
            previous = Some(&relation.relation);
        }
        if sampled != self.sampled_rows
            || matched != self.matched_rows
            || mismatched != self.mismatched_rows
            || matched + mismatched != sampled
            || self.parity != (mismatched == 0)
            || self.component_sha256 != component_digest(self)?
        {
            return Err("invalid projection parity report closure".into());
        }
        Ok(())
    }
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("projection parity check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> AnyResult<bool> {
    if args.samples_per_relation == 0 || args.samples_per_relation > 256 {
        return Err("samples per relation must be between 1 and 256".into());
    }
    let repository = args.repository.canonicalize()?;
    let reader = LocalSnapshotReader::open(&args.snapshot)?;
    let identity = reader.identity();
    let context = ProjectionContext {
        snapshot: rag_projection::ComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: identity.snapshot_sha256.to_string(),
            uri: None,
        },
        mapping_pack: rag_projection::ComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: identity.mapping_sha256.to_string(),
            uri: None,
        },
    };
    let mut samples = Vec::new();
    for relation in reader.typed_relations() {
        if project_document_summary(&relation.name, "{}", &context)?
            .document
            .is_none()
        {
            continue;
        }
        samples.extend(load_relation_samples(
            &reader,
            relation,
            args.samples_per_relation,
            identity.snapshot_sha256.as_str(),
        )?);
    }
    samples.sort_by(|left, right| {
        left.relation
            .cmp(&right.relation)
            .then_with(|| left.sample_rank_sha256.cmp(&right.sample_rank_sha256))
            .then_with(|| left.row_ordinal.cmp(&right.row_ordinal))
    });
    if samples.is_empty() {
        return Err("snapshot has no searchable relation rows".into());
    }

    let sample_ids = samples
        .iter()
        .map(sample_id)
        .collect::<AnyResult<Vec<_>>>()?;
    let rust = samples
        .iter()
        .zip(&sample_ids)
        .map(|(sample, id)| rust_projection_digest(sample, id, &context))
        .collect::<AnyResult<Vec<_>>>()?;
    let python = python_projection_digests(&args, &repository, &samples, &sample_ids)?;
    let report = build_report(
        &repository,
        &reader,
        args.samples_per_relation,
        &samples,
        rust,
        python,
    )?;
    report.validate()?;
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    atomic_write(&args.out, &bytes)?;
    println!(
        "projection parity: {} sampled rows, {} mismatches, report {}",
        report.sampled_rows,
        report.mismatched_rows,
        args.out.display()
    );
    Ok(report.parity)
}

fn sample_rank(snapshot_sha256: &str, relation: &str, ordinal: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SAMPLE_RANK_DOMAIN);
    hasher.update(snapshot_sha256.as_bytes());
    hasher.update([0]);
    hasher.update(relation.as_bytes());
    hasher.update([0]);
    hasher.update(ordinal.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn selected_ordinals(
    snapshot_sha256: &str,
    relation: &str,
    rows: u64,
    count: u32,
) -> Vec<(String, u64)> {
    let mut heap = BinaryHeap::with_capacity(count as usize + 1);
    for ordinal in 0..rows {
        heap.push((sample_rank(snapshot_sha256, relation, ordinal), ordinal));
        if heap.len() > count as usize {
            heap.pop();
        }
    }
    let mut selected = heap.into_vec();
    selected.sort();
    selected
}

fn load_relation_samples(
    reader: &LocalSnapshotReader,
    relation: &rag_ocsf::OcsfRelation,
    count: u32,
    snapshot_sha256: &str,
) -> AnyResult<Vec<SampleRow>> {
    let selected = selected_ordinals(
        snapshot_sha256,
        &relation.name,
        relation.rows,
        count.min(u32::try_from(relation.rows).unwrap_or(u32::MAX)),
    );
    let selected_by_ordinal = selected
        .iter()
        .map(|(rank, ordinal)| (*ordinal, rank.as_str()))
        .collect::<BTreeMap<_, _>>();
    let object = reader.admit_object(relation)?;
    let mut found = BTreeMap::new();
    for group in object.row_groups() {
        let group_end = group.first_row + group.rows;
        if selected_by_ordinal
            .range(group.first_row..group_end)
            .next()
            .is_none()
        {
            continue;
        }
        let mut cursor = group.first_row;
        for batch in object.scan_row_group(
            group.ordinal,
            &["event_id", "typed_event_json", "support_ref"],
        )? {
            let batch = batch?;
            let schema = batch.schema();
            let strings = |name: &str| -> AnyResult<&StringArray> {
                let index = schema.index_of(name)?;
                batch
                    .column(index)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| format!("column {name} is not UTF-8").into())
            };
            let event_ids = strings("event_id")?;
            let typed_events = strings("typed_event_json")?;
            let support_refs = strings("support_ref")?;
            let batch_end = cursor + u64::try_from(batch.num_rows())?;
            for (&ordinal, rank) in selected_by_ordinal.range(cursor..batch_end) {
                let index = usize::try_from(ordinal - cursor)?;
                found.insert(
                    ordinal,
                    SampleRow {
                        relation: relation.name.clone(),
                        relation_rows: relation.rows,
                        row_ordinal: ordinal,
                        sample_rank_sha256: (*rank).to_owned(),
                        event_id: event_ids.value(index).to_owned(),
                        typed_event_json: typed_events.value(index).to_owned(),
                        support_ref: support_refs.value(index).to_owned(),
                    },
                );
            }
            cursor = batch_end;
        }
    }
    if found.len() != selected.len() {
        return Err("selected snapshot rows were not read exactly once".into());
    }
    Ok(selected
        .into_iter()
        .map(|(_, ordinal)| {
            found
                .remove(&ordinal)
                .expect("checked selected row coverage")
        })
        .collect())
}

fn sample_id(sample: &SampleRow) -> AnyResult<String> {
    Ok(canonical_digest(&json!({
        "schema_version": "livefire.rag.projection-parity-sample/1",
        "relation": sample.relation,
        "relation_rows": sample.relation_rows,
        "row_ordinal": sample.row_ordinal,
        "sample_rank_sha256": sample.sample_rank_sha256,
        "event_id_sha256": digest_bytes(sample.event_id.as_bytes()),
        "typed_event_json_sha256": digest_bytes(sample.typed_event_json.as_bytes()),
        "support_ref_sha256": digest_bytes(sample.support_ref.as_bytes()),
    }))?
    .to_string())
}

fn rust_projection_digest(
    sample: &SampleRow,
    sample_id: &str,
    context: &ProjectionContext,
) -> AnyResult<ProjectionDigest> {
    let output = project(ProjectionInput {
        relation_name: &sample.relation,
        event_id: &sample.event_id,
        typed_event_json: &sample.typed_event_json,
        support_ref: &sample.support_ref,
        context,
    })?;
    let searchable = output.document.is_some();
    let document_id = output
        .document
        .as_ref()
        .map_or("", |document| document.document_id.as_str());
    let semantic_text = output
        .document
        .as_ref()
        .map_or("", |document| document.semantic_text.as_str());
    let facets = output.document.as_ref().map_or_else(
        || json!({"action":"","target":"","context":"","outcome":""}),
        |document| serde_json::to_value(&document.facets).expect("serializable facets"),
    );
    let kind = serde_json::to_value(output.occurrence.document_kind)?
        .as_str()
        .ok_or("document kind is not a string")?
        .to_owned();
    let event_summary = json!({
        "event_time": output.occurrence.event_time,
        "event_time_availability": output.occurrence.event_time_availability,
    });
    Ok(ProjectionDigest {
        sample_id: sample_id.to_owned(),
        searchable,
        document_kind: kind,
        document_id_sha256: digest_bytes(document_id.as_bytes()).to_string(),
        semantic_group_id_sha256: digest_bytes(output.occurrence.semantic_group_id.as_bytes())
            .to_string(),
        semantic_group_sha256_sha256: digest_bytes(
            output.occurrence.semantic_group_sha256.as_bytes(),
        )
        .to_string(),
        semantic_text_sha256: digest_bytes(semantic_text.as_bytes()).to_string(),
        facets_sha256: canonical_digest(&facets)?.to_string(),
        event_time_summary_sha256: canonical_digest(&event_summary)?.to_string(),
    })
}

fn python_projection_digests(
    args: &Args,
    repository: &Path,
    samples: &[SampleRow],
    sample_ids: &[String],
) -> AnyResult<Vec<ProjectionDigest>> {
    let worker_input = python_worker_input(samples, sample_ids)?;
    let mut child = Command::new(&args.uv)
        .args(["run", "python"])
        .arg(repository.join("tools/projection_parity_worker.py"))
        .current_dir(repository)
        .env("PYTHONPATH", repository.join("src"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("python worker stdin")?;
    let writer = match std::thread::Builder::new()
        .name("projection-parity-stdin".into())
        .spawn(move || stdin.write_all(&worker_input))
    {
        Ok(writer) => writer,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error.into());
        }
    };
    // Drain stdout while the writer thread feeds stdin. Waiting until the
    // complete request was written can deadlock when both OS pipes fill.
    let output_result = child.wait_with_output();
    let writer_result = writer
        .join()
        .map_err(|_| "python projection worker stdin thread panicked")?;
    writer_result?;
    let output = output_result?;
    if !output.status.success() {
        return Err("python projection worker failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let values = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<ProjectionDigest>, _>>()?;
    if values.len() != samples.len()
        || values
            .iter()
            .zip(sample_ids)
            .any(|(value, expected)| &value.sample_id != expected)
    {
        return Err("python projection worker response coverage".into());
    }
    Ok(values)
}

fn python_worker_input(samples: &[SampleRow], sample_ids: &[String]) -> AnyResult<Vec<u8>> {
    if samples.len() != sample_ids.len() {
        return Err("python projection worker request coverage".into());
    }
    let mut input = Vec::new();
    for (sample, sample_id) in samples.iter().zip(sample_ids) {
        serde_json::to_writer(
            &mut input,
            &WorkerRequest {
                sample_id,
                relation: &sample.relation,
                event_id: &sample.event_id,
                typed_event_json: &sample.typed_event_json,
                support_ref: &sample.support_ref,
            },
        )?;
        input.push(b'\n');
    }
    Ok(input)
}

fn implementation_ref(repository: &Path, id: &str, relative: &str) -> AnyResult<ComponentRef> {
    Ok(ComponentRef {
        id: id.into(),
        version: "working-tree".into(),
        sha256: digest_bytes(&fs::read(repository.join(relative))?),
    })
}

fn build_report(
    repository: &Path,
    reader: &LocalSnapshotReader,
    samples_per_relation: u32,
    samples: &[SampleRow],
    rust: Vec<ProjectionDigest>,
    python: Vec<ProjectionDigest>,
) -> AnyResult<ProjectionParityReport> {
    let identity = reader.identity();
    let policy_value: serde_json::Value = serde_json::from_slice(&fs::read(
        repository.join("specs/evidence-projection-policy.v2.json"),
    )?)?;
    let projection_policy = ComponentRef {
        id: PROJECTION_POLICY_ID.into(),
        version: PROJECTION_POLICY_VERSION.into(),
        sha256: canonical_digest(&policy_value)?,
    };
    let sampler_material = json!({
        "id": "livefire.rag.projection-parity-sampler",
        "version": "1",
        "algorithm": SAMPLE_ALGORITHM,
    });
    let mut relations = Vec::new();
    let mut sample_start = 0_usize;
    while sample_start < samples.len() {
        let relation_name = &samples[sample_start].relation;
        let sample_end = samples[sample_start..]
            .iter()
            .position(|sample| &sample.relation != relation_name)
            .map_or(samples.len(), |offset| sample_start + offset);
        let relation_samples = &samples[sample_start..sample_end];
        let rust_values = &rust[sample_start..sample_end];
        let python_values = &python[sample_start..sample_end];
        let mut mismatch_field_counts = MISMATCH_FIELDS
            .into_iter()
            .map(|field| (field.to_owned(), 0_u64))
            .collect::<BTreeMap<_, _>>();
        let mut mismatched_rows = 0_u64;
        for (rust, python) in rust_values.iter().zip(python_values) {
            let differences = field_differences(rust, python);
            if !differences.is_empty() {
                mismatched_rows += 1;
                for field in differences {
                    *mismatch_field_counts.get_mut(field).expect("closed fields") += 1;
                }
            }
        }
        relations.push(RelationParity {
            relation: relation_name.clone(),
            source_rows: relation_samples[0].relation_rows,
            sampled_rows: u64::try_from(relation_samples.len())?,
            matched_rows: u64::try_from(relation_samples.len())? - mismatched_rows,
            mismatched_rows,
            sample_membership_sha256: canonical_digest(
                &relation_samples
                    .iter()
                    .map(sample_id)
                    .collect::<AnyResult<Vec<_>>>()?,
            )?,
            rust_projection_sha256: canonical_digest(&rust_values)?,
            python_projection_sha256: canonical_digest(&python_values)?,
            mismatch_field_counts,
        });
        sample_start = sample_end;
    }
    let sampled_rows = u64::try_from(samples.len())?;
    let mismatched_rows = relations
        .iter()
        .map(|relation| relation.mismatched_rows)
        .sum();
    let mut report = ProjectionParityReport {
        schema_version: REPORT_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed projection parity report"),
        source_snapshot: ComponentRef {
            id: identity.snapshot_id.clone(),
            version: identity.snapshot_version.clone(),
            sha256: Digest::new(identity.snapshot_sha256.to_string())?,
        },
        mapping: ComponentRef {
            id: identity.mapping_id.clone(),
            version: identity.mapping_version.clone(),
            sha256: Digest::new(identity.mapping_sha256.to_string())?,
        },
        projection_policy,
        sampler: ComponentRef {
            id: "livefire.rag.projection-parity-sampler".into(),
            version: "1".into(),
            sha256: canonical_digest(&sampler_material)?,
        },
        rust_implementation: implementation_ref(
            repository,
            "livefire.rag.rust-projection-implementation",
            "crates/rag-projection/src/lib.rs",
        )?,
        python_implementation: implementation_ref(
            repository,
            "livefire.rag.python-projection-implementation",
            "src/livefire_rag/evidence_projection.py",
        )?,
        samples_per_relation,
        relation_count: u64::try_from(relations.len())?,
        sampled_rows,
        matched_rows: sampled_rows - mismatched_rows,
        mismatched_rows,
        sample_membership_sha256: canonical_digest(
            &samples
                .iter()
                .map(sample_id)
                .collect::<AnyResult<Vec<_>>>()?,
        )?,
        rust_projection_sha256: canonical_digest(&rust)?,
        python_projection_sha256: canonical_digest(&python)?,
        relations,
        parity: mismatched_rows == 0,
    };
    report.component_sha256 = component_digest(&report)?;
    Ok(report)
}

fn field_differences(rust: &ProjectionDigest, python: &ProjectionDigest) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if rust.searchable != python.searchable {
        fields.push("searchable");
    }
    if rust.document_kind != python.document_kind {
        fields.push("document_kind");
    }
    if rust.document_id_sha256 != python.document_id_sha256 {
        fields.push("document_id_sha256");
    }
    if rust.semantic_group_id_sha256 != python.semantic_group_id_sha256 {
        fields.push("semantic_group_id_sha256");
    }
    if rust.semantic_group_sha256_sha256 != python.semantic_group_sha256_sha256 {
        fields.push("semantic_group_sha256_sha256");
    }
    if rust.semantic_text_sha256 != python.semantic_text_sha256 {
        fields.push("semantic_text_sha256");
    }
    if rust.facets_sha256 != python.facets_sha256 {
        fields.push("facets_sha256");
    }
    if rust.event_time_summary_sha256 != python.event_time_summary_sha256 {
        fields.push("event_time_summary_sha256");
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinal_sampler_is_deterministic_bounded_and_unique() {
        let first = selected_ordinals(&"a".repeat(64), "ocsf_process_activity", 10_000, 16);
        let second = selected_ordinals(&"a".repeat(64), "ocsf_process_activity", 10_000, 16);
        assert_eq!(first, second);
        assert_eq!(first.len(), 16);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            first
                .iter()
                .map(|(_, ordinal)| ordinal)
                .collect::<BTreeSet<_>>()
                .len(),
            16
        );
        assert_eq!(selected_ordinals(&"a".repeat(64), "events", 3, 16).len(), 3);
    }

    #[test]
    fn mismatch_field_reporting_is_closed_and_sanitized() {
        let base = ProjectionDigest {
            sample_id: "a".repeat(64),
            searchable: true,
            document_kind: "activity".into(),
            document_id_sha256: "b".repeat(64),
            semantic_group_id_sha256: "c".repeat(64),
            semantic_group_sha256_sha256: "d".repeat(64),
            semantic_text_sha256: "e".repeat(64),
            facets_sha256: "f".repeat(64),
            event_time_summary_sha256: "0".repeat(64),
        };
        assert!(field_differences(&base, &base).is_empty());
        let mut changed = base.clone();
        changed.semantic_text_sha256 = "1".repeat(64);
        changed.searchable = false;
        assert_eq!(
            field_differences(&base, &changed),
            ["searchable", "semantic_text_sha256"]
        );
        let encoded = serde_json::to_string(&changed).unwrap();
        assert!(!encoded.contains("PowerShell"));
        assert!(!encoded.contains("typed_event_json"));
    }

    #[test]
    fn report_schema_is_parseable() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../specs/projection-parity-report.v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["$id"],
            "https://livefire.dev/rag/projection-parity-report.v1.schema.json"
        );
    }

    #[test]
    fn synthetic_same_row_matches_python_and_rust_without_exposing_content() {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let sample = SampleRow {
            relation: "ocsf_process_activity".into(),
            relation_rows: 1,
            row_ordinal: 0,
            sample_rank_sha256: "a".repeat(64),
            event_id: "synthetic-event".into(),
            typed_event_json: serde_json::to_string(&json!({
                "activity_name": "Create",
                "process": {"cmd_line": "pwsh -c Get-Process"},
                "status": "success",
                "time": 1_700_000_000,
            }))
            .unwrap(),
            support_ref: "synthetic-support".into(),
        };
        let id = sample_id(&sample).unwrap();
        let context = ProjectionContext {
            snapshot: rag_projection::ComponentRef {
                id: "snapshot".into(),
                version: "1".into(),
                sha256: "b".repeat(64),
                uri: None,
            },
            mapping_pack: rag_projection::ComponentRef {
                id: "mapping".into(),
                version: "1".into(),
                sha256: "c".repeat(64),
                uri: None,
            },
        };
        let rust = rust_projection_digest(&sample, &id, &context).unwrap();
        let args = Args {
            snapshot: PathBuf::new(),
            out: PathBuf::new(),
            samples_per_relation: 1,
            repository: repository.clone(),
            uv: PathBuf::from("uv"),
        };
        let python = python_projection_digests(
            &args,
            &repository,
            std::slice::from_ref(&sample),
            std::slice::from_ref(&id),
        )
        .unwrap();
        assert_eq!(python, [rust]);
        let sanitized = serde_json::to_string(&python).unwrap();
        assert!(!sanitized.contains("Get-Process"));
        assert!(!sanitized.contains("synthetic-event"));
        assert!(!sanitized.contains("synthetic-support"));
    }

    #[test]
    fn python_worker_drains_output_while_input_exceeds_pipe_capacity() {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let samples = (0..256_u64)
            .map(|ordinal| SampleRow {
                relation: "ocsf_process_activity".into(),
                relation_rows: 256,
                row_ordinal: ordinal,
                sample_rank_sha256: format!("{ordinal:064x}"),
                event_id: format!("event-{ordinal}"),
                typed_event_json: serde_json::to_string(&json!({
                    "activity_name": "Create",
                    "message": "x".repeat(2_048),
                    "process": {"name": format!("process-{ordinal}")},
                    "time": 1_700_000_000_u64 + ordinal,
                }))
                .unwrap(),
                support_ref: format!("support-{ordinal}"),
            })
            .collect::<Vec<_>>();
        let sample_ids = samples
            .iter()
            .map(sample_id)
            .collect::<AnyResult<Vec<_>>>()
            .unwrap();
        let worker_input = python_worker_input(&samples, &sample_ids).unwrap();
        assert!(worker_input.len() > 64 * 1_024);
        let args = Args {
            snapshot: PathBuf::new(),
            out: PathBuf::new(),
            samples_per_relation: 16,
            repository: repository.clone(),
            uv: PathBuf::from("uv"),
        };
        let output = python_projection_digests(&args, &repository, &samples, &sample_ids).unwrap();
        assert_eq!(output.len(), samples.len());
        let output_bytes = output
            .iter()
            .map(|value| serde_json::to_vec(value).unwrap().len() + 1)
            .sum::<usize>();
        assert!(output_bytes > 64 * 1_024);
    }
}
