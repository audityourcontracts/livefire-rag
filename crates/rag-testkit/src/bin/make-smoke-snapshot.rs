//! Generate a tiny format-faithful OCSF snapshot for the real builder smoke.

use std::{
    fs,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use rag_projection::{ComponentRef, ProjectionContext, ProjectionInput, project};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(name = "make-smoke-snapshot")]
struct Arguments {
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone)]
struct EventFixture {
    relation: &'static str,
    event_id: &'static str,
    support_ref: &'static str,
    typed_event_json: &'static str,
    query_id: &'static str,
    query: &'static str,
}

#[derive(Serialize)]
struct ObjectRef {
    relation: String,
    path: String,
    rows: u64,
    sha256: String,
    logical_sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if arguments.out.exists() {
        return Err("output already exists".into());
    }
    fs::create_dir_all(arguments.out.join("semantic"))?;
    let events = fixtures();
    let mut objects = Vec::new();

    let event_ids = events.iter().map(|row| row.event_id).collect::<Vec<_>>();
    objects.push(write_string_relation(
        &arguments.out,
        "events",
        &["event_id"],
        &[event_ids],
    )?);
    for relation in [
        "event_facets",
        "entities",
        "observables",
        "participants",
        "event_observables",
        "relationships",
    ] {
        objects.push(write_string_relation(
            &arguments.out,
            relation,
            &["id"],
            &[Vec::new()],
        )?);
    }
    for event in &events {
        objects.push(write_string_relation(
            &arguments.out,
            event.relation,
            &["event_id", "typed_event_json", "support_ref"],
            &[
                vec![event.event_id],
                vec![event.typed_event_json],
                vec![event.support_ref],
            ],
        )?);
    }
    objects.sort_by(|left, right| left.relation.cmp(&right.relation));
    let snapshot_sha256 = digest(&serde_json::to_vec(&objects)?);
    let mapping_sha256 = "b".repeat(64);
    let receipt = serde_json::json!({
        "schema_version": 1,
        "snapshot_manifest": {
            "schema_version": 1,
            "dataset_sha256": "a".repeat(64),
            "ocsf_schema_sha256": "c".repeat(64),
            "extension_pack_sha256": "d".repeat(64),
            "mapping_pack_sha256": mapping_sha256,
            "relation_contract_sha256": "e".repeat(64),
            "objects": objects,
            "logical_sha256": snapshot_sha256,
        },
        "output_logical_sha256": snapshot_sha256,
        "runnable_snapshot": {
            "component":{"id":"com.ayc.livefire-ocsf.synthetic-smoke-snapshot","version":"1","sha256":snapshot_sha256},
            "dataset_sha256":"a".repeat(64),
            "mapping_pack":{"id":"com.ayc.livefire-ocsf.synthetic-mapping-pack","version":"1","sha256":mapping_sha256},
            "relation_contract":{"id":"com.ayc.livefire-ocsf.synthetic-relation-contract","version":"1","sha256":"e".repeat(64)},
            "normalized_events":events.len(),
            "source_rows":events.len()
        },
        "closure":{
            "input_rows":events.len(),"mapped_source_records":events.len(),
            "mapped_events":events.len(),"event_rows":events.len(),
            "rejected_malformed_records":0,"unsupported_records":0,
            "unresolved_provenance_fields":0,"provenance_digest_mismatches":0
        },
        "completeness_receipt": {
            "dataset_sha256":"a".repeat(64),
            "mapping_pack_sha256":mapping_sha256,
            "normalized_snapshot_sha256":snapshot_sha256,
            "relation_contract_sha256":"e".repeat(64),
            "metrics":{
                "source_rows":events.len(),"mapped_source_records":events.len(),
                "rejected_malformed_records":0,"normalized_events":events.len()
            }
        }
    });
    fs::write(
        arguments.out.join("build-receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    write_evaluation_fixture(&arguments.out, &events, &snapshot_sha256, &mapping_sha256)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "snapshot": arguments.out,
            "events": events.len(),
            "snapshot_sha256": snapshot_sha256,
            "queries": "smoke-queries.jsonl",
            "qrels": "smoke-qrels.jsonl"
        }))?
    );
    Ok(())
}

fn fixtures() -> Vec<EventFixture> {
    vec![
        EventFixture {
            relation: "ocsf_process_activity",
            event_id: "evt_smoke_process",
            support_ref: "sup_smoke_process",
            typed_event_json: r#"{"semantic_class":"process","ocsf":{"activity_id":1,"time":1534778062000,"process":{"name":"powershell.exe","cmd_line":"powershell -EncodedCommand SQBFAFgA"},"device":{"hostname":"workstation-7"}},"image":"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"}"#,
            query_id: "q-process-encoded",
            query: "encoded PowerShell command execution on a workstation",
        },
        EventFixture {
            relation: "ocsf_authentication",
            event_id: "evt_smoke_auth",
            support_ref: "sup_smoke_auth",
            typed_event_json: r#"{"semantic_class":"authentication","ocsf":{"activity_id":2,"time":1534779000000,"status":"failure","actor":{"user":{"name":"contractor"}},"src_endpoint":{"ip":"203.0.113.44"},"device":{"hostname":"vpn-gateway"}},"mfa_authenticated":false}"#,
            query_id: "q-auth-failure",
            query: "failed authentication without MFA from an external address",
        },
        EventFixture {
            relation: "ocsf_api_activity",
            event_id: "evt_smoke_api",
            support_ref: "sup_smoke_api",
            typed_event_json: r#"{"semantic_class":"api","ocsf":{"activity_id":3,"time":1534780000000,"api":{"operation":"PutBucketAcl"},"actor":{"user":{"name":"deployment-role"}}},"service":"s3.amazonaws.com","resource":"research-exports","state_transitions":[{"field":"acl","after":"public-read"}]}"#,
            query_id: "q-public-storage",
            query: "object storage bucket changed to public read access",
        },
        EventFixture {
            relation: "ocsf_detection_finding",
            event_id: "evt_smoke_detection",
            support_ref: "sup_smoke_detection",
            typed_event_json: r#"{"semantic_class":"detection","ocsf":{"activity_id":1,"time":1534781000000,"finding_info":{"title":"Suspicious archive downloader"},"severity_id":4,"message":"Downloader behavior detected"},"disposition":"quarantined"}"#,
            query_id: "q-detection",
            query: "high severity downloader detection quarantined",
        },
        EventFixture {
            relation: "ocsf_file_activity",
            event_id: "evt_smoke_file",
            support_ref: "sup_smoke_file",
            typed_event_json: r#"{"semantic_class":"file","ocsf":{"activity_id":3,"time":1534782000000,"file":{"name":"authorized_keys","path":"/home/service/.ssh/authorized_keys"},"actor":{"user":{"name":"service"}},"device":{"hostname":"linux-app-2"}}}"#,
            query_id: "q-file-change",
            query: "SSH authorized keys file changed for a service account",
        },
        EventFixture {
            relation: "ocsf_network_activity",
            event_id: "evt_smoke_network",
            support_ref: "sup_smoke_network",
            typed_event_json: r#"{"semantic_class":"network","ocsf":{"activity_id":6,"time":1534783000000,"src_endpoint":{"ip":"10.0.4.12"},"dst_endpoint":{"ip":"198.51.100.20","port":443},"protocol_name":"TLS"},"bytes_out":73400320,"action":"connect"}"#,
            query_id: "q-network-egress",
            query: "large outbound TLS connection to an external endpoint",
        },
    ]
}

fn write_string_relation(
    root: &Path,
    relation: &str,
    names: &[&str],
    columns: &[Vec<&str>],
) -> Result<ObjectRef, Box<dyn std::error::Error>> {
    let schema = Schema::new(
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Utf8, false))
            .collect::<Vec<_>>(),
    );
    let arrays = columns
        .iter()
        .map(|values| Arc::new(StringArray::from_iter_values(values.iter().copied())) as ArrayRef)
        .collect::<Vec<_>>();
    let rows = columns.first().map_or(0, Vec::len);
    if columns.iter().any(|column| column.len() != rows) {
        return Err("relation columns have different lengths".into());
    }
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays)?;
    let relative = format!("semantic/{relation}.parquet");
    let path = root.join(&relative);
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path)?, Arc::new(schema), Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    let sha256 = digest(&fs::read(&path)?);
    Ok(ObjectRef {
        relation: relation.to_owned(),
        path: relative,
        rows: rows as u64,
        sha256: sha256.clone(),
        logical_sha256: sha256,
    })
}

fn write_evaluation_fixture(
    root: &Path,
    events: &[EventFixture],
    snapshot_sha256: &str,
    mapping_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = ProjectionContext {
        snapshot: ComponentRef {
            id: "smoke-snapshot".to_owned(),
            version: "1".to_owned(),
            sha256: snapshot_sha256.to_owned(),
            uri: None,
        },
        mapping_pack: ComponentRef {
            id: "smoke-mapping".to_owned(),
            version: "1".to_owned(),
            sha256: mapping_sha256.to_owned(),
            uri: None,
        },
    };
    let mut query_lines = String::new();
    let mut qrel_lines = String::new();
    for event in events {
        let output = project(ProjectionInput {
            relation_name: event.relation,
            event_id: event.event_id,
            typed_event_json: event.typed_event_json,
            support_ref: event.support_ref,
            context: &context,
        })?;
        let document = output.document.ok_or("smoke event was not searchable")?;
        query_lines.push_str(&serde_json::to_string(&serde_json::json!({
            "query_id": event.query_id, "query": event.query
        }))?);
        query_lines.push('\n');
        qrel_lines.push_str(&serde_json::to_string(&serde_json::json!({
            "query_id": event.query_id,
            "document_id": document.document_id,
            "relevance": 3
        }))?);
        qrel_lines.push('\n');
    }
    fs::write(root.join("smoke-queries.jsonl"), query_lines)?;
    fs::write(root.join("smoke-qrels.jsonl"), qrel_lines)?;
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
