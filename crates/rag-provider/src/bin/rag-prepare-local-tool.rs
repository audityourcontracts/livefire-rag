use std::{
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use rag_embedding::normalize_loopback_http_endpoint;
use rag_provider::{
    PROTOCOL, format_ref, input_schema_ref, output_schema_ref, retrieval_policy_ref, tool_ref,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(about = "Wrap a fast index and prepare an explicitly local-test Livefire tool loadout")]
struct Arguments {
    #[arg(long)]
    index: PathBuf,
    #[arg(long)]
    bundle: PathBuf,
    #[arg(long)]
    embedding_profile: PathBuf,
    #[arg(long)]
    source_receipt: PathBuf,
    #[arg(long, default_value = "http://127.0.0.1:1234")]
    embedding_endpoint: String,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "encoded PowerShell command")]
    query: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Arguments::parse();
    if a.out.exists() {
        return Err("refusing to overwrite loadout output".into());
    }
    let embedding_endpoint = normalize_loopback_http_endpoint(&a.embedding_endpoint)?;
    let physical = read_json(&a.index.join("index.json"))?;
    if physical["schema_version"] != "livefire.rag.fast-index/2" {
        return Err("fast index v2 is required".into());
    }
    let plugin = read_json(&a.bundle.join("plugin.json"))?;
    let provider = plugin
        .pointer("/entrypoints/provider/component")
        .ok_or("bundle provider")?
        .clone();
    let executable = plugin
        .pointer("/entrypoints/provider/executable")
        .ok_or("bundle executable")?
        .clone();
    let profile_bytes = fs::read(&a.embedding_profile)?;
    if physical
        .pointer("/embedding_profile/sha256")
        .and_then(Value::as_str)
        != Some(&sha256(&profile_bytes))
    {
        return Err("embedding profile bytes differ from indexed profile".into());
    }
    let profile = physical["embedding_profile"].clone();
    let profile_ref =
        json!({"id":profile["id"],"version":profile["version"],"sha256":profile["sha256"]});
    let source_receipt_bytes = fs::read(&a.source_receipt)?;
    let source_receipt: Value = serde_json::from_slice(&source_receipt_bytes)?;
    let snapshot = source_component(
        &source_receipt,
        physical["source"]["snapshot_sha256"]
            .as_str()
            .ok_or("physical snapshot digest")?,
    )?;
    let mapping = source_mapping_component(
        &source_receipt,
        physical["source"]["mapping_sha256"]
            .as_str()
            .ok_or("physical mapping digest")?,
    )?;
    let build_report = read_json(&a.index.join("build-report.json"))?;
    let source_profile = material_ref(
        "com.ayc.livefire-ocsf.snapshot-profile",
        &json!({"scope":"local-test","hydration":"event_id"}),
    );
    let source_admission = material_ref(
        "com.ayc.livefire-ocsf.source-admission",
        &json!({
            "scope":"local-test-not-production",
            "source_receipt_sha256":sha256(&source_receipt_bytes),
            "source_snapshot":snapshot
        }),
    );
    let record_identity = material_ref(
        "com.ayc.livefire-ocsf.event-id-policy",
        &json!({"record_id":"event_id"}),
    );
    let projection = material_ref(
        "com.ayc.livefire-rag.fast-projection-policy",
        &json!({"scope":"generic_ocsf_projection","incident_answers":false}),
    );
    let physical_ref = json!({"id":"com.ayc.livefire-rag.fast-physical-index","version":"2","sha256":physical["component_sha256"]});
    let builder = material_ref(
        "com.ayc.livefire-rag.fast-builder",
        &json!({"implementation":"rust","format":"livefire.rag.fast-index/2"}),
    );
    let objects = physical_objects(&a.index, &physical)?;
    let pointer = objects
        .iter()
        .find(|object| object["path"] == "occurrences.parquet")
        .ok_or("pointer object")?
        .clone();
    let sdk_index = json!({
      "schema_version":"livefire.index/1","index_id":"com.ayc.livefire-rag.fast-evidence.local-test","index_version":"0.2.0","index_kind":"generic_ocsf_evidence_candidates",
      "format":format_ref(),"builder":builder,
      "source_bindings":[{"source_snapshot":snapshot,"source_snapshot_profile":source_profile,"source_admission_receipt":source_admission,"record_identity_policy":record_identity}],
      "policies":{"embedding":profile_ref,"projection":projection,"retrieval":retrieval_policy_ref(),"physical_index":physical_ref,"mapping_pack":mapping},
      "objects":objects,"source_pointer_table":pointer,
      "coverage":coverage(&physical,&build_report)?,
      "query_time_contract":{"mode":"local_component","network":[format!("loopback:{embedding_endpoint}")],"secret_handles":[],"vendor_services":[],"required_local_components":[profile_ref]},
      "governance":{"inherits_source_confidentiality":true,"inherits_source_retention":true}
    });
    write_canonical(a.index.join("sdk-index-manifest.json"), &sdk_index)?;
    let index_ref = json!({"id":sdk_index["index_id"],"version":sdk_index["index_version"],"sha256":canonical_sha256(&sdk_index)});
    fs::create_dir_all(&a.out)?;
    let checks = json!({"object_digests":true,"source_binding":true,"safe_paths":true,"schema_profiles":true,"coverage_closure":true,"pointer_closure":true,"offline_query_conformance":true,"conformance":true,"deterministic_rebuild":false});
    let unsigned = json!({"schema_version":"livefire.index-admission/1","receipt_id":"com.ayc.livefire-rag.local-test-index-admission","receipt_version":"1","build_request_sha256":canonical_sha256(&json!({"index":index_ref,"scope":"local-test"})),"build_report_sha256":sha256(&fs::read(a.index.join("build-report.json"))?),"index_manifest_sha256":index_ref["sha256"],"verifier":material_ref("com.ayc.livefire-rag.local-test-index-verifier",&checks),"checks":checks,"disposition":"admitted","reason_codes":["local_test_only_not_production_admitted"]});
    let mut receipt = unsigned.clone();
    receipt["authority_signature"] =
        Value::String(format!("local-test:{}", canonical_sha256(&unsigned)));
    write_canonical(a.out.join("index-admission-receipt.json"), &receipt)?;
    let receipt_ref = json!({"id":"com.ayc.livefire-rag.local-test-index-admission","version":"1","sha256":canonical_sha256(&receipt)});
    let contract = json!({"mode":"local_component","network":[format!("loopback:{embedding_endpoint}")],"secret_handles":[],"vendor_services":[]});
    // Search data is currently loaded once into the provider process. The
    // local-test binding declares a realistic ceiling for a representative
    // index instead of a misleading 256 MiB sandbox claim.
    let limits = json!({"request_bytes":65536,"result_bytes":1048576,"wall_time_ms":30000,"memory_bytes":2147483648_u64,"max_candidates":100});
    let lock = json!({"schema_version":"livefire.tool-binding-lock/1","descriptor":tool_ref(),"provider":provider,"executable":executable,"input_schema":input_schema_ref(),"output_schema":output_schema_ref(),"index":index_ref,"index_format":format_ref(),"index_admission_receipt":receipt_ref,"source_snapshots":[snapshot],"retrieval_policy":retrieval_policy_ref(),"query_time_contract":contract,"protocol":PROTOCOL,"limits":limits});
    write_canonical(a.out.join("tool-binding-lock.json"), &lock)?;
    let lock_sha = canonical_sha256(&lock);
    let lock_ref = json!({"id":"com.ayc.livefire-rag.local-test-tool-binding","version":"1","sha256":lock_sha});
    let mounts = json!([
      {"logical_name":"evidence-index","role":"index","component":index_ref,"access":"read_only","process_path":absolute(&a.index)?},
      {"logical_name":"tool-binding-lock","role":"policy","component":lock_ref,"access":"read_only","process_path":absolute(&a.out.join("tool-binding-lock.json"))?},
      {"logical_name":"index-admission-receipt","role":"policy","component":receipt_ref,"access":"read_only","process_path":absolute(&a.out.join("index-admission-receipt.json"))?},
      {"logical_name":"embedding-profile","role":"model","component":profile_ref,"access":"read_only","process_path":absolute(&a.embedding_profile)?}
    ]);
    let deadline = 4_102_444_800_000_u64;
    let requests = [
        request("1", "handshake", json!({}), deadline),
        request(
            "2",
            "open",
            json!({"provider":provider,"tools":[tool_ref()],"indexes":[index_ref],"source_snapshots":[snapshot],"binding_lock_sha256":lock_sha,"query_time_contract":contract,"limits":limits,"mounts":mounts}),
            deadline,
        ),
        request(
            "3",
            "call",
            json!({"session_id":"${session_id}","tool":tool_ref(),"arguments":{"schema_version":"livefire.rag.fast-search.input/1","query":a.query,"mode":"lexical","top_n":10}}),
            deadline,
        ),
        request(
            "4",
            "health",
            json!({"session_id":"${session_id}"}),
            deadline,
        ),
        request(
            "5",
            "close",
            json!({"session_id":"${session_id}"}),
            deadline,
        ),
    ];
    let mut lines = Vec::new();
    for row in requests {
        lines.extend(serde_json_canonicalizer::to_vec(&row)?);
        lines.push(b'\n');
    }
    fs::write(a.out.join("requests.jsonl"), lines)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"local_test_only_not_production_admitted","index":index_ref,"binding_lock":lock_ref,"receipt":receipt_ref,"requests":a.out.join("requests.jsonl")})
        )?
    );
    Ok(())
}

fn request(id: &str, method: &str, params: Value, deadline: u64) -> Value {
    json!({"protocol":PROTOCOL,"id":id,"method":method,"params":params,"context":{"trace_id":format!("rag-local-{id}"),"deadline_unix_ms":deadline}})
}
fn source_component(
    receipt: &Value,
    expected_sha: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let component = receipt
        .pointer("/runnable_snapshot/component")
        .ok_or("source receipt has no runnable snapshot component")?;
    if component["sha256"] != expected_sha {
        return Err("source receipt snapshot digest differs from the physical index".into());
    }
    let id = component["id"]
        .as_str()
        .or_else(|| {
            receipt
                .pointer("/snapshot_manifest/snapshot_id")
                .and_then(Value::as_str)
        })
        .ok_or("source receipt has no snapshot component id")?;
    let version = component["version"]
        .as_str()
        .or_else(|| {
            receipt
                .pointer("/snapshot_manifest/snapshot_version")
                .and_then(Value::as_str)
        })
        .ok_or("source receipt has no snapshot component version")?;
    Ok(json!({"id":id,"version":version,"sha256":expected_sha}))
}
fn source_mapping_component(
    receipt: &Value,
    expected_sha: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let component = receipt
        .pointer("/runnable_snapshot/mapping_pack")
        .ok_or("source receipt has no runnable mapping-pack component")?;
    let id = component["id"]
        .as_str()
        .ok_or("source receipt mapping pack has no component id")?;
    let version = component["version"]
        .as_str()
        .ok_or("source receipt mapping pack has no component version")?;
    if component["sha256"] != expected_sha {
        return Err("source receipt mapping digest differs from the physical index".into());
    }
    Ok(json!({"id":id,"version":version,"sha256":expected_sha}))
}
fn coverage(physical: &Value, report: &Value) -> Result<Value, Box<dyn std::error::Error>> {
    let source = report
        .pointer("/accounting/source_rows_scanned")
        .and_then(Value::as_u64)
        .ok_or("build report has no source_rows_scanned")?;
    let selected = report
        .pointer("/accounting/sampling/selected_occurrences")
        .and_then(Value::as_u64)
        .ok_or("build report has no selected_occurrences")?;
    let structured = report
        .pointer("/accounting/structured_only_occurrences")
        .and_then(Value::as_u64)
        .ok_or("build report has no structured_only_occurrences")?;
    let indexed = physical
        .pointer("/occurrences/rows")
        .and_then(Value::as_u64)
        .ok_or("physical occurrence count")?;
    if selected != indexed || source < selected.saturating_add(structured) {
        return Err("build accounting is inconsistent with physical occurrence rows".into());
    }
    let excluded = source - selected;
    let mut reasons = serde_json::Map::new();
    if structured > 0 {
        reasons.insert(
            "structured_only_no_searchable_projection".into(),
            Value::from(structured),
        );
    }
    let sampled = excluded.saturating_sub(structured);
    if sampled > 0 {
        reasons.insert(
            "scenario_blind_sample_exclusion".into(),
            Value::from(sampled),
        );
    }
    Ok(
        json!({"source_records":source,"indexed_documents":physical["documents"]["rows"],"excluded_records":excluded,"reason_counts":reasons}),
    )
}
fn physical_objects(root: &Path, p: &Value) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let paths = [
        ("index.json", "application/json"),
        (
            p["documents"]["path"].as_str().ok_or("documents path")?,
            "application/vnd.apache.parquet",
        ),
        (
            p["occurrences"]["path"]
                .as_str()
                .ok_or("occurrences path")?,
            "application/vnd.apache.parquet",
        ),
        (
            p["vectors"]["path"].as_str().ok_or("vectors path")?,
            "application/octet-stream",
        ),
        (
            p["lexical"]["path"].as_str().ok_or("lexical path")?,
            "application/json",
        ),
        (
            p["occurrence_lookup"]["path"]
                .as_str()
                .ok_or("occurrence lookup path")?,
            "application/vnd.sqlite3",
        ),
        ("build-report.json", "application/json"),
    ];
    let mut v = Vec::new();
    for (path, media) in paths {
        v.push(artifact(&root.join(path), path, media)?)
    }
    Ok(v)
}
fn artifact(path: &Path, relative: &str, media: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let b = fs::read(path)?;
    Ok(json!({"path":relative,"media_type":media,"sha256":sha256(&b),"bytes":b.len()}))
}
fn material_ref(id: &str, v: &Value) -> Value {
    json!({"id":id,"version":"1","sha256":canonical_sha256(v)})
}
fn read_json(p: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn write_canonical(p: impl AsRef<Path>, v: &Value) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(p, serde_json_canonicalizer::to_vec(v)?)?;
    Ok(())
}
fn canonical_sha256(v: &Value) -> String {
    sha256(&serde_json_canonicalizer::to_vec(v).unwrap())
}
fn sha256(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn absolute(p: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(p.canonicalize()?.to_string_lossy().into_owned())
}
