use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use clap::{Parser, ValueEnum};
use rag_embedding::{EmbeddingProfile, normalize_loopback_http_endpoint, try_compose_query};
use rag_pipeline::{ComponentRef, SealedQueryVectorSet};
use rag_provider::{
    PROTOCOL, QUERY_VECTOR_SET_COMPONENT_ID, QUERY_VECTOR_SET_COMPONENT_VERSION, format_ref,
    format_ref_v3, retrieval_policy_ref, search_input_schema_ref, search_output_schema_ref,
    search_tool_ref, similar_input_schema_ref, similar_output_schema_ref, similar_tool_ref,
    similarity_policy_ref,
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
    /// Loopback model server for search. Defaults to LM Studio on port 1234
    /// when neither query-vector option is supplied.
    #[arg(long, conflicts_with = "query_vector_set")]
    embedding_endpoint: Option<String>,
    /// Complete sealed query-vector-set directory for offline dense/fused
    /// search. Mutually exclusive with --embedding-endpoint.
    #[arg(long, conflicts_with = "embedding_endpoint")]
    query_vector_set: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "encoded PowerShell command")]
    query: String,
    /// Prepare a loadout for free-text search or stored-document similarity.
    #[arg(long, value_enum, default_value_t = ToolKind::Search)]
    tool: ToolKind,
    /// Required when --tool similar is selected.
    #[arg(long)]
    document_id: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum ToolKind {
    #[default]
    Search,
    Similar,
}

static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct StagedOutput {
    path: PathBuf,
    destination: PathBuf,
    published: bool,
}

impl StagedOutput {
    fn new(destination: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let parent = destination.parent().ok_or("loadout output parent")?;
        fs::create_dir_all(parent)?;
        let name = destination
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("loadout output name")?;
        let sequence = OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{name}.livefire-rag-partial-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self {
            path,
            destination: destination.to_owned(),
            published: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn publish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        fs::rename(&self.path, &self.destination)?;
        self.published = true;
        Ok(())
    }
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Arguments::parse();
    if a.out.exists() {
        return Err("refusing to overwrite loadout output".into());
    }
    let staging = StagedOutput::new(&a.out)?;
    let output = staging.path();
    if matches!(a.tool, ToolKind::Similar) && a.query_vector_set.is_some() {
        return Err("--query-vector-set is only valid for --tool search".into());
    }
    let embedding_endpoint = if matches!(a.tool, ToolKind::Search) && a.query_vector_set.is_none() {
        Some(normalize_loopback_http_endpoint(
            a.embedding_endpoint
                .as_deref()
                .unwrap_or("http://127.0.0.1:1234"),
        )?)
    } else {
        None
    };
    let (tool, input_schema, output_schema, retrieval_policy) = match a.tool {
        ToolKind::Search => (
            search_tool_ref(),
            search_input_schema_ref(),
            search_output_schema_ref(),
            retrieval_policy_ref(),
        ),
        ToolKind::Similar => {
            a.document_id
                .as_deref()
                .filter(|value| !value.is_empty())
                .ok_or("--document-id is required for --tool similar")?;
            (
                similar_tool_ref(),
                similar_input_schema_ref(),
                similar_output_schema_ref(),
                similarity_policy_ref(),
            )
        }
    };
    let physical = read_json(&a.index.join("index.json"))?;
    refuse_test_only_index(&physical)?;
    let index_contract = index_contract(&physical)?;
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
    let query_vector_component = if let Some(root) = a.query_vector_set.as_deref() {
        let expected_profile: ComponentRef = serde_json::from_value(profile_ref.clone())?;
        let indexed_profile: EmbeddingProfile = serde_json::from_value(profile.clone())?;
        let sealed = SealedQueryVectorSet::open(
            root,
            &expected_profile,
            profile["model"].as_str().ok_or("embedding profile model")?,
            u32::try_from(
                profile["dimensions"]
                    .as_u64()
                    .ok_or("embedding profile dimensions")?,
            )?,
            profile["normalization"]
                .as_str()
                .ok_or("embedding profile normalization")?,
            None,
        )?;
        let composed = try_compose_query(&indexed_profile, &a.query)?;
        sealed.vector_for_unique_request(&a.query, &composed, "dense", 10, &[])?;
        Some(json!({
            "id":QUERY_VECTOR_SET_COMPONENT_ID,
            "version":QUERY_VECTOR_SET_COMPONENT_VERSION,
            "sha256":sealed.manifest.component_sha256
        }))
    } else {
        None
    };
    let (contract, call_arguments) = match a.tool {
        ToolKind::Search if query_vector_component.is_some() => (
            json!({"mode":"offline_closed","network":[],"secret_handles":[],"vendor_services":[]}),
            json!({"schema_version":"livefire.rag.fast-search.input/1","query":a.query,"mode":"dense","top_n":10}),
        ),
        ToolKind::Search => {
            let endpoint = embedding_endpoint.as_deref().expect("search endpoint");
            (
                json!({"mode":"local_component","network":[format!("loopback:{endpoint}")],"secret_handles":[],"vendor_services":[]}),
                json!({"schema_version":"livefire.rag.fast-search.input/1","query":a.query,"mode":"lexical","top_n":10}),
            )
        }
        ToolKind::Similar => (
            json!({"mode":"offline_closed","network":[],"secret_handles":[],"vendor_services":[]}),
            json!({"schema_version":"livefire.rag.fast-similar.input/1","document_id":a.document_id.as_deref().expect("validated document ID"),"top_n":10}),
        ),
    };
    // The SDK index manifest extends the query-time contract with the exact
    // local components needed by the index. The tool binding lock uses the
    // smaller protocol contract and deliberately rejects that extra field.
    let index_query_contract = match a.tool {
        ToolKind::Search if query_vector_component.is_some() => {
            json!({"mode":"offline_closed","network":[],"secret_handles":[],"vendor_services":[],"required_local_components":[profile_ref,query_vector_component.as_ref().expect("query vector component")]})
        }
        ToolKind::Search => {
            let endpoint = embedding_endpoint.as_deref().expect("search endpoint");
            json!({"mode":"local_component","network":[format!("loopback:{endpoint}")],"secret_handles":[],"vendor_services":[],"required_local_components":[profile_ref]})
        }
        ToolKind::Similar => {
            json!({"mode":"offline_closed","network":[],"secret_handles":[],"vendor_services":[],"required_local_components":[]})
        }
    };
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
    let physical_ref = json!({"id":"com.ayc.livefire-rag.fast-physical-index","version":index_contract.physical_version,"sha256":physical["component_sha256"]});
    let builder = material_ref(
        "com.ayc.livefire-rag.fast-builder",
        &json!({"implementation":"rust","format":index_contract.manifest_schema}),
    );
    let objects = physical_objects(&a.index, &physical, index_contract.lexical_media_type)?;
    let pointer = objects
        .iter()
        .find(|object| object["path"] == "occurrences.parquet")
        .ok_or("pointer object")?
        .clone();
    let sdk_index = json!({
      "schema_version":"livefire.index/1","index_id":"com.ayc.livefire-rag.fast-evidence.local-test","index_version":index_contract.index_version,"index_kind":"generic_ocsf_evidence_candidates",
      "format":index_contract.format,"builder":builder,
      "source_bindings":[{"source_snapshot":snapshot,"source_snapshot_profile":source_profile,"source_admission_receipt":source_admission,"record_identity_policy":record_identity}],
      "policies":{"embedding":profile_ref,"projection":projection,"retrieval":retrieval_policy,"physical_index":physical_ref,"mapping_pack":mapping},
      "objects":objects,"source_pointer_table":pointer,
      "coverage":coverage(&physical,&build_report)?,
      "query_time_contract":index_query_contract,
      "governance":{"inherits_source_confidentiality":true,"inherits_source_retention":true}
    });
    let wrapped_index = output.join("evidence-index");
    copy_physical_index(&a.index, &wrapped_index, &objects)?;
    write_canonical(wrapped_index.join("sdk-index-manifest.json"), &sdk_index)?;
    let index_ref = json!({"id":sdk_index["index_id"],"version":sdk_index["index_version"],"sha256":canonical_sha256(&sdk_index)});
    let checks = json!({"object_digests":true,"source_binding":true,"safe_paths":true,"schema_profiles":true,"coverage_closure":true,"pointer_closure":true,"offline_query_conformance":true,"conformance":true,"deterministic_rebuild":false});
    let unsigned = json!({"schema_version":"livefire.index-admission/1","receipt_id":"com.ayc.livefire-rag.local-test-index-admission","receipt_version":"1","build_request_sha256":canonical_sha256(&json!({"index":index_ref,"scope":"local-test"})),"build_report_sha256":sha256(&fs::read(a.index.join("build-report.json"))?),"index_manifest_sha256":index_ref["sha256"],"verifier":material_ref("com.ayc.livefire-rag.local-test-index-verifier",&checks),"checks":checks,"disposition":"admitted","reason_codes":["local_test_only_not_production_admitted"]});
    let mut receipt = unsigned.clone();
    receipt["authority_signature"] =
        Value::String(format!("local-test:{}", canonical_sha256(&unsigned)));
    write_canonical(output.join("index-admission-receipt.json"), &receipt)?;
    let receipt_ref = json!({"id":"com.ayc.livefire-rag.local-test-index-admission","version":"1","sha256":canonical_sha256(&receipt)});
    // Search data is currently loaded once into the provider process. The
    // local-test binding declares a realistic ceiling for a representative
    // index instead of a misleading 256 MiB sandbox claim.
    let limits = json!({"request_bytes":65536,"result_bytes":1048576,"wall_time_ms":30000,"memory_bytes":2147483648_u64,"max_candidates":100});
    let lock = json!({"schema_version":"livefire.tool-binding-lock/1","descriptor":tool,"provider":provider,"executable":executable,"input_schema":input_schema,"output_schema":output_schema,"index":index_ref,"index_format":index_contract.format,"index_admission_receipt":receipt_ref,"source_snapshots":[snapshot],"retrieval_policy":retrieval_policy,"query_time_contract":contract,"protocol":PROTOCOL,"limits":limits});
    write_canonical(output.join("tool-binding-lock.json"), &lock)?;
    let lock_sha = canonical_sha256(&lock);
    let lock_ref = json!({"id":"com.ayc.livefire-rag.local-test-tool-binding","version":"1","sha256":lock_sha});
    let mut mounts = vec![
        json!({"logical_name":"evidence-index","role":"index","component":index_ref,"access":"read_only","process_path":future_absolute(&a.out.join("evidence-index"))?}),
        json!({"logical_name":"tool-binding-lock","role":"policy","component":lock_ref,"access":"read_only","process_path":future_absolute(&a.out.join("tool-binding-lock.json"))?}),
        json!({"logical_name":"index-admission-receipt","role":"policy","component":receipt_ref,"access":"read_only","process_path":future_absolute(&a.out.join("index-admission-receipt.json"))?}),
        json!({"logical_name":"embedding-profile","role":"model","component":profile_ref,"access":"read_only","process_path":absolute(&a.embedding_profile)?}),
    ];
    if let (Some(root), Some(component)) = (
        a.query_vector_set.as_deref(),
        query_vector_component.as_ref(),
    ) {
        mounts.push(json!({
            "logical_name":"query-vector-set",
            "role":"model",
            "component":component,
            "access":"read_only",
            "process_path":absolute(root)?
        }));
    }
    let mounts = Value::Array(mounts);
    let deadline = 4_102_444_800_000_u64;
    let requests = [
        request("1", "handshake", json!({}), deadline),
        request(
            "2",
            "open",
            json!({"provider":provider,"tools":[tool],"indexes":[index_ref],"source_snapshots":[snapshot],"binding_lock_sha256":lock_sha,"query_time_contract":contract,"limits":limits,"mounts":mounts}),
            deadline,
        ),
        request(
            "3",
            "call",
            json!({"session_id":"${session_id}","tool":tool,"arguments":call_arguments}),
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
    fs::write(output.join("requests.jsonl"), lines)?;
    staging.publish()?;
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
    let semantics = report
        .pointer("/accounting/coverage_semantics")
        .and_then(Value::as_str)
        .ok_or("build report has no coverage semantics")?;
    let source = match semantics {
        "searchable_projection_only_not_source_row_coverage" => report
            .pointer("/accounting/source_rows_scanned")
            .and_then(Value::as_u64)
            .ok_or("build report has no source_rows_scanned")?,
        "dataset_scope_only_not_source_corpus_coverage" => report
            .pointer("/accounting/source_records")
            .and_then(Value::as_u64)
            .ok_or("portable build report has no source_records")?,
        _ => return Err("build report coverage is not source-aware".into()),
    };
    let selected = match semantics {
        "searchable_projection_only_not_source_row_coverage" => report
            .pointer("/accounting/sampling/selected_occurrences")
            .and_then(Value::as_u64)
            .ok_or("build report has no selected_occurrences")?,
        "dataset_scope_only_not_source_corpus_coverage" => report
            .pointer("/accounting/indexed_occurrences")
            .and_then(Value::as_u64)
            .ok_or("portable build report has no indexed_occurrences")?,
        _ => return Err("build report coverage is not source-aware".into()),
    };
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
    let remaining = excluded.saturating_sub(structured);
    if remaining > 0 {
        let reason = if semantics == "dataset_scope_only_not_source_corpus_coverage" {
            let scoped = report
                .pointer("/accounting/excluded_by_scope_occurrences")
                .and_then(Value::as_u64)
                .ok_or("portable build report has no scoped exclusion count")?;
            if scoped != remaining {
                return Err("portable scope exclusions do not close".into());
            }
            "dataset_scope_exclusion"
        } else {
            "scenario_blind_sample_exclusion"
        };
        reasons.insert(reason.into(), Value::from(remaining));
    }
    Ok(
        json!({"source_records":source,"indexed_documents":physical["documents"]["rows"],"excluded_records":excluded,"reason_counts":reasons}),
    )
}
fn physical_objects(
    root: &Path,
    p: &Value,
    lexical_media_type: &str,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
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
            lexical_media_type,
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

struct IndexContract {
    format: Value,
    manifest_schema: &'static str,
    index_version: &'static str,
    physical_version: &'static str,
    lexical_media_type: &'static str,
}

fn index_contract(physical: &Value) -> Result<IndexContract, Box<dyn std::error::Error>> {
    match physical["schema_version"].as_str() {
        Some("livefire.rag.fast-index/2") => Ok(IndexContract {
            format: format_ref(),
            manifest_schema: "livefire.rag.fast-index/2",
            index_version: "0.2.0",
            physical_version: "2",
            lexical_media_type: "application/json",
        }),
        Some("livefire.rag.fast-index/3") => Ok(IndexContract {
            format: format_ref_v3(),
            manifest_schema: "livefire.rag.fast-index/3",
            index_version: "0.3.0",
            physical_version: "3",
            lexical_media_type: "application/vnd.sqlite3",
        }),
        _ => Err("fast index version 2 or 3 is required".into()),
    }
}

fn refuse_test_only_index(physical: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if physical.get("test_only").and_then(Value::as_bool) == Some(true) {
        return Err("test-only indexes cannot be prepared as provider loadouts".into());
    }
    Ok(())
}

fn copy_physical_index(
    source: &Path,
    destination: &Path,
    objects: &[Value],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir(destination)?;
    for object in objects {
        let relative = object["path"].as_str().ok_or("physical object path")?;
        if relative.starts_with('/')
            || relative
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err("physical object path is unsafe".into());
        }
        let from = source.join(relative);
        let to = destination.join(relative);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::hard_link(&from, &to)?;
        let (bytes, digest) = file_identity(&to)?;
        if object["bytes"].as_u64() != Some(bytes)
            || object["sha256"].as_str() != Some(digest.as_str())
        {
            return Err("copied physical object identity differs".into());
        }
    }
    Ok(())
}
fn artifact(path: &Path, relative: &str, media: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let (bytes, digest) = file_identity(path)?;
    Ok(json!({"path":relative,"media_type":media,"sha256":digest,"bytes":bytes}))
}
fn file_identity(path: &Path) -> Result<(u64, String), Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or("file is too large")?;
        digest.update(&buffer[..read]);
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
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
fn future_absolute(p: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut cursor = p;
    let mut missing = Vec::new();
    while !cursor.exists() {
        missing.push(cursor.file_name().ok_or("future path name")?.to_owned());
        cursor = cursor.parent().ok_or("future path parent")?;
    }
    let mut resolved = cursor.canonicalize()?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_manifest_selects_the_exact_sdk_format() {
        let v2 = index_contract(&json!({"schema_version":"livefire.rag.fast-index/2"})).unwrap();
        assert_eq!(v2.format, format_ref());
        assert_eq!(v2.lexical_media_type, "application/json");

        let v3 = index_contract(&json!({"schema_version":"livefire.rag.fast-index/3"})).unwrap();
        assert_eq!(v3.format, format_ref_v3());
        assert_eq!(v3.lexical_media_type, "application/vnd.sqlite3");

        assert!(index_contract(&json!({"schema_version":"livefire.rag.fast-index/4"})).is_err());
    }

    #[test]
    fn provider_loadout_preparation_refuses_test_only_index() {
        assert!(refuse_test_only_index(&json!({"test_only":true})).is_err());
        assert!(refuse_test_only_index(&json!({})).is_ok());
    }
}
