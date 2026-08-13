//! Native provider for the fast experimental evidence index.
//!
//! The transport follows the language-neutral Livefire SDK JSONL lifecycle.
//! Index admission remains a host responsibility; this development provider
//! verifies the exact identities and read-only mount supplied at `open`.

#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, Read, Write},
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rag_embedding::{
    Embedder, EmbeddingError, LmStudioEmbedder, normalize_loopback_http_endpoint,
    try_compose_query, validate_embedding_profile, validate_vector,
};
use rag_index::{FastIndex, IndexError, SearchFilters, SearchHit, SearchMode};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub const PROTOCOL: &str = "livefire.tool/1";
pub const PROVIDER_ID: &str = "com.ayc.livefire-rag.fast-evidence-provider";
pub const TOOL_ID: &str = "com.ayc.livefire-rag.fast-evidence.search";
pub const FORMAT_ID: &str = "com.ayc.livefire-rag.fast-index-format";
pub const VERSION: &str = "0.2.0";

#[must_use]
pub fn provider_ref() -> Value {
    packaged_provider_ref().unwrap_or_else(|| {
        component(
            PROVIDER_ID,
            VERSION,
            &canonical_sha256(&json!({
                "schema_version":"livefire.rag.fast-provider-development-identity/1",
                "provider_id":PROVIDER_ID,
                "provider_version":VERSION,
                "scope":"unpackaged_test_only"
            })),
        )
    })
}

fn packaged_provider_ref() -> Option<Value> {
    let executable = std::env::current_exe().ok()?;
    let root = executable.parent()?.parent()?;
    let lock_path = root.join("provider.objects.lock.json");
    let lock_bytes = fs::read(lock_path).ok()?;
    let lock: Value = serde_json::from_slice(&lock_bytes).ok()?;
    if lock["schema_version"] != "livefire.object-lock/1" {
        return None;
    }
    let executable_bytes = fs::read(&executable).ok()?;
    let executable_name = executable.strip_prefix(root).ok()?.to_str()?;
    let valid = lock["objects"].as_array()?.iter().any(|object| {
        object["path"] == executable_name
            && object["sha256"] == sha256(&executable_bytes)
            && object["bytes"].as_u64() == Some(executable_bytes.len() as u64)
    });
    valid.then(|| component(PROVIDER_ID, VERSION, &canonical_sha256(&lock)))
}

#[must_use]
pub fn tool_ref() -> Value {
    tool_descriptor()["tool"].clone()
}

#[must_use]
pub fn format_ref() -> Value {
    index_format_descriptor()["format"].clone()
}

fn component(id: &str, version: &str, sha256: &str) -> Value {
    json!({"id":id,"version":version,"sha256":sha256})
}

fn schema_ref(schema: &str) -> Value {
    let value: Value = serde_json::from_str(schema).expect("embedded schema");
    component(
        value["$id"].as_str().expect("schema id"),
        "1",
        &canonical_sha256(&value),
    )
}

#[must_use]
pub fn input_schema_ref() -> Value {
    schema_ref(include_str!(
        "../../../specs/fast-evidence-search.input.v1.schema.json"
    ))
}

#[must_use]
pub fn output_schema_ref() -> Value {
    schema_ref(include_str!(
        "../../../specs/fast-evidence-search.output.v1.schema.json"
    ))
}

#[must_use]
pub fn hydration_ref_schema_ref() -> Value {
    schema_ref(include_str!(
        "../../../specs/ocsf-hydration-ref.v1.schema.json"
    ))
}

fn profile_ref(id: &str, material: &str) -> Value {
    let value: Value = serde_json::from_str(material).expect("embedded profile");
    component(id, "1", &canonical_sha256(&value))
}

#[must_use]
pub fn physical_profile() -> Value {
    json!({
        "schema_version":"livefire.rag.fast-index-physical-profile/1",
        "manifest":"livefire.rag.fast-index/2",
        "document_order":"document_id_asc",
        "vector_dtype":"f32le",
        "vector_header_bytes":64,
        "lexical_tokenizer":"ascii_camel_lower_v1",
        "pointer_table":"sqlite-occurrence-lookup-v1"
    })
}

#[must_use]
pub fn physical_profile_ref() -> Value {
    component(
        "com.ayc.livefire-rag.fast-index-physical-profile",
        "1",
        &canonical_sha256(&physical_profile()),
    )
}

#[must_use]
pub fn validator_profile() -> Value {
    json!({
        "schema_version":"livefire.rag.fast-index-validator/1",
        "checks":["component_identity","object_digests","vector_header","document_order","occurrence_closure","source_binding"],
        "candidate_handoff":"authoritative_ocsf_hydration_required"
    })
}

#[must_use]
pub fn validator_ref() -> Value {
    component(
        "com.ayc.livefire-rag.fast-index-validator",
        "1",
        &canonical_sha256(&validator_profile()),
    )
}

#[must_use]
pub fn retrieval_policy() -> Value {
    json!({
        "schema_version":"livefire.rag.fast-retrieval-policy/1",
        "dense":"cosine",
        "lexical":{"algorithm":"bm25","k1":1.2,"b":0.75},
        "fusion":{"algorithm":"reciprocal_rank","rank_constant":60},
        "tie_break":"score_desc_document_id_asc",
        "max_candidates":100,
        "result_semantics":"candidate_pointer_not_evidence"
    })
}

#[must_use]
pub fn retrieval_policy_ref() -> Value {
    component(
        "com.ayc.livefire-rag.fast-retrieval-policy",
        "1",
        &canonical_sha256(&retrieval_policy()),
    )
}

#[must_use]
pub fn tool_descriptor() -> Value {
    let mut value = json!({
        "schema_version":"livefire.tool-descriptor/1",
        "tool":{"id":TOOL_ID,"version":VERSION,"sha256":""},
        "name":"evidence.search",
        "description":"Rank generic OCSF evidence candidates and return hydration-only references; every candidate requires authoritative OCSF hydration and verification.",
        "input_schema":input_schema_ref(),
        "output_schema":output_schema_ref(),
        "result_semantics":"candidate_pointer",
        "evidence_policy":"pointer_only",
        "required_indexes":[{"format_id":FORMAT_ID,"accepted_versions":[VERSION]}],
        "limits":{"request_bytes":65536,"result_bytes":1048576,"wall_time_ms":30000,"max_candidates":100},
        "determinism":"ranked_deterministic"
    });
    let digest = canonical_sha256_omitting(&value, "/tool/sha256");
    value["tool"]["sha256"] = Value::String(digest);
    value
}

#[must_use]
pub fn index_format_descriptor() -> Value {
    let manifest_schema = schema_ref(include_str!(
        "../../../specs/fast-index-manifest.v2.schema.json"
    ));
    let document_schema = schema_ref(include_str!(
        "../../../specs/fast-document-row.v1.schema.json"
    ));
    let occurrence_schema = schema_ref(include_str!(
        "../../../specs/fast-occurrence-row.v1.schema.json"
    ));
    let report_schema = schema_ref(include_str!(
        "../../../specs/fast-build-report.v1.schema.json"
    ));
    let vector_profile = profile_ref(
        "com.ayc.livefire-rag.fast-vector-binary-profile",
        include_str!("../../../specs/fast-vector-binary-profile.v1.json"),
    );
    let lexical_profile = profile_ref(
        "com.ayc.livefire-rag.fast-lexical-profile",
        include_str!("../../../specs/fast-lexical-profile.v1.json"),
    );
    let lookup_profile = profile_ref(
        "com.ayc.livefire-rag.fast-occurrence-lookup-profile",
        include_str!("../../../specs/fast-occurrence-lookup-profile.v1.json"),
    );
    let mut value = json!({
        "schema_version":"livefire.index-format-descriptor/1",
        "format":{"id":FORMAT_ID,"version":VERSION,"sha256":""},
        "compatibility":{"rule":"exact_format_id_and_listed_version","accepted_versions":[VERSION]},
        "objects":[
            {"role":"fast_manifest","required":true,"media_type":"application/json","row_schema":manifest_schema},
            {"role":"documents","required":true,"media_type":"application/vnd.apache.parquet","row_schema":document_schema},
            {"role":"occurrences","required":true,"media_type":"application/vnd.apache.parquet","row_schema":occurrence_schema},
            {"role":"vectors","required":true,"media_type":"application/octet-stream","row_schema":vector_profile},
            {"role":"lexical","required":true,"media_type":"application/json","row_schema":lexical_profile},
            {"role":"occurrence_lookup","required":true,"media_type":"application/vnd.sqlite3","row_schema":lookup_profile},
            {"role":"build_report","required":true,"media_type":"application/json","row_schema":report_schema}
        ],
        "pointer_table":{"required":true,"schema":hydration_ref_schema_ref()},
        "physical_profile":physical_profile_ref(),
        "validator":validator_ref()
    });
    let digest = canonical_sha256_omitting(&value, "/format/sha256");
    value["format"]["sha256"] = Value::String(digest);
    value
}

fn canonical_sha256(value: &Value) -> String {
    let bytes = serde_json_canonicalizer::to_vec(value).expect("JSON canonicalizes");
    sha256(&bytes)
}

fn canonical_sha256_omitting(value: &Value, pointer: &str) -> String {
    let mut copy = value.clone();
    let (parent, field) = pointer.rsplit_once('/').expect("self digest field");
    copy.pointer_mut(parent)
        .and_then(Value::as_object_mut)
        .expect("self digest parent")
        .remove(field);
    canonical_sha256(&copy)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Session {
    index: FastIndex,
    index_ref: Value,
    source_snapshots: Vec<Value>,
    mapping_pack: Value,
    binding_lock_sha256: String,
    embedding_endpoint: Option<String>,
    request_bytes: usize,
    result_bytes: usize,
    wall_time_ms: u64,
    max_candidates: usize,
}

fn verify_sdk_index_wrapper(index_path: &str, index_ref: &Value) -> Result<Value, ProviderError> {
    let wrapper = read_json_mount(index_path, "sdk-index-manifest.json")?;
    if wrapper["schema_version"] != "livefire.index/1"
        || wrapper["index_id"] != index_ref["id"]
        || wrapper["index_version"] != index_ref["version"]
        || canonical_sha256(&wrapper) != index_ref["sha256"]
    {
        return Err(ProviderError::new(
            "invalid_binding",
            "SDK index manifest identity is incompatible",
        ));
    }
    let root = Path::new(index_path);
    let objects = wrapper["objects"]
        .as_array()
        .ok_or_else(|| ProviderError::new("corrupt_artifact", "SDK index objects are absent"))?;
    for object in objects {
        let relative = object["path"].as_str().ok_or_else(|| {
            ProviderError::new("corrupt_artifact", "SDK index object path is invalid")
        })?;
        if relative.starts_with('/')
            || relative
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err(ProviderError::new(
                "corrupt_artifact",
                "SDK index object path is unsafe",
            ));
        }
        let (bytes, digest) = stream_file_identity(&root.join(relative))?;
        if object["bytes"].as_u64() != Some(bytes) || object["sha256"] != digest {
            return Err(ProviderError::new(
                "corrupt_artifact",
                "SDK index object identity mismatch",
            ));
        }
    }
    let pointer = &wrapper["source_pointer_table"];
    if !objects.contains(pointer) {
        return Err(ProviderError::new(
            "invalid_binding",
            "SDK pointer table is outside the object inventory",
        ));
    }
    Ok(wrapper)
}

fn stream_file_identity(path: &Path) -> Result<(u64, String), ProviderError> {
    let mut file = fs::File::open(path)
        .map_err(|_| ProviderError::new("corrupt_artifact", "SDK index object is unreadable"))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|_| {
            ProviderError::new("corrupt_artifact", "SDK index object is unreadable")
        })?;
        if read == 0 {
            break;
        }
        total = total.checked_add(read as u64).ok_or_else(|| {
            ProviderError::new("corrupt_artifact", "SDK index object is too large")
        })?;
        hasher.update(&buffer[..read]);
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

pub struct Provider {
    handshaken: bool,
    next_session: u64,
    sessions: BTreeMap<String, Session>,
    provider_ref: Value,
}

impl Default for Provider {
    fn default() -> Self {
        Self {
            handshaken: false,
            next_session: 0,
            sessions: BTreeMap::new(),
            provider_ref: provider_ref(),
        }
    }
}

#[derive(Debug)]
pub struct ProviderError {
    code: &'static str,
    message: String,
    retryable: bool,
}

impl ProviderError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: false,
        }
    }

    fn retryable(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: true,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

impl Provider {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn handle(&mut self, request: &Value) -> Result<Value, ProviderError> {
        let request = exact_object(
            request,
            &["protocol", "id", "method", "params", "context"],
            &["protocol", "id", "method", "params", "context"],
            "request",
        )?;
        if string(request, "protocol", "request")? != PROTOCOL {
            return Err(ProviderError::new("protocol_error", "unsupported protocol"));
        }
        let context = exact_object(
            request.get("context").expect("required"),
            &["trace_id", "deadline_unix_ms"],
            &["trace_id", "deadline_unix_ms"],
            "context",
        )?;
        if string(context, "trace_id", "context")?.is_empty() {
            return Err(ProviderError::new(
                "invalid_request",
                "context.trace_id must be non-empty",
            ));
        }
        let deadline = positive_u64(context, "deadline_unix_ms", "context")?;
        if now_millis() >= deadline {
            return Err(ProviderError::new(
                "deadline_exceeded",
                "request deadline has expired",
            ));
        }
        let method = string(request, "method", "request")?;
        let params = request.get("params").expect("required");
        match method {
            "handshake" => self.handshake(params),
            "open" => self.open(params),
            "call" => {
                let encoded_bytes = serde_json::to_vec(request)
                    .map_err(|_| {
                        ProviderError::new("invalid_request", "request cannot be encoded")
                    })?
                    .len();
                self.call(params, deadline, encoded_bytes).await
            }
            "health" => self.health(params),
            "close" => self.close(params),
            _ => Err(ProviderError::new(
                "invalid_request",
                "unsupported provider method",
            )),
        }
    }

    fn handshake(&mut self, params: &Value) -> Result<Value, ProviderError> {
        exact_object(params, &[], &[], "params")?;
        if self.handshaken {
            return Err(ProviderError::new(
                "protocol_error",
                "handshake has already completed",
            ));
        }
        self.handshaken = true;
        Ok(json!({
            "response_kind":"handshake",
            "provider":self.provider_ref,
            "protocol":PROTOCOL,
            "tools":[tool_ref()],
            "accepted_index_formats":[format_ref()]
        }))
    }

    fn open(&mut self, params: &Value) -> Result<Value, ProviderError> {
        self.require_handshake()?;
        let fields = [
            "provider",
            "tools",
            "indexes",
            "source_snapshots",
            "binding_lock_sha256",
            "query_time_contract",
            "limits",
            "mounts",
        ];
        let params = exact_object(params, &fields, &fields, "params")?;
        if params.get("provider") != Some(&self.provider_ref)
            || params.get("tools") != Some(&Value::Array(vec![tool_ref()]))
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "provider or tool identity is incompatible",
            ));
        }
        let indexes = array(params, "indexes", "params")?;
        if indexes.len() != 1 {
            return Err(ProviderError::new(
                "invalid_binding",
                "open requires exactly one index",
            ));
        }
        let index_ref = indexes[0].clone();
        validate_component(&index_ref, "index")?;
        let source_snapshots = array(params, "source_snapshots", "params")?.to_vec();
        if source_snapshots.len() != 1 {
            return Err(ProviderError::new(
                "invalid_binding",
                "fast index requires exactly one source snapshot",
            ));
        }
        validate_component(&source_snapshots[0], "source snapshot")?;
        let binding_lock_sha256 = string(params, "binding_lock_sha256", "params")?;
        validate_sha256(binding_lock_sha256, "binding_lock_sha256")?;

        let mounts = mounts_by_name(array(params, "mounts", "params")?)?;
        let index_mount = required_mount(&mounts, "evidence-index", "index")?;
        if index_mount.get("component") != Some(&index_ref) {
            return Err(ProviderError::new(
                "invalid_binding",
                "index mount component is incompatible",
            ));
        }
        let lock_mount = required_mount(&mounts, "tool-binding-lock", "policy")?;
        let receipt_mount = required_mount(&mounts, "index-admission-receipt", "policy")?;
        let embedding_mount = required_mount(&mounts, "embedding-profile", "model")?;

        let lock = read_json_mount(
            string(lock_mount, "process_path", "binding mount")?,
            "tool-binding-lock.json",
        )?;
        if canonical_sha256(&lock) != binding_lock_sha256
            || mount_component_sha(lock_mount) != Some(binding_lock_sha256)
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "mounted binding lock identity does not match its bytes",
            ));
        }
        let required_lock = [
            "schema_version",
            "descriptor",
            "provider",
            "executable",
            "input_schema",
            "output_schema",
            "index",
            "index_format",
            "index_admission_receipt",
            "source_snapshots",
            "retrieval_policy",
            "query_time_contract",
            "protocol",
            "limits",
        ];
        let lock_object = lock.as_object().ok_or_else(|| {
            ProviderError::new("corrupt_artifact", "binding lock must be an object")
        })?;
        if required_lock
            .iter()
            .any(|field| !lock_object.contains_key(*field))
            || lock["schema_version"] != "livefire.tool-binding-lock/1"
            || lock["descriptor"] != tool_ref()
            || lock["provider"] != self.provider_ref
            || lock["input_schema"] != input_schema_ref()
            || lock["output_schema"] != output_schema_ref()
            || lock["index"] != index_ref
            || lock["index_format"] != format_ref()
            || lock["retrieval_policy"] != retrieval_policy_ref()
            || lock["protocol"] != PROTOCOL
            || lock["source_snapshots"] != params["source_snapshots"]
            || lock["query_time_contract"] != params["query_time_contract"]
            || lock["limits"] != params["limits"]
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "runtime binding differs from the mounted lock",
            ));
        }
        let executable = lock["executable"].as_object().ok_or_else(|| {
            ProviderError::new("invalid_binding", "binding executable must be an artifact")
        })?;
        let running_bytes = std::env::current_exe()
            .ok()
            .and_then(|path| fs::read(path).ok())
            .ok_or_else(|| {
                ProviderError::new("corrupt_artifact", "running executable is unreadable")
            })?;
        if executable.get("sha256").and_then(Value::as_str) != Some(sha256(&running_bytes).as_str())
            || executable.get("bytes").and_then(Value::as_u64) != Some(running_bytes.len() as u64)
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "binding executable differs from the running provider",
            ));
        }

        let receipt = read_json_mount(
            string(receipt_mount, "process_path", "receipt mount")?,
            "index-admission-receipt.json",
        )?;
        let receipt_digest = canonical_sha256(&receipt);
        if mount_component_sha(receipt_mount) != Some(receipt_digest.as_str())
            || lock
                .pointer("/index_admission_receipt/sha256")
                .and_then(Value::as_str)
                != Some(receipt_digest.as_str())
            || receipt["schema_version"] != "livefire.index-admission/1"
            || receipt["receipt_id"] != lock["index_admission_receipt"]["id"]
            || receipt["receipt_version"] != lock["index_admission_receipt"]["version"]
            || receipt["disposition"] != "admitted"
            || receipt["index_manifest_sha256"] != index_ref["sha256"]
            || !receipt["authority_signature"]
                .as_str()
                .is_some_and(|value| !value.is_empty())
            || !admission_checks_pass(&receipt)
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "mounted index admission receipt is invalid or incompatible",
            ));
        }

        let index_path = string(index_mount, "process_path", "index mount")?;
        let sdk_index = verify_sdk_index_wrapper(index_path, &index_ref)?;
        if sdk_index["format"] != format_ref()
            || sdk_index.pointer("/source_bindings/0/source_snapshot") != Some(&source_snapshots[0])
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "SDK index format or source binding is incompatible",
            ));
        }
        let physical_manifest = read_json_mount(index_path, "index.json")?;
        if physical_manifest["schema_version"] != "livefire.rag.fast-index/2"
            || sdk_index.pointer("/policies/physical_index/sha256")
                != physical_manifest.get("component_sha256")
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "mounted fast index component identity is incompatible",
            ));
        }
        let index = FastIndex::open(Path::new(index_path)).map_err(|_| {
            ProviderError::new("corrupt_artifact", "mounted index failed fast open")
        })?;
        let mapping_pack = sdk_index
            .pointer("/policies/mapping_pack")
            .cloned()
            .ok_or_else(|| {
                ProviderError::new("invalid_binding", "SDK index mapping pack is absent")
            })?;
        validate_component(&mapping_pack, "mapping pack")?;
        if mapping_pack.get("sha256").and_then(Value::as_str)
            != Some(index.manifest.source.mapping_sha256.as_str())
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "mapping pack does not match the mounted index",
            ));
        }
        validate_embedding_profile(&index.manifest.embedding_profile).map_err(|_| {
            ProviderError::new(
                "invalid_binding",
                "index embedding profile has an invalid query contract",
            )
        })?;
        if source_snapshots[0].get("sha256").and_then(Value::as_str)
            != Some(index.manifest.source.snapshot_sha256.as_str())
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "source snapshot does not match the mounted index",
            ));
        }
        let embedding_path = string(embedding_mount, "process_path", "embedding mount")?;
        let embedding_bytes = read_mount_bytes(embedding_path, "embedding-profile.json")?;
        if sha256(&embedding_bytes) != index.manifest.embedding_profile.sha256
            || embedding_mount.get("component")
                != Some(&component(
                    &index.manifest.embedding_profile.id,
                    &index.manifest.embedding_profile.version,
                    &index.manifest.embedding_profile.sha256,
                ))
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "embedding profile mount differs from the indexed profile",
            ));
        }
        let limits = exact_object(
            params.get("limits").expect("required"),
            &[
                "request_bytes",
                "result_bytes",
                "wall_time_ms",
                "memory_bytes",
                "max_candidates",
            ],
            &[
                "request_bytes",
                "result_bytes",
                "wall_time_ms",
                "memory_bytes",
                "max_candidates",
            ],
            "limits",
        )?;
        let request_bytes = optional_positive_usize(limits, "request_bytes")?.unwrap_or(1_048_576);
        let result_bytes = optional_positive_usize(limits, "result_bytes")?.unwrap_or(1_048_576);
        let wall_time_ms = optional_positive_u64(limits, "wall_time_ms")?.unwrap_or(300_000);
        // The SDK host enforces this process-level limit. Parsing it here keeps
        // the direct harness compatible with a complete binding lock.
        let _memory_bytes = optional_positive_u64(limits, "memory_bytes")?.unwrap_or(u64::MAX);
        let max_candidates = optional_positive_usize(limits, "max_candidates")?.unwrap_or(100);
        if max_candidates > 100 {
            return Err(ProviderError::new(
                "invalid_binding",
                "limits.max_candidates exceeds the fast-index bound",
            ));
        }
        let embedding_endpoint =
            loopback_endpoint(params.get("query_time_contract").expect("required"))?;

        self.next_session += 1;
        let session_id = format!("fast-evidence-{}", self.next_session);
        self.sessions.insert(
            session_id.clone(),
            Session {
                index,
                index_ref,
                source_snapshots,
                mapping_pack,
                binding_lock_sha256: binding_lock_sha256.to_owned(),
                embedding_endpoint,
                request_bytes,
                result_bytes,
                wall_time_ms,
                max_candidates,
            },
        );
        Ok(json!({
            "response_kind":"open",
            "session_id":session_id,
            "binding_lock_sha256":binding_lock_sha256
        }))
    }

    async fn call(
        &self,
        params: &Value,
        request_deadline_ms: u64,
        encoded_request_bytes: usize,
    ) -> Result<Value, ProviderError> {
        self.require_handshake()?;
        let params = exact_object(
            params,
            &["session_id", "tool", "arguments"],
            &["session_id", "tool", "arguments"],
            "params",
        )?;
        let session_id = string(params, "session_id", "params")?;
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ProviderError::new("not_found", "session was not found"))?;
        if encoded_request_bytes > session.request_bytes {
            return Err(ProviderError::new(
                "resource_exhausted",
                "request exceeds limits.request_bytes",
            ));
        }
        let started = Instant::now();
        let remaining_deadline_ms = request_deadline_ms.saturating_sub(now_millis());
        let effective_wall_ms = remaining_deadline_ms.min(session.wall_time_ms);
        if effective_wall_ms == 0 {
            return Err(ProviderError::new(
                "deadline_exceeded",
                "call deadline has expired",
            ));
        }
        if params.get("tool") != Some(&tool_ref()) {
            return Err(ProviderError::new(
                "policy_denied",
                "tool is not granted to this session",
            ));
        }
        let arguments = exact_object(
            params.get("arguments").expect("required"),
            &["schema_version", "query", "mode", "top_n", "filters"],
            &["schema_version", "query", "mode", "top_n"],
            "arguments",
        )?;
        if string(arguments, "schema_version", "arguments")? != "livefire.rag.fast-search.input/1" {
            return Err(ProviderError::new(
                "invalid_request",
                "unsupported search input schema",
            ));
        }
        let query = string(arguments, "query", "arguments")?;
        if query.is_empty() || query.len() > 8_192 {
            return Err(ProviderError::new(
                "invalid_request",
                "query length is invalid",
            ));
        }
        let mode = match string(arguments, "mode", "arguments")? {
            "dense" => SearchMode::Dense,
            "lexical" => SearchMode::Lexical,
            "fused" => SearchMode::Fused,
            _ => {
                return Err(ProviderError::new(
                    "invalid_request",
                    "search mode is invalid",
                ));
            }
        };
        let top_n = positive_usize(arguments, "top_n", "arguments")?;
        if top_n > session.max_candidates || top_n > 100 {
            return Err(ProviderError::new(
                "resource_exhausted",
                "top_n exceeds the bound max_candidates",
            ));
        }
        let filters = parse_filters(arguments.get("filters"))?;
        let vector = if matches!(mode, SearchMode::Dense | SearchMode::Fused) {
            let endpoint = session.embedding_endpoint.as_deref().ok_or_else(|| {
                ProviderError::new(
                    "unavailable",
                    "dense retrieval requires a bound embedding endpoint",
                )
            })?;
            let timeout = Duration::from_millis(effective_wall_ms);
            let embedder = LmStudioEmbedder::with_timeout(
                endpoint,
                &session.index.manifest.embedding_profile.model,
                timeout,
            )
            .map_err(|_| ProviderError::new("invalid_binding", "embedding client is invalid"))?;
            let composed = try_compose_query(&session.index.manifest.embedding_profile, query)
                .map_err(|_| {
                    ProviderError::new("invalid_binding", "query composition contract is invalid")
                })?;
            let embedded = tokio::time::timeout(timeout, embedder.embed(&[composed]))
                .await
                .map_err(|_| {
                    ProviderError::new(
                        "deadline_exceeded",
                        "query embedding exceeded the call deadline",
                    )
                })?
                .map_err(|error| {
                    if matches!(&error, EmbeddingError::Http(source) if source.is_timeout()) {
                        ProviderError::new(
                            "deadline_exceeded",
                            "query embedding exceeded the call deadline",
                        )
                    } else {
                        ProviderError::retryable("unavailable", "query embedding failed")
                    }
                })?;
            let vector = embedded.into_iter().next().ok_or_else(|| {
                ProviderError::new("unavailable", "query embedding response was empty")
            })?;
            validate_vector(
                &vector,
                session.index.manifest.embedding_profile.dimensions as usize,
                &session.index.manifest.embedding_profile.normalization,
            )
            .map_err(|_| {
                ProviderError::retryable("unavailable", "query embedding violated its profile")
            })?;
            Some(vector)
        } else {
            None
        };
        let hits = session
            .index
            .search(mode, query, vector.as_deref(), &filters, top_n)
            .map_err(|error| match error {
                IndexError::Corrupt(_)
                | IndexError::Io(_)
                | IndexError::Parquet(_)
                | IndexError::Arrow(_)
                | IndexError::Sqlite(_)
                | IndexError::Json(_) => {
                    ProviderError::new("corrupt_artifact", "bound index artifact is invalid")
                }
                IndexError::Invalid(_) => {
                    ProviderError::new("invalid_request", "search request is invalid")
                }
            })?;
        let output = search_output(session, query, top_n, hits);
        if started.elapsed() >= Duration::from_millis(effective_wall_ms)
            || now_millis() >= request_deadline_ms
        {
            return Err(ProviderError::new(
                "deadline_exceeded",
                "call exceeded its wall-time limit",
            ));
        }
        if serde_json::to_vec(&output)
            .expect("output serializes")
            .len()
            > session.result_bytes
        {
            return Err(ProviderError::new(
                "resource_exhausted",
                "tool output exceeds limits.result_bytes",
            ));
        }
        Ok(json!({"response_kind":"call","output":output}))
    }

    fn health(&self, params: &Value) -> Result<Value, ProviderError> {
        self.require_handshake()?;
        let session_id = session_only(params)?;
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| ProviderError::new("not_found", "session was not found"))?;
        Ok(json!({
            "response_kind":"health",
            "status":"ready",
            "binding_lock_sha256":session.binding_lock_sha256
        }))
    }

    fn close(&mut self, params: &Value) -> Result<Value, ProviderError> {
        self.require_handshake()?;
        let session_id = session_only(params)?;
        if self.sessions.remove(session_id).is_none() {
            return Err(ProviderError::new("not_found", "session was not found"));
        }
        Ok(json!({"response_kind":"close","closed":true}))
    }

    fn require_handshake(&self) -> Result<(), ProviderError> {
        if self.handshaken {
            Ok(())
        } else {
            Err(ProviderError::new(
                "protocol_error",
                "handshake is required before session methods",
            ))
        }
    }
}

fn search_output(session: &Session, query: &str, top_n: usize, hits: Vec<SearchHit>) -> Value {
    // A complete physical build is still a semantic candidate index: typed
    // rows with no searchable projection remain outside retrieval. It must
    // never be presented as exhaustive source coverage.
    let coverage_status = "partial";
    let mut reason_codes = vec![
        "candidate_occurrences_require_authoritative_hydration",
        "semantic_candidate_index_not_source_coverage",
    ];
    if !session.index.manifest.complete {
        reason_codes.push("sample_not_corpus_coverage");
    }
    let common = json!({
        "schema_version":"livefire.rag.fast-search.output/1",
        "tool":"evidence.search",
        "index":session.index_ref,
        "source_snapshots":session.source_snapshots,
        "query":query,
        "coverage":{
            "status":coverage_status,
            "indexed_documents":session.index.manifest.documents.rows,
            "definitive":false,
            "reason_codes":reason_codes
        },
        "selection":{
            "requested_top_n":top_n,
            "returned_count":hits.len(),
            "deterministic":true,
            "tie_break":"score_desc_document_id_asc"
        }
    });
    let mut output = common.as_object().expect("common object").clone();
    if hits.is_empty() {
        output.insert("kind".to_owned(), Value::String("miss".to_owned()));
        output.insert(
            "miss".to_owned(),
            json!({"reason":"no_ranked_candidates","message":"No indexed semantic document matched the query."}),
        );
    } else {
        output.insert("kind".to_owned(), Value::String("pointer".to_owned()));
        let candidates = hits
            .into_iter()
            .map(|hit| {
                let evidence = hit
                    .occurrences
                    .into_iter()
                    .map(|occurrence| {
                        let snapshot = session
                            .source_snapshots
                            .iter()
                            .find(|snapshot| {
                                snapshot.get("sha256").and_then(Value::as_str)
                                    == Some(occurrence.snapshot_sha256.as_str())
                            })
                            .cloned()
                            .expect("occurrence source is bound to session");
                        json!({
                            "schema_version":"livefire.ocsf-hydration-ref/1",
                            "snapshot":snapshot,
                            "mapping":session.mapping_pack,
                            "relation":occurrence.relation,
                            "event_id":occurrence.event_id,
                            "support_ref":occurrence.support_ref
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "rank":hit.rank,
                    "document_id":hit.document_id,
                    "scores":{
                        "retrieval":hit.score,
                        "dense":hit.dense_score,
                        "lexical":hit.lexical_score
                    },
                    "eligible_evidence_count":hit.eligible_occurrence_count,
                    "evidence_exhausted":hit.occurrences_exhausted,
                    "evidence":evidence
                })
            })
            .collect::<Vec<_>>();
        output.insert(
            "candidates".to_owned(),
            serde_json::to_value(candidates).expect("candidates serialize"),
        );
    }
    Value::Object(output)
}

fn parse_filters(value: Option<&Value>) -> Result<SearchFilters, ProviderError> {
    let Some(value) = value else {
        return Ok(SearchFilters::default());
    };
    let value = exact_object(
        value,
        &["relations", "time_start_ms", "time_end_ms"],
        &[],
        "filters",
    )?;
    let relations = match value.get("relations") {
        None => BTreeSet::new(),
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| {
                row.as_str().map(str::to_owned).ok_or_else(|| {
                    ProviderError::new("invalid_request", "filters.relations must be strings")
                })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(ProviderError::new(
                "invalid_request",
                "filters.relations must be an array",
            ));
        }
    };
    let time_start_ms = optional_u64(value, "time_start_ms")?;
    let time_end_ms = optional_u64(value, "time_end_ms")?;
    if matches!((time_start_ms, time_end_ms), (Some(start), Some(end)) if start >= end) {
        return Err(ProviderError::new(
            "invalid_request",
            "filter time range is empty",
        ));
    }
    Ok(SearchFilters {
        relations,
        time_start_ms,
        time_end_ms,
    })
}

fn mounts_by_name(rows: &[Value]) -> Result<BTreeMap<&str, &Map<String, Value>>, ProviderError> {
    if rows.len() != 4 {
        return Err(ProviderError::new(
            "invalid_binding",
            "open requires exactly four immutable mounts",
        ));
    }
    let mut mounts = BTreeMap::new();
    let fields = [
        "logical_name",
        "role",
        "component",
        "access",
        "process_path",
    ];
    for row in rows {
        let mount = exact_object(row, &fields, &fields, "mount")?;
        let name = string(mount, "logical_name", "mount")?;
        if string(mount, "access", "mount")? != "read_only" || mounts.insert(name, mount).is_some()
        {
            return Err(ProviderError::new(
                "policy_denied",
                "mounts must be uniquely named and read-only",
            ));
        }
        validate_component(mount.get("component").expect("required"), "mount component")?;
        if string(mount, "process_path", "mount")?.is_empty() {
            return Err(ProviderError::new(
                "invalid_binding",
                "mount path must be non-empty",
            ));
        }
    }
    let expected = [
        "embedding-profile",
        "evidence-index",
        "index-admission-receipt",
        "tool-binding-lock",
    ];
    if mounts.keys().copied().collect::<Vec<_>>() != expected {
        return Err(ProviderError::new(
            "invalid_binding",
            "mount set is not the exact provider contract",
        ));
    }
    Ok(mounts)
}

fn required_mount<'a>(
    mounts: &'a BTreeMap<&str, &'a Map<String, Value>>,
    name: &str,
    role: &str,
) -> Result<&'a Map<String, Value>, ProviderError> {
    let mount = mounts
        .get(name)
        .copied()
        .ok_or_else(|| ProviderError::new("invalid_binding", "required mount is absent"))?;
    if string(mount, "role", "mount")? != role {
        return Err(ProviderError::new(
            "invalid_binding",
            format!("{name} mount role is invalid"),
        ));
    }
    Ok(mount)
}

fn mount_component_sha(mount: &Map<String, Value>) -> Option<&str> {
    mount.get("component")?.get("sha256")?.as_str()
}

fn read_mount_bytes(path_text: &str, filename: &str) -> Result<Vec<u8>, ProviderError> {
    let path = Path::new(path_text);
    let path = if path.is_dir() {
        path.join(filename)
    } else {
        path.to_path_buf()
    };
    fs::read(path).map_err(|_| {
        ProviderError::new(
            "corrupt_artifact",
            format!("mounted {filename} is unreadable"),
        )
    })
}

fn read_json_mount(path_text: &str, filename: &str) -> Result<Value, ProviderError> {
    serde_json::from_slice(&read_mount_bytes(path_text, filename)?).map_err(|_| {
        ProviderError::new(
            "corrupt_artifact",
            format!("mounted {filename} is invalid JSON"),
        )
    })
}

fn admission_checks_pass(receipt: &Value) -> bool {
    let required = [
        "object_digests",
        "source_binding",
        "safe_paths",
        "schema_profiles",
        "coverage_closure",
        "pointer_closure",
        "offline_query_conformance",
        "conformance",
    ];
    receipt
        .get("checks")
        .and_then(Value::as_object)
        .is_some_and(|checks| {
            required
                .iter()
                .all(|name| checks.get(*name) == Some(&Value::Bool(true)))
        })
}

fn loopback_endpoint(contract: &Value) -> Result<Option<String>, ProviderError> {
    let fields = ["mode", "network", "secret_handles", "vendor_services"];
    let contract = exact_object(contract, &fields, &fields, "query_time_contract")?;
    if contract.get("secret_handles") != Some(&Value::Array(Vec::new()))
        || contract.get("vendor_services") != Some(&Value::Array(Vec::new()))
    {
        return Err(ProviderError::new(
            "policy_denied",
            "embedding contract must be secret-free and vendor-free",
        ));
    }
    let mode = string(contract, "mode", "query_time_contract")?;
    let network = array(contract, "network", "query_time_contract")?;
    if mode == "offline_closed" && network.is_empty() {
        return Ok(None);
    }
    if mode != "local_component" || network.len() != 1 {
        return Err(ProviderError::new(
            "invalid_binding",
            "local_component requires exactly one loopback endpoint",
        ));
    }
    let endpoint = network[0]
        .as_str()
        .and_then(|value| value.strip_prefix("loopback:"))
        .ok_or_else(|| {
            ProviderError::new("policy_denied", "embedding endpoint must be loopback")
        })?;
    normalize_loopback_http_endpoint(endpoint)
        .map(Some)
        .map_err(|_| {
            ProviderError::new(
                "policy_denied",
                "embedding endpoint is not exact loopback HTTP",
            )
        })
}

pub async fn serve<R: BufRead, W: Write>(reader: R, mut writer: W) -> std::io::Result<()> {
    let mut provider = Provider::new();
    for line in reader.lines() {
        let line = line?;
        let mut request_id = "unknown".to_owned();
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => {
                if let Some(id) = request.get("id").and_then(Value::as_str) {
                    request_id = id.to_owned();
                }
                match provider.handle(&request).await {
                    Ok(result) => json!({"protocol":PROTOCOL,"id":request_id,"result":result}),
                    Err(error) => error_response(&request_id, &error),
                }
            }
            Err(_) => error_response(
                &request_id,
                &ProviderError::new("invalid_request", "request is not valid JSON"),
            ),
        };
        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

fn error_response(request_id: &str, error: &ProviderError) -> Value {
    json!({
        "protocol":PROTOCOL,
        "id":request_id,
        "error":{"code":error.code,"message":error.message,"retryable":error.retryable}
    })
}

fn exact_object<'a>(
    value: &'a Value,
    allowed: &[&str],
    required: &[&str],
    label: &str,
) -> Result<&'a Map<String, Value>, ProviderError> {
    let object = value.as_object().ok_or_else(|| {
        ProviderError::new("invalid_request", format!("{label} must be an object"))
    })?;
    if object.keys().any(|key| !allowed.contains(&key.as_str()))
        || required.iter().any(|key| !object.contains_key(*key))
    {
        return Err(ProviderError::new(
            "invalid_request",
            format!("{label} has unknown or missing fields"),
        ));
    }
    Ok(object)
}

fn string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, ProviderError> {
    object.get(field).and_then(Value::as_str).ok_or_else(|| {
        ProviderError::new(
            "invalid_request",
            format!("{label}.{field} must be a string"),
        )
    })
}

fn array<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], ProviderError> {
    object
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            ProviderError::new(
                "invalid_request",
                format!("{label}.{field} must be an array"),
            )
        })
}

fn positive_u64(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<u64, ProviderError> {
    let value = object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        ProviderError::new(
            "invalid_request",
            format!("{label}.{field} must be a positive integer"),
        )
    })?;
    if value == 0 {
        return Err(ProviderError::new(
            "invalid_request",
            format!("{label}.{field} must be positive"),
        ));
    }
    Ok(value)
}

fn positive_usize(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<usize, ProviderError> {
    usize::try_from(positive_u64(object, field, label)?)
        .map_err(|_| ProviderError::new("invalid_request", format!("{label}.{field} is too large")))
}

fn optional_positive_usize(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<usize>, ProviderError> {
    object
        .get(field)
        .map(|_| positive_usize(object, field, "limits"))
        .transpose()
}

fn optional_positive_u64(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, ProviderError> {
    object
        .get(field)
        .map(|_| positive_u64(object, field, "limits"))
        .transpose()
}

fn optional_u64(object: &Map<String, Value>, field: &str) -> Result<Option<u64>, ProviderError> {
    object
        .get(field)
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                ProviderError::new(
                    "invalid_request",
                    format!("{field} must be an unsigned integer"),
                )
            })
        })
        .transpose()
}

fn session_only(params: &Value) -> Result<&str, ProviderError> {
    let params = exact_object(params, &["session_id"], &["session_id"], "params")?;
    string(params, "session_id", "params")
}

fn validate_component(value: &Value, label: &str) -> Result<(), ProviderError> {
    let component = exact_object(
        value,
        &["id", "version", "sha256", "uri"],
        &["id", "version", "sha256"],
        label,
    )?;
    if string(component, "id", label)?.is_empty() || string(component, "version", label)?.is_empty()
    {
        return Err(ProviderError::new(
            "invalid_binding",
            format!("{label} identity is empty"),
        ));
    }
    validate_sha256(string(component, "sha256", label)?, label)
}

fn validate_sha256(value: &str, label: &str) -> Result<(), ProviderError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProviderError::new(
            "invalid_binding",
            format!("{label} is not a canonical SHA-256"),
        ));
    }
    Ok(())
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
