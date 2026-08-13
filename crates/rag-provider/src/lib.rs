//! Native provider for the fast experimental evidence index.
//!
//! The transport follows the language-neutral Livefire SDK JSONL lifecycle.
//! Index admission remains a host responsibility; this development provider
//! verifies the exact identities and read-only mount supplied at `open`.

#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, Write},
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rag_embedding::{
    Embedder, EmbeddingError, LmStudioEmbedder, try_compose_query, validate_embedding_profile,
    validate_vector,
};
use rag_index::{FastIndex, IndexError, SearchFilters, SearchHit, SearchMode};
use serde_json::{Map, Value, json};

pub const PROTOCOL: &str = "livefire.tool/1";
pub const PROVIDER_ID: &str = "com.ayc.livefire-rag.fast-evidence-provider";
pub const TOOL_ID: &str = "com.ayc.livefire-rag.fast-evidence.search";
pub const FORMAT_ID: &str = "com.ayc.livefire-rag.fast-index-format";

const PROVIDER_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const TOOL_SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const FORMAT_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";

#[must_use]
pub fn provider_ref() -> Value {
    component(PROVIDER_ID, PROVIDER_SHA256)
}

#[must_use]
pub fn tool_ref() -> Value {
    component(TOOL_ID, TOOL_SHA256)
}

#[must_use]
pub fn format_ref() -> Value {
    component(FORMAT_ID, FORMAT_SHA256)
}

fn component(id: &str, sha256: &str) -> Value {
    json!({"id":id,"version":"0.1.0","sha256":sha256})
}

struct Session {
    index: FastIndex,
    index_ref: Value,
    source_snapshots: Vec<Value>,
    binding_lock_sha256: String,
    embedding_endpoint: Option<String>,
    request_bytes: usize,
    result_bytes: usize,
    wall_time_ms: u64,
    max_candidates: usize,
}

#[derive(Default)]
pub struct Provider {
    handshaken: bool,
    next_session: u64,
    sessions: BTreeMap<String, Session>,
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
            "provider":provider_ref(),
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
        if params.get("provider") != Some(&provider_ref()) {
            return Err(ProviderError::new(
                "invalid_binding",
                "provider identity does not match the executable",
            ));
        }
        if params.get("tools") != Some(&Value::Array(vec![tool_ref()])) {
            return Err(ProviderError::new(
                "invalid_binding",
                "open requires the advertised singleton tool",
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

        let mounts = array(params, "mounts", "params")?;
        if mounts.len() != 1 {
            return Err(ProviderError::new(
                "invalid_binding",
                "fast provider accepts exactly one index mount",
            ));
        }
        let mount = exact_object(
            &mounts[0],
            &[
                "logical_name",
                "role",
                "component",
                "access",
                "process_path",
            ],
            &[
                "logical_name",
                "role",
                "component",
                "access",
                "process_path",
            ],
            "index mount",
        )?;
        if string(mount, "role", "index mount")? != "index"
            || string(mount, "access", "index mount")? != "read_only"
            || mount.get("component") != Some(&index_ref)
        {
            return Err(ProviderError::new(
                "invalid_binding",
                "index mount role, access, or component is invalid",
            ));
        }
        let index_path = string(mount, "process_path", "index mount")?;
        let index = FastIndex::open(Path::new(index_path)).map_err(|_| {
            ProviderError::new("corrupt_artifact", "mounted index failed fast open")
        })?;
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
        let limits = exact_object(
            params.get("limits").expect("required"),
            &[
                "request_bytes",
                "result_bytes",
                "wall_time_ms",
                "memory_bytes",
                "max_candidates",
            ],
            &[],
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
        let query_contract = params
            .get("query_time_contract")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ProviderError::new("invalid_binding", "query_time_contract must be an object")
            })?;
        let embedding_endpoint = query_contract
            .get("embedding_endpoint")
            .and_then(Value::as_str)
            .map(str::to_owned);

        self.next_session += 1;
        let session_id = format!("fast-evidence-{}", self.next_session);
        self.sessions.insert(
            session_id.clone(),
            Session {
                index,
                index_ref,
                source_snapshots,
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
                        json!({
                            "snapshot_sha256":occurrence.snapshot_sha256,
                            "mapping_sha256":occurrence.mapping_sha256,
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
