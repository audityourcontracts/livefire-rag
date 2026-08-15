//! Resumable, content-bound embedding batches for the experimental indexer.

use std::{future::Future, path::Path, time::Duration};

use reqwest::Client;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use url::{Host, Url};

pub use rag_contracts::{MAX_DOCUMENT_FORMAT_BYTES, MAX_FORMATTED_DOCUMENT_BYTES};

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding cache failed: {0}")]
    Cache(#[from] rusqlite::Error),
    #[error("embedding request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("embedding file operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("embedding response JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("embedding task failed: {0}")]
    Task(String),
    #[error("temporary embedding backend failure: {0}")]
    Temporary(String),
    #[error("embedding response violated the profile: {0}")]
    Invalid(&'static str),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    Temporary,
    Permanent,
}

impl EmbeddingError {
    #[must_use]
    pub fn retry_class(&self) -> RetryClass {
        match self {
            Self::Http(error)
                if error.is_timeout()
                    || error.is_connect()
                    || error.status().is_some_and(|status| {
                        matches!(status.as_u16(), 408 | 429) || status.is_server_error()
                    }) =>
            {
                RetryClass::Temporary
            }
            Self::Temporary(_) => RetryClass::Temporary,
            Self::Cache(_)
            | Self::Http(_)
            | Self::Io(_)
            | Self::Json(_)
            | Self::Task(_)
            | Self::Invalid(_) => RetryClass::Permanent,
        }
    }
}

mod shard;
mod task;

pub use shard::{
    AtomicFilePublication, AtomicPublishOutcome, EMBEDDING_SHARD_HEADER_BYTES,
    EMBEDDING_SHARD_MAGIC, EmbeddingShard, EmbeddingShardExpectation, EmbeddingShardMetadata,
    EmbeddingShardVectorReader, EmbeddingShardWriter, EmbeddingTaskPartPreparation,
    VerifiedEmbeddingTaskPart, complete_embedding_task_part_recovery, decode_sha256_hex,
    prepare_embedding_task_part, restore_quarantined_embedding_task_part,
    verify_embedding_task_part,
};
pub use task::{
    EmbeddingAttemptOutcome, EmbeddingAttemptReport, EmbeddingBatchReport, EmbeddingTaskOptions,
    EmbeddingTaskReport, EmbeddingTaskStats, RetryPolicy, TaskSelection, execute_embedding_task,
    execute_embedding_task_reported,
};

pub const MAX_QUERY_BYTES: usize = 8_192;
pub const MAX_QUERY_INSTRUCTION_BYTES: usize = 8_192;
pub const MAX_QUERY_COMPOSITION_BYTES: usize = 1_024;
pub const MAX_COMPOSED_QUERY_BYTES: usize = 16_384;

/// Parse and canonicalize the only network endpoint permitted by the local
/// embedding contract. Keeping this policy beside the HTTP client prevents a
/// caller from validating one URL interpretation and requesting another.
pub fn normalize_loopback_http_endpoint(endpoint: &str) -> Result<String> {
    let parsed =
        Url::parse(endpoint).map_err(|_| EmbeddingError::Invalid("embedding endpoint URL"))?;
    if parsed.scheme() != "http"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.path() != "/"
    {
        return Err(EmbeddingError::Invalid("embedding endpoint policy"));
    }
    let port = explicit_port(endpoint, &parsed)?;
    let host = match parsed.host() {
        Some(Host::Domain("localhost")) => "localhost".to_owned(),
        Some(Host::Ipv4(address)) if address.octets() == [127, 0, 0, 1] => "127.0.0.1".to_owned(),
        Some(Host::Ipv6(address)) if address.is_loopback() => "[::1]".to_owned(),
        _ => return Err(EmbeddingError::Invalid("embedding endpoint host")),
    };
    Ok(format!("http://{host}:{port}"))
}

fn explicit_port(endpoint: &str, parsed: &Url) -> Result<u16> {
    let authority = endpoint
        .strip_prefix("http://")
        .and_then(|rest| rest.split('/').next())
        .ok_or(EmbeddingError::Invalid("embedding endpoint port"))?;
    let port_text = if authority.starts_with('[') {
        authority
            .split_once("]:")
            .map(|(_, port)| port)
            .filter(|_| authority.ends_with(|character: char| character.is_ascii_digit()))
    } else {
        authority.rsplit_once(':').map(|(_, port)| port)
    }
    .ok_or(EmbeddingError::Invalid("embedding endpoint port"))?;
    if port_text.is_empty() || !port_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(EmbeddingError::Invalid("embedding endpoint port"));
    }
    let port = port_text
        .parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
        .ok_or(EmbeddingError::Invalid("embedding endpoint port"))?;
    if parsed.port_or_known_default() != Some(port) {
        return Err(EmbeddingError::Invalid("embedding endpoint port"));
    }
    Ok(port)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProfile {
    pub id: String,
    pub version: String,
    pub sha256: String,
    pub model: String,
    pub dimensions: u32,
    pub normalization: String,
    /// A deterministic post-processing step applied to the model's full
    /// output. Reduced profiles keep their own identity and name the exact
    /// parent profile whose vectors they were derived from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_derivation: Option<VectorDerivation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_instruction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_composition: Option<String>,
}

pub const PREFIX_L2_NORMALIZE_V1: &str = "prefix_then_l2_normalize_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorDerivation {
    pub parent_embedding_profile_sha256: String,
    pub parent_dimensions: u32,
    pub transformation: String,
}

/// Read either the compact fast-index profile or the existing bound embedding
/// policy. The emitted index always carries the compact, content-bound form.
pub fn parse_embedding_profile(bytes: &[u8]) -> Result<EmbeddingProfile> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| EmbeddingError::Invalid("embedding profile JSON"))?;
    if let Ok(profile) = serde_json::from_value::<EmbeddingProfile>(value.clone()) {
        validate_embedding_profile(&profile)?;
        return Ok(profile);
    }
    let object = value
        .as_object()
        .ok_or(EmbeddingError::Invalid("embedding policy object"))?;
    let artifact = object
        .get("model_artifact_set")
        .and_then(serde_json::Value::as_object)
        .ok_or(EmbeddingError::Invalid("model artifact set"))?;
    let profile = EmbeddingProfile {
        id: artifact
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or(EmbeddingError::Invalid("profile id"))?
            .to_owned(),
        version: artifact
            .get("version")
            .and_then(serde_json::Value::as_str)
            .ok_or(EmbeddingError::Invalid("profile version"))?
            .to_owned(),
        sha256: hex_digest(bytes),
        model: object
            .get("api_model_key")
            .and_then(serde_json::Value::as_str)
            .ok_or(EmbeddingError::Invalid("API model key"))?
            .to_owned(),
        dimensions: u32::try_from(
            object
                .get("dimensions")
                .and_then(serde_json::Value::as_u64)
                .ok_or(EmbeddingError::Invalid("profile dimensions"))?,
        )
        .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?,
        normalization: object
            .get("normalization")
            .and_then(serde_json::Value::as_str)
            .ok_or(EmbeddingError::Invalid("normalization"))?
            .to_owned(),
        vector_derivation: object
            .get("vector_derivation")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| EmbeddingError::Invalid("vector derivation"))?,
        query_instruction: object
            .get("query_instruction")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        query_composition: object
            .get("query_composition")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    };
    validate_embedding_profile(&profile)?;
    Ok(profile)
}

/// Parse a profile only when the supplied bytes themselves have the digest
/// frozen by a plan. This must be used at durable execution/assembly
/// boundaries; comparing only the compact profile's declared `sha256` lets a
/// caller relabel different model settings with another profile's digest.
pub fn parse_bound_embedding_profile(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<EmbeddingProfile> {
    if hex_digest(bytes) != expected_sha256 {
        return Err(EmbeddingError::Invalid("embedding profile byte digest"));
    }
    let profile = parse_embedding_profile(bytes)?;
    if profile.sha256 != expected_sha256 {
        return Err(EmbeddingError::Invalid("embedding profile binding"));
    }
    Ok(profile)
}

pub fn validate_embedding_profile(profile: &EmbeddingProfile) -> Result<()> {
    if profile.id.is_empty()
        || profile.version.is_empty()
        || profile.model.is_empty()
        || profile.dimensions == 0
        || !matches!(profile.normalization.as_str(), "l2" | "none")
        || profile.sha256.len() != 64
        || !profile
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || profile.query_instruction.is_some() != profile.query_composition.is_some()
    {
        return Err(EmbeddingError::Invalid("embedding profile fields"));
    }
    if let Some(derivation) = &profile.vector_derivation
        && (derivation.transformation != PREFIX_L2_NORMALIZE_V1
            || derivation.parent_dimensions != 4_096
            || !matches!(profile.dimensions, 1_024 | 2_048)
            || profile.normalization != "l2"
            || derivation.parent_embedding_profile_sha256.len() != 64
            || !derivation
                .parent_embedding_profile_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(EmbeddingError::Invalid("vector derivation"));
    }
    match (&profile.query_instruction, &profile.query_composition) {
        (None, None) => {}
        (Some(instruction), Some(composition)) => {
            if instruction.len() > MAX_QUERY_INSTRUCTION_BYTES
                || composition.len() > MAX_QUERY_COMPOSITION_BYTES
                || composition.matches("{query}").count() != 1
                || composition.matches("{query_instruction}").count() != 1
            {
                return Err(EmbeddingError::Invalid("query composition contract"));
            }
            let remaining = composition
                .replace("{query_instruction}", "")
                .replace("{query}", "");
            if remaining.contains('{') || remaining.contains('}') {
                return Err(EmbeddingError::Invalid("query composition placeholder"));
            }
            let expanded = composition
                .len()
                .checked_sub("{query_instruction}".len() + "{query}".len())
                .and_then(|fixed| fixed.checked_add(instruction.len()))
                .and_then(|size| size.checked_add(MAX_QUERY_BYTES))
                .ok_or(EmbeddingError::Invalid("query composition size"))?;
            if expanded > MAX_COMPOSED_QUERY_BYTES {
                return Err(EmbeddingError::Invalid("query composition expansion"));
            }
        }
        _ => return Err(EmbeddingError::Invalid("embedding profile fields")),
    }
    Ok(())
}

/// Convert the model's response into the vector represented by `profile`.
/// Normal profiles are only validated. A reduced profile takes the declared
/// prefix and normalizes it again, because truncation changes its length.
pub fn adapt_model_vector(profile: &EmbeddingProfile, vector: Vec<f32>) -> Result<Vec<f32>> {
    validate_embedding_profile(profile)?;
    let Some(derivation) = &profile.vector_derivation else {
        validate_vector(
            &vector,
            usize::try_from(profile.dimensions)
                .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?,
            &profile.normalization,
        )?;
        return Ok(vector);
    };
    if vector.len()
        != usize::try_from(derivation.parent_dimensions)
            .map_err(|_| EmbeddingError::Invalid("parent vector dimensions"))?
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(EmbeddingError::Invalid("parent vector dimensions"));
    }
    let target = usize::try_from(profile.dimensions)
        .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?;
    let mut reduced = vector[..target].to_vec();
    let squared_norm = reduced
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if !squared_norm.is_finite() || squared_norm <= f64::EPSILON {
        return Err(EmbeddingError::Invalid("reduced vector norm"));
    }
    let norm = squared_norm.sqrt();
    for value in &mut reduced {
        *value = (f64::from(*value) / norm) as f32;
    }
    validate_vector(&reduced, target, "l2")?;
    Ok(reduced)
}

pub fn try_compose_query(profile: &EmbeddingProfile, query: &str) -> Result<String> {
    validate_embedding_profile(profile)?;
    if query.is_empty() || query.len() > MAX_QUERY_BYTES {
        return Err(EmbeddingError::Invalid("query length"));
    }
    match (&profile.query_instruction, &profile.query_composition) {
        (Some(instruction), Some(composition)) => {
            let instruction_token = "{query_instruction}";
            let query_token = "{query}";
            let instruction_at = composition
                .find(instruction_token)
                .ok_or(EmbeddingError::Invalid("query composition contract"))?;
            let query_at = composition
                .find(query_token)
                .ok_or(EmbeddingError::Invalid("query composition contract"))?;
            let mut value =
                String::with_capacity(composition.len() + instruction.len() + query.len());
            if instruction_at < query_at {
                value.push_str(&composition[..instruction_at]);
                value.push_str(instruction);
                value.push_str(&composition[instruction_at + instruction_token.len()..query_at]);
                value.push_str(query);
                value.push_str(&composition[query_at + query_token.len()..]);
            } else {
                value.push_str(&composition[..query_at]);
                value.push_str(query);
                value.push_str(&composition[query_at + query_token.len()..instruction_at]);
                value.push_str(instruction);
                value.push_str(&composition[instruction_at + instruction_token.len()..]);
            }
            if value.len() > MAX_COMPOSED_QUERY_BYTES {
                return Err(EmbeddingError::Invalid("composed query length"));
            }
            Ok(value)
        }
        (None, None) => Ok(query.to_owned()),
        _ => Err(EmbeddingError::Invalid("embedding profile fields")),
    }
}

/// Apply the exact document-input format frozen in an embedding plan. The
/// format is deliberately closed to one placeholder so a worker cannot
/// silently interpret templates differently.
pub fn format_document_input(document_format: &str, semantic_text: &str) -> Result<String> {
    rag_contracts::format_document_input(document_format, semantic_text)
        .map_err(|_| EmbeddingError::Invalid("document input format"))
}

#[derive(Debug, Clone)]
pub struct EmbeddingInput {
    pub document_id: String,
    pub document_sha256: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddedDocument {
    pub document_id: String,
    pub vector: Vec<f32>,
    pub from_cache: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingCacheStats {
    pub cache_hits: usize,
    pub embedded: usize,
}

pub trait Embedder: Sync {
    fn embed(&self, texts: &[String]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifiedEmbeddingBatch {
    pub vectors: Vec<Vec<f32>>,
    pub returned_model: String,
}

/// Embedding backends used for durable task execution must return the model
/// identity supplied by the server, not a copy of the requested model key.
pub trait IdentifiedEmbedder: Embedder {
    fn embed_identified(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<IdentifiedEmbeddingBatch>> + Send;
}

pub struct LmStudioEmbedder {
    client: Client,
    endpoint: String,
    model: String,
}

impl LmStudioEmbedder {
    #[must_use]
    pub fn new(endpoint: &str, model: &str) -> Self {
        Self::with_timeout(endpoint, model, Duration::from_secs(300))
            .expect("static LM Studio HTTP client configuration")
    }

    pub fn with_timeout(endpoint: &str, model: &str, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(EmbeddingError::Invalid("embedding timeout"));
        }
        Ok(Self {
            client: Client::builder().timeout(timeout).build()?,
            endpoint: normalize_loopback_http_endpoint(endpoint)?,
            model: model.to_owned(),
        })
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    model: String,
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

impl LmStudioEmbedder {
    async fn request_with_digest(
        &self,
        texts: &[String],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        let response_bytes = self
            .client
            .post(format!("{}/v1/embeddings", self.endpoint))
            .json(&EmbeddingRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        let response_value: serde_json::Value = serde_json::from_slice(&response_bytes)?;
        let mut normalized = response_value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .ok_or(EmbeddingError::Invalid("response data"))?
            .iter()
            .map(|datum| {
                let index = datum
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or(EmbeddingError::Invalid("response index"))?;
                let embedding = datum
                    .get("embedding")
                    .cloned()
                    .ok_or(EmbeddingError::Invalid("response embedding"))?;
                Ok((index, embedding))
            })
            .collect::<Result<Vec<_>>>()?;
        normalized.sort_by_key(|(index, _)| *index);
        let normalized = serde_json::Value::Array(
            normalized
                .into_iter()
                .map(|(_, embedding)| embedding)
                .collect(),
        );
        let mut normalized_bytes = serde_json_canonicalizer::to_vec(&normalized)?;
        normalized_bytes.push(b'\n');
        let normalized_sha256 = hex_digest(&normalized_bytes);
        let response: EmbeddingResponse = serde_json::from_slice(&response_bytes)?;
        if response.data.len() != texts.len() {
            return Err(EmbeddingError::Invalid("response cardinality"));
        }
        let mut ordered = vec![None; texts.len()];
        for datum in response.data {
            let slot = ordered
                .get_mut(datum.index)
                .ok_or(EmbeddingError::Invalid("response index"))?;
            if slot.replace(datum.embedding).is_some() {
                return Err(EmbeddingError::Invalid("duplicate response index"));
            }
        }
        let vectors = ordered
            .into_iter()
            .map(|value| value.ok_or(EmbeddingError::Invalid("missing response index")))
            .collect::<Result<Vec<_>>>()?;
        if response.model.is_empty() {
            return Err(EmbeddingError::Invalid("response model identity"));
        }
        Ok((
            IdentifiedEmbeddingBatch {
                vectors,
                returned_model: response.model,
            },
            normalized_sha256,
        ))
    }

    async fn request(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        Ok(self.request_with_digest(texts).await?.0)
    }

    /// Execute the exact server response normalization used by the bound
    /// conformance profile without first rounding response numbers to f32.
    pub async fn conformance_probe(
        &self,
        texts: &[String],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        self.request_with_digest(texts).await
    }
}

impl Embedder for LmStudioEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.request(texts).await?.vectors)
    }
}

impl IdentifiedEmbedder for LmStudioEmbedder {
    async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        self.request(texts).await
    }
}

pub struct EmbeddingCache {
    connection: Connection,
}

impl EmbeddingCache {
    pub fn open(path: &Path) -> Result<Self> {
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS vectors (
               cache_key TEXT PRIMARY KEY,
               dimensions INTEGER NOT NULL,
               vector BLOB NOT NULL
             );",
        )?;
        Ok(Self { connection })
    }

    fn get(&self, key: &str, dimensions: usize) -> Result<Option<Vec<f32>>> {
        let mut statement = self
            .connection
            .prepare("SELECT dimensions, vector FROM vectors WHERE cache_key = ?1")?;
        let mut rows = statement.query([key])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let stored_dimensions = usize::try_from(row.get::<_, i64>(0)?)
            .map_err(|_| EmbeddingError::Invalid("cached dimensions"))?;
        let bytes: Vec<u8> = row.get(1)?;
        if stored_dimensions != dimensions || bytes.len() != dimensions * 4 {
            return Err(EmbeddingError::Invalid("cached vector shape"));
        }
        Ok(Some(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four byte chunk")))
                .collect(),
        ))
    }

    pub fn vector(&self, profile: &EmbeddingProfile, input: &EmbeddingInput) -> Result<Vec<f32>> {
        let dimensions = usize::try_from(profile.dimensions)
            .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?;
        let vector = self
            .get(&cache_key(profile, input), dimensions)?
            .ok_or(EmbeddingError::Invalid("missing cached vector"))?;
        validate_vector(&vector, dimensions, &profile.normalization)?;
        Ok(vector)
    }

    fn put_batch(&mut self, rows: &[(String, Vec<f32>)]) -> Result<()> {
        let transaction = self.connection.transaction()?;
        {
            let mut statement = transaction.prepare(
                "INSERT OR REPLACE INTO vectors(cache_key, dimensions, vector) VALUES (?1, ?2, ?3)",
            )?;
            for (key, vector) in rows {
                let bytes = vector
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                statement.execute(params![key, vector.len() as i64, bytes])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    #[cfg(test)]
    fn row_count(&self) -> Result<u64> {
        let count: i64 = self
            .connection
            .query_row("SELECT count(*) FROM vectors", [], |row| row.get(0))?;
        u64::try_from(count).map_err(|_| EmbeddingError::Invalid("cached row count"))
    }
}

/// Populate every missing cache row without retaining the corpus vector matrix
/// in memory. A model response is fully validated before its whole batch is
/// committed in one SQLite transaction.
pub async fn ensure_cached<E: Embedder>(
    embedder: &E,
    cache: &mut EmbeddingCache,
    profile: &EmbeddingProfile,
    inputs: &[EmbeddingInput],
    batch_size: usize,
) -> Result<EmbeddingCacheStats> {
    if batch_size == 0 || batch_size > 32 {
        return Err(EmbeddingError::Invalid("batch size"));
    }
    validate_embedding_profile(profile)?;
    let dimensions = usize::try_from(profile.dimensions)
        .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?;
    let mut missing = Vec::new();
    let mut cache_hits = 0;
    for (ordinal, input) in inputs.iter().enumerate() {
        let key = cache_key(profile, input);
        if let Some(vector) = cache.get(&key, dimensions)? {
            validate_vector(&vector, dimensions, &profile.normalization)?;
            cache_hits += 1;
        } else {
            missing.push((ordinal, key));
        }
    }
    for batch in missing.chunks(batch_size) {
        let texts = batch
            .iter()
            .map(|(ordinal, _)| inputs[*ordinal].text.clone())
            .collect::<Vec<_>>();
        let vectors = embedder.embed(&texts).await?;
        if vectors.len() != batch.len() {
            return Err(EmbeddingError::Invalid("batch cardinality"));
        }
        let mut cache_rows = Vec::with_capacity(batch.len());
        for ((_, key), vector) in batch.iter().zip(vectors) {
            validate_vector(&vector, dimensions, &profile.normalization)?;
            cache_rows.push((key.clone(), vector));
        }
        cache.put_batch(&cache_rows)?;
    }
    Ok(EmbeddingCacheStats {
        cache_hits,
        embedded: missing.len(),
    })
}

#[must_use]
pub fn cache_key(profile: &EmbeddingProfile, input: &EmbeddingInput) -> String {
    let mut hasher = Sha256::new();
    let profile_material = serde_json::to_vec(profile).expect("embedding profile serializes");
    for value in [
        profile.sha256.as_str(),
        input.document_id.as_str(),
        input.document_sha256.as_str(),
        hex_digest(input.text.as_bytes()).as_str(),
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    hasher.update(profile_material);
    hasher.update([0]);
    format!("{:x}", hasher.finalize())
}

pub async fn embed_resumable<E: Embedder>(
    embedder: &E,
    cache: &mut EmbeddingCache,
    profile: &EmbeddingProfile,
    inputs: &[EmbeddingInput],
    batch_size: usize,
) -> Result<Vec<EmbeddedDocument>> {
    if batch_size == 0 || batch_size > 32 {
        return Err(EmbeddingError::Invalid("batch size"));
    }
    let dimensions = usize::try_from(profile.dimensions)
        .map_err(|_| EmbeddingError::Invalid("profile dimensions"))?;
    let mut output = Vec::with_capacity(inputs.len());
    let mut missing = Vec::new();
    for (ordinal, input) in inputs.iter().enumerate() {
        let key = cache_key(profile, input);
        if let Some(vector) = cache.get(&key, dimensions)? {
            validate_vector(&vector, dimensions, &profile.normalization)?;
            output.push(Some(EmbeddedDocument {
                document_id: input.document_id.clone(),
                vector,
                from_cache: true,
            }));
        } else {
            output.push(None);
            missing.push((ordinal, key));
        }
    }
    for batch in missing.chunks(batch_size) {
        let texts = batch
            .iter()
            .map(|(ordinal, _)| inputs[*ordinal].text.clone())
            .collect::<Vec<_>>();
        let vectors = embedder.embed(&texts).await?;
        if vectors.len() != batch.len() {
            return Err(EmbeddingError::Invalid("batch cardinality"));
        }
        let mut cache_rows = Vec::with_capacity(batch.len());
        for ((_, key), vector) in batch.iter().zip(&vectors) {
            validate_vector(vector, dimensions, &profile.normalization)?;
            cache_rows.push((key.clone(), vector.clone()));
        }
        cache.put_batch(&cache_rows)?;
        for ((ordinal, _), vector) in batch.iter().zip(vectors) {
            output[*ordinal] = Some(EmbeddedDocument {
                document_id: inputs[*ordinal].document_id.clone(),
                vector,
                from_cache: false,
            });
        }
    }
    output
        .into_iter()
        .map(|value| value.ok_or(EmbeddingError::Invalid("missing vector")))
        .collect()
}

pub fn validate_vector(vector: &[f32], dimensions: usize, normalization: &str) -> Result<()> {
    if vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()) {
        return Err(EmbeddingError::Invalid("vector shape or finiteness"));
    }
    if normalization == "l2" {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-4 {
            return Err(EmbeddingError::Invalid("vector norm"));
        }
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tempfile::tempdir;

    use super::*;

    struct FakeEmbedder {
        calls: AtomicUsize,
    }

    impl Embedder for FakeEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(texts
                .iter()
                .map(|text| {
                    if text == "a" {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }
    }

    fn profile() -> EmbeddingProfile {
        EmbeddingProfile {
            id: "test".into(),
            version: "1".into(),
            sha256: "a".repeat(64),
            model: "fake".into(),
            dimensions: 2,
            normalization: "l2".into(),
            vector_derivation: None,
            query_instruction: None,
            query_composition: None,
        }
    }

    #[test]
    fn prefix_derivation_is_normalized_ordered_and_deterministic() {
        let mut reduced = profile();
        reduced.dimensions = 1_024;
        reduced.vector_derivation = Some(VectorDerivation {
            parent_embedding_profile_sha256: "b".repeat(64),
            parent_dimensions: 4_096,
            transformation: PREFIX_L2_NORMALIZE_V1.into(),
        });
        let mut first_parent = vec![0.0; 4_096];
        first_parent[0] = 3.0;
        first_parent[1] = 4.0;
        first_parent[2_048] = 99.0;
        let mut second_parent = first_parent.clone();
        second_parent[2_048] = -77.0;
        let first = adapt_model_vector(&reduced, first_parent).unwrap();
        let second = adapt_model_vector(&reduced, second_parent).unwrap();
        assert_eq!(&first[..2], &[0.6, 0.8]);
        assert_eq!(first.len(), 1_024);
        assert_eq!(first, second);
        validate_vector(&first, 1_024, "l2").unwrap();
    }

    #[test]
    fn prefix_derivation_refuses_wrong_parent_size_and_zero_prefix() {
        let mut reduced = profile();
        reduced.dimensions = 1_024;
        reduced.vector_derivation = Some(VectorDerivation {
            parent_embedding_profile_sha256: "b".repeat(64),
            parent_dimensions: 4_096,
            transformation: PREFIX_L2_NORMALIZE_V1.into(),
        });
        assert!(adapt_model_vector(&reduced, vec![1.0, 0.0]).is_err());
        assert!(adapt_model_vector(&reduced, vec![0.0; 4_096]).is_err());
    }

    #[tokio::test]
    async fn cache_prevents_reembedding_and_preserves_order() {
        let root = tempdir().unwrap();
        let mut cache = EmbeddingCache::open(&root.path().join("cache.sqlite3")).unwrap();
        let embedder = FakeEmbedder {
            calls: AtomicUsize::new(0),
        };
        let inputs = [
            EmbeddingInput {
                document_id: "a".into(),
                document_sha256: "1".into(),
                text: "a".into(),
            },
            EmbeddingInput {
                document_id: "b".into(),
                document_sha256: "2".into(),
                text: "b".into(),
            },
        ];
        let first = embed_resumable(&embedder, &mut cache, &profile(), &inputs, 1)
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert_eq!(cache.row_count().unwrap(), 2);
        assert!(first.iter().all(|item| !item.from_cache));
        let second = embed_resumable(&embedder, &mut cache, &profile(), &inputs, 16)
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert!(second.iter().all(|item| item.from_cache));
        assert_eq!(second[0].document_id, "a");
        assert_eq!(second[1].vector, vec![0.0, 1.0]);
    }

    struct PartlyInvalidEmbedder;

    impl Embedder for PartlyInvalidEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            assert_eq!(texts.len(), 2);
            Ok(vec![vec![1.0, 0.0], vec![f32::NAN, 0.0]])
        }
    }

    #[tokio::test]
    async fn invalid_model_batch_writes_no_partial_cache_rows() {
        let root = tempdir().unwrap();
        let mut cache = EmbeddingCache::open(&root.path().join("cache.sqlite3")).unwrap();
        let inputs = [
            EmbeddingInput {
                document_id: "a".into(),
                document_sha256: "1".into(),
                text: "a".into(),
            },
            EmbeddingInput {
                document_id: "b".into(),
                document_sha256: "2".into(),
                text: "b".into(),
            },
        ];
        assert!(
            ensure_cached(&PartlyInvalidEmbedder, &mut cache, &profile(), &inputs, 2)
                .await
                .is_err()
        );
        assert_eq!(cache.row_count().unwrap(), 0);
    }

    #[test]
    fn compact_profile_validation_is_closed() {
        let mut value = serde_json::to_value(profile()).unwrap();
        value["normalization"] = serde_json::json!("mystery");
        assert!(parse_embedding_profile(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = serde_json::to_value(profile()).unwrap();
        value["query_instruction"] = serde_json::json!("instruction");
        assert!(parse_embedding_profile(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn query_composition_is_exact_and_bounded() {
        let mut profile = profile();
        profile.query_instruction = Some("Retrieve relevant security evidence".into());
        profile.query_composition = Some("Instruct: {query_instruction}\nQuery: {query}".into());
        assert_eq!(
            try_compose_query(&profile, "encoded PowerShell").unwrap(),
            "Instruct: Retrieve relevant security evidence\nQuery: encoded PowerShell"
        );

        for invalid in [
            "Query: {query} and {query}",
            "Query: no placeholder",
            "{query_instruction} {query} {unknown}",
            "{query_instruction} only",
        ] {
            profile.query_composition = Some(invalid.into());
            assert!(validate_embedding_profile(&profile).is_err(), "{invalid}");
        }

        profile.query_composition = Some("{query_instruction} {query}".into());
        assert!(try_compose_query(&profile, &"x".repeat(MAX_QUERY_BYTES + 1)).is_err());
        profile.query_instruction = Some("x".repeat(MAX_QUERY_INSTRUCTION_BYTES));
        assert!(validate_embedding_profile(&profile).is_err());

        profile.query_instruction = Some("literal {query} instruction".into());
        profile.query_composition = Some("Instruct: {query_instruction}\nQuery: {query}".into());
        assert_eq!(
            try_compose_query(&profile, "literal {query_instruction} query").unwrap(),
            "Instruct: literal {query} instruction\nQuery: literal {query_instruction} query"
        );
    }

    #[test]
    fn cache_key_binds_all_effective_profile_fields() {
        let input = EmbeddingInput {
            document_id: "doc".into(),
            document_sha256: "d".repeat(64),
            text: "activity".into(),
        };
        let first = profile();
        let mut second = first.clone();
        second.model = "different-model".into();
        assert_ne!(cache_key(&first, &input), cache_key(&second, &input));
    }

    #[test]
    fn durable_profile_binding_rejects_a_relabelled_compact_profile() {
        let compact = serde_json::to_vec(&profile()).unwrap();
        assert!(parse_embedding_profile(&compact).is_ok());
        assert!(parse_bound_embedding_profile(&compact, &"a".repeat(64)).is_err());
    }

    #[test]
    fn document_input_format_is_exact_and_closed() {
        assert_eq!(
            format_document_input("passage: {semantic_text}", "PowerShell download").unwrap(),
            "passage: PowerShell download"
        );
        for invalid in [
            "no placeholder",
            "{semantic_text} {semantic_text}",
            "{semantic_text} {unknown}",
        ] {
            assert!(format_document_input(invalid, "text").is_err());
        }
        assert!(format_document_input("{semantic_text}", "").is_err());
    }

    #[test]
    fn planner_and_executor_document_formatters_have_identical_results() {
        let exact_limit = "x".repeat(MAX_FORMATTED_DOCUMENT_BYTES);
        let over_limit = "x".repeat(MAX_FORMATTED_DOCUMENT_BYTES + 1);
        let maximum_format = format!(
            "{}{semantic_text}",
            "x".repeat(MAX_DOCUMENT_FORMAT_BYTES - "{semantic_text}".len()),
            semantic_text = "{semantic_text}",
        );
        let overlong_format = format!("x{maximum_format}");
        let fixtures = [
            ("{semantic_text}", "text"),
            ("passage: {semantic_text}", "PowerShell download"),
            ("prefix {semantic_text} suffix", "é 👩‍💻"),
            ("no placeholder", "text"),
            ("{semantic_text} {semantic_text}", "text"),
            ("{semantic_text} {unknown}", "text"),
            ("left { {semantic_text}", "text"),
            ("{semantic_text} right }", "text"),
            ("{semantic_text}", ""),
            (maximum_format.as_str(), "x"),
            (overlong_format.as_str(), "x"),
            ("{semantic_text}", exact_limit.as_str()),
            ("{semantic_text}", over_limit.as_str()),
        ];
        for (format, text) in fixtures {
            let executor = format_document_input(format, text);
            let planner = rag_pipeline::format_document_input_exact(format, text);
            assert_eq!(executor.is_ok(), planner.is_ok(), "format={format:?}");
            if let (Ok(executor), Ok(planner)) = (executor, planner) {
                assert_eq!(executor, planner, "format={format:?}");
            }
        }
    }

    #[test]
    fn loopback_endpoint_parser_normalizes_exact_local_origins() {
        assert_eq!(
            normalize_loopback_http_endpoint("http://LOCALHOST:1234/").unwrap(),
            "http://localhost:1234"
        );
        assert_eq!(
            normalize_loopback_http_endpoint("http://127.0.0.1:1234").unwrap(),
            "http://127.0.0.1:1234"
        );
        assert_eq!(
            normalize_loopback_http_endpoint("http://[0:0:0:0:0:0:0:1]:1234").unwrap(),
            "http://[::1]:1234"
        );
    }

    #[test]
    fn loopback_endpoint_parser_rejects_ambiguous_or_nonlocal_urls() {
        for endpoint in [
            "http://localhost:1234@evil.example",
            "http://local%68ost@evil.example:1234",
            "http://localhost.evil.example:1234",
            "http://127.0.0.2:1234",
            "http://localhost",
            "https://localhost:1234",
            "http://localhost:1234?target=evil",
            "http://localhost:1234#evil",
            "http://localhost:1234/v1",
            "http://localhost:0",
        ] {
            assert!(
                normalize_loopback_http_endpoint(endpoint).is_err(),
                "accepted {endpoint}"
            );
        }
    }

    #[tokio::test]
    async fn lmstudio_adapter_restores_server_response_order() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = connection.read(&mut buffer).unwrap();
                request.extend_from_slice(&buffer[..read]);
                let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap();
                if request.len() >= header_end + 4 + content_length {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request);
            assert!(request.starts_with("POST /v1/embeddings HTTP/1.1"));
            assert!(request.contains("\"model\":\"fixture\""));
            let body = r#"{"model":"fixture","data":[{"index":1,"embedding":[0.0,1.0]},{"index":0,"embedding":[1.0,0.0]}]}"#;
            write!(
                connection,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            connection.flush().unwrap();
        });
        let embedder = LmStudioEmbedder::with_timeout(
            &format!("http://{address}"),
            "fixture",
            Duration::from_secs(1),
        )
        .unwrap();
        let (batch, normalized_sha256) = embedder
            .conformance_probe(&["first".into(), "second".into()])
            .await
            .unwrap();
        assert_eq!(batch.returned_model, "fixture");
        assert_eq!(batch.vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(normalized_sha256, hex_digest(b"[[1,0],[0,1]]\n"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn lmstudio_adapter_enforces_http_timeout() {
        use std::{net::TcpListener, thread, time::Duration};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let _connection = listener.accept().unwrap();
            thread::sleep(Duration::from_millis(200));
        });
        let embedder = LmStudioEmbedder::with_timeout(
            &format!("http://{address}"),
            "fixture",
            Duration::from_millis(20),
        )
        .unwrap();
        let error = embedder.embed(&["query".into()]).await.unwrap_err();
        assert!(matches!(error, EmbeddingError::Http(source) if source.is_timeout()));
        server.join().unwrap();
    }
}
