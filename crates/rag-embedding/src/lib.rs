//! Resumable, content-bound embedding batches for the experimental indexer.

use std::{future::Future, path::Path, time::Duration};

use reqwest::Client;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding cache failed: {0}")]
    Cache(#[from] rusqlite::Error),
    #[error("embedding request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("embedding response violated the profile: {0}")]
    Invalid(&'static str),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

pub const MAX_QUERY_BYTES: usize = 8_192;
pub const MAX_QUERY_INSTRUCTION_BYTES: usize = 8_192;
pub const MAX_QUERY_COMPOSITION_BYTES: usize = 1_024;
pub const MAX_COMPOSED_QUERY_BYTES: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProfile {
    pub id: String,
    pub version: String,
    pub sha256: String,
    pub model: String,
    pub dimensions: u32,
    pub normalization: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_instruction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_composition: Option<String>,
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

pub trait Embedder: Sync {
    fn embed(&self, texts: &[String]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;
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
            endpoint: endpoint.trim_end_matches('/').to_owned(),
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
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

impl Embedder for LmStudioEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.endpoint))
            .json(&EmbeddingRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await?
            .error_for_status()?
            .json::<EmbeddingResponse>()
            .await?;
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
        ordered
            .into_iter()
            .map(|value| value.ok_or(EmbeddingError::Invalid("missing response index")))
            .collect()
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

    fn put(&self, key: &str, vector: &[f32]) -> Result<()> {
        let bytes = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        self.connection.execute(
            "INSERT OR REPLACE INTO vectors(cache_key, dimensions, vector) VALUES (?1, ?2, ?3)",
            params![key, vector.len() as i64, bytes],
        )?;
        Ok(())
    }
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
    cache: &EmbeddingCache,
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
        for ((ordinal, key), vector) in batch.iter().zip(vectors) {
            validate_vector(&vector, dimensions, &profile.normalization)?;
            cache.put(key, &vector)?;
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
            query_instruction: None,
            query_composition: None,
        }
    }

    #[tokio::test]
    async fn cache_prevents_reembedding_and_preserves_order() {
        let root = tempdir().unwrap();
        let cache = EmbeddingCache::open(&root.path().join("cache.sqlite3")).unwrap();
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
        let first = embed_resumable(&embedder, &cache, &profile(), &inputs, 1)
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert!(first.iter().all(|item| !item.from_cache));
        let second = embed_resumable(&embedder, &cache, &profile(), &inputs, 16)
            .await
            .unwrap();
        assert_eq!(embedder.calls.load(Ordering::Relaxed), 2);
        assert!(second.iter().all(|item| item.from_cache));
        assert_eq!(second[0].document_id, "a");
        assert_eq!(second[1].vector, vec![0.0, 1.0]);
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
