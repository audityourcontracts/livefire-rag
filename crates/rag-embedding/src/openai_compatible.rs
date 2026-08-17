use std::{env, fmt, time::Duration};

use reqwest::{
    Client, RequestBuilder,
    header::{AUTHORIZATION, HeaderValue},
};
use serde::{Deserialize, Serialize};

use crate::{
    Embedder, EmbeddingError, IdentifiedEmbedder, IdentifiedEmbeddingBatch, Result, hex_digest,
    normalize_loopback_http_endpoint,
};

/// A generous ceiling for a normal batch of JSON vectors. Callers may lower
/// this when their model dimensions and batch size are known.
pub const DEFAULT_MAX_EMBEDDING_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Prevent a configuration mistake from turning one response into an
/// unbounded allocation.
pub const HARD_MAX_EMBEDDING_RESPONSE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BearerAuthorization {
    #[default]
    None,
    /// Read the token once when the client is constructed. Only the variable
    /// name is retained in this public configuration value.
    FromEnvironment { variable: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiCompatibleOptions {
    pub timeout: Duration,
    pub max_response_bytes: usize,
    pub authorization: BearerAuthorization,
}

impl Default for OpenAiCompatibleOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            max_response_bytes: DEFAULT_MAX_EMBEDDING_RESPONSE_BYTES,
            authorization: BearerAuthorization::None,
        }
    }
}

/// A token value that cannot be serialized and never reveals its contents in
/// debug output. There is deliberately no public constructor from a string.
#[derive(Clone)]
struct BearerSecret {
    header: HeaderValue,
}

impl BearerSecret {
    fn from_environment(variable: &str) -> Result<Self> {
        if !valid_environment_variable_name(variable) {
            return Err(EmbeddingError::Invalid(
                "bearer token environment variable name",
            ));
        }
        let value = env::var(variable)
            .map_err(|_| EmbeddingError::Invalid("bearer token environment variable"))?;
        Self::from_value(&value)
    }

    fn from_value(value: &str) -> Result<Self> {
        if value.is_empty() || value.len() > 8_192 {
            return Err(EmbeddingError::Invalid("bearer token value"));
        }
        let mut header = HeaderValue::from_str(&format!("Bearer {value}"))
            .map_err(|_| EmbeddingError::Invalid("bearer token value"))?;
        header.set_sensitive(true);
        Ok(Self { header })
    }
}

impl fmt::Debug for BearerSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BearerSecret([REDACTED])")
    }
}

fn valid_environment_variable_name(variable: &str) -> bool {
    let mut bytes = variable.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// An identified OpenAI-compatible embeddings client. The only admitted
/// endpoint policy in this slice is a plain-HTTP loopback origin with an
/// explicit port. Remote HTTPS admission can be added later without weakening
/// the safe local default.
pub struct OpenAiCompatibleEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    max_response_bytes: usize,
    authorization: Option<BearerSecret>,
}

impl OpenAiCompatibleEmbedder {
    pub fn loopback(endpoint: &str, model: &str) -> Result<Self> {
        Self::loopback_with_options(endpoint, model, OpenAiCompatibleOptions::default())
    }

    pub fn loopback_with_options(
        endpoint: &str,
        model: &str,
        options: OpenAiCompatibleOptions,
    ) -> Result<Self> {
        if options.timeout.is_zero() {
            return Err(EmbeddingError::Invalid("embedding timeout"));
        }
        if options.max_response_bytes == 0
            || options.max_response_bytes > HARD_MAX_EMBEDDING_RESPONSE_BYTES
        {
            return Err(EmbeddingError::Invalid("embedding response byte limit"));
        }
        let authorization = match &options.authorization {
            BearerAuthorization::None => None,
            BearerAuthorization::FromEnvironment { variable } => {
                Some(BearerSecret::from_environment(variable)?)
            }
        };
        Ok(Self {
            client: Client::builder().timeout(options.timeout).build()?,
            endpoint: normalize_loopback_http_endpoint(endpoint)?,
            model: model.to_owned(),
            max_response_bytes: options.max_response_bytes,
            authorization,
        })
    }

    #[must_use]
    pub fn requested_model(&self) -> &str {
        &self.model
    }

    fn authorize(&self, request: RequestBuilder) -> RequestBuilder {
        match &self.authorization {
            Some(secret) => request.header(AUTHORIZATION, secret.header.clone()),
            None => request,
        }
    }

    async fn bounded_success_body(&self, mut response: reqwest::Response) -> Result<Vec<u8>> {
        response = response.error_for_status()?;
        if response.content_length().is_some_and(|length| {
            length > u64::try_from(self.max_response_bytes).unwrap_or(u64::MAX)
        }) {
            return Err(EmbeddingError::Invalid("embedding response too large"));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            let next_len = bytes
                .len()
                .checked_add(chunk.len())
                .ok_or(EmbeddingError::Invalid("embedding response too large"))?;
            if next_len > self.max_response_bytes {
                return Err(EmbeddingError::Invalid("embedding response too large"));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    async fn request_with_digest(
        &self,
        texts: &[String],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        let request = self
            .client
            .post(format!("{}/v1/embeddings", self.endpoint))
            .json(&EmbeddingRequest {
                model: &self.model,
                input: texts,
            });
        let response = self.authorize(request).send().await?;
        let response_bytes = self.bounded_success_body(response).await?;
        parse_embedding_response(&response_bytes, texts.len())
    }

    async fn request(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        Ok(self.request_with_digest(texts).await?.0)
    }

    /// Return the server-reported model and the digest of parsed `f32` vectors
    /// ordered by request index. This is the same representation stored by the
    /// worker, so equivalent JSON number spellings cannot create two identities.
    pub async fn conformance_probe(
        &self,
        texts: &[String],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        self.request_with_digest(texts).await
    }

    pub(crate) async fn health(&self, path: &str) -> Result<()> {
        if !path.starts_with('/') || path[1..].contains('/') || path.contains(['?', '#']) {
            return Err(EmbeddingError::Invalid("embedding health path"));
        }
        let request = self.client.get(format!("{}{}", self.endpoint, path));
        self.authorize(request).send().await?.error_for_status()?;
        Ok(())
    }
}

impl Embedder for OpenAiCompatibleEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.request(texts).await?.vectors)
    }
}

impl IdentifiedEmbedder for OpenAiCompatibleEmbedder {
    async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        self.request(texts).await
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

fn parse_embedding_response(
    response_bytes: &[u8],
    expected_count: usize,
) -> Result<(IdentifiedEmbeddingBatch, String)> {
    let response: EmbeddingResponse = serde_json::from_slice(response_bytes)?;
    if response.data.len() != expected_count {
        return Err(EmbeddingError::Invalid("response cardinality"));
    }
    let mut ordered = vec![None; expected_count];
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
    let (_, normalized_sha256) = canonical_f32_vectors(&vectors)?;
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

/// Encode conformance vectors using one exact representation shared by the
/// bootstrap result and every later checkpoint probe.
pub fn canonical_f32_vectors(vectors: &[Vec<f32>]) -> Result<(Vec<u8>, String)> {
    if vectors
        .iter()
        .flat_map(|vector| vector.iter())
        .any(|value| !value.is_finite())
    {
        return Err(EmbeddingError::Invalid("non-finite embedding value"));
    }
    let mut bytes = serde_json_canonicalizer::to_vec(&vectors)?;
    bytes.push(b'\n');
    let digest = hex_digest(&bytes);
    Ok((bytes, digest))
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    use super::*;
    use crate::RetryClass;

    fn serve_once(
        status: &'static str,
        body: &'static str,
        delay: Duration,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut buffer = [0_u8; 4096];
            let _ = connection.read(&mut buffer).unwrap();
            if !delay.is_zero() {
                thread::sleep(delay);
            }
            let _ = write!(
                connection,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = connection.flush();
        });
        (format!("http://{address}"), server)
    }

    fn serve_close_delimited_once(body: &'static str) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut buffer = [0_u8; 4096];
            let _ = connection.read(&mut buffer).unwrap();
            let _ = write!(
                connection,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}"
            );
            let _ = connection.flush();
        });
        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn restores_response_index_order_and_preserves_digest() {
        let body = r#"{"model":"fixture","data":[{"index":1,"embedding":[0.0,1.0]},{"index":0,"embedding":[1.0,0.0]}]}"#;
        let (endpoint, server) = serve_once("200 OK", body, Duration::ZERO);
        let client = OpenAiCompatibleEmbedder::loopback(&endpoint, "fixture").unwrap();
        let (batch, digest) = client
            .conformance_probe(&["first".into(), "second".into()])
            .await
            .unwrap();
        assert_eq!(batch.returned_model, "fixture");
        assert_eq!(batch.vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(digest, hex_digest(b"[[1,0],[0,1]]\n"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn conformance_digest_uses_the_parsed_f32_representation() {
        let body = r#"{"model":"fixture","data":[{"index":0,"embedding":[0.100000001,-0.0]}]}"#;
        let (endpoint, server) = serve_once("200 OK", body, Duration::ZERO);
        let client = OpenAiCompatibleEmbedder::loopback(&endpoint, "fixture").unwrap();
        let (batch, digest) = client.conformance_probe(&["input".into()]).await.unwrap();
        assert_eq!(batch.vectors, vec![vec![0.1, -0.0]]);
        let (expected, _) = canonical_f32_vectors(&batch.vectors).unwrap();
        assert_eq!(digest, hex_digest(&expected));
        assert_ne!(digest, hex_digest(b"[[0.100000001,0]]\n"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn temporary_http_statuses_are_retryable() {
        for status in [
            "408 Request Timeout",
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
        ] {
            let (endpoint, server) = serve_once(status, "{}", Duration::ZERO);
            let client = OpenAiCompatibleEmbedder::loopback(&endpoint, "fixture").unwrap();
            let error = client.embed(&["query".into()]).await.unwrap_err();
            assert_eq!(error.retry_class(), RetryClass::Temporary, "{status}");
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn transport_timeout_is_retryable() {
        let body = r#"{"model":"fixture","data":[]}"#;
        let (endpoint, server) = serve_once("200 OK", body, Duration::from_millis(100));
        let client = OpenAiCompatibleEmbedder::loopback_with_options(
            &endpoint,
            "fixture",
            OpenAiCompatibleOptions {
                timeout: Duration::from_millis(10),
                ..OpenAiCompatibleOptions::default()
            },
        )
        .unwrap();
        let error = client.embed(&["query".into()]).await.unwrap_err();
        assert_eq!(error.retry_class(), RetryClass::Temporary);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn rejects_response_larger_than_configured_limit() {
        let body = r#"{"model":"fixture","data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
        let (endpoint, server) = serve_close_delimited_once(body);
        let client = OpenAiCompatibleEmbedder::loopback_with_options(
            &endpoint,
            "fixture",
            OpenAiCompatibleOptions {
                max_response_bytes: body.len() - 1,
                ..OpenAiCompatibleOptions::default()
            },
        )
        .unwrap();
        let error = client.embed(&["query".into()]).await.unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Invalid("embedding response too large")
        ));
        server.join().unwrap();
    }

    #[test]
    fn authorization_header_is_sensitive_and_uses_the_secret() {
        let client = OpenAiCompatibleEmbedder {
            client: Client::new(),
            endpoint: "http://127.0.0.1:1234".into(),
            model: "fixture".into(),
            max_response_bytes: DEFAULT_MAX_EMBEDDING_RESPONSE_BYTES,
            authorization: Some(BearerSecret::from_value("test-secret").unwrap()),
        };
        let request = client
            .authorize(client.client.get("http://127.0.0.1:1234/health"))
            .build()
            .unwrap();
        let authorization = request.headers().get(AUTHORIZATION).unwrap();
        assert_eq!(authorization.to_str().unwrap(), "Bearer test-secret");
        assert!(authorization.is_sensitive());
    }

    #[test]
    fn bearer_secret_debug_output_is_redacted() {
        let secret = BearerSecret::from_value("do-not-print-me").unwrap();
        let debug = format!("{secret:?}");
        assert!(!debug.contains("do-not-print-me"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn authorization_configuration_contains_only_an_environment_name() {
        let options = OpenAiCompatibleOptions {
            authorization: BearerAuthorization::FromEnvironment {
                variable: "EMBEDDING_API_TOKEN".into(),
            },
            ..OpenAiCompatibleOptions::default()
        };
        let debug = format!("{options:?}");
        assert!(debug.contains("EMBEDDING_API_TOKEN"));
        assert!(!debug.contains("Bearer "));
    }
}
