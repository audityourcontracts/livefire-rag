use crate::{
    BearerAuthorization, Embedder, EmbeddingError, EmbeddingProfile, IdentifiedEmbedder,
    IdentifiedEmbeddingBatch, OpenAiCompatibleEmbedder, OpenAiCompatibleOptions, Result,
    adapt_model_vector, hex_digest, parse_tei_checkpoint_profile_v3, validate_embedding_profile,
};
use serde::{Deserialize, Serialize};

pub const TEI_CONFORMANCE_FIXTURE_SCHEMA_V1: &str = "livefire.rag.tei-conformance-fixture/1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiConformanceFixtureV1 {
    pub schema_version: String,
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundTeiConformance {
    fixture_sha256: String,
    input_count: usize,
    normalized_output_sha256: String,
}

/// The model and vector contract that a TEI worker must satisfy. This value is
/// derived from an embedding profile rather than accepted as a second source
/// of configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeiIdentityPolicy {
    expected_model: String,
    output_dimensions: u32,
    normalization: String,
}

impl TeiIdentityPolicy {
    fn from_profile(profile: &EmbeddingProfile) -> Result<Self> {
        validate_embedding_profile(profile)?;
        Ok(Self {
            expected_model: profile.model.clone(),
            output_dimensions: profile.dimensions,
            normalization: profile.normalization.clone(),
        })
    }

    #[must_use]
    pub fn expected_model(&self) -> &str {
        &self.expected_model
    }

    #[must_use]
    pub fn output_dimensions(&self) -> u32 {
        self.output_dimensions
    }

    #[must_use]
    pub fn normalization(&self) -> &str {
        &self.normalization
    }
}

/// A Text Embeddings Inference (TEI) backend using its OpenAI-compatible
/// embeddings endpoint. Construction currently permits loopback HTTP only.
pub struct TeiEmbedder {
    client: OpenAiCompatibleEmbedder,
    profile: EmbeddingProfile,
    identity: TeiIdentityPolicy,
    checkpoint_conformance: Option<BoundTeiConformance>,
}

impl TeiEmbedder {
    pub fn loopback(endpoint: &str, profile: EmbeddingProfile) -> Result<Self> {
        Self::loopback_with_options(endpoint, profile, OpenAiCompatibleOptions::default())
    }

    pub fn loopback_with_options(
        endpoint: &str,
        profile: EmbeddingProfile,
        options: OpenAiCompatibleOptions,
    ) -> Result<Self> {
        let identity = TeiIdentityPolicy::from_profile(&profile)?;
        let client = OpenAiCompatibleEmbedder::loopback_with_options(
            endpoint,
            identity.expected_model(),
            options,
        )?;
        Ok(Self {
            client,
            profile,
            identity,
            checkpoint_conformance: None,
        })
    }

    /// Construct the executable direct-checkpoint worker path. Unlike the
    /// lower-level loopback constructor, this refuses a policy whose complete
    /// checkpoint, tokenizer, runtime, limits, or measured conformance binding
    /// is absent or inconsistent.
    pub fn checkpoint_profile_loopback(
        endpoint: &str,
        policy_bytes: &[u8],
        authorization: BearerAuthorization,
    ) -> Result<Self> {
        let policy = parse_tei_checkpoint_profile_v3(policy_bytes)?;
        let profile = policy.embedding_profile(policy_bytes)?;
        let options = policy.client_options(authorization);
        let input_count = usize::try_from(policy.conformance.input_count)
            .map_err(|_| EmbeddingError::Invalid("TEI conformance input count"))?;
        let checkpoint_conformance = BoundTeiConformance {
            fixture_sha256: policy.conformance.fixture.sha256.clone(),
            input_count,
            normalized_output_sha256: policy.conformance.normalized_output_sha256.clone(),
        };
        let mut embedder = Self::loopback_with_options(endpoint, profile, options)?;
        embedder.checkpoint_conformance = Some(checkpoint_conformance);
        Ok(embedder)
    }

    #[must_use]
    pub fn identity_policy(&self) -> &TeiIdentityPolicy {
        &self.identity
    }

    /// Ask TEI's lightweight `/health` endpoint whether the worker can accept
    /// requests. HTTP 408, 429, and server errors retain the shared temporary
    /// retry classification.
    pub async fn health(&self) -> Result<()> {
        self.client.health("/health").await
    }

    fn enforce_identity(
        &self,
        batch: IdentifiedEmbeddingBatch,
    ) -> Result<IdentifiedEmbeddingBatch> {
        if batch.returned_model != self.identity.expected_model {
            return Err(EmbeddingError::Invalid("response model identity mismatch"));
        }
        let vectors = batch
            .vectors
            .into_iter()
            .map(|vector| adapt_model_vector(&self.profile, vector))
            .collect::<Result<Vec<_>>>()?;
        Ok(IdentifiedEmbeddingBatch {
            vectors,
            returned_model: batch.returned_model,
        })
    }

    async fn request(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        let batch = self.client.embed_identified(texts).await?;
        self.enforce_identity(batch)
    }

    /// Exercise the full worker contract and return the canonical digest of
    /// the raw model response. Returned vectors have already been converted to
    /// the profile representation when that profile declares a derivation.
    pub async fn conformance_probe(
        &self,
        texts: &[String],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        let (batch, normalized_sha256) = self.client.conformance_probe(texts).await?;
        Ok((self.enforce_identity(batch)?, normalized_sha256))
    }

    /// Run the exact fixture bound by a v3 checkpoint policy. The fixture is
    /// parsed here so callers cannot hash one file while submitting different
    /// text. Success means both the fixture digest and normalized model-output
    /// digest match the measured policy.
    pub async fn checkpoint_conformance_probe(
        &self,
        fixture_bytes: &[u8],
    ) -> Result<(IdentifiedEmbeddingBatch, String)> {
        let expected = self
            .checkpoint_conformance
            .as_ref()
            .ok_or(EmbeddingError::Invalid(
                "TEI checkpoint conformance binding",
            ))?;
        if hex_digest(fixture_bytes) != expected.fixture_sha256 {
            return Err(EmbeddingError::Invalid("TEI conformance fixture digest"));
        }
        let fixture: TeiConformanceFixtureV1 = serde_json::from_slice(fixture_bytes)
            .map_err(|_| EmbeddingError::Invalid("TEI conformance fixture JSON"))?;
        if fixture.schema_version != TEI_CONFORMANCE_FIXTURE_SCHEMA_V1
            || fixture.inputs.len() != expected.input_count
            || fixture
                .inputs
                .iter()
                .any(|input| input.is_empty() || input.len() > crate::MAX_FORMATTED_DOCUMENT_BYTES)
        {
            return Err(EmbeddingError::Invalid("TEI conformance fixture"));
        }
        let (batch, normalized_sha256) = self.conformance_probe(&fixture.inputs).await?;
        if normalized_sha256 != expected.normalized_output_sha256 {
            return Err(EmbeddingError::Invalid("TEI conformance output digest"));
        }
        Ok((batch, normalized_sha256))
    }
}

impl Embedder for TeiEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.request(texts).await?.vectors)
    }
}

impl IdentifiedEmbedder for TeiEmbedder {
    async fn embed_identified(&self, texts: &[String]) -> Result<IdentifiedEmbeddingBatch> {
        self.request(texts).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::*;

    #[test]
    fn tracked_qwen_conformance_fixture_has_the_executable_shape() {
        let fixture: TeiConformanceFixtureV1 = serde_json::from_slice(include_bytes!(
            "../../../fixtures/qwen3-embedding-8b-tei-conformance.v1.json"
        ))
        .unwrap();
        assert_eq!(fixture.schema_version, TEI_CONFORMANCE_FIXTURE_SCHEMA_V1);
        assert_eq!(fixture.inputs.len(), 1);
        assert!(!fixture.inputs[0].is_empty());
    }

    fn profile() -> EmbeddingProfile {
        EmbeddingProfile {
            id: "test".into(),
            version: "1".into(),
            sha256: "a".repeat(64),
            model: "expected-model".into(),
            dimensions: 2,
            normalization: "l2".into(),
            vector_derivation: None,
            query_instruction: None,
            query_composition: None,
        }
    }

    fn serve_once(body: &'static str) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = connection.read(&mut buffer).unwrap();
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|part| part == b"\r\n\r\n") {
                    break;
                }
            }
            write!(
                connection,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            connection.flush().unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });
        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn rejects_a_server_reported_model_that_differs_from_the_profile() {
        let body = r#"{"model":"other-model","data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
        let (endpoint, server) = serve_once(body);
        let embedder = TeiEmbedder::loopback(&endpoint, profile()).unwrap();
        let error = embedder.embed(&["query".into()]).await.unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Invalid("response model identity mismatch")
        ));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn profile_validation_rejects_wrong_vector_dimensions() {
        let body = r#"{"model":"expected-model","data":[{"index":0,"embedding":[1.0]}]}"#;
        let (endpoint, server) = serve_once(body);
        let embedder = TeiEmbedder::loopback(&endpoint, profile()).unwrap();
        let error = embedder.embed(&["query".into()]).await.unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Invalid("vector shape or finiteness")
        ));
        server.join().unwrap();
    }

    #[test]
    fn profile_validation_rejects_non_finite_vector_values() {
        let embedder = TeiEmbedder {
            client: OpenAiCompatibleEmbedder::loopback("http://127.0.0.1:1234", "expected-model")
                .unwrap(),
            identity: TeiIdentityPolicy::from_profile(&profile()).unwrap(),
            profile: profile(),
            checkpoint_conformance: None,
        };
        let error = embedder
            .enforce_identity(IdentifiedEmbeddingBatch {
                returned_model: "expected-model".into(),
                vectors: vec![vec![f32::NAN, 0.0]],
            })
            .unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Invalid("vector shape or finiteness")
        ));
    }

    #[tokio::test]
    async fn health_uses_the_tei_health_endpoint() {
        let (endpoint, server) = serve_once("healthy");
        let embedder = TeiEmbedder::loopback(&endpoint, profile()).unwrap();
        embedder.health().await.unwrap();
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /health HTTP/1.1"));
    }

    #[tokio::test]
    async fn checkpoint_probe_binds_fixture_text_and_exact_output_digest() {
        let fixture = serde_json::to_vec(&serde_json::json!({
            "schema_version": TEI_CONFORMANCE_FIXTURE_SCHEMA_V1,
            "inputs": ["fixed conformance input"]
        }))
        .unwrap();
        let body = r#"{"model":"expected-model","data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
        let (endpoint, server) = serve_once(body);
        let mut embedder = TeiEmbedder::loopback(&endpoint, profile()).unwrap();
        embedder.checkpoint_conformance = Some(BoundTeiConformance {
            fixture_sha256: hex_digest(&fixture),
            input_count: 1,
            normalized_output_sha256: hex_digest(b"[[1,0]]\n"),
        });
        let (batch, digest) = embedder
            .checkpoint_conformance_probe(&fixture)
            .await
            .unwrap();
        assert_eq!(batch.vectors, vec![vec![1.0, 0.0]]);
        assert_eq!(digest, hex_digest(b"[[1,0]]\n"));
        server.join().unwrap();

        let mut changed_fixture = fixture;
        changed_fixture.push(b' ');
        let error = embedder
            .checkpoint_conformance_probe(&changed_fixture)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            EmbeddingError::Invalid("TEI conformance fixture digest")
        ));
    }
}
