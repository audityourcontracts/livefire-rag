use std::{collections::BTreeSet, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    BearerAuthorization, EmbeddingError, EmbeddingProfile, HARD_MAX_EMBEDDING_RESPONSE_BYTES,
    OpenAiCompatibleOptions, Result, format_document_input, hex_digest, validate_embedding_profile,
};

pub const TEI_CHECKPOINT_PROFILE_SCHEMA_V3: &str = "livefire.rag.embedding-policy/3";
pub const QWEN3_EMBEDDING_8B_REPOSITORY: &str = "Qwen/Qwen3-Embedding-8B";
pub const QWEN3_EMBEDDING_8B_REVISION: &str = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af";
pub const QWEN3_EMBEDDING_8B_ARTIFACT_SET_SHA256: &str =
    "99beb578f3ca8c20eb204484178bf08fea6f0d7f016ab49ca33a8590e1af2dcb";
pub const QWEN3_EMBEDDING_8B_MAX_POSITION_EMBEDDINGS: u64 = 40_960;
pub const TEI_MODEL_ARTIFACT_SET_SCHEMA_V1: &str = "livefire.rag.tei-model-artifact-set/1";
pub const MAX_TEI_CHECKPOINT_PROFILE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TEI_MODEL_OBJECTS: usize = 17;
pub const QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS: [&str; 17] = [
    ".gitattributes",
    "1_Pooling/config.json",
    "LICENSE",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiComponentIdentity {
    pub id: String,
    pub version: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiArtifactObject {
    pub path: String,
    pub media_type: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiModelArtifactSetV1 {
    pub schema_version: String,
    pub repository: String,
    pub revision: String,
    pub objects: Vec<TeiArtifactObject>,
}

impl TeiModelArtifactSetV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TEI_MODEL_ARTIFACT_SET_SCHEMA_V1
            || self.repository != QWEN3_EMBEDDING_8B_REPOSITORY
            || self.revision != QWEN3_EMBEDDING_8B_REVISION
            || self.objects.len() != QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS.len()
        {
            return Err(EmbeddingError::Invalid("TEI model artifact set"));
        }
        for (object, expected_path) in self.objects.iter().zip(QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS) {
            validate_artifact(object)?;
            if object.path != expected_path {
                return Err(EmbeddingError::Invalid("TEI model artifact set path"));
            }
        }
        if self.sha256()? != QWEN3_EMBEDDING_8B_ARTIFACT_SET_SHA256 {
            return Err(EmbeddingError::Invalid("TEI model artifact set digest"));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String> {
        Ok(hex_digest(&serde_json_canonicalizer::to_vec(self)?))
    }
}

pub fn parse_tei_model_artifact_set_v1(bytes: &[u8]) -> Result<TeiModelArtifactSetV1> {
    let manifest: TeiModelArtifactSetV1 = serde_json::from_slice(bytes)
        .map_err(|_| EmbeddingError::Invalid("TEI model artifact set JSON"))?;
    manifest.validate()?;
    Ok(manifest)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableTokenizerV3 {
    pub repository: String,
    pub revision: String,
    pub format: String,
    pub object: TeiArtifactObject,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiImageIdentityV3 {
    pub component: TeiComponentIdentity,
    pub repository: String,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiLoadPolicyV3 {
    pub component: TeiComponentIdentity,
    pub model_source: String,
    pub revision_policy: String,
    pub local_files_only: bool,
    pub trust_remote_code: bool,
    pub safetensors_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiBatchingV3 {
    pub maximum_batch_items: u32,
    pub maximum_batch_tokens: u64,
    pub maximum_concurrent_requests: u32,
    pub order: String,
    pub overlength: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiResponseLimitsV3 {
    pub request_timeout_ms: u64,
    pub maximum_response_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiOutputProcessingV3 {
    pub client_normalization: String,
    pub required_l2_norm_tolerance_millionths: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiAcceleratorPolicyV3 {
    pub provider: String,
    pub gpu_model_id: String,
    pub compute_capability: String,
    pub architecture_image_class: String,
    pub gpu_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiConformanceV3 {
    pub mode: String,
    pub measured: bool,
    pub fixture: TeiArtifactObject,
    pub input_count: u32,
    pub returned_model: String,
    pub normalized_output_sha256: String,
    pub accelerator: TeiAcceleratorPolicyV3,
    pub candidate_sha256: String,
    pub initial_result_sha256: String,
    pub fresh_pod_replay_result_sha256: String,
}

/// Complete identity and execution policy for serving the upstream Qwen
/// checkpoint through a local TEI worker. No field in this type represents an
/// unmeasured or pending conformance state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeiCheckpointProfileV3 {
    pub schema_version: String,
    pub admission_status: String,
    pub purpose: String,
    pub model_repository: String,
    pub model_revision: String,
    pub model_snapshot_completeness: String,
    pub model_artifact_set: TeiComponentIdentity,
    pub model_objects: Vec<TeiArtifactObject>,
    pub tokenizer: TeiComponentIdentity,
    pub executable_tokenizer: ExecutableTokenizerV3,
    pub tei_image: TeiImageIdentityV3,
    /// Custom derivative image containing the fixed Rust worker executable.
    pub executor_image: TeiImageIdentityV3,
    /// Sealed receipt binding the custom image to its base, Dockerfile, and worker binary.
    pub executor_image_build: TeiComponentIdentity,
    pub runtime: TeiComponentIdentity,
    pub inference_engine: TeiComponentIdentity,
    pub load_policy: TeiLoadPolicyV3,
    pub runtime_mode: String,
    pub api_contract: String,
    pub api_model_key: String,
    pub dimensions: u32,
    pub checkpoint_compute_dtype: String,
    pub api_vector_dtype: String,
    pub stored_vector_dtype: String,
    pub pooling: String,
    pub normalization: String,
    pub maximum_tokens: u32,
    pub document_format: String,
    pub query_instruction: String,
    pub query_composition: String,
    pub batching: TeiBatchingV3,
    pub response_limits: TeiResponseLimitsV3,
    pub output_processing: TeiOutputProcessingV3,
    pub accelerator: TeiAcceleratorPolicyV3,
    pub conformance: TeiConformanceV3,
}

impl TeiCheckpointProfileV3 {
    /// Produce the compact profile carried by plans and indexes. The digest is
    /// over the exact policy bytes, so the compact value remains bound to all
    /// runtime, checkpoint, tokenizer, and conformance fields.
    pub fn embedding_profile(&self, policy_bytes: &[u8]) -> Result<EmbeddingProfile> {
        let reparsed = parse_tei_checkpoint_profile_v3(policy_bytes)?;
        if reparsed != *self {
            return Err(EmbeddingError::Invalid("TEI profile byte binding"));
        }
        let profile = EmbeddingProfile {
            id: self.model_artifact_set.id.clone(),
            version: self.model_artifact_set.version.clone(),
            sha256: hex_digest(policy_bytes),
            model: self.api_model_key.clone(),
            dimensions: self.dimensions,
            normalization: self.normalization.clone(),
            vector_derivation: None,
            query_instruction: Some(self.query_instruction.clone()),
            query_composition: Some(self.query_composition.clone()),
        };
        validate_embedding_profile(&profile)?;
        Ok(profile)
    }

    #[must_use]
    pub fn client_options(&self, authorization: BearerAuthorization) -> OpenAiCompatibleOptions {
        OpenAiCompatibleOptions {
            timeout: Duration::from_millis(self.response_limits.request_timeout_ms),
            max_response_bytes: usize::try_from(self.response_limits.maximum_response_bytes)
                .expect("validated TEI response byte limit fits usize"),
            authorization,
        }
    }
}

pub fn parse_tei_checkpoint_profile_v3(bytes: &[u8]) -> Result<TeiCheckpointProfileV3> {
    if bytes.is_empty() || bytes.len() > MAX_TEI_CHECKPOINT_PROFILE_BYTES {
        return Err(EmbeddingError::Invalid("TEI checkpoint profile size"));
    }
    let profile: TeiCheckpointProfileV3 = serde_json::from_slice(bytes)
        .map_err(|_| EmbeddingError::Invalid("TEI checkpoint profile JSON"))?;
    validate_tei_checkpoint_profile_v3(&profile)?;
    Ok(profile)
}

pub fn validate_tei_checkpoint_profile_v3(profile: &TeiCheckpointProfileV3) -> Result<()> {
    if profile.schema_version != TEI_CHECKPOINT_PROFILE_SCHEMA_V3
        || !matches!(
            profile.admission_status.as_str(),
            "development_only" | "production_candidate"
        )
        || !matches!(
            profile.purpose.as_str(),
            "semantic_search" | "action_novelty" | "target_novelty"
        )
    {
        return Err(EmbeddingError::Invalid("TEI profile identity"));
    }
    if profile.model_repository != QWEN3_EMBEDDING_8B_REPOSITORY
        || profile.model_revision != QWEN3_EMBEDDING_8B_REVISION
        || profile.model_snapshot_completeness != "complete_hugging_face_snapshot"
    {
        return Err(EmbeddingError::Invalid("TEI checkpoint identity"));
    }
    validate_component(&profile.model_artifact_set)?;
    validate_component(&profile.tokenizer)?;
    validate_component(&profile.tei_image.component)?;
    validate_component(&profile.executor_image.component)?;
    validate_component(&profile.executor_image_build)?;
    validate_component(&profile.runtime)?;
    validate_component(&profile.inference_engine)?;
    validate_component(&profile.load_policy.component)?;
    if profile.model_artifact_set.version != profile.model_revision
        || profile.model_artifact_set.sha256 != QWEN3_EMBEDDING_8B_ARTIFACT_SET_SHA256
        || profile.tokenizer.version != profile.model_revision
        || profile.executable_tokenizer.repository != profile.model_repository
        || profile.executable_tokenizer.revision != profile.model_revision
        || profile.executable_tokenizer.format != "hugging_face_tokenizer_json"
        || profile.executable_tokenizer.object.path != "tokenizer.json"
        || profile.executable_tokenizer.object.sha256 != profile.tokenizer.sha256
    {
        return Err(EmbeddingError::Invalid("TEI tokenizer revision binding"));
    }
    validate_artifact(&profile.executable_tokenizer.object)?;
    validate_model_objects(profile)?;
    validate_runtime(profile)?;
    validate_vector_contract(profile)?;
    validate_execution_limits(profile)?;
    validate_conformance(profile)?;
    Ok(())
}

fn validate_model_objects(profile: &TeiCheckpointProfileV3) -> Result<()> {
    if profile.model_objects.len() != QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS.len() {
        return Err(EmbeddingError::Invalid("TEI model object set"));
    }
    let mut paths = BTreeSet::new();
    let mut previous = None;
    for (object, expected_path) in profile
        .model_objects
        .iter()
        .zip(QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS)
    {
        validate_artifact(object)?;
        if object.path != expected_path
            || previous.is_some_and(|path: &str| path >= object.path.as_str())
            || !paths.insert(object.path.as_str())
        {
            return Err(EmbeddingError::Invalid("TEI model object order"));
        }
        previous = Some(object.path.as_str());
    }
    let tokenizer = profile
        .model_objects
        .iter()
        .find(|object| object.path == "tokenizer.json")
        .ok_or(EmbeddingError::Invalid("TEI model snapshot completeness"))?;
    if tokenizer.sha256 != profile.executable_tokenizer.object.sha256 {
        return Err(EmbeddingError::Invalid("TEI model snapshot completeness"));
    }
    if profile.model_artifact_set.sha256
        != tei_model_artifact_set_sha256_v3(
            &profile.model_repository,
            &profile.model_revision,
            &profile.model_objects,
        )?
    {
        return Err(EmbeddingError::Invalid("TEI model artifact set binding"));
    }
    Ok(())
}

/// Digest the complete, ordered object list represented by the model artifact
/// component. Plans can carry this digest without copying every checkpoint
/// object.
pub fn tei_model_artifact_set_sha256_v3(
    repository: &str,
    revision: &str,
    objects: &[TeiArtifactObject],
) -> Result<String> {
    let value = serde_json::json!({
        "schema_version": TEI_MODEL_ARTIFACT_SET_SCHEMA_V1,
        "repository": repository,
        "revision": revision,
        "objects": objects,
    });
    Ok(hex_digest(&serde_json_canonicalizer::to_vec(&value)?))
}

fn validate_runtime(profile: &TeiCheckpointProfileV3) -> Result<()> {
    let image_digest = profile
        .tei_image
        .digest
        .strip_prefix("sha256:")
        .ok_or(EmbeddingError::Invalid("TEI image digest"))?;
    let executor_digest = profile
        .executor_image
        .digest
        .strip_prefix("sha256:")
        .ok_or(EmbeddingError::Invalid("TEI executor image digest"))?;
    if profile.tei_image.repository != "ghcr.io/huggingface/text-embeddings-inference"
        || image_digest != profile.tei_image.component.sha256
        || profile.executor_image.repository.is_empty()
        || profile.executor_image.repository.len() > 512
        || profile.executor_image.repository.contains("://")
        || executor_digest != profile.executor_image.component.sha256
        || profile.runtime_mode != "tei_loopback_worker"
        || profile.api_contract != "openai_compatible_v1_embeddings"
        || profile.api_model_key.is_empty()
        || profile.api_model_key.len() > 1_024
        || profile.load_policy.model_source != "mounted_complete_snapshot"
        || profile.load_policy.revision_policy != "exact"
        || !profile.load_policy.local_files_only
        || profile.load_policy.trust_remote_code
        || !profile.load_policy.safetensors_only
    {
        return Err(EmbeddingError::Invalid("TEI runtime contract"));
    }
    let ids = [
        profile.model_artifact_set.id.as_str(),
        profile.tokenizer.id.as_str(),
        profile.tei_image.component.id.as_str(),
        profile.executor_image.component.id.as_str(),
        profile.executor_image_build.id.as_str(),
        profile.runtime.id.as_str(),
        profile.inference_engine.id.as_str(),
        profile.load_policy.component.id.as_str(),
    ];
    if ids.into_iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(EmbeddingError::Invalid("TEI component identity confusion"));
    }
    Ok(())
}

fn validate_vector_contract(profile: &TeiCheckpointProfileV3) -> Result<()> {
    if profile.dimensions != 4_096
        || !matches!(
            profile.checkpoint_compute_dtype.as_str(),
            "float32" | "float16"
        )
        || profile.api_vector_dtype != "float32"
        || profile.stored_vector_dtype != "f32le"
        || profile.pooling != "last_token"
        || profile.normalization != "l2"
        || profile.maximum_tokens != 8_192
        || profile.query_composition != "Instruct: {query_instruction}\nQuery: {query}"
        || profile.output_processing.client_normalization != "none"
        || profile
            .output_processing
            .required_l2_norm_tolerance_millionths
            != 100
    {
        return Err(EmbeddingError::Invalid("TEI vector contract"));
    }
    format_document_input(&profile.document_format, "probe")?;
    let compact = EmbeddingProfile {
        id: profile.model_artifact_set.id.clone(),
        version: profile.model_artifact_set.version.clone(),
        sha256: "a".repeat(64),
        model: profile.api_model_key.clone(),
        dimensions: profile.dimensions,
        normalization: profile.normalization.clone(),
        vector_derivation: None,
        query_instruction: Some(profile.query_instruction.clone()),
        query_composition: Some(profile.query_composition.clone()),
    };
    validate_embedding_profile(&compact)
}

fn validate_execution_limits(profile: &TeiCheckpointProfileV3) -> Result<()> {
    let maximum_response_bytes = usize::try_from(profile.response_limits.maximum_response_bytes)
        .map_err(|_| EmbeddingError::Invalid("TEI response limits"))?;
    if !(1..=32).contains(&profile.batching.maximum_batch_items)
        || profile.batching.maximum_batch_tokens < QWEN3_EMBEDDING_8B_MAX_POSITION_EMBEDDINGS
        || profile.batching.maximum_batch_tokens
            > u64::from(profile.maximum_tokens) * u64::from(profile.batching.maximum_batch_items)
        || !(1..=256).contains(&profile.batching.maximum_concurrent_requests)
        || profile.batching.order != "preserve_input_order"
        || profile.batching.overlength != "reject"
        || !(1..=3_600_000).contains(&profile.response_limits.request_timeout_ms)
        || !(1_024..=HARD_MAX_EMBEDDING_RESPONSE_BYTES).contains(&maximum_response_bytes)
    {
        return Err(EmbeddingError::Invalid("TEI execution limits"));
    }
    Ok(())
}

fn validate_conformance(profile: &TeiCheckpointProfileV3) -> Result<()> {
    validate_artifact(&profile.conformance.fixture)?;
    if profile.conformance.mode != "exact_digest"
        || !profile.conformance.measured
        || profile.conformance.fixture.media_type != "application/json"
        || profile.conformance.input_count == 0
        || profile.conformance.input_count > profile.batching.maximum_batch_items
        || profile.conformance.returned_model != profile.api_model_key
        || !valid_sha256(&profile.conformance.normalized_output_sha256)
        || profile.conformance.accelerator != profile.accelerator
        || !valid_sha256(&profile.conformance.candidate_sha256)
        || !valid_sha256(&profile.conformance.initial_result_sha256)
        || !valid_sha256(&profile.conformance.fresh_pod_replay_result_sha256)
        || profile.conformance.initial_result_sha256
            == profile.conformance.fresh_pod_replay_result_sha256
    {
        return Err(EmbeddingError::Invalid("TEI conformance contract"));
    }
    validate_accelerator(&profile.accelerator)?;
    Ok(())
}

fn validate_accelerator(accelerator: &TeiAcceleratorPolicyV3) -> Result<()> {
    if accelerator.provider != "runpod"
        || accelerator.gpu_model_id.is_empty()
        || accelerator.gpu_model_id.len() > 256
        || accelerator.architecture_image_class.is_empty()
        || accelerator.architecture_image_class.len() > 128
        || accelerator.gpu_count != 1
        || accelerator.compute_capability.len() < 3
        || accelerator.compute_capability.len() > 8
        || accelerator.compute_capability.matches('.').count() != 1
        || !accelerator
            .compute_capability
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.')
    {
        return Err(EmbeddingError::Invalid("TEI accelerator policy"));
    }
    Ok(())
}

fn validate_component(component: &TeiComponentIdentity) -> Result<()> {
    if component.id.is_empty() || component.version.is_empty() || !valid_sha256(&component.sha256) {
        return Err(EmbeddingError::Invalid("TEI component identity"));
    }
    Ok(())
}

fn validate_artifact(artifact: &TeiArtifactObject) -> Result<()> {
    if !valid_relative_path(&artifact.path)
        || artifact.media_type.is_empty()
        || artifact.bytes == 0
        || artifact.bytes > 9_007_199_254_740_991
        || !valid_sha256(&artifact.sha256)
    {
        return Err(EmbeddingError::Invalid("TEI artifact object"));
    }
    Ok(())
}

fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains(['\\', ':', '\0'])
        && path
            .split('/')
            .all(|component| !component.is_empty() && !matches!(component, "." | ".."))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn digest(label: &str) -> String {
        hex_digest(label.as_bytes())
    }

    fn component(id: &str, version: &str) -> serde_json::Value {
        json!({"id": id, "version": version, "sha256": digest(id)})
    }

    fn artifact(path: &str, media_type: &str) -> serde_json::Value {
        json!({"path": path, "media_type": media_type, "bytes": 10, "sha256": digest(path)})
    }

    fn valid_value() -> serde_json::Value {
        let manifest = parse_tei_model_artifact_set_v1(include_bytes!(
            "../../../profiles/qwen3-embedding-8b-upstream-model-artifacts.v1.json"
        ))
        .unwrap();
        let revision = manifest.revision.clone();
        let image_sha = digest("tei-image");
        let executor_image_sha = digest("executor-image");
        let model_objects = serde_json::to_value(&manifest.objects).unwrap();
        let tokenizer = model_objects
            .as_array()
            .unwrap()
            .iter()
            .find(|object| object["path"] == "tokenizer.json")
            .unwrap()
            .clone();
        let model_artifact_set_sha = manifest.sha256().unwrap();
        let accelerator = json!({
            "provider": "runpod",
            "gpu_model_id": "NVIDIA H100 PCIe",
            "compute_capability": "9.0",
            "architecture_image_class": "sm90-cuda12",
            "gpu_count": 1
        });
        let mut value = json!({
            "schema_version": TEI_CHECKPOINT_PROFILE_SCHEMA_V3,
            "admission_status": "development_only",
            "purpose": "semantic_search",
            "model_repository": QWEN3_EMBEDDING_8B_REPOSITORY,
            "model_revision": revision,
            "model_snapshot_completeness": "complete_hugging_face_snapshot",
            "model_artifact_set": {
                "id": "qwen-checkpoint",
                "version": revision,
                "sha256": model_artifact_set_sha
            },
            "model_objects": model_objects,
            "tokenizer": {"id": "qwen-tokenizer", "version": revision, "sha256": tokenizer["sha256"]},
            "executable_tokenizer": {
                "repository": QWEN3_EMBEDDING_8B_REPOSITORY,
                "revision": revision,
                "format": "hugging_face_tokenizer_json",
                "object": tokenizer,
                "add_special_tokens": true
            }
        });
        let runtime = json!({
            "tei_image": {
                "component": {"id": "tei-image", "version": "1.8.0", "sha256": image_sha},
                "repository": "ghcr.io/huggingface/text-embeddings-inference",
                "digest": format!("sha256:{image_sha}")
            },
            "executor_image": {
                "component": {"id": "executor-image", "version": "1", "sha256": executor_image_sha},
                "repository": "ghcr.io/example/livefire-rag-worker",
                "digest": format!("sha256:{executor_image_sha}")
            },
            "executor_image_build": component("executor-image-build", "1"),
            "runtime": component("oci-runtime", "1.0.0"),
            "inference_engine": component("tei-engine", "1.8.0"),
            "load_policy": {
                "component": component("tei-load-policy", "1"),
                "model_source": "mounted_complete_snapshot",
                "revision_policy": "exact",
                "local_files_only": true,
                "trust_remote_code": false,
                "safetensors_only": true
            },
            "runtime_mode": "tei_loopback_worker",
            "api_contract": "openai_compatible_v1_embeddings",
            "api_model_key": "qwen3-embedding-8b-direct"
        });
        let vector = json!({
            "dimensions": 4096,
            "checkpoint_compute_dtype": "float16",
            "api_vector_dtype": "float32",
            "stored_vector_dtype": "f32le",
            "pooling": "last_token",
            "normalization": "l2",
            "maximum_tokens": 8192,
            "document_format": "{semantic_text}",
            "query_instruction": "Retrieve relevant security evidence",
            "query_composition": "Instruct: {query_instruction}\nQuery: {query}",
            "batching": {
                "maximum_batch_items": 32,
                "maximum_batch_tokens": 262144,
                "maximum_concurrent_requests": 4,
                "order": "preserve_input_order",
                "overlength": "reject"
            },
            "response_limits": {"request_timeout_ms": 300000, "maximum_response_bytes": 67108864},
            "output_processing": {"client_normalization": "none", "required_l2_norm_tolerance_millionths": 100},
            "accelerator": accelerator.clone(),
            "conformance": {
                "mode": "exact_digest",
                "measured": true,
                "fixture": artifact("conformance/tei-v3.json", "application/json"),
                "input_count": 2,
                "returned_model": "qwen3-embedding-8b-direct",
                "normalized_output_sha256": digest("measured-test-output"),
                "accelerator": accelerator,
                "candidate_sha256": digest("candidate"),
                "initial_result_sha256": digest("initial result"),
                "fresh_pod_replay_result_sha256": digest("fresh replay result")
            }
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(runtime.as_object().unwrap().clone());
        value
            .as_object_mut()
            .unwrap()
            .extend(vector.as_object().unwrap().clone());
        value
    }

    fn parse_value(value: &serde_json::Value) -> Result<TeiCheckpointProfileV3> {
        parse_tei_checkpoint_profile_v3(&serde_json::to_vec(value).unwrap())
    }

    #[test]
    fn complete_measured_profile_is_executable() {
        let bytes = serde_json::to_vec(&valid_value()).unwrap();
        let profile = parse_tei_checkpoint_profile_v3(&bytes).unwrap();
        let compact = profile.embedding_profile(&bytes).unwrap();
        assert_eq!(compact.model, "qwen3-embedding-8b-direct");
        assert_eq!(compact.dimensions, 4_096);
        assert_eq!(
            profile
                .client_options(BearerAuthorization::None)
                .max_response_bytes,
            67_108_864
        );
        let legacy_error = crate::parse_embedding_profile(&bytes).unwrap_err();
        assert!(matches!(
            legacy_error,
            EmbeddingError::Invalid("TEI checkpoint profile requires the TEI worker path")
        ));
    }

    #[test]
    fn tracked_runpod_policy_fixture_is_executable() {
        let bytes = include_bytes!("../../../rust-fixtures/runpod/embedding-policy.v3.json");
        let profile = parse_tei_checkpoint_profile_v3(bytes).unwrap();
        assert_eq!(profile.batching.maximum_batch_items, 8);
        assert_eq!(profile.batching.maximum_batch_tokens, 65_536);
        assert_eq!(profile.batching.maximum_concurrent_requests, 4);
        assert_eq!(profile.response_limits.request_timeout_ms, 120_000);
        assert_eq!(profile.response_limits.maximum_response_bytes, 1_048_576);
    }

    #[test]
    fn rejects_unmeasured_or_placeholder_conformance() {
        let mut value = valid_value();
        value["conformance"]["measured"] = json!(false);
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["conformance"]["normalized_output_sha256"] = json!("0".repeat(64));
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["conformance"]["mode"] = json!("pending");
        assert!(parse_value(&value).is_err());
    }

    #[test]
    fn rejects_execution_limits_that_cannot_serve_the_pinned_checkpoint() {
        let mut value = valid_value();
        value["batching"]["maximum_batch_tokens"] = json!(32_768);
        assert!(parse_value(&value).is_err());

        let mut value = valid_value();
        value["batching"]["maximum_concurrent_requests"] = json!(0);
        assert!(parse_value(&value).is_err());

        let mut value = valid_value();
        value["response_limits"]["request_timeout_ms"] = json!(0);
        assert!(parse_value(&value).is_err());
    }

    #[test]
    fn rejects_model_tokenizer_runtime_and_dtype_confusion() {
        let mutations = [
            ("model_repository", json!("local/model")),
            ("runtime_mode", json!("lmstudio_loopback")),
            ("api_vector_dtype", json!("bfloat16")),
            ("stored_vector_dtype", json!("bfloat16")),
            ("checkpoint_compute_dtype", json!("bfloat16")),
            ("checkpoint_compute_dtype", json!("f32le")),
        ];
        for (field, replacement) in mutations {
            let mut value = valid_value();
            value[field] = replacement;
            assert!(parse_value(&value).is_err(), "accepted {field}");
        }
        let mut value = valid_value();
        value["executor_image"]["digest"] = json!(format!("sha256:{}", digest("other-image")));
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["executor_image"] = value["tei_image"].clone();
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["executable_tokenizer"]["revision"] = json!(digest("other")[..40].to_owned());
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["runtime"]["id"] = value["model_artifact_set"]["id"].clone();
        assert!(parse_value(&value).is_err());
    }

    #[test]
    fn rejects_incomplete_or_ambiguous_model_object_sets() {
        let mut value = valid_value();
        value["model_objects"] = json!([artifact("config.json", "application/json")]);
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["model_objects"].as_array_mut().unwrap().reverse();
        assert!(parse_value(&value).is_err());
        let mut value = valid_value();
        value["model_objects"][14]["sha256"] = json!(digest("wrong tokenizer"));
        assert!(parse_value(&value).is_err());
    }

    #[test]
    fn schema_file_is_valid_json_and_names_the_v3_contract() {
        let schema: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../specs/embedding-policy.v3.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            TEI_CHECKPOINT_PROFILE_SCHEMA_V3
        );
        assert_eq!(
            schema["properties"]["runtime_mode"]["const"],
            "tei_loopback_worker"
        );
        assert_eq!(
            schema["properties"]["conformance"]["$ref"],
            "#/$defs/conformance"
        );
    }

    #[test]
    fn tracked_upstream_manifest_is_exact_and_digest_bound() {
        let manifest = parse_tei_model_artifact_set_v1(include_bytes!(
            "../../../profiles/qwen3-embedding-8b-upstream-model-artifacts.v1.json"
        ))
        .unwrap();
        assert_eq!(
            manifest.sha256().unwrap(),
            QWEN3_EMBEDDING_8B_ARTIFACT_SET_SHA256
        );
        assert_eq!(
            manifest
                .objects
                .iter()
                .map(|object| object.bytes)
                .sum::<u64>(),
            15_150_575_778
        );
        assert_eq!(
            manifest
                .objects
                .iter()
                .map(|object| object.path.as_str())
                .collect::<Vec<_>>(),
            QWEN3_EMBEDDING_8B_SNAPSHOT_PATHS
        );
        let schema: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../specs/tei-model-artifact-set.v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["properties"]["revision"]["const"],
            QWEN3_EMBEDDING_8B_REVISION
        );
    }
}
