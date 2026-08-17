//! Sealed, provider-independent artifacts for measuring a TEI checkpoint on
//! RunPod before an embedding profile is allowed to claim exact conformance.
//!
//! These contracts contain no endpoint URL, storage URL, environment variable,
//! credential, or token. Provider adapters supply transport details separately.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    ComponentRef, Digest, PipelineError, Result, SafeRelativePath, canonical_digest,
    component_digest, require_safe_u64, require_text,
};

pub const RUNPOD_TEI_CONFORMANCE_CANDIDATE_SCHEMA: &str =
    "livefire.rag.runpod-tei-conformance-candidate/1";
pub const RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA: &str =
    "livefire.rag.runpod-tei-conformance-result/1";
pub const RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA: &str =
    "livefire.rag.runpod-executor-image-build-receipt/1";
pub const TEI_MODEL_ARTIFACT_SET_SCHEMA: &str = "livefire.rag.tei-model-artifact-set/1";
pub const EMBEDDING_POLICY_V3_CONFORMANCE_MODE: &str = "exact_digest";

const MAX_MODEL_OBJECTS: usize = 10_000;
const QWEN3_MAX_POSITION_EMBEDDINGS: u64 = 40_960;
const MAXIMUM_BATCH_TOKENS: u64 = 262_144;
const MAXIMUM_CONCURRENT_REQUESTS: u32 = 256;
const MAXIMUM_REQUEST_TIMEOUT_MS: u64 = 3_600_000;
const MAXIMUM_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiArtifactObject {
    pub path: SafeRelativePath,
    pub media_type: String,
    pub bytes: u64,
    pub sha256: Digest,
}

impl RunpodTeiArtifactObject {
    fn validate(&self) -> Result<()> {
        require_text(&self.media_type)?;
        require_safe_u64(self.bytes)?;
        if self.bytes == 0 {
            return Err(PipelineError::Invalid("TEI conformance artifact size"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiBoundArtifact {
    pub component_sha256: Digest,
    pub object: RunpodTeiArtifactObject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiBoundBuildReceipt {
    pub component: ComponentRef,
    pub object: RunpodTeiArtifactObject,
}

impl RunpodTeiBoundBuildReceipt {
    fn validate(&self) -> Result<()> {
        validate_component(&self.component)?;
        self.object.validate()?;
        if self.object.media_type != "application/json" {
            return Err(PipelineError::Invalid(
                "executor image build receipt object",
            ));
        }
        Ok(())
    }
}

impl RunpodTeiBoundArtifact {
    fn validate(&self) -> Result<()> {
        self.object.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiTokenizerIdentity {
    pub component: ComponentRef,
    pub repository: String,
    pub revision: String,
    pub format: String,
    pub object: RunpodTeiArtifactObject,
    pub add_special_tokens: bool,
}

impl RunpodTeiTokenizerIdentity {
    fn validate(&self, model_repository: &str, model_revision: &str) -> Result<()> {
        validate_component(&self.component)?;
        self.object.validate()?;
        if self.repository != model_repository
            || self.revision != model_revision
            || self.component.version != model_revision
            || self.format != "hugging_face_tokenizer_json"
            || self.object.path.as_str() != "tokenizer.json"
            || self.object.sha256 != self.component.sha256
        {
            return Err(PipelineError::Invalid("TEI conformance tokenizer binding"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiImageIdentity {
    pub component: ComponentRef,
    pub repository: String,
    pub digest: String,
}

impl RunpodTeiImageIdentity {
    fn validate(&self) -> Result<()> {
        validate_component(&self.component)?;
        require_plain_text(&self.repository, 512)?;
        let digest = self
            .digest
            .strip_prefix("sha256:")
            .ok_or(PipelineError::Invalid("TEI conformance image digest"))?;
        if digest != self.component.sha256.as_str() {
            return Err(PipelineError::Invalid("TEI conformance image binding"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodExecutorImageBuildReceipt {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub executor_image: RunpodTeiImageIdentity,
    pub tei_base_image: RunpodTeiImageIdentity,
    pub platform: String,
    pub dockerfile: RunpodTeiArtifactObject,
    pub worker_binary: RunpodTeiBoundArtifact,
}

impl RunpodExecutorImageBuildReceipt {
    pub fn validate(&self) -> Result<()> {
        self.executor_image.validate()?;
        self.tei_base_image.validate()?;
        self.dockerfile.validate()?;
        self.worker_binary.validate()?;
        if self.schema_version != RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA
            || self.platform != "linux/amd64"
            || self.dockerfile.media_type != "text/x-dockerfile"
            || self.executor_image.component == self.tei_base_image.component
            || self.executor_image.repository == self.tei_base_image.repository
            || self.component_sha256 != component_digest(self)?
        {
            return Err(PipelineError::Invalid("executor image build receipt"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiLoadPolicy {
    pub component: ComponentRef,
    pub model_source: String,
    pub revision_policy: String,
    pub local_files_only: bool,
    pub trust_remote_code: bool,
    pub safetensors_only: bool,
}

impl RunpodTeiLoadPolicy {
    fn validate(&self) -> Result<()> {
        validate_component(&self.component)?;
        if self.model_source != "mounted_complete_snapshot"
            || self.revision_policy != "exact"
            || !self.local_files_only
            || self.trust_remote_code
            || !self.safetensors_only
        {
            return Err(PipelineError::Invalid("TEI conformance load policy"));
        }
        Ok(())
    }
}

/// Accelerator details that affect exact output reproducibility. A different
/// GPU model or architecture/image class requires a different profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiAcceleratorPolicy {
    pub provider: String,
    pub gpu_model_id: String,
    pub compute_capability: String,
    pub architecture_image_class: String,
    pub gpu_count: u32,
}

impl RunpodTeiAcceleratorPolicy {
    fn validate(&self) -> Result<()> {
        require_plain_text(&self.provider, 64)?;
        require_plain_text(&self.gpu_model_id, 256)?;
        require_identifier(&self.architecture_image_class, 128)?;
        if self.provider != "runpod"
            || self.gpu_count != 1
            || self.compute_capability.len() < 3
            || self.compute_capability.len() > 8
            || self.compute_capability.matches('.').count() != 1
            || !self
                .compute_capability
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'.')
        {
            return Err(PipelineError::Invalid("TEI conformance accelerator policy"));
        }
        Ok(())
    }
}

/// Identity reported by the worker. Results repeat this value so comparison
/// never relies only on a candidate digest hidden elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiExecutionIdentity {
    pub model_artifact_set: ComponentRef,
    pub tokenizer: ComponentRef,
    pub tei_image: ComponentRef,
    pub executor_image: ComponentRef,
    pub executor_image_build: ComponentRef,
    pub runtime: ComponentRef,
    pub inference_engine: ComponentRef,
    pub load_policy: ComponentRef,
    pub worker_binary: ComponentRef,
    pub served_model: String,
    pub dimensions: u32,
    pub pooling: String,
    pub forced_runtime_dtype: String,
    pub api_vector_dtype: String,
    pub normalization: String,
    pub maximum_tokens: u32,
    /// Value passed to TEI as `--max-client-batch-size`.
    pub maximum_client_batch_size: u32,
    /// Value passed to TEI as `--max-batch-tokens`. It must cover the pinned
    /// Qwen checkpoint's full positional window even when the fixture is tiny.
    pub maximum_batch_tokens: u64,
    /// Value passed to TEI as `--max-concurrent-requests`.
    pub maximum_concurrent_requests: u32,
    /// Bound used by the loopback embeddings client for each request.
    pub request_timeout_ms: u64,
    /// Maximum response body accepted by the loopback embeddings client.
    pub maximum_response_bytes: u64,
}

impl RunpodTeiExecutionIdentity {
    fn validate(&self) -> Result<()> {
        for component in [
            &self.model_artifact_set,
            &self.tokenizer,
            &self.tei_image,
            &self.executor_image,
            &self.executor_image_build,
            &self.runtime,
            &self.inference_engine,
            &self.load_policy,
            &self.worker_binary,
        ] {
            validate_component(component)?;
        }
        require_plain_text(&self.served_model, 1_024)?;
        let ids = [
            self.model_artifact_set.id.as_str(),
            self.tokenizer.id.as_str(),
            self.tei_image.id.as_str(),
            self.executor_image.id.as_str(),
            self.executor_image_build.id.as_str(),
            self.runtime.id.as_str(),
            self.inference_engine.id.as_str(),
            self.load_policy.id.as_str(),
            self.worker_binary.id.as_str(),
        ];
        if ids.into_iter().collect::<BTreeSet<_>>().len() != ids.len()
            || self.dimensions == 0
            || self.pooling != "last_token"
            || !matches!(self.forced_runtime_dtype.as_str(), "float16" | "float32")
            || self.api_vector_dtype != "float32"
            || self.normalization != "l2"
            || self.maximum_tokens == 0
            || self.maximum_tokens > 8_192
            || !(1..=32).contains(&self.maximum_client_batch_size)
            || !(QWEN3_MAX_POSITION_EMBEDDINGS..=MAXIMUM_BATCH_TOKENS)
                .contains(&self.maximum_batch_tokens)
            || self.maximum_batch_tokens
                > u64::from(self.maximum_tokens) * u64::from(self.maximum_client_batch_size)
            || !(1..=MAXIMUM_CONCURRENT_REQUESTS).contains(&self.maximum_concurrent_requests)
            || !(1..=MAXIMUM_REQUEST_TIMEOUT_MS).contains(&self.request_timeout_ms)
            || !(1_024..=MAXIMUM_RESPONSE_BYTES).contains(&self.maximum_response_bytes)
        {
            return Err(PipelineError::Invalid("TEI conformance execution identity"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiConformanceFixture {
    pub object: RunpodTeiArtifactObject,
    pub input_count: u32,
}

impl RunpodTeiConformanceFixture {
    fn validate(&self) -> Result<()> {
        self.object.validate()?;
        if self.object.media_type != "application/json" || !(1..=32).contains(&self.input_count) {
            return Err(PipelineError::Invalid("TEI conformance fixture"));
        }
        Ok(())
    }
}

/// Immutable input to the first measurement. It intentionally has no expected
/// or claimed output digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiConformanceCandidate {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub model_repository: String,
    pub model_revision: String,
    pub model_snapshot_completeness: String,
    pub model_manifest: RunpodTeiBoundArtifact,
    pub model_objects: Vec<RunpodTeiArtifactObject>,
    pub tokenizer: RunpodTeiTokenizerIdentity,
    pub tei_image: RunpodTeiImageIdentity,
    /// Custom derivative image that contains the fixed Rust worker entrypoint.
    pub executor_image: RunpodTeiImageIdentity,
    pub executor_image_build: RunpodTeiBoundBuildReceipt,
    pub runtime: ComponentRef,
    pub inference_engine: ComponentRef,
    pub load_policy: RunpodTeiLoadPolicy,
    pub worker_binary: RunpodTeiBoundArtifact,
    pub execution: RunpodTeiExecutionIdentity,
    pub accelerator: RunpodTeiAcceleratorPolicy,
    pub fixture: RunpodTeiConformanceFixture,
    pub expected_output_key: SafeRelativePath,
    pub output_format: String,
}

impl RunpodTeiConformanceCandidate {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != RUNPOD_TEI_CONFORMANCE_CANDIDATE_SCHEMA
            || self.model_snapshot_completeness != "complete_upstream_snapshot"
            || !valid_revision(&self.model_revision)
        {
            return Err(PipelineError::Invalid("TEI conformance candidate identity"));
        }
        require_plain_text(&self.model_repository, 512)?;
        self.model_manifest.validate()?;
        self.tokenizer
            .validate(&self.model_repository, &self.model_revision)?;
        self.tei_image.validate()?;
        self.executor_image.validate()?;
        self.executor_image_build.validate()?;
        validate_component(&self.runtime)?;
        validate_component(&self.inference_engine)?;
        self.load_policy.validate()?;
        self.worker_binary.validate()?;
        self.execution.validate()?;
        self.accelerator.validate()?;
        self.fixture.validate()?;
        if self.fixture.input_count > self.execution.maximum_client_batch_size {
            return Err(PipelineError::Invalid(
                "TEI conformance fixture exceeds client batch limit",
            ));
        }
        if self.output_format != "rfc8785_f32_vectors_lf_v1" {
            return Err(PipelineError::Invalid("TEI conformance output format"));
        }
        validate_model_objects(self)?;
        if self.model_manifest.component_sha256 != self.execution.model_artifact_set.sha256
            || self.tokenizer.component != self.execution.tokenizer
            || self.tei_image.component != self.execution.tei_image
            || self.executor_image.component != self.execution.executor_image
            || self.executor_image_build.component != self.execution.executor_image_build
            || self.runtime != self.execution.runtime
            || self.inference_engine != self.execution.inference_engine
            || self.load_policy.component != self.execution.load_policy
            || self.worker_binary.component_sha256 != self.execution.worker_binary.sha256
        {
            return Err(PipelineError::Invalid(
                "TEI conformance candidate execution binding",
            ));
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("TEI conformance candidate digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

fn validate_model_objects(candidate: &RunpodTeiConformanceCandidate) -> Result<()> {
    if candidate.model_objects.is_empty() || candidate.model_objects.len() > MAX_MODEL_OBJECTS {
        return Err(PipelineError::Invalid("TEI conformance model objects"));
    }
    let mut previous = None;
    let mut paths = BTreeSet::new();
    let mut tokenizer_bound = false;
    for object in &candidate.model_objects {
        object.validate()?;
        if previous.is_some_and(|path: &str| path >= object.path.as_str())
            || !paths.insert(object.path.as_str())
        {
            return Err(PipelineError::Invalid("TEI conformance model object order"));
        }
        tokenizer_bound |= object.path == candidate.tokenizer.object.path
            && object.sha256 == candidate.tokenizer.object.sha256;
        previous = Some(object.path.as_str());
    }
    let digest = model_artifact_set_digest(
        &candidate.model_repository,
        &candidate.model_revision,
        &candidate.model_objects,
    )?;
    if !tokenizer_bound || digest != candidate.execution.model_artifact_set.sha256 {
        return Err(PipelineError::Invalid(
            "TEI conformance model manifest binding",
        ));
    }
    Ok(())
}

pub fn model_artifact_set_digest(
    repository: &str,
    revision: &str,
    objects: &[RunpodTeiArtifactObject],
) -> Result<Digest> {
    canonical_digest(&serde_json::json!({
        "schema_version": TEI_MODEL_ARTIFACT_SET_SCHEMA,
        "repository": repository,
        "revision": revision,
        "objects": objects,
    }))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiMachineIdentity {
    pub pod_id: String,
    pub machine_id: String,
    pub gpu_device_id: String,
    pub accelerator: RunpodTeiAcceleratorPolicy,
}

impl RunpodTeiMachineIdentity {
    fn validate(&self) -> Result<()> {
        require_identifier(&self.pod_id, 128)?;
        require_identifier(&self.machine_id, 256)?;
        require_identifier(&self.gpu_device_id, 256)?;
        self.accelerator.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunpodTeiConformanceOutcome {
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiNormalizedOutput {
    pub object: RunpodTeiArtifactObject,
    pub format: String,
    pub vector_count: u32,
    pub dimensions: u32,
    pub dtype: String,
    pub normalized_output_sha256: Digest,
}

impl RunpodTeiNormalizedOutput {
    fn validate_against(&self, candidate: &RunpodTeiConformanceCandidate) -> Result<()> {
        self.object.validate()?;
        if self.object.path != candidate.expected_output_key
            || self.object.media_type != "application/json"
            || self.object.sha256 != self.normalized_output_sha256
            || self.format != candidate.output_format
            || self.vector_count != candidate.fixture.input_count
            || self.dimensions != candidate.execution.dimensions
            || self.dtype != candidate.execution.api_vector_dtype
        {
            return Err(PipelineError::Invalid("TEI conformance output binding"));
        }
        Ok(())
    }
}

/// One immutable measurement attempt. Failed attempts remain accountable but
/// cannot be promoted into an embedding profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTeiConformanceResult {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub candidate_sha256: Digest,
    pub run_id: String,
    pub machine: RunpodTeiMachineIdentity,
    pub execution: RunpodTeiExecutionIdentity,
    pub outcome: RunpodTeiConformanceOutcome,
    pub returned_model: Option<String>,
    pub normalized_output: Option<RunpodTeiNormalizedOutput>,
    pub model_load_ms: u64,
    pub request_ms: u64,
    pub failure_code: Option<String>,
}

impl RunpodTeiConformanceResult {
    pub fn validate_against(&self, candidate: &RunpodTeiConformanceCandidate) -> Result<()> {
        candidate.validate()?;
        if self.schema_version != RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA
            || self.candidate_sha256 != candidate.component_sha256
        {
            return Err(PipelineError::Invalid("TEI conformance result identity"));
        }
        require_identifier(&self.run_id, 128)?;
        self.machine.validate()?;
        self.execution.validate()?;
        require_safe_u64(self.model_load_ms)?;
        require_safe_u64(self.request_ms)?;
        if self.machine.accelerator != candidate.accelerator
            || self.execution != candidate.execution
        {
            return Err(PipelineError::Invalid(
                "TEI conformance result execution binding",
            ));
        }
        match self.outcome {
            RunpodTeiConformanceOutcome::Completed => {
                let returned_model = self
                    .returned_model
                    .as_deref()
                    .ok_or(PipelineError::Invalid("TEI conformance returned model"))?;
                let output = self
                    .normalized_output
                    .as_ref()
                    .ok_or(PipelineError::Invalid("TEI conformance completed output"))?;
                if returned_model != candidate.execution.served_model || self.failure_code.is_some()
                {
                    return Err(PipelineError::Invalid("TEI conformance completion outcome"));
                }
                output.validate_against(candidate)?;
            }
            RunpodTeiConformanceOutcome::Failed => {
                if self.returned_model.is_some()
                    || self.normalized_output.is_some()
                    || self
                        .failure_code
                        .as_deref()
                        .is_none_or(|code| require_identifier(code, 128).is_err())
                {
                    return Err(PipelineError::Invalid("TEI conformance failure outcome"));
                }
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("TEI conformance result digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self, candidate: &RunpodTeiConformanceCandidate) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate_against(candidate)
    }
}

/// Fields copied directly into `embedding-policy/3.conformance` after two
/// successful measurements agree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingPolicyV3ConformanceFields {
    pub mode: String,
    pub measured: bool,
    pub fixture: RunpodTeiArtifactObject,
    pub input_count: u32,
    pub returned_model: String,
    pub normalized_output_sha256: Digest,
    pub accelerator: RunpodTeiAcceleratorPolicy,
    pub candidate_sha256: Digest,
    pub initial_result_sha256: Digest,
    pub fresh_pod_replay_result_sha256: Digest,
}

impl EmbeddingPolicyV3ConformanceFields {
    pub fn validate_against(&self, candidate: &RunpodTeiConformanceCandidate) -> Result<()> {
        candidate.validate()?;
        if self.mode != EMBEDDING_POLICY_V3_CONFORMANCE_MODE
            || !self.measured
            || self.fixture != candidate.fixture.object
            || self.input_count != candidate.fixture.input_count
            || self.returned_model != candidate.execution.served_model
            || self.accelerator != candidate.accelerator
            || self.candidate_sha256 != candidate.component_sha256
            || self.initial_result_sha256 == self.fresh_pod_replay_result_sha256
        {
            return Err(PipelineError::Invalid(
                "embedding policy v3 conformance fields",
            ));
        }
        Ok(())
    }
}

/// Promote an initial measurement and a fresh-Pod replay only when exact
/// output and all execution-affecting identities agree. Pod, machine, and GPU
/// device instance IDs may differ; the accelerator class itself may not.
pub fn seal_embedding_policy_v3_conformance(
    candidate: &RunpodTeiConformanceCandidate,
    first: &RunpodTeiConformanceResult,
    fresh_pod_replay: &RunpodTeiConformanceResult,
) -> Result<EmbeddingPolicyV3ConformanceFields> {
    candidate.validate()?;
    first.validate_against(candidate)?;
    fresh_pod_replay.validate_against(candidate)?;
    if first.outcome != RunpodTeiConformanceOutcome::Completed
        || fresh_pod_replay.outcome != RunpodTeiConformanceOutcome::Completed
        || first.run_id == fresh_pod_replay.run_id
        || first.machine.pod_id == fresh_pod_replay.machine.pod_id
        || first.execution != fresh_pod_replay.execution
        || first.machine.accelerator != fresh_pod_replay.machine.accelerator
    {
        return Err(PipelineError::Invalid("TEI conformance replay identity"));
    }
    let first_output = first
        .normalized_output
        .as_ref()
        .ok_or(PipelineError::Invalid("TEI conformance completed output"))?;
    let replay_output = fresh_pod_replay
        .normalized_output
        .as_ref()
        .ok_or(PipelineError::Invalid("TEI conformance completed output"))?;
    if first.returned_model != fresh_pod_replay.returned_model
        || first_output.normalized_output_sha256 != replay_output.normalized_output_sha256
        || first_output.object.sha256 != replay_output.object.sha256
        || first_output.object.bytes != replay_output.object.bytes
    {
        return Err(PipelineError::Invalid("TEI conformance replay output"));
    }
    let fields = EmbeddingPolicyV3ConformanceFields {
        mode: EMBEDDING_POLICY_V3_CONFORMANCE_MODE.into(),
        measured: true,
        fixture: candidate.fixture.object.clone(),
        input_count: candidate.fixture.input_count,
        returned_model: candidate.execution.served_model.clone(),
        normalized_output_sha256: first_output.normalized_output_sha256.clone(),
        accelerator: candidate.accelerator.clone(),
        candidate_sha256: candidate.component_sha256.clone(),
        initial_result_sha256: first.component_sha256.clone(),
        fresh_pod_replay_result_sha256: fresh_pod_replay.component_sha256.clone(),
    };
    fields.validate_against(candidate)?;
    Ok(fields)
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn require_plain_text(value: &str, maximum_bytes: usize) -> Result<()> {
    require_text(value)?;
    if value.len() > maximum_bytes
        || value.contains("://")
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(PipelineError::Invalid("TEI conformance plain text"));
    }
    Ok(())
}

fn require_identifier(value: &str, maximum_bytes: usize) -> Result<()> {
    require_text(value)?;
    if value.len() > maximum_bytes
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PipelineError::Invalid("TEI conformance identifier"));
    }
    Ok(())
}

fn validate_component(component: &ComponentRef) -> Result<()> {
    component.validate()?;
    require_plain_text(&component.id, 512)?;
    require_plain_text(&component.version, 512)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest_bytes;

    fn digest(label: &str) -> Digest {
        digest_bytes(label.as_bytes())
    }

    fn component(id: &str, version: &str) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: version.into(),
            sha256: digest(id),
        }
    }

    fn object(path: &str, media_type: &str) -> RunpodTeiArtifactObject {
        RunpodTeiArtifactObject {
            path: SafeRelativePath::new(path).unwrap(),
            media_type: media_type.into(),
            bytes: 16,
            sha256: digest(path),
        }
    }

    fn candidate() -> RunpodTeiConformanceCandidate {
        let revision = digest("revision").as_str()[..40].to_owned();
        let model_objects = vec![
            object("config.json", "application/json"),
            object("model.safetensors", "application/vnd.safetensors"),
            object("tokenizer.json", "application/json"),
        ];
        let model_sha =
            model_artifact_set_digest("upstream/model", &revision, &model_objects).unwrap();
        let tokenizer_object = model_objects[2].clone();
        let tokenizer = RunpodTeiTokenizerIdentity {
            component: ComponentRef {
                id: "tokenizer".into(),
                version: revision.clone(),
                sha256: tokenizer_object.sha256.clone(),
            },
            repository: "upstream/model".into(),
            revision: revision.clone(),
            format: "hugging_face_tokenizer_json".into(),
            object: tokenizer_object,
            add_special_tokens: true,
        };
        let image_component = component("tei-image", "1.9.3");
        let runtime = component("container-runtime", "1");
        let inference_engine = component("tei-engine", "1.9.3");
        let load_component = component("load-policy", "1");
        let worker_component = component("rust-worker", "1");
        let model_component = ComponentRef {
            id: "model-artifact-set".into(),
            version: revision.clone(),
            sha256: model_sha.clone(),
        };
        let execution = RunpodTeiExecutionIdentity {
            model_artifact_set: model_component.clone(),
            tokenizer: tokenizer.component.clone(),
            tei_image: image_component.clone(),
            executor_image: component("executor-image", "1"),
            executor_image_build: component("executor-image-build", "1"),
            runtime: runtime.clone(),
            inference_engine: inference_engine.clone(),
            load_policy: load_component.clone(),
            worker_binary: worker_component.clone(),
            served_model: "served-model".into(),
            dimensions: 4_096,
            pooling: "last_token".into(),
            forced_runtime_dtype: "float16".into(),
            api_vector_dtype: "float32".into(),
            normalization: "l2".into(),
            maximum_tokens: 8_192,
            maximum_client_batch_size: 8,
            maximum_batch_tokens: 65_536,
            maximum_concurrent_requests: 4,
            request_timeout_ms: 120_000,
            maximum_response_bytes: 64 * 1024 * 1024,
        };
        let accelerator = RunpodTeiAcceleratorPolicy {
            provider: "runpod".into(),
            gpu_model_id: "NVIDIA H100 PCIe".into(),
            compute_capability: "9.0".into(),
            architecture_image_class: "sm90-cuda12".into(),
            gpu_count: 1,
        };
        let mut candidate = RunpodTeiConformanceCandidate {
            schema_version: RUNPOD_TEI_CONFORMANCE_CANDIDATE_SCHEMA.into(),
            component_sha256: digest("placeholder"),
            model_repository: "upstream/model".into(),
            model_revision: revision,
            model_snapshot_completeness: "complete_upstream_snapshot".into(),
            model_manifest: RunpodTeiBoundArtifact {
                component_sha256: model_sha,
                object: object("model/manifest.json", "application/json"),
            },
            model_objects,
            tokenizer,
            tei_image: RunpodTeiImageIdentity {
                digest: format!("sha256:{}", image_component.sha256),
                component: image_component,
                repository: "ghcr.io/huggingface/text-embeddings-inference".into(),
            },
            executor_image: {
                let component = execution.executor_image.clone();
                RunpodTeiImageIdentity {
                    digest: format!("sha256:{}", component.sha256),
                    component,
                    repository: "ghcr.io/example/livefire-rag-worker".into(),
                }
            },
            executor_image_build: RunpodTeiBoundBuildReceipt {
                component: execution.executor_image_build.clone(),
                object: object("executor-image-build.json", "application/json"),
            },
            runtime,
            inference_engine,
            load_policy: RunpodTeiLoadPolicy {
                component: load_component,
                model_source: "mounted_complete_snapshot".into(),
                revision_policy: "exact".into(),
                local_files_only: true,
                trust_remote_code: false,
                safetensors_only: true,
            },
            worker_binary: RunpodTeiBoundArtifact {
                component_sha256: worker_component.sha256.clone(),
                object: object("bin/rag-tei-conformance-worker", "application/octet-stream"),
            },
            execution,
            accelerator,
            fixture: RunpodTeiConformanceFixture {
                object: object("fixtures/tei-conformance.json", "application/json"),
                input_count: 1,
            },
            expected_output_key: SafeRelativePath::new("results/normalized-vectors.json").unwrap(),
            output_format: "rfc8785_f32_vectors_lf_v1".into(),
        };
        candidate.seal().unwrap();
        candidate
    }

    fn completed_result(
        candidate: &RunpodTeiConformanceCandidate,
        run_id: &str,
        pod_id: &str,
        machine_id: &str,
        device_id: &str,
    ) -> RunpodTeiConformanceResult {
        let output_digest = digest("identical normalized f32 output");
        let mut result = RunpodTeiConformanceResult {
            schema_version: RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA.into(),
            component_sha256: digest("placeholder-result"),
            candidate_sha256: candidate.component_sha256.clone(),
            run_id: run_id.into(),
            machine: RunpodTeiMachineIdentity {
                pod_id: pod_id.into(),
                machine_id: machine_id.into(),
                gpu_device_id: device_id.into(),
                accelerator: candidate.accelerator.clone(),
            },
            execution: candidate.execution.clone(),
            outcome: RunpodTeiConformanceOutcome::Completed,
            returned_model: Some(candidate.execution.served_model.clone()),
            normalized_output: Some(RunpodTeiNormalizedOutput {
                object: RunpodTeiArtifactObject {
                    path: candidate.expected_output_key.clone(),
                    media_type: "application/json".into(),
                    bytes: 128,
                    sha256: output_digest.clone(),
                },
                format: candidate.output_format.clone(),
                vector_count: candidate.fixture.input_count,
                dimensions: candidate.execution.dimensions,
                dtype: candidate.execution.api_vector_dtype.clone(),
                normalized_output_sha256: output_digest,
            }),
            model_load_ms: 10_000,
            request_ms: 500,
            failure_code: None,
        };
        result.seal(candidate).unwrap();
        result
    }

    #[test]
    fn candidate_and_result_self_digests_are_deterministic() {
        let first = candidate();
        let second = candidate();
        assert_eq!(first.component_sha256, second.component_sha256);
        assert_eq!(first.component_sha256, component_digest(&first).unwrap());
        let candidate_json = serde_json::to_value(&first).unwrap();
        assert!(candidate_json.get("normalized_output_sha256").is_none());
        assert!(
            !serde_json::to_string(&first)
                .unwrap()
                .contains("normalized_output_sha256")
        );
        let result = completed_result(&first, "run-1", "pod-1", "machine-1", "gpu-1");
        assert_eq!(result.component_sha256, component_digest(&result).unwrap());
        result.validate_against(&first).unwrap();

        let mut confused = candidate();
        confused.executor_image = confused.tei_image.clone();
        confused.execution.executor_image = confused.execution.tei_image.clone();
        assert!(confused.seal().is_err());
    }

    #[test]
    fn tracked_candidate_fixture_is_sealed_and_valid() {
        let candidate: RunpodTeiConformanceCandidate = serde_json::from_slice(include_bytes!(
            "../../../rust-fixtures/runpod/tei-conformance-candidate.v1.json"
        ))
        .unwrap();
        candidate.validate().unwrap();
        assert_eq!(candidate.fixture.input_count, 1);
        assert_eq!(candidate.execution.maximum_batch_tokens, 65_536);
    }

    #[test]
    fn candidate_seals_production_limits_independently_of_one_input_fixture() {
        let candidate = candidate();
        assert_eq!(candidate.fixture.input_count, 1);
        assert_eq!(candidate.execution.maximum_client_batch_size, 8);
        assert_eq!(candidate.execution.maximum_batch_tokens, 65_536);
        assert_eq!(candidate.execution.maximum_concurrent_requests, 4);

        let mut too_small_for_checkpoint = candidate.clone();
        too_small_for_checkpoint.execution.maximum_batch_tokens = 32_768;
        assert!(too_small_for_checkpoint.seal().is_err());

        let mut no_concurrency = candidate.clone();
        no_concurrency.execution.maximum_concurrent_requests = 0;
        assert!(no_concurrency.seal().is_err());

        let mut fixture_exceeds_client_batch = candidate;
        fixture_exceeds_client_batch.fixture.input_count = 9;
        assert!(fixture_exceeds_client_batch.seal().is_err());
    }

    #[test]
    fn executor_image_build_receipt_binds_platform_base_source_and_binary() {
        let candidate = candidate();
        let mut receipt = RunpodExecutorImageBuildReceipt {
            schema_version: RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA.into(),
            component_sha256: digest("unsealed build receipt"),
            executor_image: candidate.executor_image.clone(),
            tei_base_image: candidate.tei_image.clone(),
            platform: "linux/amd64".into(),
            dockerfile: object("container/Dockerfile", "text/x-dockerfile"),
            worker_binary: candidate.worker_binary.clone(),
        };
        receipt.seal().unwrap();
        let sealed = receipt.component_sha256.clone();
        receipt.platform = "linux/arm64".into();
        assert!(receipt.validate().is_err());
        receipt.platform = "linux/amd64".into();
        receipt.worker_binary.component_sha256 = digest("different worker");
        receipt.component_sha256 = sealed;
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn fresh_pod_replay_may_change_only_machine_instance_and_timings() {
        let candidate = candidate();
        let first = completed_result(&candidate, "run-1", "pod-1", "machine-1", "gpu-1");
        let mut replay = completed_result(&candidate, "run-2", "pod-2", "machine-2", "gpu-2");
        replay.model_load_ms = 12_000;
        replay.request_ms = 450;
        replay.seal(&candidate).unwrap();
        let fields = seal_embedding_policy_v3_conformance(&candidate, &first, &replay).unwrap();
        assert!(fields.measured);
        assert_eq!(fields.accelerator, candidate.accelerator);
        assert_eq!(
            fields.normalized_output_sha256,
            first
                .normalized_output
                .as_ref()
                .unwrap()
                .normalized_output_sha256
        );
        fields.validate_against(&candidate).unwrap();
    }

    #[test]
    fn replay_rejects_same_pod_output_or_execution_mismatch() {
        let candidate = candidate();
        let first = completed_result(&candidate, "run-1", "pod-1", "machine-1", "gpu-1");

        let same_pod = completed_result(&candidate, "run-2", "pod-1", "machine-2", "gpu-2");
        assert!(seal_embedding_policy_v3_conformance(&candidate, &first, &same_pod).is_err());

        let mut wrong_output = completed_result(&candidate, "run-2", "pod-2", "machine-2", "gpu-2");
        let output = wrong_output.normalized_output.as_mut().unwrap();
        output.normalized_output_sha256 = digest("different output");
        output.object.sha256 = output.normalized_output_sha256.clone();
        wrong_output.seal(&candidate).unwrap();
        assert!(seal_embedding_policy_v3_conformance(&candidate, &first, &wrong_output).is_err());

        let mut wrong_gpu = completed_result(&candidate, "run-2", "pod-2", "machine-2", "gpu-2");
        wrong_gpu.machine.accelerator.gpu_model_id = "NVIDIA A100".into();
        wrong_gpu.component_sha256 = component_digest(&wrong_gpu).unwrap();
        assert!(wrong_gpu.validate_against(&candidate).is_err());

        let mut wrong_execution =
            completed_result(&candidate, "run-2", "pod-2", "machine-2", "gpu-2");
        wrong_execution.execution.runtime.sha256 = digest("other runtime");
        wrong_execution.component_sha256 = component_digest(&wrong_execution).unwrap();
        assert!(wrong_execution.validate_against(&candidate).is_err());
    }

    #[test]
    fn failed_result_is_sealed_but_cannot_be_promoted() {
        let candidate = candidate();
        let first = completed_result(&candidate, "run-1", "pod-1", "machine-1", "gpu-1");
        let mut failed = RunpodTeiConformanceResult {
            schema_version: RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA.into(),
            component_sha256: digest("placeholder-result"),
            candidate_sha256: candidate.component_sha256.clone(),
            run_id: "run-2".into(),
            machine: RunpodTeiMachineIdentity {
                pod_id: "pod-2".into(),
                machine_id: "machine-2".into(),
                gpu_device_id: "gpu-2".into(),
                accelerator: candidate.accelerator.clone(),
            },
            execution: candidate.execution.clone(),
            outcome: RunpodTeiConformanceOutcome::Failed,
            returned_model: None,
            normalized_output: None,
            model_load_ms: 100,
            request_ms: 0,
            failure_code: Some("model_load_failed".into()),
        };
        failed.seal(&candidate).unwrap();
        assert!(seal_embedding_policy_v3_conformance(&candidate, &first, &failed).is_err());
    }

    #[test]
    fn contracts_reject_urls_unsafe_paths_and_unsafe_integers() {
        let mut invalid_candidate = candidate();
        invalid_candidate.model_repository = "https://example.invalid/model".into();
        invalid_candidate.component_sha256 = component_digest(&invalid_candidate).unwrap();
        assert!(invalid_candidate.validate().is_err());

        assert!(serde_json::from_str::<RunpodTeiArtifactObject>(
            r#"{"path":"../escape","media_type":"application/json","bytes":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#
        )
        .is_err());

        let candidate = candidate();
        let mut result = completed_result(&candidate, "run-1", "pod-1", "machine-1", "gpu-1");
        result.request_ms = 9_007_199_254_740_992;
        result.component_sha256 = component_digest(&result).unwrap();
        assert!(result.validate_against(&candidate).is_err());
    }

    #[test]
    fn schemas_are_closed_and_name_the_contracts() {
        for (text, id) in [
            (
                include_str!("../schema/runpod-tei-conformance-candidate.v1.schema.json"),
                "https://livefire.dev/rag/runpod-tei-conformance-candidate.v1.schema.json",
            ),
            (
                include_str!("../schema/runpod-tei-conformance-result.v1.schema.json"),
                "https://livefire.dev/rag/runpod-tei-conformance-result.v1.schema.json",
            ),
            (
                include_str!("../schema/runpod-executor-image-build-receipt.v1.schema.json"),
                "https://livefire.dev/rag/runpod-executor-image-build-receipt.v1.schema.json",
            ),
        ] {
            let schema: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(schema["$id"], id);
            assert_eq!(schema["additionalProperties"], false);
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert_closed_objects(&schema);
        }
    }

    #[test]
    fn tracked_qwen_manifest_uses_the_same_artifact_set_digest_contract() {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Manifest {
            schema_version: String,
            repository: String,
            revision: String,
            objects: Vec<RunpodTeiArtifactObject>,
        }

        let bytes =
            include_bytes!("../../../profiles/qwen3-embedding-8b-upstream-model-artifacts.v1.json");
        let manifest: Manifest = serde_json::from_slice(bytes).unwrap();
        assert_eq!(manifest.schema_version, TEI_MODEL_ARTIFACT_SET_SCHEMA);
        assert_eq!(manifest.objects.len(), 17);
        assert_eq!(
            manifest
                .objects
                .iter()
                .map(|object| object.bytes)
                .sum::<u64>(),
            15_150_575_778
        );
        assert_eq!(
            model_artifact_set_digest(&manifest.repository, &manifest.revision, &manifest.objects)
                .unwrap()
                .as_str(),
            "99beb578f3ca8c20eb204484178bf08fea6f0d7f016ab49ca33a8590e1af2dcb"
        );
    }

    fn assert_closed_objects(value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                if object.get("type").and_then(serde_json::Value::as_str) == Some("object") {
                    assert_eq!(
                        object.get("additionalProperties"),
                        Some(&serde_json::Value::Bool(false))
                    );
                }
                for child in object.values() {
                    assert_closed_objects(child);
                }
            }
            serde_json::Value::Array(array) => {
                for child in array {
                    assert_closed_objects(child);
                }
            }
            _ => {}
        }
    }
}
