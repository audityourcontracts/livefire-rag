//! Content-bound contracts for executing an existing embedding plan on remote workers.
//!
//! These types describe immutable objects and deterministic work allocation. They
//! deliberately contain no endpoint URLs, credentials, environment variables, or
//! provider access tokens. A storage or RunPod adapter supplies those at runtime.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

use super::{
    ComponentRef, Digest, EmbeddingPlanV2, PipelineError, PreparedCorpusManifest,
    RUNPOD_EMBEDDING_BUNDLE_SCHEMA, RUNPOD_RUN_REPORT_SCHEMA, RUNPOD_WORKER_ATTEMPT_SCHEMA, Result,
    SafeRelativePath, canonical_json_bytes, component_digest, digest_bytes, require_safe_u64,
    require_text, resolve_existing_artifact,
};

const MAX_WORKERS: u32 = 65_535;

/// One immutable object at a key relative to a caller-controlled storage root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudObjectRef {
    pub key: SafeRelativePath,
    pub bytes: u64,
    pub sha256: Digest,
}

impl CloudObjectRef {
    fn validate(&self) -> Result<()> {
        require_safe_u64(self.bytes)?;
        if self.bytes == 0 {
            return Err(PipelineError::Invalid("cloud object byte length"));
        }
        Ok(())
    }
}

/// An immutable file whose logical component identity is already declared by
/// the prepared corpus, plan, or embedding profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudComponentArtifact {
    pub component_sha256: Digest,
    pub object: CloudObjectRef,
}

impl CloudComponentArtifact {
    fn validate(&self) -> Result<()> {
        self.object.validate()
    }
}

/// One prepared-document object needed by the embedding plan. Occurrence
/// objects are intentionally not part of the remote embedding bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudPreparedDocumentArtifact {
    pub prepared_path: SafeRelativePath,
    pub object: CloudObjectRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodBundleArtifacts {
    pub prepared_manifest: CloudComponentArtifact,
    pub embedding_plan: CloudComponentArtifact,
    /// The exact binary u32 token-count object bound by the v2 plan.
    pub document_token_counts: CloudObjectRef,
    pub embedding_profile: CloudComponentArtifact,
    /// Sealed receipt for the custom executor image build.
    pub executor_image_build: CloudComponentArtifact,
    pub executable_tokenizer: CloudComponentArtifact,
    /// Exact input used by the worker's mandatory TEI conformance check.
    pub conformance_fixture: CloudObjectRef,
    /// Byte-exact frozen catalogue query plan used to produce the cloud
    /// profile's sealed offline query vectors.
    pub query_plan: CloudObjectRef,
    /// Verified Rust worker binary staged with the immutable bundle.
    pub worker_binary: CloudComponentArtifact,
    /// Sealed manifest whose component identity is the plan's model artifact.
    pub model_manifest: CloudComponentArtifact,
    /// Every file required to load the model described by `model_manifest`.
    /// A one-file GGUF model has one entry; Safetensors models list every shard
    /// plus their configuration files.
    pub model_objects: Vec<CloudObjectRef>,
    pub prepared_documents: Vec<CloudPreparedDocumentArtifact>,
}

/// Exact accelerator policy shared by every worker in one run. Machines may
/// differ, but the provider, model, architecture, compute capability, and
/// count may not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodAcceleratorIdentity {
    pub provider: String,
    pub model: String,
    pub architecture: String,
    pub compute_capability: String,
    pub count: u32,
}

impl RunpodAcceleratorIdentity {
    fn validate(&self) -> Result<()> {
        for value in [
            &self.provider,
            &self.model,
            &self.architecture,
            &self.compute_capability,
        ] {
            require_bounded_text(value)?;
        }
        if self.count != 1 {
            return Err(PipelineError::Invalid("cloud accelerator count"));
        }
        Ok(())
    }
}

/// Identity that must be identical for every successful worker. Pod and
/// machine IDs are recorded separately so different hosts may be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodExecutionIdentity {
    pub executor_image: ComponentRef,
    pub executor_image_build: ComponentRef,
    pub runtime: ComponentRef,
    pub worker_binary: ComponentRef,
    pub model_artifact: ComponentRef,
    pub embedding_profile: ComponentRef,
    pub accelerator: RunpodAcceleratorIdentity,
    pub returned_model: String,
}

impl RunpodExecutionIdentity {
    fn validate(&self) -> Result<()> {
        self.executor_image.validate()?;
        self.executor_image_build.validate()?;
        self.runtime.validate()?;
        self.worker_binary.validate()?;
        self.model_artifact.validate()?;
        self.embedding_profile.validate()?;
        self.accelerator.validate()?;
        require_text(&self.returned_model)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodExpectedTaskOutput {
    pub task_id: String,
    pub task_ordinal: u32,
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub token_count: u64,
    pub result_key: SafeRelativePath,
    pub receipt_key: SafeRelativePath,
    pub report_key: SafeRelativePath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodWorkerAssignment {
    pub component_sha256: Digest,
    pub worker_id: String,
    /// Half-open range in the embedding plan's task array.
    pub task_start: u32,
    pub task_end: u32,
    /// Half-open document ordinal range covered by the assigned tasks.
    pub ordinal_start: u64,
    pub ordinal_end: u64,
    pub token_count: u64,
    pub tasks: Vec<RunpodExpectedTaskOutput>,
}

impl RunpodWorkerAssignment {
    fn validate(&self) -> Result<()> {
        require_identifier(&self.worker_id)?;
        for value in [
            u64::from(self.task_start),
            u64::from(self.task_end),
            self.ordinal_start,
            self.ordinal_end,
            self.token_count,
        ] {
            require_safe_u64(value)?;
        }
        if self.task_start >= self.task_end
            || self.ordinal_start >= self.ordinal_end
            || self.token_count == 0
            || usize::try_from(self.task_end - self.task_start).ok() != Some(self.tasks.len())
        {
            return Err(PipelineError::Invalid("cloud worker assignment range"));
        }
        let mut next_task = self.task_start;
        let mut next_ordinal = self.ordinal_start;
        let mut tokens = 0_u64;
        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for task in &self.tasks {
            require_identifier(&task.task_id)?;
            for value in [
                u64::from(task.task_ordinal),
                task.ordinal_start,
                task.ordinal_end,
                task.token_count,
            ] {
                require_safe_u64(value)?;
            }
            if task.task_ordinal != next_task
                || task.ordinal_start != next_ordinal
                || task.ordinal_start >= task.ordinal_end
                || task.token_count == 0
                || !ids.insert(task.task_id.as_str())
                || !keys.insert(task.result_key.as_str())
                || !keys.insert(task.receipt_key.as_str())
                || !keys.insert(task.report_key.as_str())
            {
                return Err(PipelineError::Invalid("cloud assigned task coverage"));
            }
            next_task = next_task
                .checked_add(1)
                .ok_or(PipelineError::Invalid("cloud task ordinal"))?;
            next_ordinal = task.ordinal_end;
            tokens = tokens
                .checked_add(task.token_count)
                .ok_or(PipelineError::Invalid("cloud assignment token count"))?;
        }
        if next_task != self.task_end
            || next_ordinal != self.ordinal_end
            || tokens != self.token_count
            || self.component_sha256 != component_digest(self)?
        {
            return Err(PipelineError::Invalid("cloud worker assignment binding"));
        }
        Ok(())
    }

    fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

/// Immutable upload and execution contract. It binds only portable artifacts;
/// provider credentials and endpoint details are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodEmbeddingBundle {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub prepared_corpus_sha256: Digest,
    pub plan_sha256: Digest,
    pub embedding_profile_sha256: Digest,
    pub tokenizer_sha256: Digest,
    pub model_sha256: Digest,
    pub document_count: u64,
    pub task_count: u32,
    pub total_tokens: u64,
    pub artifacts: RunpodBundleArtifacts,
    pub execution: RunpodExecutionIdentity,
    pub query_vector_output: RunpodExpectedQueryVectorOutput,
    pub assignments: Vec<RunpodWorkerAssignment>,
}

/// Deterministic keys written only by worker-0000 while its exact TEI process
/// is still warm. Content hashes are learned from the immutable completion
/// marker because model output does not exist when the bundle is sealed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodExpectedQueryVectorOutput {
    pub worker_id: String,
    pub manifest_key: SafeRelativePath,
    pub query_plan_key: SafeRelativePath,
    pub vectors_key: SafeRelativePath,
}

impl RunpodExpectedQueryVectorOutput {
    fn validate(&self) -> Result<()> {
        if self.worker_id != "worker-0000"
            || self.manifest_key.as_str() != "query-vectors/manifest.json"
            || self.query_plan_key.as_str() != "query-vectors/queries.jsonl"
            || self.vectors_key.as_str() != "query-vectors/vectors.f32le"
        {
            return Err(PipelineError::Invalid("cloud query vector output keys"));
        }
        Ok(())
    }
}

impl RunpodEmbeddingBundle {
    pub fn validate_against(
        &self,
        prepared: &PreparedCorpusManifest,
        plan: &EmbeddingPlanV2,
    ) -> Result<()> {
        prepared.validate()?;
        plan.validate_manifest_binding(prepared)?;
        require_safe_u64(self.document_count)?;
        require_safe_u64(self.total_tokens)?;
        if self.schema_version != RUNPOD_EMBEDDING_BUNDLE_SCHEMA
            || self.prepared_corpus_sha256 != prepared.component_sha256
            || self.plan_sha256 != plan.component_sha256
            || self.embedding_profile_sha256 != plan.embedding_profile.component.sha256
            || self.tokenizer_sha256 != plan.executable_tokenizer.artifact.sha256
            || self.model_sha256 != plan.embedding_profile.model_artifact.sha256
            || self.document_count != plan.document_count
            || usize::try_from(self.task_count).ok() != Some(plan.tasks.len())
            || self.total_tokens != plan.token_statistics.total_tokens
            || self.assignments.is_empty()
        {
            return Err(PipelineError::Invalid("cloud bundle plan binding"));
        }
        self.execution.validate()?;
        self.query_vector_output.validate()?;
        if self.execution.model_artifact != plan.embedding_profile.model_artifact
            || self.execution.embedding_profile != plan.embedding_profile.component
            || self.execution.executor_image_build.sha256
                != self.artifacts.executor_image_build.component_sha256
            || self.execution.worker_binary.sha256 != self.artifacts.worker_binary.component_sha256
        {
            return Err(PipelineError::Invalid("cloud execution identity binding"));
        }
        validate_artifacts(&self.artifacts, prepared, plan)?;
        validate_assignments(&self.assignments, plan)?;
        let mut keys = BTreeSet::new();
        for key in [
            self.query_vector_output.manifest_key.as_str(),
            self.query_vector_output.query_plan_key.as_str(),
            self.query_vector_output.vectors_key.as_str(),
        ] {
            if !keys.insert(key) {
                return Err(PipelineError::Invalid("cloud bundle duplicate object key"));
            }
        }
        for key in
            artifact_keys(&self.artifacts).chain(self.assignments.iter().flat_map(|assignment| {
                assignment.tasks.iter().flat_map(|task| {
                    [
                        task.result_key.as_str(),
                        task.receipt_key.as_str(),
                        task.report_key.as_str(),
                    ]
                })
            }))
        {
            if !keys.insert(key) {
                return Err(PipelineError::Invalid("cloud bundle duplicate object key"));
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("cloud bundle component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }

    /// Verify every local input object before upload. Prepared occurrence
    /// objects are not part of this embedding-only bundle.
    pub fn validate_input_files(
        &self,
        root: &Path,
        prepared: &PreparedCorpusManifest,
        plan: &EmbeddingPlanV2,
    ) -> Result<()> {
        self.validate_against(prepared, plan)?;
        for object in input_objects(&self.artifacts) {
            validate_cloud_object_file(root, object)?;
        }
        Ok(())
    }
}

/// Create the bundle and deterministic contiguous worker ranges. Boundaries
/// are chosen nearest each worker's cumulative share of planned tokens while
/// leaving at least one task for every remaining worker.
pub fn build_runpod_embedding_bundle(
    prepared: &PreparedCorpusManifest,
    plan: &EmbeddingPlanV2,
    artifacts: RunpodBundleArtifacts,
    execution: RunpodExecutionIdentity,
    worker_count: u32,
) -> Result<RunpodEmbeddingBundle> {
    prepared.validate()?;
    plan.validate_manifest_binding(prepared)?;
    if worker_count == 0
        || worker_count > MAX_WORKERS
        || usize::try_from(worker_count)
            .map_err(|_| PipelineError::Invalid("cloud worker count"))?
            > plan.tasks.len()
    {
        return Err(PipelineError::Invalid("cloud worker count"));
    }
    let boundaries = token_balanced_boundaries(plan, worker_count)?;
    let mut assignments = Vec::with_capacity(boundaries.len() - 1);
    for (worker, range) in boundaries.windows(2).enumerate() {
        let start = range[0];
        let end = range[1];
        let tasks = plan.tasks[start..end]
            .iter()
            .enumerate()
            .map(|(offset, task)| {
                Ok(RunpodExpectedTaskOutput {
                    task_id: task.task_id.clone(),
                    task_ordinal: u32::try_from(start + offset)
                        .map_err(|_| PipelineError::Invalid("cloud task ordinal"))?,
                    ordinal_start: task.ordinal_start,
                    ordinal_end: task.ordinal_end,
                    token_count: task.token_count,
                    result_key: task.result_path.clone(),
                    receipt_key: task.receipt_path.clone(),
                    report_key: SafeRelativePath::new(format!("reports/{}.json", task.task_id))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut assignment = RunpodWorkerAssignment {
            component_sha256: digest_bytes(b"unsealed"),
            worker_id: format!("worker-{worker:04}"),
            task_start: u32::try_from(start)
                .map_err(|_| PipelineError::Invalid("cloud task ordinal"))?,
            task_end: u32::try_from(end)
                .map_err(|_| PipelineError::Invalid("cloud task ordinal"))?,
            ordinal_start: tasks
                .first()
                .ok_or(PipelineError::Invalid("cloud assignment tasks"))?
                .ordinal_start,
            ordinal_end: tasks
                .last()
                .ok_or(PipelineError::Invalid("cloud assignment tasks"))?
                .ordinal_end,
            token_count: tasks.iter().try_fold(0_u64, |sum, task| {
                sum.checked_add(task.token_count)
                    .ok_or(PipelineError::Invalid("cloud assignment token count"))
            })?,
            tasks,
        };
        assignment.seal()?;
        assignments.push(assignment);
    }
    let mut bundle = RunpodEmbeddingBundle {
        schema_version: RUNPOD_EMBEDDING_BUNDLE_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed"),
        prepared_corpus_sha256: prepared.component_sha256.clone(),
        plan_sha256: plan.component_sha256.clone(),
        embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
        tokenizer_sha256: plan.executable_tokenizer.artifact.sha256.clone(),
        model_sha256: plan.embedding_profile.model_artifact.sha256.clone(),
        document_count: plan.document_count,
        task_count: u32::try_from(plan.tasks.len())
            .map_err(|_| PipelineError::Invalid("cloud task count"))?,
        total_tokens: plan.token_statistics.total_tokens,
        artifacts,
        execution,
        query_vector_output: RunpodExpectedQueryVectorOutput {
            worker_id: "worker-0000".into(),
            manifest_key: SafeRelativePath::new("query-vectors/manifest.json")?,
            query_plan_key: SafeRelativePath::new("query-vectors/queries.jsonl")?,
            vectors_key: SafeRelativePath::new("query-vectors/vectors.f32le")?,
        },
        assignments,
    };
    bundle.seal()?;
    bundle.validate_against(prepared, plan)?;
    Ok(bundle)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodMachineIdentity {
    pub pod_id: String,
    pub machine_id: String,
}

impl RunpodMachineIdentity {
    fn validate(&self) -> Result<()> {
        require_identifier(&self.pod_id)?;
        require_identifier(&self.machine_id)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerAttemptOutcome {
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodTaskOutput {
    pub task_id: String,
    pub result: CloudObjectRef,
    pub receipt: CloudObjectRef,
    pub report: CloudObjectRef,
}

/// Exact sealed query-vector objects published by worker-0000.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodQueryVectorSetOutput {
    pub component_sha256: Digest,
    pub manifest: CloudObjectRef,
    pub query_plan: CloudObjectRef,
    pub vectors: CloudObjectRef,
}

impl RunpodQueryVectorSetOutput {
    fn validate_against(&self, bundle: &RunpodEmbeddingBundle) -> Result<()> {
        self.manifest.validate()?;
        self.query_plan.validate()?;
        self.vectors.validate()?;
        let expected = &bundle.query_vector_output;
        if self.manifest.key != expected.manifest_key
            || self.query_plan.key != expected.query_plan_key
            || self.vectors.key != expected.vectors_key
            || self.query_plan.bytes != bundle.artifacts.query_plan.bytes
            || self.query_plan.sha256 != bundle.artifacts.query_plan.sha256
        {
            return Err(PipelineError::Invalid("cloud query vector output binding"));
        }
        Ok(())
    }
}

/// Marker for one bounded worker attempt. Failed attempts contain no task
/// outputs. Completed attempts must contain every assigned task exactly once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodWorkerAttemptMarker {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub bundle_sha256: Digest,
    pub assignment_sha256: Digest,
    pub worker_id: String,
    pub attempt_id: String,
    pub attempt_number: u32,
    pub outcome: WorkerAttemptOutcome,
    pub machine: RunpodMachineIdentity,
    pub execution: RunpodExecutionIdentity,
    pub started_at_ms: u64,
    pub completed_at_ms: u64,
    pub requests: u64,
    pub retries: u64,
    pub outputs: Vec<RunpodTaskOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_vector_set: Option<RunpodQueryVectorSetOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
}

impl RunpodWorkerAttemptMarker {
    pub fn validate_against(&self, bundle: &RunpodEmbeddingBundle) -> Result<()> {
        if self.schema_version != RUNPOD_WORKER_ATTEMPT_SCHEMA
            || self.bundle_sha256 != bundle.component_sha256
        {
            return Err(PipelineError::Invalid("cloud attempt bundle binding"));
        }
        require_identifier(&self.worker_id)?;
        require_identifier(&self.attempt_id)?;
        for value in [
            u64::from(self.attempt_number),
            self.started_at_ms,
            self.completed_at_ms,
            self.requests,
            self.retries,
        ] {
            require_safe_u64(value)?;
        }
        if self.attempt_number == 0
            || self.started_at_ms > self.completed_at_ms
            || self.retries > self.requests
        {
            return Err(PipelineError::Invalid("cloud attempt accounting"));
        }
        self.machine.validate()?;
        self.execution.validate()?;
        if self.execution != bundle.execution {
            return Err(PipelineError::Invalid("cloud attempt execution identity"));
        }
        let assignment = bundle
            .assignments
            .iter()
            .find(|value| value.worker_id == self.worker_id)
            .ok_or(PipelineError::Invalid("cloud attempt worker"))?;
        if self.assignment_sha256 != assignment.component_sha256 {
            return Err(PipelineError::Invalid("cloud attempt assignment binding"));
        }
        match self.outcome {
            WorkerAttemptOutcome::Completed => {
                if self.requests == 0 || self.failure_code.is_some() {
                    return Err(PipelineError::Invalid("cloud completed attempt state"));
                }
                validate_task_outputs(&self.outputs, assignment)?;
                if self.worker_id == bundle.query_vector_output.worker_id {
                    self.query_vector_set
                        .as_ref()
                        .ok_or(PipelineError::Invalid("cloud query vector output missing"))?
                        .validate_against(bundle)?;
                } else if self.query_vector_set.is_some() {
                    return Err(PipelineError::Invalid("cloud query vector output worker"));
                }
            }
            WorkerAttemptOutcome::Failed => {
                if !self.outputs.is_empty()
                    || self.query_vector_set.is_some()
                    || self
                        .failure_code
                        .as_deref()
                        .map(require_identifier)
                        .transpose()?
                        .is_none()
                {
                    return Err(PipelineError::Invalid("cloud failed attempt state"));
                }
            }
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("cloud attempt component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }

    pub fn canonical_object(&self) -> Result<CloudObjectRef> {
        let bytes = canonical_json_bytes(self)?;
        Ok(CloudObjectRef {
            // One deterministic, no-overwrite completion key lets the host
            // fetch successful markers without granting object-list access.
            key: SafeRelativePath::new(format!("attempts/{}/completed.json", self.worker_id))?,
            bytes: u64::try_from(bytes.len())
                .map_err(|_| PipelineError::Invalid("cloud marker bytes"))?,
            sha256: digest_bytes(&bytes),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodSelectedAttempt {
    pub worker_id: String,
    pub attempt_id: String,
    pub marker_component_sha256: Digest,
    pub marker: CloudObjectRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodRunReport {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub bundle_sha256: Digest,
    pub execution: RunpodExecutionIdentity,
    pub worker_count: u32,
    pub task_count: u32,
    pub document_count: u64,
    pub total_tokens: u64,
    pub vector_objects: u32,
    pub receipt_objects: u32,
    pub report_objects: u32,
    pub query_vector_set: RunpodQueryVectorSetOutput,
    pub selected_attempts: Vec<RunpodSelectedAttempt>,
}

impl RunpodRunReport {
    /// Validate exact-one completed coverage. `attempts` must contain precisely
    /// the markers selected by the report: missing and unreported extra markers
    /// are both rejected.
    pub fn validate_against(
        &self,
        bundle: &RunpodEmbeddingBundle,
        attempts: &[RunpodWorkerAttemptMarker],
    ) -> Result<()> {
        if self.schema_version != RUNPOD_RUN_REPORT_SCHEMA
            || self.bundle_sha256 != bundle.component_sha256
            || self.execution != bundle.execution
            || usize::try_from(self.worker_count).ok() != Some(bundle.assignments.len())
            || self.task_count != bundle.task_count
            || self.document_count != bundle.document_count
            || self.total_tokens != bundle.total_tokens
            || self.vector_objects != bundle.task_count
            || self.receipt_objects != bundle.task_count
            || self.report_objects != bundle.task_count
            || self.query_vector_set.validate_against(bundle).is_err()
            || self.selected_attempts.len() != bundle.assignments.len()
            || attempts.len() != self.selected_attempts.len()
        {
            return Err(PipelineError::Invalid("cloud report bundle binding"));
        }
        for value in [self.document_count, self.total_tokens] {
            require_safe_u64(value)?;
        }
        let refs = self
            .selected_attempts
            .iter()
            .map(|entry| (entry.marker_component_sha256.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        let loaded = attempts
            .iter()
            .map(|attempt| (attempt.component_sha256.as_str(), attempt))
            .collect::<BTreeMap<_, _>>();
        if refs.len() != self.selected_attempts.len() || loaded.len() != attempts.len() {
            return Err(PipelineError::Invalid("cloud report duplicate attempt"));
        }
        let mut workers = BTreeSet::new();
        let mut tasks = BTreeSet::new();
        for assignment in &bundle.assignments {
            let attempt = attempts
                .iter()
                .find(|attempt| attempt.worker_id == assignment.worker_id)
                .ok_or(PipelineError::Invalid("cloud report missing worker"))?;
            attempt.validate_against(bundle)?;
            if attempt.outcome != WorkerAttemptOutcome::Completed
                || !workers.insert(attempt.worker_id.as_str())
            {
                return Err(PipelineError::Invalid("cloud report worker coverage"));
            }
            if attempt.worker_id == bundle.query_vector_output.worker_id
                && attempt.query_vector_set.as_ref() != Some(&self.query_vector_set)
            {
                return Err(PipelineError::Invalid(
                    "cloud report query vector selection",
                ));
            }
            let selected = refs
                .get(attempt.component_sha256.as_str())
                .ok_or(PipelineError::Invalid("cloud report unselected attempt"))?;
            let canonical = attempt.canonical_object()?;
            if selected.worker_id != attempt.worker_id
                || selected.attempt_id != attempt.attempt_id
                || selected.marker != canonical
            {
                return Err(PipelineError::Invalid(
                    "cloud report attempt object binding",
                ));
            }
            for output in &attempt.outputs {
                if !tasks.insert(output.task_id.as_str()) {
                    return Err(PipelineError::Invalid("cloud report duplicate task"));
                }
            }
        }
        let expected = bundle
            .assignments
            .iter()
            .flat_map(|assignment| assignment.tasks.iter().map(|task| task.task_id.as_str()))
            .collect::<BTreeSet<_>>();
        if tasks != expected
            || loaded.keys().copied().collect::<BTreeSet<_>>() != refs.keys().copied().collect()
        {
            return Err(PipelineError::Invalid(
                "cloud report task or attempt coverage",
            ));
        }
        if self.component_sha256 != component_digest(self)? {
            return Err(PipelineError::Invalid("cloud report component digest"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }
}

/// Build a sealed report from exactly one completed attempt per assignment.
/// Failed attempts remain separate audit artifacts and never count toward
/// successful task coverage.
pub fn build_runpod_run_report(
    bundle: &RunpodEmbeddingBundle,
    attempts: &[RunpodWorkerAttemptMarker],
) -> Result<RunpodRunReport> {
    let selected_attempts = attempts
        .iter()
        .map(|attempt| {
            Ok(RunpodSelectedAttempt {
                worker_id: attempt.worker_id.clone(),
                attempt_id: attempt.attempt_id.clone(),
                marker_component_sha256: attempt.component_sha256.clone(),
                marker: attempt.canonical_object()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let query_vector_set = attempts
        .iter()
        .find(|attempt| attempt.worker_id == bundle.query_vector_output.worker_id)
        .and_then(|attempt| attempt.query_vector_set.clone())
        .ok_or(PipelineError::Invalid("cloud query vector output missing"))?;
    let mut report = RunpodRunReport {
        schema_version: RUNPOD_RUN_REPORT_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed"),
        bundle_sha256: bundle.component_sha256.clone(),
        execution: bundle.execution.clone(),
        worker_count: u32::try_from(bundle.assignments.len())
            .map_err(|_| PipelineError::Invalid("cloud worker count"))?,
        task_count: bundle.task_count,
        document_count: bundle.document_count,
        total_tokens: bundle.total_tokens,
        vector_objects: bundle.task_count,
        receipt_objects: bundle.task_count,
        report_objects: bundle.task_count,
        query_vector_set,
        selected_attempts,
    };
    report.seal()?;
    report.validate_against(bundle, attempts)?;
    Ok(report)
}

fn validate_artifacts(
    artifacts: &RunpodBundleArtifacts,
    prepared: &PreparedCorpusManifest,
    plan: &EmbeddingPlanV2,
) -> Result<()> {
    for artifact in [
        &artifacts.prepared_manifest,
        &artifacts.embedding_plan,
        &artifacts.embedding_profile,
        &artifacts.executor_image_build,
        &artifacts.executable_tokenizer,
        &artifacts.worker_binary,
        &artifacts.model_manifest,
    ] {
        artifact.validate()?;
    }
    artifacts.document_token_counts.validate()?;
    artifacts.conformance_fixture.validate()?;
    artifacts.query_plan.validate()?;
    if artifacts.prepared_manifest.component_sha256 != prepared.component_sha256
        || artifacts.embedding_plan.component_sha256 != plan.component_sha256
        || artifacts.embedding_profile.component_sha256 != plan.embedding_profile.component.sha256
        || artifacts.executable_tokenizer.component_sha256
            != plan.executable_tokenizer.artifact.sha256
        || artifacts.model_manifest.component_sha256 != plan.embedding_profile.model_artifact.sha256
        || artifacts.document_token_counts.bytes != plan.document_token_counts_object.bytes
        || artifacts.document_token_counts.sha256 != plan.document_token_counts_object.sha256
        || artifacts.model_objects.is_empty()
        || artifacts.query_plan.key.as_str() != "input/query/queries.jsonl"
        || artifacts.prepared_documents.len() != prepared.documents.len()
    {
        return Err(PipelineError::Invalid("cloud input artifact binding"));
    }
    let expected = prepared
        .documents
        .iter()
        .map(|entry| (entry.object.path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let supplied = artifacts
        .prepared_documents
        .iter()
        .map(|entry| (entry.prepared_path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != prepared.documents.len()
        || supplied.len() != artifacts.prepared_documents.len()
    {
        return Err(PipelineError::Invalid("cloud prepared document coverage"));
    }
    for (path, object) in expected {
        let artifact = supplied
            .get(path)
            .ok_or(PipelineError::Invalid("cloud missing prepared document"))?;
        artifact.object.validate()?;
        if artifact.object.bytes != object.object.bytes
            || artifact.object.sha256 != object.object.sha256
        {
            return Err(PipelineError::Invalid("cloud prepared document binding"));
        }
    }
    let mut model_keys = BTreeSet::new();
    for object in &artifacts.model_objects {
        object.validate()?;
        if !model_keys.insert(object.key.as_str()) {
            return Err(PipelineError::Invalid("cloud duplicate model object"));
        }
    }
    Ok(())
}

fn validate_assignments(
    assignments: &[RunpodWorkerAssignment],
    plan: &EmbeddingPlanV2,
) -> Result<()> {
    let mut next_task = 0_u32;
    let mut next_ordinal = 0_u64;
    let mut ids = BTreeSet::new();
    let mut tokens = 0_u64;
    for assignment in assignments {
        assignment.validate()?;
        if assignment.task_start != next_task
            || assignment.ordinal_start != next_ordinal
            || !ids.insert(assignment.worker_id.as_str())
        {
            return Err(PipelineError::Invalid("cloud assignment overlap or gap"));
        }
        for expected in &assignment.tasks {
            let task = plan
                .tasks
                .get(
                    usize::try_from(expected.task_ordinal)
                        .map_err(|_| PipelineError::Invalid("cloud task ordinal"))?,
                )
                .ok_or(PipelineError::Invalid("cloud extra assigned task"))?;
            if expected.task_id != task.task_id
                || expected.ordinal_start != task.ordinal_start
                || expected.ordinal_end != task.ordinal_end
                || expected.token_count != task.token_count
                || expected.result_key != task.result_path
                || expected.receipt_key != task.receipt_path
                || expected.report_key.as_str() != format!("reports/{}.json", task.task_id)
            {
                return Err(PipelineError::Invalid("cloud assigned task binding"));
            }
        }
        next_task = assignment.task_end;
        next_ordinal = assignment.ordinal_end;
        tokens = tokens
            .checked_add(assignment.token_count)
            .ok_or(PipelineError::Invalid("cloud assignment token count"))?;
    }
    if usize::try_from(next_task).ok() != Some(plan.tasks.len())
        || next_ordinal != plan.document_count
        || tokens != plan.token_statistics.total_tokens
    {
        return Err(PipelineError::Invalid("cloud assignment complete coverage"));
    }
    Ok(())
}

fn validate_task_outputs(
    outputs: &[RunpodTaskOutput],
    assignment: &RunpodWorkerAssignment,
) -> Result<()> {
    if outputs.len() != assignment.tasks.len() {
        return Err(PipelineError::Invalid("cloud task output coverage"));
    }
    let values = outputs
        .iter()
        .map(|output| (output.task_id.as_str(), output))
        .collect::<BTreeMap<_, _>>();
    if values.len() != outputs.len() {
        return Err(PipelineError::Invalid("cloud duplicate task output"));
    }
    for expected in &assignment.tasks {
        let output = values
            .get(expected.task_id.as_str())
            .ok_or(PipelineError::Invalid("cloud missing task output"))?;
        output.result.validate()?;
        output.receipt.validate()?;
        output.report.validate()?;
        if output.result.key != expected.result_key || output.receipt.key != expected.receipt_key {
            return Err(PipelineError::Invalid("cloud task output key"));
        }
        if output.report.key != expected.report_key {
            return Err(PipelineError::Invalid("cloud task report key"));
        }
    }
    Ok(())
}

fn token_balanced_boundaries(plan: &EmbeddingPlanV2, workers: u32) -> Result<Vec<usize>> {
    let worker_count =
        usize::try_from(workers).map_err(|_| PipelineError::Invalid("cloud worker count"))?;
    let mut prefix = Vec::with_capacity(plan.tasks.len() + 1);
    prefix.push(0_u64);
    for task in &plan.tasks {
        prefix.push(
            prefix
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(task.token_count)
                .ok_or(PipelineError::Invalid("cloud token total"))?,
        );
    }
    let total = *prefix
        .last()
        .ok_or(PipelineError::Invalid("cloud token total"))?;
    require_safe_u64(total)?;
    let mut boundaries = vec![0_usize];
    for worker in 1..worker_count {
        let minimum = boundaries[worker - 1] + 1;
        let maximum = plan.tasks.len() - (worker_count - worker);
        let target = u128::from(total) * worker as u128;
        let mut best = minimum;
        let mut best_distance = u128::MAX;
        for (candidate, cumulative_tokens) in
            prefix.iter().enumerate().take(maximum + 1).skip(minimum)
        {
            let scaled = u128::from(*cumulative_tokens) * worker_count as u128;
            let distance = scaled.abs_diff(target);
            if distance < best_distance {
                best = candidate;
                best_distance = distance;
            }
        }
        boundaries.push(best);
    }
    boundaries.push(plan.tasks.len());
    Ok(boundaries)
}

fn artifact_keys(artifacts: &RunpodBundleArtifacts) -> impl Iterator<Item = &str> {
    [
        &artifacts.prepared_manifest,
        &artifacts.embedding_plan,
        &artifacts.embedding_profile,
        &artifacts.executor_image_build,
        &artifacts.executable_tokenizer,
        &artifacts.worker_binary,
        &artifacts.model_manifest,
    ]
    .into_iter()
    .map(|artifact| artifact.object.key.as_str())
    .chain([
        artifacts.document_token_counts.key.as_str(),
        artifacts.conformance_fixture.key.as_str(),
        artifacts.query_plan.key.as_str(),
    ])
    .chain(
        artifacts
            .model_objects
            .iter()
            .map(|object| object.key.as_str()),
    )
    .chain(
        artifacts
            .prepared_documents
            .iter()
            .map(|artifact| artifact.object.key.as_str()),
    )
}

fn input_objects(artifacts: &RunpodBundleArtifacts) -> impl Iterator<Item = &CloudObjectRef> {
    [
        &artifacts.prepared_manifest,
        &artifacts.embedding_plan,
        &artifacts.embedding_profile,
        &artifacts.executor_image_build,
        &artifacts.executable_tokenizer,
        &artifacts.worker_binary,
        &artifacts.model_manifest,
    ]
    .into_iter()
    .map(|artifact| &artifact.object)
    .chain([
        &artifacts.document_token_counts,
        &artifacts.conformance_fixture,
        &artifacts.query_plan,
    ])
    .chain(artifacts.model_objects.iter())
    .chain(
        artifacts
            .prepared_documents
            .iter()
            .map(|artifact| &artifact.object),
    )
}

fn validate_cloud_object_file(root: &Path, object: &CloudObjectRef) -> Result<()> {
    object.validate()?;
    let path = resolve_existing_artifact(root, &object.key)?;
    let metadata = path.metadata()?;
    if !metadata.is_file() || metadata.len() != object.bytes {
        return Err(PipelineError::Invalid("cloud input object metadata"));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    if Digest::new(format!("{:x}", hasher.finalize()))? != object.sha256 {
        return Err(PipelineError::Invalid("cloud input object digest"));
    }
    Ok(())
}

fn require_identifier(value: &str) -> Result<()> {
    require_text(value)?;
    if value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PipelineError::Invalid("cloud identifier"));
    }
    Ok(())
}

fn require_bounded_text(value: &str) -> Result<()> {
    require_text(value)?;
    if value.len() > 256 || value.chars().any(char::is_control) {
        return Err(PipelineError::Invalid("cloud bounded text"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DocumentKind, EmbeddingProfileRef, ExecutableTokenizerRef, ObjectEntry,
        PREPARED_CORPUS_SCHEMA, PreparedDocumentObject, PreparedDocumentRow,
        PreparedOccurrenceObject, RelationAccounting, TokenBalanceOptions, TokenizerArtifactFormat,
        build_token_balanced_plan, canonical_digest, document_order_digest,
        embedding_input_order_digest,
    };
    use std::collections::BTreeMap;

    const TEST_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"é":2,"👩‍💻":3,"<unk>":4},"unk_token":"<unk>"}}"#;

    fn digest(byte: char) -> Digest {
        Digest::new(byte.to_string().repeat(64)).unwrap()
    }

    fn component(id: &str, version: &str, sha256: Digest) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: version.into(),
            sha256,
        }
    }

    fn fixture() -> (
        PreparedCorpusManifest,
        EmbeddingPlanV2,
        RunpodBundleArtifacts,
        RunpodExecutionIdentity,
    ) {
        let texts = ["a b", "a", "é 👩‍💻", "b b", "a"];
        let rows = texts
            .iter()
            .enumerate()
            .map(|(ordinal, text)| PreparedDocumentRow {
                document_ordinal: ordinal as u64,
                document_id: format!("doc-{ordinal:02}"),
                document_sha256: digest(char::from_digit(ordinal as u32, 16).unwrap()),
                semantic_text_sha256: digest_bytes(text.as_bytes()),
                semantic_text: (*text).into(),
                document_kind: DocumentKind::Activity,
                primary_relation: "events".into(),
                facets_json: "{}".into(),
                relations_json: "[\"events\"]".into(),
                occurrence_count: 1,
            })
            .collect::<Vec<_>>();
        let dataset = crate::DatasetIdentity {
            id: "dataset".into(),
            version: "1".into(),
            source_snapshot: component("snapshot", "1", digest('a')),
            mapping: component("mapping", "1", digest('b')),
            source_admission: vec![],
            included_relations: vec!["events".into()],
            excluded_relations: vec![],
            structured_only_relations: vec![],
        };
        let document_path = SafeRelativePath::new("documents/part.parquet").unwrap();
        let mut prepared = PreparedCorpusManifest {
            schema_version: PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: digest('0'),
            dataset: dataset.clone(),
            projection_policy: component("projection", "1", digest('c')),
            document_schema: component("doc-schema", "1", digest('d')),
            occurrence_schema: component("occ-schema", "1", digest('e')),
            preparation_implementation: component("prepare", "1", digest('f')),
            document_count: rows.len() as u64,
            occurrence_count: rows.len() as u64,
            document_order_sha256: document_order_digest(
                rows.iter().map(|row| row.document_id.as_str()),
            ),
            embedding_input_order_sha256: embedding_input_order_digest(&rows),
            documents: vec![PreparedDocumentObject {
                object: ObjectEntry {
                    path: document_path.clone(),
                    rows: rows.len() as u64,
                    bytes: 100,
                    sha256: digest('3'),
                    logical_order_sha256: canonical_digest(&rows).unwrap(),
                },
                ordinal: 0,
                first_document_id: rows.first().unwrap().document_id.clone(),
                last_document_id: rows.last().unwrap().document_id.clone(),
                embedding_input_order_sha256: embedding_input_order_digest(&rows),
            }],
            occurrences: vec![PreparedOccurrenceObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("occurrences/events/part.parquet").unwrap(),
                    rows: rows.len() as u64,
                    bytes: 1,
                    sha256: digest('4'),
                    logical_order_sha256: digest('5'),
                },
                ordinal: 0,
                relation: "events".into(),
            }],
            relation_accounting: BTreeMap::from([(
                "events".into(),
                RelationAccounting {
                    source_rows: rows.len() as u64,
                    searchable_occurrences: rows.len() as u64,
                    selected_occurrences: rows.len() as u64,
                    excluded_rows: 0,
                },
            )]),
        };
        prepared.seal().unwrap();
        let profile = EmbeddingProfileRef {
            component: component("profile", "1", digest('6')),
            model_artifact: component("model", "revision-a", digest('7')),
            tokenizer: component("tokenizer-logical", "revision-a", digest('8')),
            maximum_input_tokens: 4,
            pooling: "last".into(),
            normalization: "l2".into(),
            dimensions: 4,
            dtype: "f32le".into(),
            document_format: "{semantic_text}".into(),
        };
        let executable = ExecutableTokenizerRef {
            artifact: component(
                "tokenizer-file",
                "revision-a",
                digest_bytes(TEST_TOKENIZER.as_bytes()),
            ),
            format: TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: "revision-a".into(),
            target_tokenizer: profile.tokenizer.clone(),
            add_special_tokens: false,
            maximum_input_bytes: 128,
        };
        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            profile.clone(),
            executable.clone(),
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 3,
                maximum_task_documents: 3,
            },
        )
        .unwrap();
        let object = |key: &str, sha: Digest, bytes| CloudObjectRef {
            key: SafeRelativePath::new(key).unwrap(),
            bytes,
            sha256: sha,
        };
        let bound = |key: &str, component_sha256: Digest| CloudComponentArtifact {
            component_sha256: component_sha256.clone(),
            object: object(key, digest('1'), 10),
        };
        let artifacts = RunpodBundleArtifacts {
            prepared_manifest: bound("input/prepared.json", prepared.component_sha256.clone()),
            embedding_plan: bound("input/plan.json", plan.component_sha256.clone()),
            document_token_counts: object(
                "input/document-token-counts.u32le",
                plan.document_token_counts_object.sha256.clone(),
                plan.document_token_counts_object.bytes,
            ),
            embedding_profile: bound("input/profile.json", profile.component.sha256.clone()),
            executor_image_build: bound("input/executor-image-build.json", digest('e')),
            executable_tokenizer: bound("input/tokenizer.json", executable.artifact.sha256.clone()),
            conformance_fixture: object("input/conformance-fixture.json", digest('c'), 12),
            query_plan: object("input/query/queries.jsonl", digest('9'), 120),
            worker_binary: bound("input/bin/rag-runpod-worker", digest('b')),
            model_manifest: bound(
                "input/model/manifest.json",
                profile.model_artifact.sha256.clone(),
            ),
            model_objects: vec![object("input/model/model.gguf", digest('a'), 200)],
            prepared_documents: vec![CloudPreparedDocumentArtifact {
                prepared_path: document_path,
                object: object("input/documents/part.parquet", digest('3'), 100),
            }],
        };
        let execution = RunpodExecutionIdentity {
            executor_image: component("image", "1", digest('a')),
            executor_image_build: component("image-build", "1", digest('e')),
            runtime: component("runtime", "1", digest('b')),
            worker_binary: component("worker", "1", digest('b')),
            model_artifact: profile.model_artifact.clone(),
            embedding_profile: profile.component.clone(),
            accelerator: RunpodAcceleratorIdentity {
                provider: "runpod".into(),
                model: "NVIDIA A100 80GB PCIe".into(),
                architecture: "ampere-sm80".into(),
                compute_capability: "8.0".into(),
                count: 1,
            },
            returned_model: "model".into(),
        };
        (prepared, plan, artifacts, execution)
    }

    #[test]
    fn static_assignments_are_disjoint_complete_and_token_balanced() {
        let (prepared, plan, artifacts, execution) = fixture();
        let boundaries = token_balanced_boundaries(&plan, 2).unwrap();
        assert_eq!(boundaries, [0, 1, 3]);
        let bundle =
            build_runpod_embedding_bundle(&prepared, &plan, artifacts, execution, 2).unwrap();
        assert_eq!(bundle.assignments[0].task_start, 0);
        assert_eq!(bundle.assignments[0].task_end, 1);
        assert_eq!(bundle.assignments[1].task_start, 1);
        assert_eq!(bundle.assignments[1].task_end, 3);
        assert_eq!(
            bundle
                .assignments
                .iter()
                .map(|value| value.token_count)
                .sum::<u64>(),
            8
        );
        bundle.validate_against(&prepared, &plan).unwrap();
    }

    fn completed_attempts(bundle: &RunpodEmbeddingBundle) -> Vec<RunpodWorkerAttemptMarker> {
        bundle
            .assignments
            .iter()
            .enumerate()
            .map(|(ordinal, assignment)| {
                let outputs = assignment
                    .tasks
                    .iter()
                    .map(|task| RunpodTaskOutput {
                        task_id: task.task_id.clone(),
                        result: CloudObjectRef {
                            key: task.result_key.clone(),
                            bytes: 96,
                            sha256: digest('c'),
                        },
                        receipt: CloudObjectRef {
                            key: task.receipt_key.clone(),
                            bytes: 32,
                            sha256: digest('d'),
                        },
                        report: CloudObjectRef {
                            key: task.report_key.clone(),
                            bytes: 32,
                            sha256: digest('e'),
                        },
                    })
                    .collect();
                let mut marker = RunpodWorkerAttemptMarker {
                    schema_version: RUNPOD_WORKER_ATTEMPT_SCHEMA.into(),
                    component_sha256: digest('0'),
                    bundle_sha256: bundle.component_sha256.clone(),
                    assignment_sha256: assignment.component_sha256.clone(),
                    worker_id: assignment.worker_id.clone(),
                    attempt_id: format!("attempt-{}", ordinal + 1),
                    attempt_number: 1,
                    outcome: WorkerAttemptOutcome::Completed,
                    machine: RunpodMachineIdentity {
                        pod_id: format!("pod-{ordinal}"),
                        machine_id: format!("machine-{ordinal}"),
                    },
                    execution: bundle.execution.clone(),
                    started_at_ms: 100,
                    completed_at_ms: 200,
                    requests: 1,
                    retries: 0,
                    outputs,
                    query_vector_set: (ordinal == 0).then(|| RunpodQueryVectorSetOutput {
                        component_sha256: digest('8'),
                        manifest: CloudObjectRef {
                            key: bundle.query_vector_output.manifest_key.clone(),
                            bytes: 512,
                            sha256: digest('7'),
                        },
                        query_plan: CloudObjectRef {
                            key: bundle.query_vector_output.query_plan_key.clone(),
                            bytes: bundle.artifacts.query_plan.bytes,
                            sha256: bundle.artifacts.query_plan.sha256.clone(),
                        },
                        vectors: CloudObjectRef {
                            key: bundle.query_vector_output.vectors_key.clone(),
                            bytes: 16_384,
                            sha256: digest('6'),
                        },
                    }),
                    failure_code: None,
                };
                marker.seal().unwrap();
                marker.validate_against(bundle).unwrap();
                marker
            })
            .collect()
    }

    #[test]
    fn completed_workers_may_use_different_machines_with_one_exact_accelerator() {
        let (prepared, plan, artifacts, execution) = fixture();
        let bundle =
            build_runpod_embedding_bundle(&prepared, &plan, artifacts, execution, 2).unwrap();
        let attempts = completed_attempts(&bundle);
        assert_ne!(attempts[0].machine, attempts[1].machine);
        assert_eq!(
            attempts[0].execution.accelerator,
            attempts[1].execution.accelerator
        );
        let report = build_runpod_run_report(&bundle, &attempts).unwrap();
        report.validate_against(&bundle, &attempts).unwrap();

        let mut wrong_query_vectors = report.clone();
        wrong_query_vectors.query_vector_set.vectors.sha256 = digest('f');
        wrong_query_vectors.seal().unwrap();
        assert!(
            wrong_query_vectors
                .validate_against(&bundle, &attempts)
                .is_err()
        );

        let mut wrong_execution = attempts[0].clone();
        wrong_execution.execution.runtime.sha256 = digest('e');
        wrong_execution.seal().unwrap();
        assert!(wrong_execution.validate_against(&bundle).is_err());

        let mut mixed_gpu = attempts[1].clone();
        mixed_gpu.execution.accelerator.model = "NVIDIA H100 80GB HBM3".into();
        mixed_gpu.execution.accelerator.architecture = "hopper-sm90".into();
        mixed_gpu.execution.accelerator.compute_capability = "9.0".into();
        mixed_gpu.seal().unwrap();
        assert!(mixed_gpu.validate_against(&bundle).is_err());
    }

    #[test]
    fn coverage_rejects_overlap_gap_extra_output_extra_attempt_and_unsafe_integer() {
        let (prepared, plan, artifacts, execution) = fixture();
        let bundle =
            build_runpod_embedding_bundle(&prepared, &plan, artifacts, execution, 2).unwrap();
        let attempts = completed_attempts(&bundle);
        let report = build_runpod_run_report(&bundle, &attempts).unwrap();

        let mut overlapping = bundle.clone();
        overlapping.assignments[1] = overlapping.assignments[0].clone();
        overlapping.assignments[1].worker_id = "worker-overlap".into();
        overlapping.assignments[1].seal().unwrap();
        overlapping.seal().unwrap();
        assert!(overlapping.validate_against(&prepared, &plan).is_err());

        let mut extra_output = attempts[0].clone();
        extra_output.outputs.push(extra_output.outputs[0].clone());
        extra_output.seal().unwrap();
        assert!(extra_output.validate_against(&bundle).is_err());

        let mut unsafe_integer = attempts[0].clone();
        unsafe_integer.completed_at_ms = 9_007_199_254_740_992;
        unsafe_integer.seal().unwrap();
        assert!(unsafe_integer.validate_against(&bundle).is_err());

        let mut extra_attempts = attempts.clone();
        let mut extra = attempts[0].clone();
        extra.attempt_id = "attempt-extra".into();
        extra.attempt_number = 2;
        extra.seal().unwrap();
        extra_attempts.push(extra);
        assert!(report.validate_against(&bundle, &extra_attempts).is_err());

        assert!(serde_json::from_str::<CloudObjectRef>(
            r#"{"key":"../escape","bytes":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#
        )
        .is_err());
    }

    #[test]
    fn schemas_are_strict_json_and_name_the_public_contracts() {
        for (text, id) in [
            (
                include_str!("../schema/runpod-embedding-bundle.v1.schema.json"),
                "https://livefire.dev/rag/runpod-embedding-bundle.v1.schema.json",
            ),
            (
                include_str!("../schema/runpod-worker-attempt.v1.schema.json"),
                "https://livefire.dev/rag/runpod-worker-attempt.v1.schema.json",
            ),
            (
                include_str!("../schema/runpod-run-report.v1.schema.json"),
                "https://livefire.dev/rag/runpod-run-report.v1.schema.json",
            ),
        ] {
            let value: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(value["$id"], id);
            assert_eq!(value["additionalProperties"], false);
            assert_eq!(
                value["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert_closed_objects(&value);
            assert_eq!(
                value.pointer("/$defs/safeInteger/maximum"),
                Some(&serde_json::json!(9_007_199_254_740_991_u64))
            );
        }
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
            serde_json::Value::Array(values) => {
                for child in values {
                    assert_closed_objects(child);
                }
            }
            _ => {}
        }
    }
}
