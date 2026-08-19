#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    fs::{self, File},
    future::Future,
    io::{Read, Write},
    path::{Path, PathBuf},
    pin::Pin,
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rag_embedding::{
    BearerAuthorization, EmbeddingProfile, EmbeddingTaskOptions, EmbeddingTaskReport,
    EmbeddingTaskStats, IdentifiedEmbedder, OpenAiCompatibleOptions, RetryPolicy,
    TeiCheckpointProfileV3, TeiEmbedder, adapt_model_vector, canonical_f32_vectors,
    execute_embedding_task_reported, parse_tei_checkpoint_profile_v3,
    parse_tei_model_artifact_set_v1, prepare_embedding_task_part, try_compose_query,
    verify_embedding_task_part,
};
use rag_pipeline::{
    CloudObjectRef, ComponentRef, Digest, EmbeddingPlanV2, ExactTokenizer, ExecutableTokenizerRef,
    ExecutorReceipt, PreparedCorpusManifest, PreparedDocumentRow, QueryVectorExecutionBinding,
    QueryVectorSetInput, RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA, RUNPOD_WORKER_ATTEMPT_SCHEMA,
    RunpodAcceleratorIdentity, RunpodEmbeddingBundle, RunpodExecutorImageBuildReceipt,
    RunpodMachineIdentity, RunpodQueryVectorSetOutput, RunpodStorageChallengeFailure,
    RunpodStorageChallengeResponse, RunpodTaskOutput, RunpodTeiArtifactObject,
    RunpodTeiConformanceCandidate, RunpodTeiConformanceOutcome, RunpodTeiConformanceResult,
    RunpodTeiMachineIdentity, RunpodTeiNormalizedOutput, RunpodWorkerAssignment,
    RunpodWorkerAttemptMarker, RunpodWorkerRuntimeEvent, SafeRelativePath, SealedQueryVectorSet,
    TokenizerArtifactFormat, VECTOR_RECEIPT_SCHEMA, VectorObject, VectorResultReceipt,
    WorkerAttemptOutcome, canonical_digest, canonical_json_bytes, digest_bytes,
    embedding_input_order_digest, format_document_input_exact, query_vector_plan_queries,
    read_json, read_prepared_documents, resolve_existing_artifact, resolve_output_artifact,
    write_query_vector_set,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use tokio::{process::Child, time::sleep};

const TEI_ENTRYPOINT: &str = "/entrypoint.sh";
const PINNED_TEI_IMAGE_SHA256: &str =
    "144aaa80ddcb520d49df83f915dc188ddd7cc6b1b3b9684a829c21dd39cbe3c5";
const PINNED_TEI_COMPUTE_CAPABILITY: &str = "12.0";
const MAX_OBSERVATION_BYTES: u64 = 64 * 1024;
const MAX_CONTROL_BYTES: u64 = 16 * 1024 * 1024;
const WORKSPACE_ROOT: &str = "/workspace";
const WORKER_UID: u32 = 1000;
const WORKER_GID: u32 = 1000;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("worker input is invalid")]
    Invalid(&'static str),
    #[error("worker input/output failed")]
    Io(#[from] std::io::Error),
    #[error("worker JSON failed")]
    Json(#[from] serde_json::Error),
    #[error("embedding execution failed")]
    Embedding(#[from] rag_embedding::EmbeddingError),
    #[error("model hydration failed")]
    ModelHydration(#[from] reqwest::Error),
    #[error("pipeline validation failed")]
    Pipeline(#[from] rag_pipeline::PipelineError),
}

impl WorkerError {
    #[must_use]
    pub fn public_code(&self) -> &'static str {
        match self {
            Self::Invalid(code) => code,
            Self::Io(_) => "io_failure",
            Self::Json(_) => "json_failure",
            Self::Embedding(_) => "embedding_failure",
            Self::ModelHydration(_) => "model_hydration_failure",
            Self::Pipeline(_) => "contract_failure",
        }
    }
}

type Result<T> = std::result::Result<T, WorkerError>;

/// Try to restrict the selected run directory, then permanently leave root
/// before any model or result processing begins. Some network-volume
/// filesystems do not implement `chown`; in that case the mandatory probe
/// after the privilege drop is the admission test for the mounted prefix.
pub fn prepare_runtime_storage(root: &Path) -> Result<PathBuf> {
    let root = validate_workspace_run_root(root)?;
    let effective_uid = nix::unistd::geteuid();
    if effective_uid.is_root() {
        let ownership_hardened = nix::unistd::chown(
            &root,
            Some(nix::unistd::Uid::from_raw(WORKER_UID)),
            Some(nix::unistd::Gid::from_raw(WORKER_GID)),
        )
        .is_ok();
        #[cfg(unix)]
        if ownership_hardened {
            use std::os::unix::fs::PermissionsExt as _;
            // Permission tightening is defense in depth. The post-drop probe
            // below remains mandatory even when this succeeds.
            let _ = fs::set_permissions(&root, fs::Permissions::from_mode(0o700));
        }
        clear_supplementary_groups()?;
        nix::unistd::setgid(nix::unistd::Gid::from_raw(WORKER_GID))
            .map_err(|_| WorkerError::Invalid("privilege_drop"))?;
        nix::unistd::setuid(nix::unistd::Uid::from_raw(WORKER_UID))
            .map_err(|_| WorkerError::Invalid("privilege_drop"))?;
    } else if effective_uid.as_raw() != WORKER_UID {
        return Err(WorkerError::Invalid("worker_user"));
    }
    if nix::unistd::geteuid().as_raw() != WORKER_UID
        || nix::unistd::getegid().as_raw() != WORKER_GID
    {
        return Err(WorkerError::Invalid("privilege_drop"));
    }
    verify_permanent_privilege_drop()?;
    #[cfg(unix)]
    nix::sys::stat::umask(nix::sys::stat::Mode::from_bits_truncate(0o077));
    storage_probe(root.clone())?;
    Ok(root)
}

#[cfg(target_os = "linux")]
fn clear_supplementary_groups() -> Result<()> {
    nix::unistd::setgroups(&[]).map_err(|_| WorkerError::Invalid("privilege_drop"))
}

#[cfg(not(target_os = "linux"))]
fn clear_supplementary_groups() -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_permanent_privilege_drop() -> Result<()> {
    nix::sys::prctl::set_no_new_privs().map_err(|_| WorkerError::Invalid("privilege_drop"))?;
    let users = nix::unistd::getresuid().map_err(|_| WorkerError::Invalid("privilege_drop"))?;
    let groups = nix::unistd::getresgid().map_err(|_| WorkerError::Invalid("privilege_drop"))?;
    let supplementary =
        nix::unistd::getgroups().map_err(|_| WorkerError::Invalid("privilege_drop"))?;
    if [users.real, users.effective, users.saved]
        .into_iter()
        .any(|value| value.as_raw() != WORKER_UID)
        || [groups.real, groups.effective, groups.saved]
            .into_iter()
            .any(|value| value.as_raw() != WORKER_GID)
        || !supplementary.is_empty()
    {
        return Err(WorkerError::Invalid("privilege_drop"));
    }
    let status = fs::read_to_string("/proc/self/status")?;
    for field in ["CapInh", "CapPrm", "CapEff", "CapAmb"] {
        let value = status
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{field}:\t")))
            .ok_or(WorkerError::Invalid("privilege_drop"))?;
        if value.bytes().any(|byte| byte != b'0') {
            return Err(WorkerError::Invalid("privilege_drop"));
        }
    }
    if !status.lines().any(|line| line == "NoNewPrivs:\t1") {
        return Err(WorkerError::Invalid("privilege_drop"));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn verify_permanent_privilege_drop() -> Result<()> {
    Ok(())
}

/// Create a nested directory, then exercise both immutable hard-link and
/// atomic-rename publication under the exact run directory.
pub fn storage_probe(root: PathBuf) -> Result<()> {
    let nonce = unique_stage_nonce()?;
    let probe_dir = root.join(format!(".storage-probe-{nonce}"));
    fs::create_dir(&probe_dir)?;
    let hard_linked = probe_dir.join("hard-link.complete");
    let staged = probe_dir.join("rename.partial");
    let published = probe_dir.join("rename.complete");
    let payload = b"livefire-rag-storage-probe-v1\n";
    let probe = (|| -> Result<()> {
        publish_no_overwrite(&hard_linked, payload)?;
        if fs::read(&hard_linked)? != payload {
            return Err(WorkerError::Invalid("storage_probe"));
        }
        let mut file = File::options().write(true).create_new(true).open(&staged)?;
        file.write_all(payload)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&staged, &published)?;
        if fs::read(&published)? != payload {
            return Err(WorkerError::Invalid("storage_probe"));
        }
        fs::remove_file(&hard_linked)?;
        fs::remove_file(&published)?;
        fs::remove_dir(&probe_dir)?;
        File::open(&root)?.sync_all()?;
        Ok(())
    })();
    if probe.is_err() {
        let _ = fs::remove_file(&hard_linked);
        let _ = fs::remove_file(&staged);
        let _ = fs::remove_file(&published);
        let _ = fs::remove_dir(&probe_dir);
    }
    probe
}

/// Verify every named staged object's exact byte count and SHA-256 after the
/// process has dropped privileges. Each flat specification is PATH, BYTES,
/// SHA256; paths must be sorted and unique.
pub fn verify_storage_objects(root: &Path, required_objects: &[String]) -> Result<(usize, u64)> {
    let mut chunks = required_objects.chunks_exact(3);
    let mut previous: Option<&str> = None;
    let mut bytes = 0_u64;
    let mut count = 0_usize;
    for specification in &mut chunks {
        let relative = &specification[0];
        if previous.is_some_and(|value| value >= relative.as_str()) {
            return Err(WorkerError::Invalid("required_object_order"));
        }
        previous = Some(relative);
        let expected_bytes = specification[1]
            .parse::<u64>()
            .map_err(|_| WorkerError::Invalid("required_object_bytes"))?;
        let expected_sha = Digest::new(specification[2].clone())?;
        let safe = SafeRelativePath::new(relative.clone())?;
        let path = resolve_existing_artifact(root, &safe)?;
        if !fs::metadata(&path)?.is_file() {
            return Err(WorkerError::Invalid("required_object_type"));
        }
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut copied = 0_u64;
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            copied = copied
                .checked_add(
                    u64::try_from(read)
                        .map_err(|_| WorkerError::Invalid("required_object_bytes"))?,
                )
                .ok_or(WorkerError::Invalid("required_object_bytes"))?;
        }
        if copied != expected_bytes || hex_bytes(&hasher.finalize()) != expected_sha.as_str() {
            return Err(WorkerError::Invalid("required_object_digest"));
        }
        bytes = bytes
            .checked_add(copied)
            .ok_or(WorkerError::Invalid("required_object_bytes"))?;
        count += 1;
    }
    if !chunks.remainder().is_empty() {
        return Err(WorkerError::Invalid("required_object_shape"));
    }
    Ok((count, bytes))
}

fn unique_stage_nonce() -> Result<String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| WorkerError::Invalid("system_time"))?
        .as_nanos();
    Ok(format!("{}-{nanos}", std::process::id()))
}

fn validate_workspace_run_root(root: &Path) -> Result<PathBuf> {
    validate_run_root_under(Path::new(WORKSPACE_ROOT), root)
}

fn validate_run_root_under(workspace_path: &Path, root: &Path) -> Result<PathBuf> {
    if !workspace_path.is_absolute()
        || !root.is_absolute()
        || fs::symlink_metadata(workspace_path)?
            .file_type()
            .is_symlink()
    {
        return Err(WorkerError::Invalid("run_root"));
    }
    let workspace = fs::canonicalize(workspace_path)?;
    if root == workspace_path || !root.starts_with(workspace_path) {
        return Err(WorkerError::Invalid("run_root"));
    }
    let relative = root
        .strip_prefix(workspace_path)
        .map_err(|_| WorkerError::Invalid("run_root"))?;
    let mut current = workspace_path.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(WorkerError::Invalid("run_root"));
        };
        current.push(component);
        if fs::symlink_metadata(&current)?.file_type().is_symlink() {
            return Err(WorkerError::Invalid("run_root"));
        }
    }
    let root = fs::canonicalize(root)?;
    if root == workspace || !root.starts_with(&workspace) || !root.is_dir() {
        return Err(WorkerError::Invalid("run_root"));
    }
    Ok(root)
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub root: PathBuf,
    pub bundle: String,
    pub worker_id: String,
    pub assignment_count: u32,
    pub attempt_id: String,
    pub attempt_number: u32,
    pub observation: String,
    pub observation_wait_seconds: u64,
    pub port: u16,
    pub batch_size: usize,
    pub requests_in_flight: usize,
    pub health_wait_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct ConformanceOptions {
    pub root: PathBuf,
    pub candidate: String,
    pub run_id: String,
    pub observation: String,
    pub observation_wait_seconds: u64,
    pub port: u16,
    pub health_wait_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct StorageChallengeOptions {
    pub root: PathBuf,
    pub executor_image: String,
    pub challenge: String,
    pub challenge_bytes: u64,
    pub challenge_sha256: String,
    pub response: String,
    pub wait_seconds: u64,
}

/// Wait for one exact host-uploaded object, read it through the mounted
/// filesystem, then publish the content-bound response using the same
/// immutable hard-link publication used by production worker outputs.
pub async fn storage_challenge(options: StorageChallengeOptions) -> Result<()> {
    if options.wait_seconds == 0 || options.wait_seconds > 3600 {
        return Err(WorkerError::Invalid("storage_challenge_wait"));
    }
    let challenge = CloudObjectRef {
        key: SafeRelativePath::new(options.challenge)?,
        bytes: options.challenge_bytes,
        sha256: Digest::new(options.challenge_sha256)?,
    };
    let response_key = SafeRelativePath::new(options.response)?;
    if challenge.bytes == 0
        || challenge.bytes > 1024 * 1024
        || challenge.key == response_key
        || resolve_output_artifact(&options.root, &response_key)?.exists()
    {
        return Err(WorkerError::Invalid("storage_challenge_contract"));
    }
    let expected = RunpodStorageChallengeResponse::new(options.executor_image, challenge.clone())?;
    let started = Instant::now();
    loop {
        match resolve_existing_artifact(&options.root, &challenge.key) {
            Ok(_) => {
                verified_object(&options.root, &challenge)?;
                break;
            }
            Err(rag_pipeline::PipelineError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound
                    && started.elapsed() < Duration::from_secs(options.wait_seconds) =>
            {
                sleep(Duration::from_millis(250)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
    let output = resolve_output_artifact(&options.root, &response_key)?;
    publish_no_overwrite(&output, &canonical_json_bytes(&expected)?)
}

/// Best-effort diagnostic publication for a failed storage challenge. This is
/// deliberately separate from the success path and can run either before or
/// after the permanent privilege drop, provided the exact run root itself is
/// valid and writable.
pub fn publish_storage_challenge_failure(
    root: &Path,
    failure_prefix: &str,
    executor_image: String,
    challenge: String,
    challenge_bytes: u64,
    challenge_sha256: String,
    failure_code: &str,
) -> Result<()> {
    if !rag_pipeline::RUNPOD_STORAGE_CHALLENGE_FAILURE_CODES.contains(&failure_code) {
        return Err(WorkerError::Invalid("storage_challenge_failure_code"));
    }
    let root = validate_workspace_run_root(root)?;
    let challenge = CloudObjectRef {
        key: SafeRelativePath::new(challenge)?,
        bytes: challenge_bytes,
        sha256: Digest::new(challenge_sha256)?,
    };
    let failure = RunpodStorageChallengeFailure::new(executor_image, challenge)?;
    let key = SafeRelativePath::new(format!("{failure_prefix}-{failure_code}.json"))?;
    let output = resolve_output_artifact(&root, &key)?;
    publish_no_overwrite(&output, &canonical_json_bytes(&failure)?)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerObservation {
    pub schema_version: String,
    pub machine: RunpodMachineIdentity,
    pub accelerator: RunpodAcceleratorIdentity,
}

impl WorkerObservation {
    fn validate(&self, expected: &RunpodAcceleratorIdentity) -> Result<()> {
        if self.schema_version != "livefire.rag.runpod-worker-observation/1"
            || self.accelerator != *expected
            || !valid_identifier(&self.machine.pod_id)
            || !valid_identifier(&self.machine.machine_id)
        {
            return Err(WorkerError::Invalid("observation_binding"));
        }
        Ok(())
    }
}

struct ValidatedWork {
    root: PathBuf,
    bundle: RunpodEmbeddingBundle,
    assignment: RunpodWorkerAssignment,
    additional_assignments: Vec<RunpodWorkerAssignment>,
    prepared: PreparedCorpusManifest,
    plan: EmbeddingPlanV2,
    backend_version: String,
    checkpoint_dtype: String,
    maximum_batch_items: u32,
    maximum_batch_tokens: u64,
    maximum_concurrent_requests: u32,
    served_model: String,
    policy_bytes: Vec<u8>,
    compact_profile: EmbeddingProfile,
    fixture_bytes: Vec<u8>,
    query_plan_path: PathBuf,
    query_plan_bytes: Vec<u8>,
    model_dir: PathBuf,
    document_objects: BTreeMap<String, CloudObjectRef>,
    observation: WorkerObservation,
}

struct RuntimeReporter {
    root: PathBuf,
    bundle_file_sha256: Digest,
    worker_id: String,
    attempt_id: String,
    attempt_number: u32,
    sequence: u32,
    current_phase: &'static str,
}

impl RuntimeReporter {
    fn open(options: &RunOptions) -> Result<Self> {
        let root = fs::canonicalize(&options.root)?;
        let bundle_path = contained_existing(&root, &options.bundle)?;
        ensure_bounded(&bundle_path, MAX_CONTROL_BYTES)?;
        Ok(Self {
            root,
            bundle_file_sha256: digest_bytes(&fs::read(bundle_path)?),
            worker_id: options.worker_id.clone(),
            attempt_id: options.attempt_id.clone(),
            attempt_number: options.attempt_number,
            sequence: 0,
            current_phase: "worker_started",
        })
    }

    fn begin(&mut self, phase: &'static str) {
        self.current_phase = phase;
    }

    fn complete(&mut self, phase: &'static str) -> Result<()> {
        self.begin(phase);
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(WorkerError::Invalid("runtime_event_sequence"))?;
        let event = RunpodWorkerRuntimeEvent::progress(
            self.bundle_file_sha256.clone(),
            self.worker_id.clone(),
            self.attempt_id.clone(),
            self.attempt_number,
            self.sequence,
            phase,
        )?;
        let key = SafeRelativePath::new(format!(
            "runtime/{}/attempts/{}/phases/{:02}-{phase}.json",
            self.worker_id, self.attempt_id, self.sequence
        ))?;
        let path = resolve_output_artifact(&self.root, &key)?;
        publish_no_overwrite(&path, &canonical_json_bytes(&event)?)
    }

    fn fail(&mut self, failure_code: &str) -> Result<()> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(WorkerError::Invalid("runtime_event_sequence"))?;
        let event = RunpodWorkerRuntimeEvent::failure(
            self.bundle_file_sha256.clone(),
            self.worker_id.clone(),
            self.attempt_id.clone(),
            self.attempt_number,
            self.sequence,
            self.current_phase,
            failure_code,
        )?;
        let key = SafeRelativePath::new(format!(
            "runtime/{}/attempts/{}/failed.json",
            self.worker_id, self.attempt_id
        ))?;
        let path = resolve_output_artifact(&self.root, &key)?;
        publish_no_overwrite(&path, &canonical_json_bytes(&event)?)
    }
}

pub async fn run(options: RunOptions) -> Result<()> {
    validate_cli(&options)?;
    let mut reporter = RuntimeReporter::open(&options)?;
    reporter.complete("worker_started")?;
    let result = run_reported(&options, &mut reporter).await;
    if let Err(error) = &result {
        let _ = reporter.fail(error.public_code());
    }
    result
}

async fn run_reported(options: &RunOptions, reporter: &mut RuntimeReporter) -> Result<()> {
    let executable = std::env::current_exe()?;
    let mut work = ValidatedWork::load(options, &executable, reporter).await?;
    reporter.begin("tei_started");
    let mut tei = TeiProcess::start(
        &work.model_dir,
        options.port,
        &work.checkpoint_dtype,
        work.maximum_batch_items,
        work.maximum_batch_tokens,
        work.maximum_concurrent_requests as usize,
        &work.served_model,
    )?;
    reporter.complete("tei_started")?;
    let endpoint = format!("http://127.0.0.1:{}", options.port);
    let embedder = Arc::new(TeiEmbedder::checkpoint_profile_loopback(
        &endpoint,
        &work.policy_bytes,
        BearerAuthorization::None,
    )?);
    let health = wait_for_health(
        &embedder,
        &mut tei,
        Duration::from_secs(options.health_wait_seconds),
    )
    .await;
    if let Err(error) = health {
        tei.stop().await;
        return Err(error);
    }
    reporter.complete("tei_healthy")?;
    let mut assignments = Vec::with_capacity(1 + work.additional_assignments.len());
    assignments.push(work.assignment.clone());
    assignments.append(&mut work.additional_assignments);
    let mut outcome = Ok(());
    for assignment in assignments {
        work.assignment = assignment;
        if let Err(error) = execute_assignment(
            &work,
            Arc::clone(&embedder),
            &options.attempt_id,
            options.attempt_number,
            options.batch_size,
            options.requests_in_flight,
            Some(&mut *reporter),
        )
        .await
        {
            outcome = Err(error);
            break;
        }
    }
    tei.stop().await;
    outcome
}

pub async fn conformance(options: ConformanceOptions) -> Result<()> {
    if !valid_identifier(&options.run_id)
        || options.port == 0
        || options.health_wait_seconds == 0
        || options.health_wait_seconds > 7_200
    {
        return Err(WorkerError::Invalid("conformance_arguments"));
    }
    let root = fs::canonicalize(&options.root)?;
    let candidate_path = contained_existing(&root, &options.candidate)?;
    ensure_bounded(&candidate_path, MAX_CONTROL_BYTES)?;
    let candidate: RunpodTeiConformanceCandidate = read_json(&candidate_path)?;
    candidate.validate()?;
    if candidate.tei_image.component.sha256.as_str() != PINNED_TEI_IMAGE_SHA256
        || candidate.tei_image.digest != format!("sha256:{PINNED_TEI_IMAGE_SHA256}")
        || candidate.executor_image.component == candidate.tei_image.component
        || candidate.executor_image.repository == candidate.tei_image.repository
        || candidate.accelerator.compute_capability != PINNED_TEI_COMPUTE_CAPABILITY
    {
        return Err(WorkerError::Invalid("tei_image_accelerator_binding"));
    }

    let executable = std::env::current_exe()?;
    let staged_worker = verified_tei_object(&root, &candidate.worker_binary.object)?;
    if file_digest(&staged_worker)? != candidate.worker_binary.component_sha256
        || file_digest(&executable)? != candidate.worker_binary.component_sha256
    {
        return Err(WorkerError::Invalid("worker_binary_binding"));
    }
    let build_receipt_path = verified_tei_object(&root, &candidate.executor_image_build.object)?;
    let build_receipt: RunpodExecutorImageBuildReceipt = read_json(&build_receipt_path)?;
    build_receipt.validate()?;
    if build_receipt.component_sha256 != candidate.executor_image_build.component.sha256
        || build_receipt.executor_image != candidate.executor_image
        || build_receipt.tei_base_image != candidate.tei_image
        || build_receipt.worker_binary != candidate.worker_binary
    {
        return Err(WorkerError::Invalid("executor_image_build_binding"));
    }
    verified_tei_object(&root, &build_receipt.dockerfile)?;
    let model_manifest_path = verified_tei_object(&root, &candidate.model_manifest.object)?;
    let model_manifest = parse_tei_model_artifact_set_v1(&fs::read(model_manifest_path)?)?;
    if model_manifest.repository != candidate.model_repository
        || model_manifest.revision != candidate.model_revision
        || model_manifest.objects.len() != candidate.model_objects.len()
        || model_manifest
            .objects
            .iter()
            .zip(&candidate.model_objects)
            .any(|(left, right)| {
                left.path != right.path.as_str()
                    || left.media_type != right.media_type
                    || left.bytes != right.bytes
                    || left.sha256 != right.sha256.as_str()
            })
    {
        return Err(WorkerError::Invalid("model_manifest_binding"));
    }
    hydrate_candidate_model_tree(&root, &candidate).await?;
    let tokenizer_path = verified_tei_object(&root, &candidate.tokenizer.object)?;
    let tokenizer = ExactTokenizer::from_bytes(
        ExecutableTokenizerRef {
            artifact: candidate.tokenizer.component.clone(),
            format: TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: candidate.tokenizer.revision.clone(),
            target_tokenizer: candidate.tokenizer.component.clone(),
            add_special_tokens: candidate.tokenizer.add_special_tokens,
            maximum_input_bytes: candidate.fixture.object.bytes,
        },
        &fs::read(tokenizer_path)?,
    )?;
    let fixture_path = verified_tei_object(&root, &candidate.fixture.object)?;
    let fixture_bytes = fs::read(fixture_path)?;
    let fixture: rag_embedding::TeiConformanceFixtureV1 = serde_json::from_slice(&fixture_bytes)?;
    if fixture.schema_version != rag_embedding::TEI_CONFORMANCE_FIXTURE_SCHEMA_V1
        || fixture.inputs.len() != candidate.fixture.input_count as usize
        || fixture.inputs.iter().any(String::is_empty)
    {
        return Err(WorkerError::Invalid("conformance_fixture_binding"));
    }
    for input in &fixture.inputs {
        if tokenizer.count(input)? > u64::from(candidate.execution.maximum_tokens) {
            return Err(WorkerError::Invalid("conformance_fixture_token_limit"));
        }
    }
    let model_dir = validate_candidate_model_tree(&root, &candidate)?;
    let observation = wait_for_observation(
        &root,
        &options.observation,
        Duration::from_secs(options.observation_wait_seconds),
    )
    .await?;
    validate_candidate_observation(&observation, &candidate)?;
    let gpu_device_id = validate_local_gpu_identity(
        &candidate.accelerator.gpu_model_id,
        &candidate.accelerator.compute_capability,
        candidate.accelerator.gpu_count,
    )
    .await?;

    let model_started = Instant::now();
    let mut tei = TeiProcess::start(
        &model_dir,
        options.port,
        &candidate.execution.forced_runtime_dtype,
        candidate.execution.maximum_client_batch_size,
        candidate.execution.maximum_batch_tokens,
        candidate.execution.maximum_concurrent_requests as usize,
        &candidate.execution.served_model,
    )?;
    let profile = EmbeddingProfile {
        id: candidate.execution.model_artifact_set.id.clone(),
        version: candidate.execution.model_artifact_set.version.clone(),
        sha256: candidate.component_sha256.to_string(),
        model: candidate.execution.served_model.clone(),
        dimensions: candidate.execution.dimensions,
        normalization: candidate.execution.normalization.clone(),
        vector_derivation: None,
        query_instruction: None,
        query_composition: None,
    };
    let endpoint = format!("http://127.0.0.1:{}", options.port);
    let embedder = TeiEmbedder::loopback_with_options(
        &endpoint,
        profile,
        OpenAiCompatibleOptions {
            timeout: Duration::from_millis(candidate.execution.request_timeout_ms),
            max_response_bytes: usize::try_from(candidate.execution.maximum_response_bytes)
                .map_err(|_| WorkerError::Invalid("conformance_response_limit"))?,
            authorization: BearerAuthorization::None,
        },
    )?;
    let health = wait_for_health(
        &embedder,
        &mut tei,
        Duration::from_secs(options.health_wait_seconds),
    )
    .await;
    let model_load_ms = model_started
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    if let Err(error) = health {
        tei.stop().await;
        return Err(error);
    }
    let request_started = Instant::now();
    let probe = embedder.conformance_probe(&fixture.inputs).await;
    let request_ms = request_started
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    tei.stop().await;
    let (batch, probe_digest) = probe?;
    let (normalized_bytes, normalized_digest) = canonical_f32_vectors(&batch.vectors)?;
    if normalized_digest != probe_digest {
        return Err(WorkerError::Invalid("conformance_output_canonicalization"));
    }
    let normalized_digest = Digest::new(normalized_digest)?;
    let output_path = resolve_output_artifact(&root, &candidate.expected_output_key)?;
    publish_no_overwrite(&output_path, &normalized_bytes)?;
    let mut result = RunpodTeiConformanceResult {
        schema_version: RUNPOD_TEI_CONFORMANCE_RESULT_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed"),
        candidate_sha256: candidate.component_sha256.clone(),
        run_id: options.run_id.clone(),
        machine: RunpodTeiMachineIdentity {
            pod_id: observation.machine.pod_id,
            machine_id: observation.machine.machine_id,
            gpu_device_id,
            accelerator: candidate.accelerator.clone(),
        },
        execution: candidate.execution.clone(),
        outcome: RunpodTeiConformanceOutcome::Completed,
        returned_model: Some(batch.returned_model),
        normalized_output: Some(RunpodTeiNormalizedOutput {
            object: RunpodTeiArtifactObject {
                path: candidate.expected_output_key.clone(),
                media_type: "application/json".into(),
                bytes: normalized_bytes.len() as u64,
                sha256: normalized_digest.clone(),
            },
            format: candidate.output_format.clone(),
            vector_count: candidate.fixture.input_count,
            dimensions: candidate.execution.dimensions,
            dtype: candidate.execution.api_vector_dtype.clone(),
            normalized_output_sha256: normalized_digest,
        }),
        model_load_ms,
        request_ms,
        failure_code: None,
    };
    result.seal(&candidate)?;
    let result_key = SafeRelativePath::new(format!("conformance/results/{}.json", options.run_id))?;
    let result_path = resolve_output_artifact(&root, &result_key)?;
    publish_no_overwrite(&result_path, &canonical_json_bytes(&result)?)?;
    Ok(())
}

fn validate_cli(options: &RunOptions) -> Result<()> {
    if options.attempt_number == 0
        || options.port == 0
        || options.assignment_count == 0
        || options.batch_size == 0
        || options.batch_size > 32
        || options.requests_in_flight == 0
        || options.requests_in_flight > 256
        || options.health_wait_seconds == 0
        || options.health_wait_seconds > 7_200
        || !valid_identifier(&options.worker_id)
        || !valid_identifier(&options.attempt_id)
    {
        return Err(WorkerError::Invalid("cli_arguments"));
    }
    Ok(())
}

impl ValidatedWork {
    async fn load(
        options: &RunOptions,
        executable: &Path,
        reporter: &mut RuntimeReporter,
    ) -> Result<Self> {
        reporter.begin("control_objects_verified");
        let root = fs::canonicalize(&options.root)?;
        if !root.is_dir() {
            return Err(WorkerError::Invalid("run_root"));
        }
        let bundle_path = contained_existing(&root, &options.bundle)?;
        ensure_bounded(&bundle_path, MAX_CONTROL_BYTES)?;
        let bundle: RunpodEmbeddingBundle = read_json(&bundle_path)?;

        let prepared_path = verified_object(&root, &bundle.artifacts.prepared_manifest.object)?;
        let prepared: PreparedCorpusManifest = read_json(&prepared_path)?;
        let plan_path = verified_object(&root, &bundle.artifacts.embedding_plan.object)?;
        let plan: EmbeddingPlanV2 = read_json(&plan_path)?;
        bundle.validate_against(&prepared, &plan)?;
        if bundle.artifacts.prepared_manifest.component_sha256 != prepared.component_sha256
            || bundle.artifacts.embedding_plan.component_sha256 != plan.component_sha256
        {
            return Err(WorkerError::Invalid("bundle_component_binding"));
        }

        let mut assignments =
            selected_worker_assignments(&bundle, &options.worker_id, options.assignment_count)?
                .into_iter();
        let assignment = assignments
            .next()
            .ok_or(WorkerError::Invalid("worker_assignment_count"))?;
        let additional_assignments = assignments.collect();

        let policy_path = verified_object(&root, &bundle.artifacts.embedding_profile.object)?;
        ensure_bounded(&policy_path, MAX_CONTROL_BYTES)?;
        let policy_bytes = fs::read(policy_path)?;
        let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
        if options.batch_size > policy.batching.maximum_batch_items as usize {
            return Err(WorkerError::Invalid("batch_size_policy"));
        }
        if options.requests_in_flight > policy.batching.maximum_concurrent_requests as usize {
            return Err(WorkerError::Invalid("requests_in_flight_policy"));
        }
        let compact_profile = policy.embedding_profile(&policy_bytes)?;
        validate_execution_binding(&bundle, &plan, &policy, &compact_profile)?;

        let build_receipt_path =
            verified_object(&root, &bundle.artifacts.executor_image_build.object)?;
        let build_receipt: RunpodExecutorImageBuildReceipt = read_json(&build_receipt_path)?;
        build_receipt.validate()?;
        if build_receipt.component_sha256 != bundle.artifacts.executor_image_build.component_sha256
            || build_receipt.component_sha256.as_str() != policy.executor_image_build.sha256
            || build_receipt.executor_image.component.id != policy.executor_image.component.id
            || build_receipt.executor_image.component.version
                != policy.executor_image.component.version
            || build_receipt.executor_image.component.sha256.as_str()
                != policy.executor_image.component.sha256
            || build_receipt.executor_image.repository != policy.executor_image.repository
            || build_receipt.executor_image.digest != policy.executor_image.digest
            || build_receipt.tei_base_image.component.id != policy.tei_image.component.id
            || build_receipt.tei_base_image.component.version != policy.tei_image.component.version
            || build_receipt.tei_base_image.component.sha256.as_str()
                != policy.tei_image.component.sha256
            || build_receipt.tei_base_image.repository != policy.tei_image.repository
            || build_receipt.tei_base_image.digest != policy.tei_image.digest
            || build_receipt.worker_binary.component_sha256
                != bundle.artifacts.worker_binary.component_sha256
            || build_receipt.worker_binary.object.path != bundle.artifacts.worker_binary.object.key
            || build_receipt.worker_binary.object.bytes
                != bundle.artifacts.worker_binary.object.bytes
            || build_receipt.worker_binary.object.sha256
                != bundle.artifacts.worker_binary.object.sha256
        {
            return Err(WorkerError::Invalid("executor_image_build_binding"));
        }

        let tokenizer_path = verified_object(&root, &bundle.artifacts.executable_tokenizer.object)?;
        let tokenizer_bytes = fs::read(tokenizer_path)?;
        let tokenizer =
            ExactTokenizer::from_bytes(plan.executable_tokenizer.clone(), &tokenizer_bytes)?;
        tokenizer
            .reference()
            .validate_for_profile(&plan.embedding_profile)?;

        let counts_path = verified_object(&root, &bundle.artifacts.document_token_counts)?;
        let counts = plan.read_document_token_counts(
            counts_path
                .parent()
                .and_then(Path::parent)
                .ok_or(WorkerError::Invalid("token_count_path"))?,
        );
        // Cloud storage may relocate the plan object. Validate the exact bytes
        // directly when its standard relative parent is not preserved.
        if counts.is_err() {
            let bytes = fs::read(counts_path)?;
            let decoded = rag_pipeline::decode_document_token_counts(&bytes)?;
            plan.validate_document_token_counts(&decoded)?;
        }

        let fixture_path = verified_object(&root, &bundle.artifacts.conformance_fixture)?;
        let fixture_bytes = fs::read(fixture_path)?;
        if digest_bytes(&fixture_bytes).as_str() != policy.conformance.fixture.sha256
            || u64::try_from(fixture_bytes.len()).ok() != Some(policy.conformance.fixture.bytes)
        {
            return Err(WorkerError::Invalid("conformance_fixture_binding"));
        }
        let query_plan_path = verified_object(&root, &bundle.artifacts.query_plan)?;
        ensure_bounded(&query_plan_path, 128 * 1024 * 1024)?;
        let query_plan_bytes = fs::read(&query_plan_path)?;
        query_vector_plan_queries(&query_plan_bytes)?;

        let manifest_path = verified_object(&root, &bundle.artifacts.model_manifest.object)?;
        let model_manifest_bytes = fs::read(manifest_path)?;
        let model_manifest = parse_tei_model_artifact_set_v1(&model_manifest_bytes)?;
        if model_manifest.objects != policy.model_objects
            || bundle.artifacts.model_manifest.component_sha256.as_str()
                != policy.model_artifact_set.sha256
        {
            return Err(WorkerError::Invalid("model_manifest_binding"));
        }

        let staged_worker = verified_object(&root, &bundle.artifacts.worker_binary.object)?;
        if file_digest(&staged_worker)? != bundle.execution.worker_binary.sha256
            || file_digest(executable)? != bundle.execution.worker_binary.sha256
        {
            return Err(WorkerError::Invalid("worker_binary_binding"));
        }
        reporter.complete("control_objects_verified")?;

        reporter.begin("model_tree_verified");
        let model_dir = validate_model_tree(&root, &bundle, &policy)?;
        reporter.complete("model_tree_verified")?;

        reporter.begin("observation_verified");
        let observation = wait_for_observation(
            &root,
            &options.observation,
            Duration::from_secs(options.observation_wait_seconds),
        )
        .await?;
        observation.validate(&bundle.execution.accelerator)?;
        reporter.complete("observation_verified")?;
        reporter.begin("gpu_verified");
        validate_local_gpu(&bundle.execution.accelerator).await?;
        reporter.complete("gpu_verified")?;

        let document_objects = bundle
            .artifacts
            .prepared_documents
            .iter()
            .map(|item| (item.prepared_path.as_str().to_owned(), item.object.clone()))
            .collect();
        Ok(Self {
            root,
            bundle,
            assignment,
            additional_assignments,
            prepared,
            plan,
            backend_version: policy.inference_engine.version.clone(),
            checkpoint_dtype: policy.checkpoint_compute_dtype.clone(),
            maximum_batch_items: policy.batching.maximum_batch_items,
            maximum_batch_tokens: policy.batching.maximum_batch_tokens,
            maximum_concurrent_requests: policy.batching.maximum_concurrent_requests,
            served_model: policy.api_model_key.clone(),
            policy_bytes,
            compact_profile,
            fixture_bytes,
            query_plan_path,
            query_plan_bytes,
            model_dir,
            document_objects,
            observation,
        })
    }
}

fn selected_worker_assignments(
    bundle: &RunpodEmbeddingBundle,
    worker_id: &str,
    assignment_count: u32,
) -> Result<Vec<RunpodWorkerAssignment>> {
    let start = bundle
        .assignments
        .iter()
        .position(|item| item.worker_id == worker_id)
        .ok_or(WorkerError::Invalid("worker_assignment"))?;
    let count = usize::try_from(assignment_count)
        .map_err(|_| WorkerError::Invalid("worker_assignment_count"))?;
    let end = start
        .checked_add(count)
        .filter(|end| count != 0 && *end <= bundle.assignments.len())
        .ok_or(WorkerError::Invalid("worker_assignment_count"))?;
    Ok(bundle.assignments[start..end].to_vec())
}

fn validate_execution_binding(
    bundle: &RunpodEmbeddingBundle,
    plan: &EmbeddingPlanV2,
    policy: &TeiCheckpointProfileV3,
    compact: &EmbeddingProfile,
) -> Result<()> {
    let component = |value: &rag_embedding::TeiComponentIdentity| -> Result<ComponentRef> {
        Ok(ComponentRef {
            id: value.id.clone(),
            version: value.version.clone(),
            sha256: Digest::new(value.sha256.clone())?,
        })
    };
    let accelerator = &bundle.execution.accelerator;
    if policy.tei_image.component.sha256 != PINNED_TEI_IMAGE_SHA256
        || policy.tei_image.digest != format!("sha256:{PINNED_TEI_IMAGE_SHA256}")
        || bundle.execution.executor_image != component(&policy.executor_image.component)?
        || bundle.execution.executor_image_build != component(&policy.executor_image_build)?
        || policy.executor_image.component == policy.tei_image.component
        || policy.executor_image.repository == policy.tei_image.repository
        || policy.accelerator.compute_capability != PINNED_TEI_COMPUTE_CAPABILITY
        || bundle.execution.runtime != component(&policy.runtime)?
        || bundle.execution.model_artifact != component(&policy.model_artifact_set)?
        || bundle.execution.embedding_profile != plan.embedding_profile.component
        || bundle.execution.embedding_profile.sha256.as_str() != compact.sha256
        || bundle.execution.returned_model != policy.api_model_key
        || accelerator.provider != policy.accelerator.provider
        || accelerator.model != policy.accelerator.gpu_model_id
        || accelerator.architecture != policy.accelerator.architecture_image_class
        || accelerator.compute_capability != policy.accelerator.compute_capability
        || accelerator.count != policy.accelerator.gpu_count
    {
        return Err(WorkerError::Invalid("execution_identity_binding"));
    }
    Ok(())
}

fn validate_model_tree(
    root: &Path,
    bundle: &RunpodEmbeddingBundle,
    policy: &TeiCheckpointProfileV3,
) -> Result<PathBuf> {
    if bundle.artifacts.model_objects.len() != policy.model_objects.len() {
        return Err(WorkerError::Invalid("model_object_coverage"));
    }
    let mut prefix: Option<String> = None;
    for expected in &policy.model_objects {
        let suffix = &expected.path;
        let object = bundle
            .artifacts
            .model_objects
            .iter()
            .find(|item| {
                item.key.as_str() == suffix
                    || item
                        .key
                        .as_str()
                        .strip_suffix(suffix)
                        .is_some_and(|head| head.ends_with('/'))
            })
            .ok_or(WorkerError::Invalid("model_object_coverage"))?;
        if object.bytes != expected.bytes || object.sha256.as_str() != expected.sha256 {
            return Err(WorkerError::Invalid("model_object_binding"));
        }
        verified_object(root, object)?;
        let head = object
            .key
            .as_str()
            .strip_suffix(suffix)
            .ok_or(WorkerError::Invalid("model_object_path"))?
            .trim_end_matches('/')
            .to_owned();
        if prefix.as_ref().is_some_and(|value| value != &head) {
            return Err(WorkerError::Invalid("model_object_prefix"));
        }
        prefix.get_or_insert(head);
    }
    let relative = prefix.ok_or(WorkerError::Invalid("model_object_prefix"))?;
    if relative.is_empty() {
        Ok(root.to_owned())
    } else {
        contained_existing(root, &relative)
    }
}

async fn wait_for_observation(
    root: &Path,
    relative: &str,
    timeout: Duration,
) -> Result<WorkerObservation> {
    let safe = SafeRelativePath::new(relative.to_owned())?;
    let started = Instant::now();
    loop {
        match resolve_existing_artifact(root, &safe) {
            Ok(path) => {
                ensure_bounded(&path, MAX_OBSERVATION_BYTES)?;
                return Ok(read_json(&path)?);
            }
            Err(rag_pipeline::PipelineError::Io(error))
                if error.kind() == std::io::ErrorKind::NotFound && started.elapsed() < timeout =>
            {
                sleep(Duration::from_millis(250)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

async fn wait_for_health(
    embedder: &TeiEmbedder,
    process: &mut TeiProcess,
    timeout: Duration,
) -> Result<()> {
    let started = Instant::now();
    loop {
        if process.child.try_wait()?.is_some() {
            return Err(WorkerError::Invalid("tei_exited_before_health"));
        }
        match embedder.health().await {
            Ok(()) => return Ok(()),
            Err(_) if started.elapsed() < timeout => sleep(Duration::from_millis(500)).await,
            Err(_) => return Err(WorkerError::Invalid("tei_health_timeout")),
        }
    }
}

pub trait WorkerBackend: IdentifiedEmbedder + Send + Sync + 'static {
    fn conformance<'a>(
        &'a self,
        fixture: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = rag_embedding::Result<()>> + Send + 'a>>;
}

impl WorkerBackend for TeiEmbedder {
    fn conformance<'a>(
        &'a self,
        fixture: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = rag_embedding::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.checkpoint_conformance_probe(fixture).await?;
            Ok(())
        })
    }
}

async fn execute_assignment<E: WorkerBackend>(
    work: &ValidatedWork,
    backend: Arc<E>,
    attempt_id: &str,
    attempt_number: u32,
    batch_size: usize,
    requests_in_flight: usize,
    mut reporter: Option<&mut RuntimeReporter>,
) -> Result<RunpodWorkerAttemptMarker> {
    let marker_key = SafeRelativePath::new(format!(
        "attempts/{}/completed.json",
        work.assignment.worker_id
    ))?;
    let existing_marker = marker_key.join_to(&work.root);
    if existing_marker.try_exists()? {
        let marker: RunpodWorkerAttemptMarker = read_json(&existing_marker)?;
        marker.validate_against(&work.bundle)?;
        if marker.worker_id != work.assignment.worker_id
            || marker.attempt_id != attempt_id
            || marker.attempt_number != attempt_number
        {
            return Err(WorkerError::Invalid("completion_marker_exists"));
        }
        if let Some(reporter) = reporter {
            reporter.complete("assignment_completed")?;
        }
        return Ok(marker);
    }
    let started_at_ms = unix_ms()?;
    if let Some(reporter) = reporter.as_deref_mut() {
        reporter.begin("conformance_passed");
    }
    backend.conformance(&work.fixture_bytes).await?;
    if let Some(reporter) = reporter.as_deref_mut() {
        reporter.complete("conformance_passed")?;
    }
    let mut outputs = Vec::new();
    // The exact conformance probe is one model request and is part of the
    // attempt even when all durable task outputs can be reused.
    let mut requests = 1_u64;
    let mut retries = 0_u64;
    let (query_vector_set, query_requests) =
        if work.assignment.worker_id == work.bundle.query_vector_output.worker_id {
            if let Some(reporter) = reporter.as_deref_mut() {
                reporter.begin("query_vectors_published");
            }
            let (output, requests) =
                execute_query_vector_set(work, backend.as_ref(), batch_size).await?;
            if let Some(reporter) = reporter.as_deref_mut() {
                reporter.complete("query_vectors_published")?;
            }
            (Some(output), requests)
        } else {
            (None, 0)
        };
    requests = requests
        .checked_add(query_requests)
        .ok_or(WorkerError::Invalid("request_count"))?;
    if let Some(reporter) = reporter.as_deref_mut() {
        reporter.begin("first_task_published");
    }
    let mut first_task_reported = false;
    for expected in &work.assignment.tasks {
        let task_index = usize::try_from(expected.task_ordinal)
            .map_err(|_| WorkerError::Invalid("task_ordinal"))?;
        let task = work
            .plan
            .tasks
            .get(task_index)
            .ok_or(WorkerError::Invalid("task_ordinal"))?;
        if let Some(output) = reusable_output(work, expected, task_index)? {
            outputs.push(output);
            if !first_task_reported {
                if let Some(reporter) = reporter.as_deref_mut() {
                    reporter.complete("first_task_published")?;
                }
                first_task_reported = true;
            }
            continue;
        }
        let rows = load_task_rows(work, task)?;
        let texts = rows
            .iter()
            .map(|row| {
                format_document_input_exact(
                    &work.plan.embedding_profile.document_format,
                    &row.semantic_text,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let input_bytes = texts.iter().try_fold(0_u64, |sum, text| {
            sum.checked_add(text.len() as u64)
                .ok_or(WorkerError::Invalid("input_byte_count"))
        })?;
        let result_path = resolve_output_artifact(&work.root, &expected.result_key)?;
        let receipt_path = resolve_output_artifact(&work.root, &expected.receipt_key)?;
        let report_path = resolve_output_artifact(&work.root, &expected.report_key)?;
        let order = decode_digest(task.embedding_input_order_sha256.as_str())?;
        let expectation = rag_embedding::EmbeddingShardExpectation {
            row_count: task.row_count(),
            dimensions: work.compact_profile.dimensions,
            order_sha256: order,
        };
        let _ = prepare_embedding_task_part(
            &result_path,
            expectation,
            &work.compact_profile.normalization,
            None,
        )?;
        let began_ms = unix_ms()?;
        let began = Instant::now();
        let (stats, report) = execute_embedding_task_reported(
            Arc::clone(&backend),
            &work.compact_profile,
            &texts,
            &result_path,
            order,
            EmbeddingTaskOptions {
                batch_size,
                max_in_flight: requests_in_flight,
                retry: RetryPolicy::default(),
            },
        )
        .await?;
        let verified = verify_embedding_task_part(
            &result_path,
            expectation,
            &work.compact_profile.normalization,
            None,
        )?;
        let mut receipt =
            build_receipt(work, task, &stats, &verified, input_bytes, began.elapsed())?;
        receipt.seal()?;
        receipt.validate_against_v2(&work.plan)?;
        publish_no_overwrite(&receipt_path, &canonical_json_bytes(&receipt)?)?;
        write_task_report(
            work,
            task_index,
            &receipt,
            &stats,
            &report,
            began_ms,
            unix_ms()?,
            batch_size,
            requests_in_flight,
            &result_path,
            &receipt_path,
            &report_path,
        )?;
        requests = requests
            .checked_add(stats.requests as u64)
            .ok_or(WorkerError::Invalid("request_count"))?;
        retries = retries
            .checked_add(stats.retries as u64)
            .ok_or(WorkerError::Invalid("retry_count"))?;
        outputs.push(task_output(
            expected,
            &result_path,
            &receipt_path,
            &report_path,
        )?);
        if !first_task_reported {
            if let Some(reporter) = reporter.as_deref_mut() {
                reporter.complete("first_task_published")?;
            }
            first_task_reported = true;
        }
    }
    let mut marker = RunpodWorkerAttemptMarker {
        schema_version: RUNPOD_WORKER_ATTEMPT_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed"),
        bundle_sha256: work.bundle.component_sha256.clone(),
        assignment_sha256: work.assignment.component_sha256.clone(),
        worker_id: work.assignment.worker_id.clone(),
        attempt_id: attempt_id.to_owned(),
        attempt_number,
        outcome: WorkerAttemptOutcome::Completed,
        machine: work.observation.machine.clone(),
        execution: work.bundle.execution.clone(),
        started_at_ms,
        completed_at_ms: unix_ms()?,
        requests,
        retries,
        outputs,
        query_vector_set,
        failure_code: None,
    };
    marker.seal()?;
    marker.validate_against(&work.bundle)?;
    let marker_path = resolve_output_artifact(&work.root, &marker_key)?;
    publish_no_overwrite(&marker_path, &canonical_json_bytes(&marker)?)?;
    if let Some(reporter) = reporter {
        reporter.complete("assignment_completed")?;
    }
    Ok(marker)
}

async fn execute_query_vector_set<E: IdentifiedEmbedder>(
    work: &ValidatedWork,
    backend: &E,
    batch_size: usize,
) -> Result<(RunpodQueryVectorSetOutput, u64)> {
    let destination =
        resolve_output_artifact(&work.root, &SafeRelativePath::new("query-vectors")?)?;
    if destination.try_exists()? {
        let sealed = SealedQueryVectorSet::open(
            &destination,
            &work.bundle.execution.embedding_profile,
            &work.bundle.execution.returned_model,
            work.compact_profile.dimensions,
            &work.compact_profile.normalization,
            Some(&work.query_plan_path),
        )?;
        return Ok((
            query_vector_output(work, sealed.manifest.component_sha256)?,
            0,
        ));
    }

    let queries = query_vector_plan_queries(&work.query_plan_bytes)?;
    let composed = queries
        .iter()
        .map(|query| try_compose_query(&work.compact_profile, &query.query))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut vectors = Vec::with_capacity(queries.len());
    let mut requests = 0_u64;
    for inputs in composed.chunks(batch_size) {
        let response = backend.embed_identified(inputs).await?;
        if response.returned_model != work.bundle.execution.returned_model
            || response.vectors.len() != inputs.len()
        {
            return Err(WorkerError::Invalid("query_vector_model_response"));
        }
        for vector in response.vectors {
            vectors.push(adapt_model_vector(&work.compact_profile, vector)?);
        }
        requests = requests
            .checked_add(1)
            .ok_or(WorkerError::Invalid("request_count"))?;
    }
    let inputs = queries
        .iter()
        .zip(&composed)
        .zip(&vectors)
        .map(|((query, composed), vector)| QueryVectorSetInput {
            query_id: &query.query_id,
            raw_query: &query.query,
            composed_query: composed,
            vector,
        })
        .collect::<Vec<_>>();
    let manifest = write_query_vector_set(
        &destination,
        &work.query_plan_path,
        QueryVectorExecutionBinding {
            embedding_profile: work.bundle.execution.embedding_profile.clone(),
            embedding_policy: work.bundle.execution.embedding_profile.clone(),
            execution_identity_sha256: canonical_digest(&work.bundle.execution)?,
            execution: work.bundle.execution.clone(),
            executor_image_build_receipt: work.bundle.execution.executor_image_build.clone(),
        },
        work.compact_profile.dimensions,
        &work.compact_profile.normalization,
        &inputs,
    )?;
    SealedQueryVectorSet::open(
        &destination,
        &work.bundle.execution.embedding_profile,
        &work.bundle.execution.returned_model,
        work.compact_profile.dimensions,
        &work.compact_profile.normalization,
        Some(&work.query_plan_path),
    )?;
    Ok((
        query_vector_output(work, manifest.component_sha256)?,
        requests,
    ))
}

fn query_vector_output(
    work: &ValidatedWork,
    component_sha256: Digest,
) -> Result<RunpodQueryVectorSetOutput> {
    let expected = &work.bundle.query_vector_output;
    Ok(RunpodQueryVectorSetOutput {
        component_sha256,
        manifest: local_output(
            &expected.manifest_key,
            &expected.manifest_key.join_to(&work.root),
        )?,
        query_plan: local_output(
            &expected.query_plan_key,
            &expected.query_plan_key.join_to(&work.root),
        )?,
        vectors: local_output(
            &expected.vectors_key,
            &expected.vectors_key.join_to(&work.root),
        )?,
    })
}

fn load_task_rows(
    work: &ValidatedWork,
    task: &rag_pipeline::EmbeddingTaskV2,
) -> Result<Vec<PreparedDocumentRow>> {
    let mut result = Vec::new();
    for slice in &task.input_slices {
        let object = work
            .document_objects
            .get(slice.path.as_str())
            .ok_or(WorkerError::Invalid("task_document_object"))?;
        if object.sha256 != slice.object_sha256 {
            return Err(WorkerError::Invalid("task_document_digest"));
        }
        let path = verified_object(&work.root, object)?;
        let rows = read_prepared_documents(&path)?;
        let start = usize::try_from(slice.row_offset)
            .map_err(|_| WorkerError::Invalid("task_document_range"))?;
        let count =
            usize::try_from(slice.rows).map_err(|_| WorkerError::Invalid("task_document_range"))?;
        let end = start
            .checked_add(count)
            .ok_or(WorkerError::Invalid("task_document_range"))?;
        let selected = rows
            .get(start..end)
            .ok_or(WorkerError::Invalid("task_document_range"))?;
        if embedding_input_order_digest(selected) != slice.embedding_input_order_sha256 {
            return Err(WorkerError::Invalid("task_document_order"));
        }
        result.extend_from_slice(selected);
    }
    if u64::try_from(result.len()).ok() != Some(task.row_count())
        || embedding_input_order_digest(&result) != task.embedding_input_order_sha256
        || result.first().map(|row| row.document_ordinal) != Some(task.ordinal_start)
        || result.last().map(|row| row.document_ordinal + 1) != Some(task.ordinal_end)
    {
        return Err(WorkerError::Invalid("task_document_coverage"));
    }
    Ok(result)
}

fn build_receipt(
    work: &ValidatedWork,
    task: &rag_pipeline::EmbeddingTaskV2,
    stats: &EmbeddingTaskStats,
    verified: &rag_embedding::VerifiedEmbeddingTaskPart,
    input_bytes: u64,
    elapsed: Duration,
) -> Result<VectorResultReceipt> {
    Ok(VectorResultReceipt {
        schema_version: VECTOR_RECEIPT_SCHEMA.into(),
        component_sha256: digest_bytes(b"unsealed"),
        plan_sha256: work.plan.component_sha256.clone(),
        prepared_corpus_sha256: work.plan.prepared_corpus_sha256.clone(),
        embedding_profile_sha256: work.plan.embedding_profile.component.sha256.clone(),
        task_id: task.task_id.clone(),
        ordinal_start: task.ordinal_start,
        ordinal_end: task.ordinal_end,
        embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
        vector: VectorObject {
            path: task.result_path.clone(),
            rows: task.row_count(),
            bytes: verified.bytes,
            sha256: Digest::new(hex_bytes(&verified.sha256))?,
            dimensions: work.compact_profile.dimensions,
            dtype: "f32le".into(),
            embedding_input_order_sha256: task.embedding_input_order_sha256.clone(),
        },
        executor: ExecutorReceipt {
            implementation: work.bundle.execution.worker_binary.clone(),
            runtime: work.bundle.execution.runtime.clone(),
            returned_model: stats.returned_model.clone(),
            requests: stats.requests as u64,
            retries: stats.retries as u64,
            input_bytes_upper_bound: input_bytes,
            elapsed_ms: elapsed.as_millis().try_into().unwrap_or(u64::MAX),
            conformance_passed: true,
        },
        derivation: None,
        finite_values_validated: true,
        normalization_validated: true,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_task_report(
    work: &ValidatedWork,
    task_index: usize,
    receipt: &VectorResultReceipt,
    _stats: &EmbeddingTaskStats,
    execution: &EmbeddingTaskReport,
    started: u64,
    finished: u64,
    batch_size: usize,
    requests_in_flight: usize,
    vector_path: &Path,
    receipt_path: &Path,
    report_path: &Path,
) -> Result<()> {
    let task = &work.plan.tasks[task_index];
    let accelerator = &work.bundle.execution.accelerator;
    let execution_identity = serde_json::json!({
        "backend_kind": "tei",
        "executor_image": work.bundle.execution.executor_image,
        "executor_image_build": work.bundle.execution.executor_image_build,
        "runtime": work.bundle.execution.runtime,
        "worker_binary": work.bundle.execution.worker_binary,
        "model_artifact": work.bundle.execution.model_artifact,
        "embedding_profile": work.bundle.execution.embedding_profile,
        "returned_model": work.bundle.execution.returned_model,
        "accelerator": accelerator,
    });
    let report = serde_json::json!({
        "schema_version": "livefire.rag.embedding-task-run-report/2",
        "plan_sha256": work.plan.component_sha256,
        "source_snapshot_sha256": work.prepared.dataset.source_snapshot.sha256,
        "prepared_corpus_sha256": work.prepared.component_sha256,
        "embedding_profile_sha256": work.plan.embedding_profile.component.sha256,
        "tokenizer_sha256": work.plan.executable_tokenizer.artifact.sha256,
        "task_id": task.task_id,
        "task_index": task_index,
        "ordinal_start": task.ordinal_start,
        "ordinal_end": task.ordinal_end,
        "document_count": task.row_count(),
        "token_count": task.token_count,
        "receipt_sha256": receipt.component_sha256,
        "outcome": "executed",
        "started_unix_ms": started,
        "finished_unix_ms": finished,
        "execution_identity": execution_identity,
        "git": {"status":"unavailable", "commit":null, "working_tree_dirty":null},
        "machine": {"status":"observed", "operating_system":"linux", "operating_system_version":null, "architecture":std::env::consts::ARCH, "cpu_model":null, "logical_cpu_count":std::thread::available_parallelism().ok().map(std::num::NonZeroUsize::get), "ram_bytes":null},
        "accelerator": {"status":"observed", "provider":accelerator.provider, "machine_id":work.observation.machine.machine_id, "model":accelerator.model, "architecture":accelerator.architecture, "compute_capability":accelerator.compute_capability, "count":accelerator.count},
        "backend": {"status":"observed", "kind":"tei", "version":work.backend_version, "endpoint_kind":"local_loopback", "batch_size":batch_size, "requests_in_flight":requests_in_flight, "cold_load_micros":null},
        "transport_bytes": {"status":"partial", "request_body_bytes":null, "response_body_bytes":null, "submitted_text_bytes":execution.sent_input_text_bytes, "decoded_vector_bytes":execution.vector_bytes},
        "resource_usage": {"status":"not_measured", "worker_peak_rss_bytes":null, "backend_peak_rss_bytes":null},
        "artifact_sizes": {"status":"partial", "vector_shard_bytes":fs::metadata(vector_path)?.len(), "receipt_bytes":fs::metadata(receipt_path)?.len(), "task_report_bytes":null},
        "execution": execution,
    });
    publish_no_overwrite(report_path, &canonical_json_bytes(&report)?)?;
    Ok(())
}

fn reusable_output(
    work: &ValidatedWork,
    expected: &rag_pipeline::RunpodExpectedTaskOutput,
    task_index: usize,
) -> Result<Option<RunpodTaskOutput>> {
    let result_path = expected.result_key.join_to(&work.root);
    let receipt_path = expected.receipt_key.join_to(&work.root);
    let report_path = expected.report_key.join_to(&work.root);
    if !result_path.try_exists()? || !receipt_path.try_exists()? || !report_path.try_exists()? {
        return Ok(None);
    }
    let receipt: VectorResultReceipt = read_json(&receipt_path)?;
    receipt.validate_against_v2(&work.plan)?;
    if receipt.executor.implementation != work.bundle.execution.worker_binary
        || receipt.executor.runtime != work.bundle.execution.runtime
        || receipt.executor.returned_model != work.bundle.execution.returned_model
    {
        return Err(WorkerError::Invalid("reused_receipt_identity"));
    }
    let report: serde_json::Value = read_json(&report_path)?;
    if report
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some("livefire.rag.embedding-task-run-report/2")
        || report.get("task_index").and_then(serde_json::Value::as_u64)
            != u64::try_from(task_index).ok()
        || report
            .get("receipt_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(receipt.component_sha256.as_str())
    {
        return Err(WorkerError::Invalid("reused_report_binding"));
    }
    let expectation = rag_embedding::EmbeddingShardExpectation {
        row_count: receipt.vector.rows,
        dimensions: receipt.vector.dimensions,
        order_sha256: decode_digest(receipt.embedding_input_order_sha256.as_str())?,
    };
    verify_embedding_task_part(
        &result_path,
        expectation,
        &work.compact_profile.normalization,
        Some(decode_digest(receipt.vector.sha256.as_str())?),
    )?;
    Ok(Some(task_output(
        expected,
        &result_path,
        &receipt_path,
        &report_path,
    )?))
}

fn task_output(
    expected: &rag_pipeline::RunpodExpectedTaskOutput,
    result: &Path,
    receipt: &Path,
    report: &Path,
) -> Result<RunpodTaskOutput> {
    Ok(RunpodTaskOutput {
        task_id: expected.task_id.clone(),
        result: local_output(&expected.result_key, result)?,
        receipt: local_output(&expected.receipt_key, receipt)?,
        report: local_output(&expected.report_key, report)?,
    })
}

fn local_output(key: &SafeRelativePath, path: &Path) -> Result<CloudObjectRef> {
    let bytes = fs::metadata(path)?.len();
    if bytes == 0 {
        return Err(WorkerError::Invalid("empty_output"));
    }
    Ok(CloudObjectRef {
        key: key.clone(),
        bytes,
        sha256: file_digest(path)?,
    })
}

struct TeiProcess {
    child: Child,
}

impl TeiProcess {
    fn start(
        model_dir: &Path,
        port: u16,
        dtype: &str,
        maximum_batch_items: u32,
        maximum_batch_tokens: u64,
        maximum_concurrent_requests: usize,
        served_model: &str,
    ) -> Result<Self> {
        let mut command = tokio::process::Command::new(TEI_ENTRYPOINT);
        command
            .args(tei_command_args(
                model_dir,
                port,
                dtype,
                maximum_batch_items,
                maximum_batch_tokens,
                maximum_concurrent_requests,
                served_model,
            ))
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        command.env_clear();
        for name in [
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .env("HOME", "/tmp")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1");
        Ok(Self {
            child: command.spawn()?,
        })
    }

    async fn stop(&mut self) {
        if let Some(process_id) = self.child.id() {
            let _ = tokio::process::Command::new("/bin/kill")
                .args(["-TERM", &process_id.to_string()])
                .env_clear()
                .status()
                .await;
        }
        if tokio::time::timeout(Duration::from_secs(10), self.child.wait())
            .await
            .is_err()
        {
            let _ = self.child.start_kill();
            let _ = self.child.wait().await;
        }
    }
}

fn tei_command_args(
    model_dir: &Path,
    port: u16,
    dtype: &str,
    maximum_batch_items: u32,
    maximum_batch_tokens: u64,
    maximum_concurrent_requests: usize,
    served_model: &str,
) -> Vec<String> {
    vec![
        "--model-id".into(),
        model_dir.display().to_string(),
        "--hostname".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--dtype".into(),
        dtype.to_owned(),
        "--served-model-name".into(),
        served_model.to_owned(),
        "--pooling".into(),
        "last-token".into(),
        "--max-client-batch-size".into(),
        maximum_batch_items.to_string(),
        "--max-batch-tokens".into(),
        maximum_batch_tokens.to_string(),
        "--max-concurrent-requests".into(),
        maximum_concurrent_requests.to_string(),
        "--auto-truncate".into(),
        "false".into(),
    ]
}

fn contained_existing(root: &Path, relative: &str) -> Result<PathBuf> {
    resolve_existing_artifact(root, &SafeRelativePath::new(relative.to_owned())?)
        .map_err(Into::into)
}

fn verified_object(root: &Path, object: &CloudObjectRef) -> Result<PathBuf> {
    let path = resolve_existing_artifact(root, &object.key)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() != object.bytes || file_digest(&path)? != object.sha256
    {
        return Err(WorkerError::Invalid("object_digest"));
    }
    Ok(path)
}

fn verified_tei_object(root: &Path, object: &RunpodTeiArtifactObject) -> Result<PathBuf> {
    let path = resolve_existing_artifact(root, &object.path)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file()
        || metadata.len() != object.bytes
        || file_digest(&path)?.as_str() != object.sha256.as_str()
    {
        return Err(WorkerError::Invalid("conformance_object_digest"));
    }
    Ok(path)
}

async fn hydrate_candidate_model_tree(
    root: &Path,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<()> {
    if candidate.model_repository != "Qwen/Qwen3-Embedding-8B"
        || candidate.model_revision.len() != 40
        || !candidate
            .model_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(WorkerError::Invalid("model_hydration_source"));
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 || attempt.url().scheme() != "https" {
                attempt.stop()
            } else {
                attempt.follow()
            }
        }))
        .build()?;
    let mut downloads = tokio::task::JoinSet::new();
    for object in &candidate.model_objects {
        if verified_tei_object(root, object).is_ok() {
            continue;
        }
        let client = client.clone();
        let root = root.to_path_buf();
        let repository = candidate.model_repository.clone();
        let revision = candidate.model_revision.clone();
        let object = object.clone();
        downloads.spawn(async move {
            hydrate_model_object(&client, &root, &repository, &revision, &object).await?;
            verified_tei_object(&root, &object)?;
            Ok::<(), WorkerError>(())
        });
        if downloads.len() == 4 {
            downloads
                .join_next()
                .await
                .ok_or(WorkerError::Invalid("model_hydration_task"))?
                .map_err(|_| WorkerError::Invalid("model_hydration_task"))??;
        }
    }
    while let Some(result) = downloads.join_next().await {
        result.map_err(|_| WorkerError::Invalid("model_hydration_task"))??;
    }
    Ok(())
}

async fn hydrate_model_object(
    client: &reqwest::Client,
    root: &Path,
    repository: &str,
    revision: &str,
    object: &RunpodTeiArtifactObject,
) -> Result<()> {
    let destination = resolve_output_artifact(root, &object.path)?;
    if destination.exists() {
        verified_tei_object(root, object)?;
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or(WorkerError::Invalid("model_hydration_path"))?;
    fs::create_dir_all(parent)?;
    let partial = destination.with_extension("hf.partial");
    if partial.exists() {
        let metadata = fs::symlink_metadata(&partial)?;
        if !metadata.file_type().is_file() || metadata.len() > object.bytes {
            return Err(WorkerError::Invalid("model_hydration_partial"));
        }
    } else {
        File::options()
            .write(true)
            .create_new(true)
            .open(&partial)?
            .sync_all()?;
    }
    let url = hugging_face_model_url(repository, revision, object.path.as_str())?;
    for _attempt in 0..5 {
        let offset = fs::metadata(&partial)?.len();
        if offset == object.bytes {
            break;
        }
        let mut request = client.get(url.clone());
        if offset != 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = match request.send().await {
            Ok(response) => response,
            Err(_) => continue,
        };
        let expected_status = if offset == 0 {
            reqwest::StatusCode::OK
        } else {
            reqwest::StatusCode::PARTIAL_CONTENT
        };
        if response.status() != expected_status
            || response
                .content_length()
                .is_some_and(|bytes| bytes != object.bytes - offset)
            || offset != 0
                && response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|value| value.to_str().ok())
                    != Some(
                        format!("bytes {offset}-{}/{}", object.bytes - 1, object.bytes).as_str(),
                    )
        {
            return Err(WorkerError::Invalid("model_hydration_response"));
        }
        let mut file = File::options().append(true).open(&partial)?;
        let mut downloaded = offset;
        loop {
            let chunk = match tokio::time::timeout(Duration::from_secs(60), response.chunk()).await
            {
                Ok(Ok(Some(chunk))) => chunk,
                Ok(Ok(None)) => break,
                Ok(Err(_)) | Err(_) => break,
            };
            downloaded = downloaded
                .checked_add(chunk.len() as u64)
                .ok_or(WorkerError::Invalid("model_hydration_bytes"))?;
            if downloaded > object.bytes {
                return Err(WorkerError::Invalid("model_hydration_bytes"));
            }
            file.write_all(&chunk)?;
        }
        file.sync_all()?;
        if downloaded == object.bytes {
            break;
        }
    }
    if fs::metadata(&partial)?.len() != object.bytes
        || file_digest(&partial)?.as_str() != object.sha256.as_str()
    {
        return Err(WorkerError::Invalid("model_hydration_digest"));
    }
    match fs::hard_link(&partial, &destination) {
        Ok(()) => fs::remove_file(&partial)?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verified_tei_object(root, object)?;
            fs::remove_file(&partial)?;
        }
        Err(error) => return Err(error.into()),
    }
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn hugging_face_model_url(repository: &str, revision: &str, path: &str) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse("https://huggingface.co/")
        .map_err(|_| WorkerError::Invalid("model_hydration_url"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| WorkerError::Invalid("model_hydration_url"))?;
        for segment in repository
            .split('/')
            .chain(["resolve", revision])
            .chain(path.split('/'))
        {
            if segment.is_empty() || matches!(segment, "." | "..") {
                return Err(WorkerError::Invalid("model_hydration_url"));
            }
            segments.push(segment);
        }
    }
    Ok(url)
}

fn validate_candidate_model_tree(
    root: &Path,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<PathBuf> {
    for object in &candidate.model_objects {
        verified_tei_object(root, object)?;
    }
    // Candidate model paths are the exact paths in the artifact-set digest,
    // including nested paths such as `1_Pooling/config.json`. Therefore the
    // candidate's run root is also the local model snapshot root.
    Ok(root.to_owned())
}

fn validate_candidate_observation(
    observation: &WorkerObservation,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<()> {
    if observation.schema_version != "livefire.rag.runpod-worker-observation/1"
        || !valid_identifier(&observation.machine.pod_id)
        || !valid_identifier(&observation.machine.machine_id)
        || observation.accelerator.provider != candidate.accelerator.provider
        || observation.accelerator.model != candidate.accelerator.gpu_model_id
        || observation.accelerator.architecture != candidate.accelerator.architecture_image_class
        || observation.accelerator.compute_capability != candidate.accelerator.compute_capability
        || observation.accelerator.count != candidate.accelerator.gpu_count
    {
        return Err(WorkerError::Invalid("conformance_observation_binding"));
    }
    Ok(())
}

fn ensure_bounded(path: &Path, maximum: u64) -> Result<()> {
    let bytes = fs::metadata(path)?.len();
    if bytes == 0 || bytes > maximum {
        return Err(WorkerError::Invalid("control_file_size"));
    }
    Ok(())
}

fn file_digest(path: &Path) -> Result<Digest> {
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
    Digest::new(format!("{:x}", hasher.finalize())).map_err(Into::into)
}

fn decode_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(WorkerError::Invalid("digest_decode"));
    }
    let mut result = [0_u8; 32];
    for (index, output) in result.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| WorkerError::Invalid("digest_decode"))?;
    }
    Ok(result)
}

fn hex_bytes(value: &[u8]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unix_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| WorkerError::Invalid("system_time"))?
        .as_millis()
        .try_into()
        .map_err(|_| WorkerError::Invalid("system_time"))
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

async fn validate_local_gpu(expected: &RunpodAcceleratorIdentity) -> Result<()> {
    validate_local_gpu_identity(
        &expected.model,
        &expected.compute_capability,
        expected.count,
    )
    .await
    .map(|_| ())
}

async fn validate_local_gpu_identity(
    expected_model: &str,
    expected_compute_capability: &str,
    expected_count: u32,
) -> Result<String> {
    let mut command = tokio::process::Command::new("nvidia-smi");
    command
        .args([
            "--query-gpu=name,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        command.env("PATH", path);
    }
    let output = command.output().await?;
    if !output.status.success() {
        return Err(WorkerError::Invalid("gpu_observation"));
    }
    let text =
        std::str::from_utf8(&output.stdout).map_err(|_| WorkerError::Invalid("gpu_observation"))?;
    let rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if rows.len() != 1 || expected_count != 1 {
        return Err(WorkerError::Invalid("gpu_count"));
    }
    let fields = rows[0].split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 3
        || fields[0] != expected_model
        || fields[1].is_empty()
        || fields[1].len() > 256
        || fields[2] != expected_compute_capability
    {
        return Err(WorkerError::Invalid("gpu_identity"));
    }
    Ok(fields[1].to_owned())
}

fn publish_no_overwrite(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let staging = path.with_extension(format!("{}.partial", unique_stage_nonce()?));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    {
        use std::io::Write;
        let mut file = options.open(&staging)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    match fs::hard_link(&staging, path) {
        Ok(()) => fs::remove_file(&staging)?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(&staging)?;
            if fs::read(path)? != bytes {
                return Err(WorkerError::Invalid("immutable_output_exists"));
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&staging);
            return Err(error.into());
        }
    }
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_embedding::{Embedder, IdentifiedEmbedder, IdentifiedEmbeddingBatch};
    use rag_pipeline::{
        ComponentRef, DatasetIdentity, DocumentKind, EmbeddingProfileRef, ExecutableTokenizerRef,
        ObjectEntry, PreparedDocumentObject, PreparedOccurrenceObject, RelationAccounting,
        RunpodBundleArtifacts, RunpodExecutionIdentity, RunpodExpectedQueryVectorOutput,
        RunpodExpectedTaskOutput, TokenBalanceOptions, TokenizerArtifactFormat,
        build_token_balanced_plan, canonical_digest, component_digest, document_order_digest,
        write_prepared_documents,
    };

    const TEST_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"<unk>":2},"unk_token":"<unk>"}}"#;

    struct FakeBackend;

    impl Embedder for FakeBackend {
        async fn embed(&self, texts: &[String]) -> rag_embedding::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }
    }

    impl IdentifiedEmbedder for FakeBackend {
        async fn embed_identified(
            &self,
            texts: &[String],
        ) -> rag_embedding::Result<IdentifiedEmbeddingBatch> {
            Ok(IdentifiedEmbeddingBatch {
                returned_model: "served-model".into(),
                vectors: texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect(),
            })
        }
    }

    impl WorkerBackend for FakeBackend {
        fn conformance<'a>(
            &'a self,
            _fixture: &'a [u8],
        ) -> Pin<Box<dyn Future<Output = rag_embedding::Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn test_digest(byte: u8) -> Digest {
        Digest::new(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn test_component(id: &str, digest: Digest) -> ComponentRef {
        ComponentRef {
            id: id.into(),
            version: "1".into(),
            sha256: digest,
        }
    }

    fn object(key: &str, path: &Path) -> CloudObjectRef {
        CloudObjectRef {
            key: SafeRelativePath::new(key).unwrap(),
            bytes: fs::metadata(path).unwrap().len(),
            sha256: file_digest(path).unwrap(),
        }
    }

    fn fake_work(root: &Path) -> ValidatedWork {
        fs::create_dir_all(root.join("documents")).unwrap();
        fs::create_dir_all(root.join("input/query")).unwrap();
        let query_plan_path = root.join("input/query/queries.jsonl");
        let query_plan_bytes =
            b"{\"query_id\":\"q-1\",\"query\":\"a\",\"mode\":\"dense\",\"top_n\":1,\"relations\":[]}\n"
                .to_vec();
        fs::write(&query_plan_path, &query_plan_bytes).unwrap();
        let rows = vec![PreparedDocumentRow {
            document_ordinal: 0,
            document_id: "doc-00".into(),
            document_sha256: test_digest(1),
            semantic_text_sha256: digest_bytes(b"a"),
            semantic_text: "a".into(),
            document_kind: DocumentKind::Activity,
            primary_relation: "events".into(),
            facets_json: "{}".into(),
            relations_json: "[\"events\"]".into(),
            occurrence_count: 1,
        }];
        let document_path = root.join("documents/part.parquet");
        write_prepared_documents(&document_path, &rows).unwrap();
        let document_object = object("documents/part.parquet", &document_path);
        let mut prepared = PreparedCorpusManifest {
            schema_version: rag_pipeline::PREPARED_CORPUS_SCHEMA.into(),
            component_sha256: test_digest(0),
            dataset: DatasetIdentity {
                id: "dataset".into(),
                version: "1".into(),
                source_snapshot: test_component("snapshot", test_digest(2)),
                mapping: test_component("mapping", test_digest(3)),
                source_admission: vec![],
                included_relations: vec!["events".into()],
                excluded_relations: vec![],
                structured_only_relations: vec![],
            },
            projection_policy: test_component("projection", test_digest(4)),
            document_schema: test_component("document-schema", test_digest(5)),
            occurrence_schema: test_component("occurrence-schema", test_digest(6)),
            preparation_implementation: test_component("prepare", test_digest(7)),
            document_count: 1,
            occurrence_count: 1,
            document_order_sha256: document_order_digest(["doc-00"]),
            embedding_input_order_sha256: embedding_input_order_digest(&rows),
            documents: vec![PreparedDocumentObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("documents/part.parquet").unwrap(),
                    rows: 1,
                    bytes: document_object.bytes,
                    sha256: document_object.sha256.clone(),
                    logical_order_sha256: canonical_digest(&rows).unwrap(),
                },
                ordinal: 0,
                first_document_id: "doc-00".into(),
                last_document_id: "doc-00".into(),
                embedding_input_order_sha256: embedding_input_order_digest(&rows),
            }],
            occurrences: vec![PreparedOccurrenceObject {
                object: ObjectEntry {
                    path: SafeRelativePath::new("occurrences/events.parquet").unwrap(),
                    rows: 1,
                    bytes: 1,
                    sha256: test_digest(8),
                    logical_order_sha256: test_digest(9),
                },
                ordinal: 0,
                relation: "events".into(),
            }],
            relation_accounting: BTreeMap::from([(
                "events".into(),
                RelationAccounting {
                    source_rows: 1,
                    searchable_occurrences: 1,
                    selected_occurrences: 1,
                    excluded_rows: 0,
                },
            )]),
        };
        prepared.seal().unwrap();
        let tokenizer_component = test_component("tokenizer", test_digest(10));
        let model_component = test_component("model", test_digest(11));
        let profile_component = test_component("profile", test_digest(12));
        let tokenizer = ExecutableTokenizerRef {
            artifact: ComponentRef {
                id: "tokenizer-json".into(),
                version: "1".into(),
                sha256: digest_bytes(TEST_TOKENIZER.as_bytes()),
            },
            format: TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: "1".into(),
            target_tokenizer: tokenizer_component.clone(),
            add_special_tokens: false,
            maximum_input_bytes: 128,
        };
        let plan = build_token_balanced_plan(
            &prepared,
            &rows,
            EmbeddingProfileRef {
                component: profile_component.clone(),
                model_artifact: model_component.clone(),
                tokenizer: tokenizer_component,
                maximum_input_tokens: 8,
                pooling: "last_token".into(),
                normalization: "l2".into(),
                dimensions: 4,
                dtype: "f32le".into(),
                document_format: "{semantic_text}".into(),
            },
            tokenizer,
            TEST_TOKENIZER.as_bytes(),
            TokenBalanceOptions {
                maximum_task_tokens: 8,
                maximum_task_documents: 1,
            },
        )
        .unwrap();
        let task = &plan.tasks[0];
        let mut assignment = RunpodWorkerAssignment {
            component_sha256: test_digest(0),
            worker_id: "worker-0000".into(),
            task_start: 0,
            task_end: 1,
            ordinal_start: 0,
            ordinal_end: 1,
            token_count: task.token_count,
            tasks: vec![RunpodExpectedTaskOutput {
                task_id: task.task_id.clone(),
                task_ordinal: 0,
                ordinal_start: 0,
                ordinal_end: 1,
                token_count: task.token_count,
                result_key: task.result_path.clone(),
                receipt_key: task.receipt_path.clone(),
                report_key: SafeRelativePath::new(format!("reports/{}.json", task.task_id))
                    .unwrap(),
            }],
        };
        assignment.component_sha256 = component_digest(&assignment).unwrap();
        let accelerator = RunpodAcceleratorIdentity {
            provider: "runpod".into(),
            model: "gpu".into(),
            architecture: "cuda".into(),
            compute_capability: "8.0".into(),
            count: 1,
        };
        let execution = RunpodExecutionIdentity {
            executor_image: test_component("image", test_digest(13)),
            executor_image_build: test_component("image-build", test_digest(29)),
            runtime: test_component("runtime", test_digest(14)),
            worker_binary: test_component("worker", test_digest(15)),
            model_artifact: model_component,
            embedding_profile: profile_component,
            accelerator: accelerator.clone(),
            returned_model: "served-model".into(),
        };
        let bundle = RunpodEmbeddingBundle {
            schema_version: rag_pipeline::RUNPOD_EMBEDDING_BUNDLE_SCHEMA.into(),
            component_sha256: test_digest(16),
            prepared_corpus_sha256: prepared.component_sha256.clone(),
            plan_sha256: plan.component_sha256.clone(),
            embedding_profile_sha256: plan.embedding_profile.component.sha256.clone(),
            tokenizer_sha256: plan.executable_tokenizer.artifact.sha256.clone(),
            model_sha256: plan.embedding_profile.model_artifact.sha256.clone(),
            document_count: 1,
            task_count: 1,
            total_tokens: task.token_count,
            artifacts: serde_json::from_value::<RunpodBundleArtifacts>(serde_json::json!({
                "prepared_manifest":{"component_sha256":prepared.component_sha256,"object":{"key":"input/prepared.json","bytes":1,"sha256":test_digest(20)}},
                "embedding_plan":{"component_sha256":plan.component_sha256,"object":{"key":"input/plan.json","bytes":1,"sha256":test_digest(21)}},
                "embedding_profile":{"component_sha256":plan.embedding_profile.component.sha256,"object":{"key":"input/profile.json","bytes":1,"sha256":test_digest(22)}},
                "executor_image_build":{"component_sha256":execution.executor_image_build.sha256,"object":{"key":"input/executor-image-build.json","bytes":1,"sha256":test_digest(30)}},
                "executable_tokenizer":{"component_sha256":plan.executable_tokenizer.artifact.sha256,"object":{"key":"input/tokenizer.json","bytes":1,"sha256":test_digest(23)}},
                "document_token_counts":{"key":"input/counts","bytes":1,"sha256":test_digest(24)},
                "conformance_fixture":{"key":"input/fixture","bytes":1,"sha256":test_digest(25)},
                "query_plan":{"key":"input/query/queries.jsonl","bytes":query_plan_bytes.len(),"sha256":digest_bytes(&query_plan_bytes)},
                "worker_binary":{"component_sha256":execution.worker_binary.sha256,"object":{"key":"input/worker","bytes":1,"sha256":test_digest(26)}},
                "model_manifest":{"component_sha256":plan.embedding_profile.model_artifact.sha256,"object":{"key":"input/model.json","bytes":1,"sha256":test_digest(27)}},
                "model_objects":[{"key":"model/file","bytes":1,"sha256":test_digest(28)}],
                "prepared_documents":[{"prepared_path":"documents/part.parquet","object":document_object}]
            }))
            .unwrap(),
            execution,
            query_vector_output: RunpodExpectedQueryVectorOutput {
                worker_id: "worker-0000".into(),
                manifest_key: SafeRelativePath::new("query-vectors/manifest.json").unwrap(),
                query_plan_key: SafeRelativePath::new("query-vectors/queries.jsonl").unwrap(),
                vectors_key: SafeRelativePath::new("query-vectors/vectors.f32le").unwrap(),
            },
            assignments: vec![assignment.clone()],
        };
        ValidatedWork {
            root: fs::canonicalize(root).unwrap(),
            bundle,
            assignment,
            additional_assignments: Vec::new(),
            prepared,
            plan,
            backend_version: "1.9.3".into(),
            checkpoint_dtype: "float16".into(),
            maximum_batch_items: 1,
            maximum_batch_tokens: 40_960,
            maximum_concurrent_requests: 1,
            served_model: "served-model".into(),
            policy_bytes: vec![],
            compact_profile: EmbeddingProfile {
                id: "profile".into(),
                version: "1".into(),
                sha256: test_digest(12).to_string(),
                model: "served-model".into(),
                dimensions: 4,
                normalization: "l2".into(),
                vector_derivation: None,
                query_instruction: None,
                query_composition: None,
            },
            fixture_bytes: b"fixture".to_vec(),
            query_plan_path,
            query_plan_bytes,
            model_dir: root.join("model"),
            document_objects: BTreeMap::from([("documents/part.parquet".into(), document_object)]),
            observation: WorkerObservation {
                schema_version: "livefire.rag.runpod-worker-observation/1".into(),
                machine: RunpodMachineIdentity {
                    pod_id: "pod-1".into(),
                    machine_id: "machine-1".into(),
                },
                accelerator,
            },
        }
    }

    #[test]
    fn hugging_face_model_url_is_exactly_revision_and_path_bound() {
        let url = hugging_face_model_url(
            "Qwen/Qwen3-Embedding-8B",
            "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
            "1_Pooling/config.json",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://huggingface.co/Qwen/Qwen3-Embedding-8B/resolve/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/1_Pooling/config.json"
        );
        assert!(hugging_face_model_url("Qwen//model", "revision", "config.json").is_err());
        assert!(hugging_face_model_url("Qwen/model", "revision", "../config.json").is_err());
    }

    #[test]
    fn tei_command_is_loopback_and_policy_bound() {
        let args = tei_command_args(
            Path::new("/workspace/model"),
            8080,
            "float16",
            16,
            65_536,
            4,
            "served-model",
        );
        let dockerfile = include_str!("../Dockerfile");
        assert!(dockerfile.contains(
            "ghcr.io/huggingface/text-embeddings-inference:120-1.9@sha256:\
             144aaa80ddcb520d49df83f915dc188ddd7cc6b1b3b9684a829c21dd39cbe3c5"
        ));
        assert_eq!(
            args,
            [
                "--model-id",
                "/workspace/model",
                "--hostname",
                "127.0.0.1",
                "--port",
                "8080",
                "--dtype",
                "float16",
                "--served-model-name",
                "served-model",
                "--pooling",
                "last-token",
                "--max-client-batch-size",
                "16",
                "--max-batch-tokens",
                "65536",
                "--max-concurrent-requests",
                "4",
                "--auto-truncate",
                "false"
            ]
        );
        assert!(!args.iter().any(|value| value == "0.0.0.0"));
    }

    #[test]
    fn one_input_conformance_fixture_does_not_shrink_sealed_tei_limits() {
        let fixture = rag_embedding::TeiConformanceFixtureV1 {
            schema_version: rag_embedding::TEI_CONFORMANCE_FIXTURE_SCHEMA_V1.into(),
            inputs: vec!["one deterministic probe".into()],
        };
        assert_eq!(fixture.inputs.len(), 1);

        let args = tei_command_args(
            Path::new("/workspace/model"),
            8080,
            "float16",
            8,
            65_536,
            4,
            "served-model",
        );
        let value_after = |name: &str| {
            let position = args.iter().position(|value| value == name).unwrap();
            args[position + 1].as_str()
        };
        assert_eq!(value_after("--max-client-batch-size"), "8");
        assert_eq!(value_after("--max-batch-tokens"), "65536");
        assert_eq!(value_after("--max-concurrent-requests"), "4");
    }

    #[tokio::test]
    async fn health_wait_reports_an_exited_tei_process_immediately() {
        let child = tokio::process::Command::new("/usr/bin/false")
            .spawn()
            .unwrap();
        let mut process = TeiProcess { child };
        let embedder = TeiEmbedder::loopback(
            "http://127.0.0.1:9",
            EmbeddingProfile {
                id: "profile".into(),
                version: "1".into(),
                sha256: test_digest(90).to_string(),
                model: "served-model".into(),
                dimensions: 4,
                normalization: "l2".into(),
                vector_derivation: None,
                query_instruction: None,
                query_composition: None,
            },
        )
        .unwrap();
        let started = Instant::now();
        let error = wait_for_health(&embedder, &mut process, Duration::from_secs(30))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            WorkerError::Invalid("tei_exited_before_health")
        ));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn docker_context_excludes_large_generated_trees() {
        let ignore = include_str!("../../../.dockerignore");
        for required in [
            ".git", "target", "models", "indexes", "reports", "data", ".env",
        ] {
            assert!(
                ignore.lines().any(|line| line == required),
                "missing {required}"
            );
        }
        assert!(
            !ignore
                .lines()
                .any(|line| line == "crates" || line == "Cargo.toml")
        );
    }

    #[test]
    fn refuses_a_completion_marker_with_different_bytes() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("attempts/worker/completed.json");
        publish_no_overwrite(&path, b"one").unwrap();
        publish_no_overwrite(&path, b"one").unwrap();
        assert!(matches!(
            publish_no_overwrite(&path, b"two"),
            Err(WorkerError::Invalid("immutable_output_exists"))
        ));
    }

    #[test]
    fn runtime_reporter_publishes_progress_and_a_sealed_failure() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("bundle.json"), b"exact bundle bytes").unwrap();
        let options = RunOptions {
            root: root.path().to_path_buf(),
            bundle: "bundle.json".into(),
            worker_id: "worker-0000".into(),
            assignment_count: 1,
            attempt_id: "attempt-1".into(),
            attempt_number: 1,
            observation: "observation.json".into(),
            observation_wait_seconds: 1,
            port: 8080,
            batch_size: 1,
            requests_in_flight: 1,
            health_wait_seconds: 1,
        };
        let mut reporter = RuntimeReporter::open(&options).unwrap();
        reporter.complete("worker_started").unwrap();
        reporter.begin("model_tree_verified");
        reporter.fail("model_object_binding").unwrap();

        let progress: RunpodWorkerRuntimeEvent = read_json(
            &root
                .path()
                .join("runtime/worker-0000/attempts/attempt-1/phases/01-worker_started.json"),
        )
        .unwrap();
        progress.validate().unwrap();
        let failure: RunpodWorkerRuntimeEvent = read_json(
            &root
                .path()
                .join("runtime/worker-0000/attempts/attempt-1/failed.json"),
        )
        .unwrap();
        failure.validate().unwrap();
        assert_eq!(failure.phase, "model_tree_verified");
        assert_eq!(
            failure.failure_code.as_deref(),
            Some("model_object_binding")
        );
        assert_eq!(
            failure.bundle_file_sha256,
            digest_bytes(b"exact bundle bytes")
        );
    }

    #[test]
    fn contiguous_assignments_are_bounded_by_the_sealed_bundle_order() {
        let root = tempfile::tempdir().unwrap();
        let mut bundle = fake_work(root.path()).bundle;
        let mut second = bundle.assignments[0].clone();
        second.worker_id = "worker-0001".into();
        second.component_sha256 = component_digest(&second).unwrap();
        bundle.assignments.push(second);
        let selected = selected_worker_assignments(&bundle, "worker-0000", 2).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|assignment| assignment.worker_id.as_str())
                .collect::<Vec<_>>(),
            ["worker-0000", "worker-0001"]
        );
        assert!(selected_worker_assignments(&bundle, "worker-0001", 2).is_err());
        assert!(selected_worker_assignments(&bundle, "worker-0000", 0).is_err());
        assert!(selected_worker_assignments(&bundle, "unknown", 1).is_err());
    }

    #[tokio::test]
    async fn waits_for_a_root_contained_observation() {
        let root = tempfile::tempdir().unwrap();
        let path = root
            .path()
            .join("runtime/worker/attempts/attempt-1/observation.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let observation = WorkerObservation {
            schema_version: "livefire.rag.runpod-worker-observation/1".into(),
            machine: RunpodMachineIdentity {
                pod_id: "pod-1".into(),
                machine_id: "machine-1".into(),
            },
            accelerator: RunpodAcceleratorIdentity {
                provider: "runpod".into(),
                model: "gpu".into(),
                architecture: "cuda".into(),
                compute_capability: "8.0".into(),
                count: 1,
            },
        };
        rag_pipeline::write_canonical_json(&path, &observation).unwrap();
        assert_eq!(
            wait_for_observation(
                root.path(),
                "runtime/worker/attempts/attempt-1/observation.json",
                Duration::from_secs(1)
            )
            .await
            .unwrap(),
            observation
        );
        assert!(
            wait_for_observation(root.path(), "../escape", Duration::ZERO)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn fake_backend_publishes_and_reuses_a_complete_assignment() {
        let root = tempfile::tempdir().unwrap();
        let work = fake_work(root.path());
        let first = execute_assignment(&work, Arc::new(FakeBackend), "attempt-1", 1, 1, 1, None)
            .await
            .unwrap();
        assert_eq!(first.outputs.len(), 1);
        let query_vectors = first.query_vector_set.as_ref().unwrap();
        assert_eq!(
            query_vectors.query_plan.sha256,
            work.bundle.artifacts.query_plan.sha256
        );
        assert!(root.path().join("query-vectors/manifest.json").is_file());
        assert!(root.path().join("query-vectors/vectors.f32le").is_file());
        let second = execute_assignment(&work, Arc::new(FakeBackend), "attempt-1", 1, 1, 1, None)
            .await
            .unwrap();
        assert_eq!(first, second);
        assert!(
            root.path()
                .join("attempts/worker-0000/completed.json")
                .is_file()
        );
    }

    #[test]
    fn storage_probe_publishes_reads_and_cleans_up() {
        let root = tempfile::tempdir().unwrap();
        storage_probe(root.path().to_path_buf()).unwrap();
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn storage_object_check_reads_safe_sorted_objects() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("inputs")).unwrap();
        fs::write(root.path().join("inputs/a"), b"abc").unwrap();
        fs::write(root.path().join("inputs/b"), b"defgh").unwrap();
        assert_eq!(
            verify_storage_objects(
                root.path(),
                &[
                    "inputs/a".into(),
                    "3".into(),
                    digest_bytes(b"abc").to_string(),
                    "inputs/b".into(),
                    "5".into(),
                    digest_bytes(b"defgh").to_string(),
                ]
            )
            .unwrap(),
            (2, 8)
        );
        assert!(
            verify_storage_objects(
                root.path(),
                &[
                    "inputs/b".into(),
                    "5".into(),
                    digest_bytes(b"defgh").to_string(),
                    "inputs/a".into(),
                    "3".into(),
                    digest_bytes(b"abc").to_string(),
                ]
            )
            .is_err()
        );
        assert!(
            verify_storage_objects(
                root.path(),
                &[
                    "../escape".into(),
                    "0".into(),
                    digest_bytes(b"").to_string()
                ]
            )
            .is_err()
        );
        assert!(
            verify_storage_objects(
                root.path(),
                &[
                    "inputs/a".into(),
                    "3".into(),
                    digest_bytes(b"wrong").to_string()
                ]
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn storage_challenge_waits_for_exact_bytes_and_publishes_bound_response() {
        let root = tempfile::tempdir().unwrap();
        let root_path = root.path().to_path_buf();
        let image = format!("ghcr.io/example/worker@sha256:{}", "a".repeat(64));
        let payload = b"fresh host challenge".to_vec();
        let challenge = CloudObjectRef {
            key: SafeRelativePath::new("runtime/storage-challenge/challenge.bin").unwrap(),
            bytes: payload.len() as u64,
            sha256: digest_bytes(&payload),
        };
        let expected =
            RunpodStorageChallengeResponse::new(image.clone(), challenge.clone()).unwrap();
        let task = tokio::spawn(storage_challenge(StorageChallengeOptions {
            root: root_path.clone(),
            executor_image: image,
            challenge: challenge.key.to_string(),
            challenge_bytes: challenge.bytes,
            challenge_sha256: challenge.sha256.to_string(),
            response: "runtime/storage-challenge/response.json".into(),
            wait_seconds: 2,
        }));
        sleep(Duration::from_millis(50)).await;
        let challenge_path = challenge.key.join_to(&root_path);
        fs::create_dir_all(challenge_path.parent().unwrap()).unwrap();
        fs::write(challenge_path, payload).unwrap();
        task.await.unwrap().unwrap();
        let response_path = root_path.join("runtime/storage-challenge/response.json");
        assert_eq!(
            fs::read(response_path).unwrap(),
            canonical_json_bytes(&expected).unwrap()
        );
    }

    #[tokio::test]
    async fn storage_challenge_rejects_changed_or_existing_output() {
        let root = tempfile::tempdir().unwrap();
        let challenge_path = root.path().join("challenge.bin");
        fs::write(&challenge_path, b"changed").unwrap();
        let options = StorageChallengeOptions {
            root: root.path().to_path_buf(),
            executor_image: format!("ghcr.io/example/worker@sha256:{}", "a".repeat(64)),
            challenge: "challenge.bin".into(),
            challenge_bytes: 8,
            challenge_sha256: digest_bytes(b"expected").to_string(),
            response: "response.json".into(),
            wait_seconds: 1,
        };
        assert!(storage_challenge(options.clone()).await.is_err());
        fs::write(root.path().join("response.json"), b"stale").unwrap();
        assert!(storage_challenge(options).await.is_err());
    }

    #[test]
    fn run_root_must_be_a_real_directory_strictly_below_workspace() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let run = workspace.join("run-1");
        fs::create_dir_all(&run).unwrap();
        assert_eq!(
            validate_run_root_under(&workspace, &run).unwrap(),
            fs::canonicalize(&run).unwrap()
        );
        assert!(validate_run_root_under(&workspace, &workspace).is_err());
        assert!(validate_run_root_under(&workspace, &workspace.join("run-1/../run-1")).is_err());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&run, workspace.join("linked-run")).unwrap();
            assert!(validate_run_root_under(&workspace, &workspace.join("linked-run")).is_err());
        }
    }
}
