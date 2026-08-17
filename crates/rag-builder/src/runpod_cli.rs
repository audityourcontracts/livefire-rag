//! Host-side commands for sealed RunPod embedding runs.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use clap::{Args, Subcommand};
use rag_embedding::{parse_tei_checkpoint_profile_v3, parse_tei_model_artifact_set_v1};
use rag_pipeline::{
    CloudComponentArtifact, CloudObjectRef, CloudPreparedDocumentArtifact, ComponentRef, Digest,
    ExactTokenizer, ExecutableTokenizerRef, RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA,
    RunpodAcceleratorIdentity, RunpodBundleArtifacts, RunpodEmbeddingBundle,
    RunpodExecutionIdentity, RunpodExecutorImageBuildReceipt, RunpodMachineIdentity,
    RunpodRunReport, RunpodStorageChallengeResponse, RunpodTeiArtifactObject,
    RunpodTeiBoundArtifact, RunpodTeiConformanceCandidate, RunpodTeiConformanceOutcome,
    RunpodTeiConformanceResult, RunpodTeiImageIdentity, RunpodWorkerAttemptMarker,
    SafeRelativePath, SealedQueryVectorSet, TokenizerArtifactFormat, WorkerAttemptOutcome,
    build_runpod_embedding_bundle, build_runpod_run_report, component_digest,
    query_vector_plan_queries, read_json, seal_embedding_policy_v3_conformance,
    write_canonical_json,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};
use tempfile::Builder;
use thiserror::Error;

use crate::{portable, runpod_control, runpod_s3};

const BUNDLE_FILE: &str = "bundle.json";

#[derive(Debug, Error)]
pub(crate) enum RunpodCliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Pipeline(#[from] rag_pipeline::PipelineError),
    #[error(transparent)]
    Embedding(#[from] rag_embedding::EmbeddingError),
    #[error(transparent)]
    S3(#[from] runpod_s3::RunpodS3Error),
    #[error(transparent)]
    Control(#[from] runpod_control::RunpodControlError),
    #[error("local prepared-corpus or plan validation failed: {0}")]
    Portable(String),
    #[error("RunPod bundle input is invalid: {0}")]
    Invalid(&'static str),
    #[error(
        "created Pod could not receive its sealed machine observation: {reason}; Pod cleanup {cleanup}"
    )]
    ObservationStage {
        reason: String,
        cleanup: &'static str,
    },
    #[error(
        "created RunPod resource could not be recorded locally: {reason}; resource cleanup {cleanup}"
    )]
    StateWrite {
        reason: String,
        cleanup: &'static str,
    },
    #[error("supervised RunPod execution ended with outcome {0}")]
    SupervisedRun(&'static str),
}

pub(crate) type Result<T> = std::result::Result<T, RunpodCliError>;

#[derive(Debug, Subcommand)]
pub(crate) enum RunpodCommand {
    /// Seal or validate the exact custom worker image build identity.
    ExecutorImage {
        #[command(subcommand)]
        command: ExecutorImageCommand,
    },
    /// Build or validate the immutable files needed by cloud workers.
    Bundle {
        #[command(subcommand)]
        command: BundleCommand,
    },
    /// Upload only the exact files declared by a validated bundle.
    Stage(StageOptions),
    /// Fetch deterministic completion markers and only their declared outputs.
    Fetch(FetchOptions),
    /// Verify downloaded markers, outputs, and the host-built run report.
    Verify(VerifyOptions),
    /// Create, inspect, or explicitly terminate a network volume.
    Volume {
        #[command(subcommand)]
        command: VolumeCommand,
    },
    /// Preview, create, inspect, or explicitly terminate one worker Pod.
    Pod {
        #[command(subcommand)]
        command: PodCommand,
    },
    /// Bootstrap measured TEI conformance before sealing a cloud policy.
    Conformance {
        #[command(subcommand)]
        command: ConformanceCommand,
    },
    /// Prove host upload, mounted-volume access, and immutable worker publication.
    StorageChallenge {
        #[command(subcommand)]
        command: StorageChallengeCommand,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum StorageChallengeCommand {
    /// Print the exact redacted Pod request using a fresh local challenge.
    DryRun(StorageChallengeOptions),
    /// Run one capped challenge and always request Pod termination.
    Create(StorageChallengeOptions),
}

#[derive(Debug, Subcommand)]
pub(crate) enum ExecutorImageCommand {
    /// Seal a receipt from digest-pinned images and exact local build outputs.
    Seal(SealExecutorImageOptions),
    /// Re-open a receipt and rehash its Dockerfile and exported worker binary.
    Validate(ValidateExecutorImageOptions),
}

#[derive(Debug, Args)]
pub(crate) struct SealExecutorImageOptions {
    /// Custom executor image as repository@sha256:<64 lowercase hex characters>.
    #[arg(long)]
    executor_image: String,
    /// Stable component name for the custom executor image.
    #[arg(long, default_value = "livefire.rag.runpod-executor-image")]
    executor_component_id: String,
    /// Human-readable build version for the custom executor image.
    #[arg(long)]
    executor_version: String,
    /// Official TEI base as repository@sha256:<64 lowercase hex characters>.
    #[arg(long)]
    tei_base_image: String,
    /// Stable component name for the official TEI image.
    #[arg(long, default_value = "huggingface.text-embeddings-inference")]
    tei_base_component_id: String,
    /// Exact official TEI release version, for example 1.9.3.
    #[arg(long)]
    tei_base_version: String,
    /// Dockerfile used for this exact image build.
    #[arg(long)]
    dockerfile: PathBuf,
    /// Exported Linux/AMD64 worker copied into the final image.
    #[arg(long)]
    worker_binary: PathBuf,
    /// Portable Dockerfile path recorded in the receipt.
    #[arg(long, default_value = "container/Dockerfile")]
    dockerfile_object_path: String,
    /// Portable worker path recorded in the receipt.
    #[arg(long, default_value = "bin/rag-runpod-worker")]
    worker_object_path: String,
    /// New canonical JSON receipt. Existing files are refused.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ValidateExecutorImageOptions {
    /// Receipt created by `rag runpod executor-image seal`.
    #[arg(long)]
    receipt: PathBuf,
    /// Dockerfile whose bytes must match the receipt.
    #[arg(long)]
    dockerfile: PathBuf,
    /// Exported worker binary whose bytes must match the receipt.
    #[arg(long)]
    worker_binary: PathBuf,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ConformanceCommand {
    /// Seal and copy a strict candidate template with every exact input object.
    Build(ConformanceBuildOptions),
    /// Re-open a candidate directory and hash every input object.
    Validate(ConformanceValidateOptions),
    /// Upload only the candidate and its exact declared input objects.
    Stage(ConformanceStageOptions),
    /// Fetch one deterministic result and its exact normalized-vector object.
    Fetch(ConformanceFetchOptions),
    /// Print one conformance Pod request without reading credentials.
    PodDryRun(ConformanceLaunchOptions),
    /// Launch one price-capped conformance Pod and stage its observation.
    PodCreate(ConformanceLaunchOptions),
    /// Seal policy/3 from a strict draft and two fresh-Pod results.
    Seal(ConformanceSealOptions),
}

#[derive(Debug, Subcommand)]
pub(crate) enum VolumeCommand {
    /// Create one network volume and save its returned identity.
    Create(CreateVolumeOptions),
    /// Read one network volume by its exact ID.
    Status(ControlStatusOptions),
    /// Permanently delete one network volume after an exact-ID confirmation.
    Terminate(ControlTerminateOptions),
}

#[derive(Debug, Subcommand)]
pub(crate) enum PodCommand {
    /// Print the complete redacted request without reading credentials.
    DryRun(PodLaunchOptions),
    /// Create one Secure Cloud Pod under an explicit hourly price cap.
    Create(PodLaunchOptions),
    /// Read one Pod and its admitted machine identity.
    Status(ControlStatusOptions),
    /// Permanently terminate one Pod after an exact-ID confirmation.
    Terminate(ControlTerminateOptions),
}

#[derive(Debug, Subcommand)]
pub(crate) enum BundleCommand {
    /// Build a sealed bundle directory without contacting RunPod.
    Build(Box<BuildBundleOptions>),
    /// Re-open a bundle and verify all local bytes and bindings.
    Validate(ValidateBundleOptions),
}

#[derive(Debug, Args)]
pub(crate) struct BuildBundleOptions {
    /// Prepared corpus directory containing manifest.json.
    #[arg(long)]
    prepared: PathBuf,
    /// V2 embedding plan directory containing plan.json and token counts.
    #[arg(long)]
    plan: PathBuf,
    /// Exact measured embedding-policy/3 JSON used to create the plan.
    #[arg(long)]
    embedding_policy: PathBuf,
    /// Sealed receipt binding the custom image to its base, Dockerfile, and worker.
    #[arg(long)]
    executor_image_build: PathBuf,
    /// Executable tokenizer.json bound by the plan and policy.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Complete, pinned model artifact-set JSON.
    #[arg(long)]
    model_manifest: PathBuf,
    /// Directory containing every path declared by the model manifest.
    #[arg(long)]
    model_root: PathBuf,
    /// Conformance fixture bound by the measured embedding policy.
    #[arg(long)]
    conformance_fixture: PathBuf,
    /// Frozen JSONL catalogue query plan embedded by worker 0000 while TEI is warm.
    #[arg(long)]
    query_plan: PathBuf,
    /// Rust worker binary whose digest is also present in the execution identity.
    #[arg(long)]
    worker_binary: PathBuf,
    /// Exact RunPod execution identity JSON, including image and accelerator.
    #[arg(long)]
    execution: PathBuf,
    /// Number of one-GPU workers. Tasks are split into token-balanced ranges.
    #[arg(long)]
    workers: u32,
    /// New local bundle directory. Existing paths are refused.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ValidateBundleOptions {
    /// Local bundle directory created by `rag runpod bundle build`.
    #[arg(long)]
    bundle: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct StageOptions {
    /// Validated local bundle directory.
    #[arg(long)]
    bundle: PathBuf,
    /// Unique, safe path below the RunPod network-volume bucket.
    #[arg(long)]
    run_prefix: String,
    #[arg(long)]
    network_volume_id: String,
    #[arg(long)]
    datacenter_id: String,
    /// Environment variable containing the S3 access key.
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    /// Environment variable containing the S3 secret key.
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
}

#[derive(Debug, Args)]
pub(crate) struct FetchOptions {
    #[arg(long)]
    bundle: PathBuf,
    #[arg(long)]
    run_prefix: String,
    #[arg(long)]
    network_volume_id: String,
    #[arg(long)]
    datacenter_id: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
    /// New local directory for markers, task outputs, and run-report.json.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct VerifyOptions {
    #[arg(long)]
    bundle: PathBuf,
    /// Directory produced by `rag runpod fetch`.
    #[arg(long)]
    fetched: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct CreateVolumeOptions {
    #[arg(long)]
    name: String,
    /// Volume capacity in gigabytes.
    #[arg(long)]
    size_gb: u32,
    #[arg(long)]
    datacenter_id: String,
    /// New file that receives the returned volume identity.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
}

#[derive(Debug, Args)]
pub(crate) struct ControlStatusOptions {
    #[arg(long)]
    id: String,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
}

#[derive(Debug, Args)]
pub(crate) struct ControlTerminateOptions {
    #[arg(long)]
    id: String,
    /// Must exactly equal --id. This prevents an accidental broad deletion.
    #[arg(long)]
    confirm_terminate: String,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
}

#[derive(Debug, Args)]
pub(crate) struct PodLaunchOptions {
    /// Validated local bundle directory.
    #[arg(long)]
    bundle: PathBuf,
    /// Saved JSON identity returned by `rag runpod volume create` or status.
    #[arg(long)]
    volume: PathBuf,
    /// Immutable image reference ending in @sha256:<digest>.
    #[arg(long)]
    image: String,
    #[arg(long)]
    gpu_type_id: String,
    #[arg(long)]
    name: String,
    #[arg(long)]
    worker_id: String,
    #[arg(long)]
    attempt_id: String,
    #[arg(long, default_value_t = 1)]
    attempt_number: u32,
    #[arg(long)]
    run_prefix: String,
    #[arg(long, default_value_t = 50)]
    container_disk_gb: u32,
    /// Refuse and clean up a Pod whose returned adjusted hourly price is higher.
    #[arg(long)]
    maximum_hourly_price: f64,
    /// Hard wall-clock limit after Pod creation.
    #[arg(long)]
    maximum_runtime_seconds: u64,
    /// Hard compute-spend limit calculated from the returned hourly price.
    #[arg(long)]
    maximum_total_compute_usd: f64,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
    /// New file that receives the final watchdog and termination receipt.
    #[arg(long)]
    out: Option<PathBuf>,
    /// New file that records the full admitted Pod before observation staging.
    #[arg(long)]
    launch_out: Option<PathBuf>,
    /// New file that records the scheduler-returned Pod ID and requested deletion deadline.
    #[arg(long)]
    create_out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceBuildOptions {
    #[arg(long)]
    template: PathBuf,
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceValidateOptions {
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceStageOptions {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    run_prefix: String,
    #[arg(long)]
    network_volume_id: String,
    #[arg(long)]
    datacenter_id: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceFetchOptions {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    run_prefix: String,
    #[arg(long)]
    run_id: String,
    #[arg(long)]
    network_volume_id: String,
    #[arg(long)]
    datacenter_id: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceSealOptions {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    first_result: PathBuf,
    #[arg(long)]
    fresh_pod_replay_result: PathBuf,
    #[arg(long)]
    policy_draft: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ConformanceLaunchOptions {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    volume: PathBuf,
    #[arg(long)]
    image: String,
    #[arg(long)]
    gpu_type_id: String,
    #[arg(long)]
    name: String,
    #[arg(long)]
    run_id: String,
    #[arg(long)]
    run_prefix: String,
    #[arg(long, default_value_t = 50)]
    container_disk_gb: u32,
    #[arg(long)]
    maximum_hourly_price: f64,
    #[arg(long)]
    maximum_runtime_seconds: u64,
    #[arg(long)]
    maximum_total_compute_usd: f64,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
    #[arg(long)]
    out: Option<PathBuf>,
    /// New file that records the full admitted Pod before observation staging.
    #[arg(long)]
    launch_out: Option<PathBuf>,
    /// New file that records the scheduler-returned Pod ID and requested deletion deadline.
    #[arg(long)]
    create_out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct StorageChallengeOptions {
    /// Sealed receipt for the exact custom image and Rust worker binary.
    #[arg(long)]
    executor_image_build: PathBuf,
    /// Saved JSON identity returned by `rag runpod volume create` or status.
    #[arg(long)]
    volume: PathBuf,
    /// Immutable image reference ending in @sha256:<digest>.
    #[arg(long)]
    image: String,
    #[arg(long)]
    gpu_type_id: String,
    #[arg(long)]
    name: String,
    /// Unique, unused path below the RunPod network-volume bucket.
    #[arg(long)]
    run_prefix: String,
    #[arg(long, default_value_t = 20)]
    container_disk_gb: u32,
    #[arg(long)]
    maximum_hourly_price: f64,
    #[arg(long)]
    maximum_runtime_seconds: u64,
    #[arg(long)]
    maximum_total_compute_usd: f64,
    #[arg(long, default_value = "RUNPOD_API_KEY")]
    api_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_ACCESS_KEY")]
    access_key_environment: String,
    #[arg(long, default_value = "RUNPOD_S3_SECRET_KEY")]
    secret_key_environment: String,
    /// New final content-bound challenge and watchdog receipt.
    #[arg(long)]
    out: Option<PathBuf>,
    /// New file containing the full admitted Pod identity and returned price.
    #[arg(long)]
    launch_out: Option<PathBuf>,
    /// New file containing the scheduler request digest and termination deadline.
    #[arg(long)]
    create_out: Option<PathBuf>,
}

pub(crate) async fn run(command: RunpodCommand) -> Result<()> {
    match command {
        RunpodCommand::ExecutorImage { command } => run_executor_image(command),
        RunpodCommand::Bundle { command } => match command {
            BundleCommand::Build(options) => build_bundle(*options),
            BundleCommand::Validate(options) => {
                let bundle = load_and_validate_bundle(&options.bundle)?;
                println!("{}", serde_json::to_string_pretty(&bundle)?);
                Ok(())
            }
        },
        RunpodCommand::Stage(options) => stage(options).await,
        RunpodCommand::Fetch(options) => fetch(options).await,
        RunpodCommand::Verify(options) => verify_fetched(&options.bundle, &options.fetched),
        RunpodCommand::Volume { command } => run_volume(command).await,
        RunpodCommand::Pod { command } => run_pod(command).await,
        RunpodCommand::Conformance { command } => run_conformance(command).await,
        RunpodCommand::StorageChallenge { command } => run_storage_challenge(command).await,
    }
}

fn run_executor_image(command: ExecutorImageCommand) -> Result<()> {
    match command {
        ExecutorImageCommand::Seal(options) => seal_executor_image(options),
        ExecutorImageCommand::Validate(options) => {
            let receipt = validate_executor_image_files(
                &options.receipt,
                &options.dockerfile,
                &options.worker_binary,
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            Ok(())
        }
    }
}

fn seal_executor_image(options: SealExecutorImageOptions) -> Result<()> {
    let executor_image = parse_digest_pinned_image(
        &options.executor_image,
        options.executor_component_id,
        options.executor_version,
    )?;
    let tei_base_image = parse_digest_pinned_image(
        &options.tei_base_image,
        options.tei_base_component_id,
        options.tei_base_version,
    )?;
    let dockerfile = tei_artifact_for_file(
        &options.dockerfile,
        &options.dockerfile_object_path,
        "text/x-dockerfile",
    )?;
    let worker_object = tei_artifact_for_file(
        &options.worker_binary,
        &options.worker_object_path,
        "application/octet-stream",
    )?;
    let mut receipt = RunpodExecutorImageBuildReceipt {
        schema_version: RUNPOD_EXECUTOR_IMAGE_BUILD_RECEIPT_SCHEMA.into(),
        component_sha256: digest(b"unsealed executor image build receipt"),
        executor_image,
        tei_base_image,
        platform: "linux/amd64".into(),
        dockerfile,
        worker_binary: RunpodTeiBoundArtifact {
            component_sha256: worker_object.sha256.clone(),
            object: worker_object,
        },
    };
    receipt.seal()?;
    let mut output = ReservedJsonOutput::new(&options.out)?;
    output.write(&receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn validate_executor_image_files(
    receipt_path: &Path,
    dockerfile_path: &Path,
    worker_binary_path: &Path,
) -> Result<RunpodExecutorImageBuildReceipt> {
    let receipt: RunpodExecutorImageBuildReceipt = read_json(receipt_path)?;
    receipt.validate()?;
    let dockerfile = tei_artifact_for_file(
        dockerfile_path,
        receipt.dockerfile.path.as_str(),
        "text/x-dockerfile",
    )?;
    let worker = tei_artifact_for_file(
        worker_binary_path,
        receipt.worker_binary.object.path.as_str(),
        "application/octet-stream",
    )?;
    if dockerfile != receipt.dockerfile
        || worker != receipt.worker_binary.object
        || receipt.worker_binary.component_sha256 != worker.sha256
    {
        return Err(RunpodCliError::Invalid(
            "executor image build inputs differ from the sealed receipt",
        ));
    }
    Ok(receipt)
}

fn parse_digest_pinned_image(
    reference: &str,
    component_id: String,
    version: String,
) -> Result<RunpodTeiImageIdentity> {
    let (repository, image_digest) =
        reference
            .rsplit_once("@sha256:")
            .ok_or(RunpodCliError::Invalid(
                "image reference must end in @sha256:<64 lowercase hex characters>",
            ))?;
    if repository.is_empty() || repository.contains('@') {
        return Err(RunpodCliError::Invalid(
            "image reference must contain one digest-pinned repository",
        ));
    }
    let sha256 = Digest::new(image_digest)?;
    let identity = RunpodTeiImageIdentity {
        component: ComponentRef {
            id: component_id,
            version,
            sha256: sha256.clone(),
        },
        repository: repository.into(),
        digest: format!("sha256:{sha256}"),
    };
    // The receipt validator is the single authority for component text and
    // image-reference rules. A temporary receipt would obscure errors here,
    // so validate the complete object after construction in `seal`.
    Ok(identity)
}

fn tei_artifact_for_file(
    path: &Path,
    object_path: &str,
    media_type: &str,
) -> Result<RunpodTeiArtifactObject> {
    let object = object_for_file(path, object_path)?;
    Ok(RunpodTeiArtifactObject {
        path: object.key,
        media_type: media_type.into(),
        bytes: object.bytes,
        sha256: object.sha256,
    })
}

async fn run_conformance(command: ConformanceCommand) -> Result<()> {
    match command {
        ConformanceCommand::Build(options) => build_conformance_candidate(options),
        ConformanceCommand::Validate(options) => {
            let candidate = load_conformance_candidate(&options.candidate)?;
            println!("{}", serde_json::to_string_pretty(&candidate)?);
            Ok(())
        }
        ConformanceCommand::Stage(options) => stage_conformance_candidate(options).await,
        ConformanceCommand::Fetch(options) => fetch_conformance_result(options).await,
        ConformanceCommand::PodDryRun(options) => conformance_pod_dry_run(options),
        ConformanceCommand::PodCreate(options) => conformance_pod_create(options).await,
        ConformanceCommand::Seal(options) => seal_conformance_policy(options),
    }
}

fn build_conformance_candidate(options: ConformanceBuildOptions) -> Result<()> {
    refuse_existing(&options.out)?;
    let mut candidate: RunpodTeiConformanceCandidate = read_json(&options.template)?;
    candidate.seal()?;
    let parent = options
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = Builder::new()
        .prefix(".runpod-conformance-")
        .tempdir_in(parent)?;
    for object in conformance_input_objects(&options.input_root, &candidate)? {
        let source = rag_pipeline::resolve_existing_artifact(&options.input_root, &object.key)?;
        let staged = staged_object(staging.path(), &source, object.key.as_str())?;
        if staged != object {
            return Err(RunpodCliError::Invalid(
                "conformance input differs from the candidate template",
            ));
        }
    }
    write_canonical_json(&staging.path().join("candidate.json"), &candidate)?;
    validate_conformance_candidate_values(staging.path(), &candidate)?;
    let staged_root = staging.keep();
    publish_without_overwrite(&staged_root, &options.out)?;
    println!("{}", serde_json::to_string_pretty(&candidate)?);
    Ok(())
}

fn load_conformance_candidate(root: &Path) -> Result<RunpodTeiConformanceCandidate> {
    let candidate: RunpodTeiConformanceCandidate = read_json(&root.join("candidate.json"))?;
    validate_conformance_candidate_values(root, &candidate)?;
    Ok(candidate)
}

fn validate_conformance_candidate_values(
    root: &Path,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<()> {
    candidate.validate()?;
    for object in conformance_input_objects(root, candidate)? {
        verify_local_object(root, &object)?;
    }
    let manifest_bytes = read_regular_file(&candidate.model_manifest.object.path.join_to(root))?;
    let manifest = parse_tei_model_artifact_set_v1(&manifest_bytes)?;
    if manifest.sha256()? != candidate.model_manifest.component_sha256.as_str()
        || manifest.objects.len() != candidate.model_objects.len()
    {
        return Err(RunpodCliError::Invalid(
            "candidate model manifest differs from its complete object list",
        ));
    }
    for (manifest_object, candidate_object) in
        manifest.objects.iter().zip(candidate.model_objects.iter())
    {
        if manifest_object.path != candidate_object.path.as_str()
            || manifest_object.media_type != candidate_object.media_type
            || manifest_object.bytes != candidate_object.bytes
            || manifest_object.sha256 != candidate_object.sha256.as_str()
        {
            return Err(RunpodCliError::Invalid(
                "candidate model manifest differs from its complete object list",
            ));
        }
    }
    let tokenizer_path =
        rag_pipeline::resolve_existing_artifact(root, &candidate.tokenizer.object.path)?;
    let tokenizer = ExactTokenizer::from_bytes(
        ExecutableTokenizerRef {
            artifact: candidate.tokenizer.component.clone(),
            format: TokenizerArtifactFormat::HuggingFaceTokenizerJson,
            model_revision: candidate.tokenizer.revision.clone(),
            target_tokenizer: candidate.tokenizer.component.clone(),
            add_special_tokens: candidate.tokenizer.add_special_tokens,
            maximum_input_bytes: candidate.fixture.object.bytes,
        },
        &read_regular_file(&tokenizer_path)?,
    )?;
    let fixture_path =
        rag_pipeline::resolve_existing_artifact(root, &candidate.fixture.object.path)?;
    let fixture: rag_embedding::TeiConformanceFixtureV1 =
        serde_json::from_slice(&read_regular_file(&fixture_path)?)?;
    if fixture.schema_version != rag_embedding::TEI_CONFORMANCE_FIXTURE_SCHEMA_V1
        || fixture.inputs.len() != candidate.fixture.input_count as usize
        || fixture.inputs.iter().any(String::is_empty)
    {
        return Err(RunpodCliError::Invalid(
            "conformance fixture contents differ from the candidate",
        ));
    }
    for input in &fixture.inputs {
        if tokenizer.count(input)? > u64::from(candidate.execution.maximum_tokens) {
            return Err(RunpodCliError::Invalid(
                "conformance fixture input exceeds the exact token limit",
            ));
        }
    }
    Ok(())
}

fn conformance_input_objects(
    root: &Path,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<Vec<CloudObjectRef>> {
    let mut objects = vec![
        conformance_object(&candidate.model_manifest.object),
        conformance_object(&candidate.tokenizer.object),
        conformance_object(&candidate.executor_image_build.object),
        conformance_object(&candidate.worker_binary.object),
        conformance_object(&candidate.fixture.object),
    ];
    let receipt_path =
        rag_pipeline::resolve_existing_artifact(root, &candidate.executor_image_build.object.path)?;
    let receipt: RunpodExecutorImageBuildReceipt =
        serde_json::from_slice(&read_regular_file(&receipt_path)?)?;
    validate_executor_image_build_receipt(&receipt, candidate)?;
    objects.push(conformance_object(&receipt.dockerfile));
    objects.extend(candidate.model_objects.iter().map(conformance_object));
    let mut unique = std::collections::BTreeMap::new();
    for object in objects {
        if let Some(existing) = unique.insert(object.key.clone(), object.clone())
            && existing != object
        {
            return Err(RunpodCliError::Invalid(
                "candidate repeats one path with a different identity",
            ));
        }
    }
    if unique.contains_key(&candidate.expected_output_key)
        || unique.keys().any(|path| {
            path.as_str() == "candidate.json"
                || path.as_str().starts_with("runtime/")
                || path.as_str().starts_with("conformance/results/")
        })
    {
        return Err(RunpodCliError::Invalid(
            "candidate input collides with a control or output path",
        ));
    }
    Ok(unique.into_values().collect())
}

fn conformance_object(object: &rag_pipeline::RunpodTeiArtifactObject) -> CloudObjectRef {
    CloudObjectRef {
        key: object.path.clone(),
        bytes: object.bytes,
        sha256: object.sha256.clone(),
    }
}

fn validate_executor_image_build_receipt(
    receipt: &RunpodExecutorImageBuildReceipt,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<()> {
    receipt.validate()?;
    if receipt.component_sha256 != candidate.executor_image_build.component.sha256
        || !same_json(&receipt.executor_image, &candidate.executor_image)?
        || !same_json(&receipt.tei_base_image, &candidate.tei_image)?
        || receipt.worker_binary.component_sha256 != candidate.worker_binary.component_sha256
        || !same_json(
            &receipt.worker_binary.object,
            &candidate.worker_binary.object,
        )?
    {
        return Err(RunpodCliError::Invalid(
            "executor image build receipt differs from the conformance candidate",
        ));
    }
    Ok(())
}

async fn stage_conformance_candidate(options: ConformanceStageOptions) -> Result<()> {
    let candidate = load_conformance_candidate(&options.candidate)?;
    let mut objects = conformance_input_objects(&options.candidate, &candidate)?;
    objects.push(object_for_file(
        &options.candidate.join("candidate.json"),
        "candidate.json",
    )?);
    let manifest = runpod_s3::RunpodS3Manifest::new(
        SafeRelativePath::new(options.run_prefix)?,
        objects.clone(),
    )?;
    let client = runpod_s3::RunpodS3Client::from_environment(
        &options.network_volume_id,
        &options.datacenter_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    for object in &objects {
        match client.head_object(&manifest, object).await? {
            runpod_s3::HeadObjectState::Present => {}
            runpod_s3::HeadObjectState::Missing => {
                client
                    .put_object(&manifest, object, &options.candidate)
                    .await?;
            }
        }
    }
    println!("staged {} exact conformance objects", objects.len());
    Ok(())
}

async fn fetch_conformance_result(options: ConformanceFetchOptions) -> Result<()> {
    refuse_existing(&options.out)?;
    let candidate = load_conformance_candidate(&options.candidate)?;
    let run_prefix = SafeRelativePath::new(options.run_prefix)?;
    let client = runpod_s3::RunpodS3Client::from_environment(
        &options.network_volume_id,
        &options.datacenter_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    let parent = options
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = Builder::new()
        .prefix(".runpod-conformance-result-")
        .tempdir_in(parent)?;
    let (result, _) = client
        .get_conformance_result(
            run_prefix.clone(),
            &candidate,
            &options.run_id,
            staging.path(),
        )
        .await?;
    if result.outcome != RunpodTeiConformanceOutcome::Completed {
        return Err(RunpodCliError::Invalid(
            "conformance result did not complete successfully",
        ));
    }
    let normalized = result
        .normalized_output
        .as_ref()
        .ok_or(RunpodCliError::Invalid(
            "completed conformance result has no normalized output",
        ))?;
    let output = conformance_object(&normalized.object);
    let manifest = runpod_s3::RunpodS3Manifest::new(run_prefix, [output.clone()])?;
    client
        .get_worker_output(&manifest, &output, staging.path())
        .await?;
    verify_local_object(staging.path(), &output)?;
    let staged_root = staging.keep();
    publish_without_overwrite(&staged_root, &options.out)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn seal_conformance_policy(options: ConformanceSealOptions) -> Result<()> {
    let candidate = load_conformance_candidate(&options.candidate)?;
    let (first, first_output) = load_fetched_conformance_result(&options.first_result, &candidate)?;
    let (replay, replay_output) =
        load_fetched_conformance_result(&options.fresh_pod_replay_result, &candidate)?;
    if first_output != replay_output {
        return Err(RunpodCliError::Invalid(
            "fresh-Pod conformance normalized-vector bytes differ",
        ));
    }
    let fields = seal_embedding_policy_v3_conformance(&candidate, &first, &replay)?;
    let mut draft: serde_json::Value = read_json(&options.policy_draft)?;
    let object = draft
        .as_object_mut()
        .ok_or(RunpodCliError::Invalid("policy draft is not a JSON object"))?;
    let expected = policy_draft_fields();
    if object
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>()
        != expected
    {
        return Err(RunpodCliError::Invalid(
            "policy draft fields are incomplete or contain an unknown field",
        ));
    }
    if object
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some("livefire.rag.embedding-policy-draft/3")
    {
        return Err(RunpodCliError::Invalid("policy draft schema is invalid"));
    }
    object.insert(
        "schema_version".into(),
        serde_json::Value::String("livefire.rag.embedding-policy/3".into()),
    );
    object.insert("conformance".into(), serde_json::to_value(fields)?);
    let bytes = rag_pipeline::canonical_json_bytes(&draft)?;
    let policy = parse_tei_checkpoint_profile_v3(&bytes)?;
    validate_sealed_policy_candidate_binding(&policy, &candidate)?;
    let mut output = ReservedJsonOutput::new(&options.out)?;
    output.write(&policy)?;
    println!("{}", serde_json::to_string_pretty(&policy)?);
    Ok(())
}

fn load_fetched_conformance_result(
    path: &Path,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<(RunpodTeiConformanceResult, Vec<u8>)> {
    reject_symlink(path)?;
    let (root, result_path) = if path.is_dir() {
        let results = path.join("conformance/results");
        reject_symlink(&results)?;
        let mut result_paths = fs::read_dir(&results)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        result_paths.sort();
        result_paths.retain(|candidate| {
            candidate
                .extension()
                .is_some_and(|extension| extension == "json")
        });
        if result_paths.len() != 1 {
            return Err(RunpodCliError::Invalid(
                "fetched conformance directory must contain exactly one result JSON",
            ));
        }
        (path.to_path_buf(), result_paths.remove(0))
    } else {
        let results = path.parent().ok_or(RunpodCliError::Invalid(
            "conformance result path has no parent",
        ))?;
        let conformance = results.parent().ok_or(RunpodCliError::Invalid(
            "conformance result is not in conformance/results",
        ))?;
        if results.file_name().and_then(|name| name.to_str()) != Some("results")
            || conformance.file_name().and_then(|name| name.to_str()) != Some("conformance")
        {
            return Err(RunpodCliError::Invalid(
                "conformance result file must be below conformance/results",
            ));
        }
        let root = conformance.parent().ok_or(RunpodCliError::Invalid(
            "conformance result path has no fetched root",
        ))?;
        (root.to_path_buf(), path.to_path_buf())
    };
    let result: RunpodTeiConformanceResult = read_json(&result_path)?;
    result.validate_against(candidate)?;
    if result.outcome != RunpodTeiConformanceOutcome::Completed {
        return Err(RunpodCliError::Invalid(
            "conformance policy sealing requires completed results",
        ));
    }
    let normalized = result
        .normalized_output
        .as_ref()
        .ok_or(RunpodCliError::Invalid(
            "completed conformance result has no normalized output",
        ))?;
    let object = conformance_object(&normalized.object);
    verify_local_object(&root, &object)?;
    let output_path = rag_pipeline::resolve_existing_artifact(&root, &object.key)?;
    let bytes = read_regular_file(&output_path)?;
    Ok((result, bytes))
}

fn policy_draft_fields() -> std::collections::BTreeSet<&'static str> {
    [
        "schema_version",
        "admission_status",
        "purpose",
        "model_repository",
        "model_revision",
        "model_snapshot_completeness",
        "model_artifact_set",
        "model_objects",
        "tokenizer",
        "executable_tokenizer",
        "tei_image",
        "executor_image",
        "executor_image_build",
        "runtime",
        "inference_engine",
        "load_policy",
        "runtime_mode",
        "api_contract",
        "api_model_key",
        "dimensions",
        "checkpoint_compute_dtype",
        "api_vector_dtype",
        "stored_vector_dtype",
        "pooling",
        "normalization",
        "maximum_tokens",
        "document_format",
        "query_instruction",
        "query_composition",
        "batching",
        "response_limits",
        "output_processing",
        "accelerator",
    ]
    .into_iter()
    .collect()
}

fn validate_sealed_policy_candidate_binding(
    policy: &rag_embedding::TeiCheckpointProfileV3,
    candidate: &RunpodTeiConformanceCandidate,
) -> Result<()> {
    if policy.model_repository != candidate.model_repository
        || policy.model_revision != candidate.model_revision
        || !same_json(
            &policy.model_artifact_set,
            &candidate.execution.model_artifact_set,
        )?
        || !same_json(&policy.model_objects, &candidate.model_objects)?
        || !same_json(&policy.tei_image, &candidate.tei_image)?
        || !same_json(&policy.executor_image, &candidate.executor_image)?
        || policy.executor_image_build.id != candidate.executor_image_build.component.id
        || policy.executor_image_build.version != candidate.executor_image_build.component.version
        || policy.executor_image_build.sha256
            != candidate.executor_image_build.component.sha256.as_str()
        || !same_json(&policy.runtime, &candidate.runtime)?
        || !same_json(&policy.inference_engine, &candidate.inference_engine)?
        || !same_json(&policy.load_policy, &candidate.load_policy)?
        || !same_json(&policy.accelerator, &candidate.accelerator)?
        || policy.tokenizer.id != candidate.tokenizer.component.id
        || policy.tokenizer.version != candidate.tokenizer.component.version
        || policy.tokenizer.sha256 != candidate.tokenizer.component.sha256.as_str()
        || policy.executable_tokenizer.repository != candidate.tokenizer.repository
        || policy.executable_tokenizer.revision != candidate.tokenizer.revision
        || policy.executable_tokenizer.format != candidate.tokenizer.format
        || !same_json(
            &policy.executable_tokenizer.object,
            &candidate.tokenizer.object,
        )?
        || policy.executable_tokenizer.add_special_tokens != candidate.tokenizer.add_special_tokens
        || policy.api_model_key != candidate.execution.served_model
        || policy.dimensions != candidate.execution.dimensions
        || policy.pooling != candidate.execution.pooling
        || policy.checkpoint_compute_dtype != candidate.execution.forced_runtime_dtype
        || policy.api_vector_dtype != candidate.execution.api_vector_dtype
        || policy.normalization != candidate.execution.normalization
        || policy.maximum_tokens != candidate.execution.maximum_tokens
        || policy_execution_limits(policy) != candidate_execution_limits(candidate)
    {
        return Err(RunpodCliError::Invalid(
            "sealed policy differs from the measured conformance candidate",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SealedExecutionLimits {
    maximum_client_batch_size: u32,
    maximum_batch_tokens: u64,
    maximum_concurrent_requests: u32,
    request_timeout_ms: u64,
    maximum_response_bytes: u64,
}

fn policy_execution_limits(
    policy: &rag_embedding::TeiCheckpointProfileV3,
) -> SealedExecutionLimits {
    SealedExecutionLimits {
        maximum_client_batch_size: policy.batching.maximum_batch_items,
        maximum_batch_tokens: policy.batching.maximum_batch_tokens,
        maximum_concurrent_requests: policy.batching.maximum_concurrent_requests,
        request_timeout_ms: policy.response_limits.request_timeout_ms,
        maximum_response_bytes: policy.response_limits.maximum_response_bytes,
    }
}

fn candidate_execution_limits(candidate: &RunpodTeiConformanceCandidate) -> SealedExecutionLimits {
    SealedExecutionLimits {
        maximum_client_batch_size: candidate.execution.maximum_client_batch_size,
        maximum_batch_tokens: candidate.execution.maximum_batch_tokens,
        maximum_concurrent_requests: candidate.execution.maximum_concurrent_requests,
        request_timeout_ms: candidate.execution.request_timeout_ms,
        maximum_response_bytes: candidate.execution.maximum_response_bytes,
    }
}

fn same_json<L: Serialize, R: Serialize>(left: &L, right: &R) -> Result<bool> {
    Ok(serde_json::to_value(left)? == serde_json::to_value(right)?)
}

fn validate_watchdog_limits(hourly_price: f64, runtime_seconds: u64, total_usd: f64) -> Result<()> {
    if !(300..=604_800).contains(&runtime_seconds)
        || !hourly_price.is_finite()
        || hourly_price <= 0.0
        || !total_usd.is_finite()
        || total_usd <= 0.0
    {
        return Err(RunpodCliError::Invalid(
            "watchdog runtime, hourly price, or total compute USD limit is invalid",
        ));
    }
    Ok(())
}

const STORAGE_CHALLENGE_BOOTSTRAP_KEY: &str = ".storage-challenge-bootstrap";
const STORAGE_CHALLENGE_INPUT_KEY: &str = ".storage-challenge.bin";
// Keep the response directly below the run root. The S3 service may create
// challenge parent directories with provider-selected ownership, while the
// worker explicitly owns the exact run root and must create its own outputs.
const STORAGE_CHALLENGE_RESPONSE_KEY: &str = ".storage-challenge-response.json";

struct PreparedStorageChallenge {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    bootstrap: CloudObjectRef,
    challenge: CloudObjectRef,
    response: CloudObjectRef,
    expected_response: RunpodStorageChallengeResponse,
}

fn prepare_storage_challenge(executor_image: &str) -> Result<PreparedStorageChallenge> {
    let temporary = tempfile::tempdir()?;
    let root = temporary.path().to_path_buf();
    fs::write(
        root.join(STORAGE_CHALLENGE_BOOTSTRAP_KEY),
        b"livefire-rag-storage-challenge-bootstrap-v1\n",
    )?;
    let mut challenge_bytes = [0_u8; 32];
    File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut challenge_bytes))
        .map_err(|_| RunpodCliError::Invalid("operating-system random source is unavailable"))?;
    let challenge_path = root.join(STORAGE_CHALLENGE_INPUT_KEY);
    fs::create_dir_all(
        challenge_path
            .parent()
            .ok_or(RunpodCliError::Invalid("storage challenge input parent"))?,
    )?;
    fs::write(&challenge_path, challenge_bytes)?;
    let bootstrap = object_for_file(
        &root.join(STORAGE_CHALLENGE_BOOTSTRAP_KEY),
        STORAGE_CHALLENGE_BOOTSTRAP_KEY,
    )?;
    let challenge = object_for_file(&challenge_path, STORAGE_CHALLENGE_INPUT_KEY)?;
    let expected_response =
        RunpodStorageChallengeResponse::new(executor_image.to_owned(), challenge.clone())?;
    let response_path = root.join(STORAGE_CHALLENGE_RESPONSE_KEY);
    write_canonical_json(&response_path, &expected_response)?;
    let response = object_for_file(&response_path, STORAGE_CHALLENGE_RESPONSE_KEY)?;
    Ok(PreparedStorageChallenge {
        _temporary: temporary,
        root,
        bootstrap,
        challenge,
        response,
        expected_response,
    })
}

fn storage_challenge_specification(
    options: &StorageChallengeOptions,
    challenge: &CloudObjectRef,
) -> Result<runpod_control::PodCreateSpec> {
    validate_watchdog_limits(
        options.maximum_hourly_price,
        options.maximum_runtime_seconds,
        options.maximum_total_compute_usd,
    )?;
    let receipt: RunpodExecutorImageBuildReceipt = read_json(&options.executor_image_build)?;
    receipt.validate()?;
    let expected_image = format!(
        "{}@{}",
        receipt.executor_image.repository, receipt.executor_image.digest
    );
    if options.image != expected_image {
        return Err(RunpodCliError::Invalid(
            "storage challenge image differs from the sealed executor build",
        ));
    }
    let run_prefix = SafeRelativePath::new(options.run_prefix.clone())?;
    let volume: runpod_control::NetworkVolume = read_json(&options.volume)?;
    Ok(runpod_control::PodCreateSpec {
        name: options.name.clone(),
        image: options.image.clone(),
        gpu_type_id: options.gpu_type_id.clone(),
        network_volume: volume,
        worker_binary: ComponentRef {
            id: "livefire.rag.runpod-worker".into(),
            version: receipt.component_sha256.to_string(),
            sha256: receipt.worker_binary.component_sha256,
        },
        worker_arguments: vec![
            "storage-challenge".into(),
            "--root".into(),
            format!("{}/{}", runpod_control::WORKSPACE_MOUNT, run_prefix),
            "--executor-image-repository".into(),
            receipt.executor_image.repository,
            "--executor-image-digest".into(),
            receipt.executor_image.digest,
            "--challenge".into(),
            challenge.key.to_string(),
            "--challenge-bytes".into(),
            challenge.bytes.to_string(),
            "--challenge-sha256".into(),
            challenge.sha256.to_string(),
            "--response".into(),
            STORAGE_CHALLENGE_RESPONSE_KEY.into(),
            "--wait-seconds".into(),
            options.maximum_runtime_seconds.to_string(),
        ],
        container_disk_gb: options.container_disk_gb,
        maximum_adjusted_hourly_price: options.maximum_hourly_price,
    })
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct RunpodStorageChallengeReceipt {
    schema_version: &'static str,
    component_sha256: Digest,
    executor_image: String,
    pod_id: String,
    machine_id: String,
    network_volume_id: String,
    run_prefix: String,
    bootstrap: CloudObjectRef,
    challenge: CloudObjectRef,
    response: CloudObjectRef,
    response_verified: bool,
    requested_terminate_after: String,
    termination_binding: &'static str,
    schedule_receipt_sha256: Digest,
    launch_receipt_sha256: Digest,
    returned_hourly_price_usd: f64,
    maximum_hourly_price_usd: f64,
    maximum_runtime_seconds: u64,
    maximum_total_compute_usd: f64,
    started_at_ms: u64,
    completed_at_ms: u64,
    elapsed_ms: u64,
    observed_compute_usd: f64,
    outcome: &'static str,
    pod_termination: &'static str,
}

impl RunpodStorageChallengeReceipt {
    fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        Ok(())
    }
}

struct StorageChallengeWatchdog {
    outcome: &'static str,
    pod_termination: &'static str,
    completed_at_ms: u64,
    elapsed_ms: u64,
    observed_compute_usd: f64,
}

struct StorageChallengeSupervisionRequest {
    run_prefix: SafeRelativePath,
    response_key: SafeRelativePath,
    started: Instant,
    maximum_runtime_seconds: u64,
    maximum_total_compute_usd: f64,
}

async fn supervise_storage_challenge(
    control: &runpod_control::RunpodClient,
    s3: &runpod_s3::RunpodS3Client,
    pod: &runpod_control::Pod,
    request: StorageChallengeSupervisionRequest,
) -> StorageChallengeWatchdog {
    let mut outcome = loop {
        let elapsed = request.started.elapsed();
        let observed_cost = pod.adjusted_cost_per_hr * elapsed.as_secs_f64() / 3600.0;
        if elapsed >= Duration::from_secs(request.maximum_runtime_seconds) {
            break "runtime_limit";
        }
        if observed_cost >= request.maximum_total_compute_usd {
            break "cost_limit";
        }
        match s3
            .head_worker_key(&request.run_prefix, &request.response_key)
            .await
        {
            Ok(runpod_s3::HeadObjectState::Present) => break "response_published",
            Ok(runpod_s3::HeadObjectState::Missing) => {}
            Err(_) => break "storage_poll_failed",
        }
        match control.get_pod(&pod.id).await {
            Ok(observed)
                if observed.desired_status == runpod_control::PodDesiredStatus::Running => {}
            Ok(_) => break "pod_exited",
            Err(_) => break "control_poll_failed",
        }
        let runtime_remaining = Duration::from_secs(request.maximum_runtime_seconds)
            .saturating_sub(request.started.elapsed());
        let cost_remaining_seconds = ((request.maximum_total_compute_usd - observed_cost).max(0.0)
            * 3600.0
            / pod.adjusted_cost_per_hr.max(f64::MIN_POSITIVE))
        .max(0.001);
        tokio::time::sleep(
            Duration::from_secs_f64(cost_remaining_seconds.min(5.0)).min(runtime_remaining),
        )
        .await;
    };
    let pod_termination = if control.delete_pod(&pod.id).await.is_ok() {
        "succeeded"
    } else {
        outcome = "cleanup_failed";
        "failed"
    };
    let elapsed_ms = u64::try_from(request.started.elapsed().as_millis()).unwrap_or(u64::MAX);
    StorageChallengeWatchdog {
        outcome,
        pod_termination,
        completed_at_ms: unix_time_ms(),
        elapsed_ms,
        observed_compute_usd: pod.adjusted_cost_per_hr * elapsed_ms as f64 / 3_600_000.0,
    }
}

async fn run_storage_challenge(command: StorageChallengeCommand) -> Result<()> {
    let (options, create) = match command {
        StorageChallengeCommand::DryRun(options) => (options, false),
        StorageChallengeCommand::Create(options) => (options, true),
    };
    let prepared = prepare_storage_challenge(&options.image)?;
    let specification = storage_challenge_specification(&options, &prepared.challenge)?;
    if !create {
        let request = runpod_control::dry_run_schedule_pod(
            &specification,
            &termination_deadline(options.maximum_runtime_seconds)?,
            &options.api_key_environment,
        )?;
        println!("{}", serde_json::to_string_pretty(&request)?);
        return Ok(());
    }
    storage_challenge_create(options, prepared, specification).await
}

async fn storage_challenge_create(
    options: StorageChallengeOptions,
    prepared: PreparedStorageChallenge,
    specification: runpod_control::PodCreateSpec,
) -> Result<()> {
    let output_path = options.out.as_ref().ok_or(RunpodCliError::Invalid(
        "storage challenge create requires --out",
    ))?;
    let launch_path = options.launch_out.as_ref().ok_or(RunpodCliError::Invalid(
        "storage challenge create requires --launch-out",
    ))?;
    let create_path = options.create_out.as_ref().ok_or(RunpodCliError::Invalid(
        "storage challenge create requires --create-out",
    ))?;
    if launch_path == output_path || create_path == output_path || create_path == launch_path {
        return Err(RunpodCliError::Invalid(
            "schedule, launch, and storage challenge receipt paths must differ",
        ));
    }
    let mut output = ReservedJsonOutput::new(output_path)?;
    let mut launch_output = ReservedJsonOutput::new(launch_path)?;
    let mut create_output = ReservedJsonOutput::new(create_path)?;
    let run_prefix = SafeRelativePath::new(options.run_prefix.clone())?;
    let transfer = runpod_s3::RunpodS3Manifest::new(
        run_prefix.clone(),
        [
            prepared.bootstrap.clone(),
            prepared.challenge.clone(),
            prepared.response.clone(),
        ],
    )?;
    let control = runpod_control::RunpodClient::from_environment(
        &options.api_key_environment,
        runpod_control::RunpodClientLimits::default(),
    )?;
    let s3 = runpod_s3::RunpodS3Client::from_environment(
        &specification.network_volume.id,
        &specification.network_volume.data_center_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    for object in [&prepared.bootstrap, &prepared.challenge, &prepared.response] {
        if s3.head_object(&transfer, object).await? != runpod_s3::HeadObjectState::Missing {
            return Err(RunpodCliError::Invalid(
                "storage challenge run prefix is not unused",
            ));
        }
    }
    // This direct child of the run prefix makes the exact run directory exist
    // before the root process changes only that directory's ownership. The
    // actual challenge remains absent until the admitted Pod is running.
    s3.put_object(&transfer, &prepared.bootstrap, &prepared.root)
        .await?;
    let started = Instant::now();
    let started_at_ms = unix_time_ms();
    let requested_terminate_after = termination_deadline(options.maximum_runtime_seconds)?;
    let scheduled = control
        .schedule_pod_with_termination(&specification, &requested_terminate_after)
        .await?;
    if let Err(error) = create_output.write(&scheduled) {
        let cleanup = if control.delete_pod(&scheduled.pod_id).await.is_ok() {
            "succeeded"
        } else {
            "failed"
        };
        return Err(RunpodCliError::StateWrite {
            reason: error.to_string(),
            cleanup,
        });
    }
    let schedule_receipt_sha256 = object_for_file(create_path, "schedule-receipt.json")?.sha256;
    let pod = control
        .admit_scheduled_pod(&scheduled, &specification)
        .await?;
    if let Err(error) = launch_output.write(&pod) {
        let cleanup = if control.delete_pod(&pod.id).await.is_ok() {
            "succeeded"
        } else {
            "failed"
        };
        return Err(RunpodCliError::StateWrite {
            reason: error.to_string(),
            cleanup,
        });
    }
    let launch_receipt_sha256 = object_for_file(launch_path, "launch-receipt.json")?.sha256;
    if let Err(error) = s3
        .put_object(&transfer, &prepared.challenge, &prepared.root)
        .await
    {
        let pod_termination = if control.delete_pod(&pod.id).await.is_ok() {
            "succeeded"
        } else {
            "failed"
        };
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut receipt = RunpodStorageChallengeReceipt {
            schema_version: "livefire.rag.runpod-storage-challenge-receipt/1",
            component_sha256: digest(b"unsealed storage challenge receipt"),
            executor_image: options.image,
            pod_id: pod.id,
            machine_id: pod.machine.id,
            network_volume_id: pod.network_volume.id,
            run_prefix: run_prefix.to_string(),
            bootstrap: prepared.bootstrap,
            challenge: prepared.challenge,
            response: prepared.response,
            response_verified: false,
            requested_terminate_after,
            termination_binding: "requested_unobservable",
            schedule_receipt_sha256,
            launch_receipt_sha256,
            returned_hourly_price_usd: pod.adjusted_cost_per_hr,
            maximum_hourly_price_usd: options.maximum_hourly_price,
            maximum_runtime_seconds: options.maximum_runtime_seconds,
            maximum_total_compute_usd: options.maximum_total_compute_usd,
            started_at_ms,
            completed_at_ms: unix_time_ms(),
            elapsed_ms,
            observed_compute_usd: pod.adjusted_cost_per_hr * elapsed_ms as f64 / 3_600_000.0,
            outcome: "challenge_upload_failed",
            pod_termination,
        };
        receipt.seal()?;
        output.write(&receipt)?;
        eprintln!("storage challenge upload failed: {error}");
        return Err(RunpodCliError::SupervisedRun(receipt.outcome));
    }
    let watchdog = supervise_storage_challenge(
        &control,
        &s3,
        &pod,
        StorageChallengeSupervisionRequest {
            run_prefix: run_prefix.clone(),
            response_key: prepared.response.key.clone(),
            started,
            maximum_runtime_seconds: options.maximum_runtime_seconds,
            maximum_total_compute_usd: options.maximum_total_compute_usd,
        },
    )
    .await;
    let mut outcome = watchdog.outcome;
    let mut response_verified = false;
    if outcome == "response_published" {
        let destination = tempfile::tempdir()?;
        match s3
            .get_worker_output(&transfer, &prepared.response, destination.path())
            .await
        {
            Ok(path) => {
                let response: RunpodStorageChallengeResponse = read_json(&path)?;
                if response == prepared.expected_response && response.validate().is_ok() {
                    response_verified = true;
                    outcome = "completed";
                } else {
                    outcome = "response_binding_failed";
                }
            }
            Err(_) => outcome = "response_fetch_failed",
        }
    }
    let mut receipt = RunpodStorageChallengeReceipt {
        schema_version: "livefire.rag.runpod-storage-challenge-receipt/1",
        component_sha256: digest(b"unsealed storage challenge receipt"),
        executor_image: options.image,
        pod_id: pod.id,
        machine_id: pod.machine.id,
        network_volume_id: pod.network_volume.id,
        run_prefix: run_prefix.to_string(),
        bootstrap: prepared.bootstrap,
        challenge: prepared.challenge,
        response: prepared.response,
        response_verified,
        requested_terminate_after,
        termination_binding: "requested_unobservable",
        schedule_receipt_sha256,
        launch_receipt_sha256,
        returned_hourly_price_usd: pod.adjusted_cost_per_hr,
        maximum_hourly_price_usd: options.maximum_hourly_price,
        maximum_runtime_seconds: options.maximum_runtime_seconds,
        maximum_total_compute_usd: options.maximum_total_compute_usd,
        started_at_ms,
        completed_at_ms: watchdog.completed_at_ms,
        elapsed_ms: watchdog.elapsed_ms,
        observed_compute_usd: watchdog.observed_compute_usd,
        outcome,
        pod_termination: watchdog.pod_termination,
    };
    receipt.seal()?;
    output.write(&receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if receipt.outcome == "completed" && receipt.pod_termination == "succeeded" {
        Ok(())
    } else {
        Err(RunpodCliError::SupervisedRun(receipt.outcome))
    }
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct RunpodSupervisedRunReceipt {
    schema_version: &'static str,
    pod_id: String,
    machine_id: String,
    network_volume_id: String,
    run_prefix: String,
    completion_key: String,
    requested_terminate_after: String,
    termination_binding: &'static str,
    schedule_receipt_sha256: Digest,
    launch_receipt_sha256: Digest,
    returned_hourly_price_usd: f64,
    maximum_hourly_price_usd: f64,
    maximum_runtime_seconds: u64,
    maximum_total_compute_usd: f64,
    started_at_ms: u64,
    completed_at_ms: u64,
    elapsed_ms: u64,
    observed_compute_usd: f64,
    declared_staged_input_bytes: u64,
    observation_staged_bytes: u64,
    fetched_output_bytes: u64,
    outcome: &'static str,
    pod_termination: &'static str,
}

struct SupervisionRequest {
    started: Instant,
    started_at_ms: u64,
    run_prefix: SafeRelativePath,
    completion_key: SafeRelativePath,
    maximum_hourly_price: f64,
    maximum_runtime_seconds: u64,
    maximum_total_compute_usd: f64,
    declared_staged_input_bytes: u64,
    observation_staged_bytes: u64,
    requested_terminate_after: String,
    schedule_receipt_sha256: Digest,
    launch_receipt_sha256: Digest,
}

async fn supervise_pod(
    control: &runpod_control::RunpodClient,
    s3: &runpod_s3::RunpodS3Client,
    pod: &runpod_control::Pod,
    request: SupervisionRequest,
) -> RunpodSupervisedRunReceipt {
    let mut outcome = loop {
        let elapsed = request.started.elapsed();
        let observed_cost = pod.adjusted_cost_per_hr * elapsed.as_secs_f64() / 3600.0;
        if elapsed >= Duration::from_secs(request.maximum_runtime_seconds) {
            break "runtime_limit";
        }
        if observed_cost >= request.maximum_total_compute_usd {
            break "cost_limit";
        }
        match s3
            .head_worker_key(&request.run_prefix, &request.completion_key)
            .await
        {
            Ok(runpod_s3::HeadObjectState::Present) => break "completed",
            Ok(runpod_s3::HeadObjectState::Missing) => {}
            Err(_) => break "poll_failed",
        }
        match control.get_pod(&pod.id).await {
            Ok(observed)
                if observed.desired_status == runpod_control::PodDesiredStatus::Running => {}
            Ok(_) => break "pod_exited",
            Err(_) => break "poll_failed",
        }
        let runtime_remaining = Duration::from_secs(request.maximum_runtime_seconds)
            .saturating_sub(request.started.elapsed());
        let cost_remaining_seconds = if pod.adjusted_cost_per_hr > 0.0 {
            ((request.maximum_total_compute_usd - observed_cost).max(0.0) * 3600.0
                / pod.adjusted_cost_per_hr)
                .max(0.001)
        } else {
            5.0
        };
        tokio::time::sleep(
            Duration::from_secs_f64(cost_remaining_seconds.min(5.0)).min(runtime_remaining),
        )
        .await;
    };
    let termination = if control.delete_pod(&pod.id).await.is_ok() {
        "succeeded"
    } else {
        outcome = "cleanup_failed";
        "failed"
    };
    let elapsed_ms = u64::try_from(request.started.elapsed().as_millis()).unwrap_or(u64::MAX);
    RunpodSupervisedRunReceipt {
        schema_version: "livefire.rag.runpod-supervised-run/1",
        pod_id: pod.id.clone(),
        machine_id: pod.machine.id.clone(),
        network_volume_id: pod.network_volume.id.clone(),
        run_prefix: request.run_prefix.to_string(),
        completion_key: request.completion_key.to_string(),
        requested_terminate_after: request.requested_terminate_after,
        termination_binding: "requested_unobservable",
        schedule_receipt_sha256: request.schedule_receipt_sha256,
        launch_receipt_sha256: request.launch_receipt_sha256,
        returned_hourly_price_usd: pod.adjusted_cost_per_hr,
        maximum_hourly_price_usd: request.maximum_hourly_price,
        maximum_runtime_seconds: request.maximum_runtime_seconds,
        maximum_total_compute_usd: request.maximum_total_compute_usd,
        started_at_ms: request.started_at_ms,
        completed_at_ms: unix_time_ms(),
        elapsed_ms,
        observed_compute_usd: pod.adjusted_cost_per_hr * elapsed_ms as f64 / 3_600_000.0,
        declared_staged_input_bytes: request.declared_staged_input_bytes,
        observation_staged_bytes: request.observation_staged_bytes,
        fetched_output_bytes: 0,
        outcome,
        pod_termination: termination,
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

fn termination_deadline(runtime_seconds: u64) -> Result<String> {
    let deadline = SystemTime::now()
        .checked_add(Duration::from_secs(runtime_seconds))
        .ok_or(RunpodCliError::Invalid(
            "Pod termination deadline overflowed",
        ))?;
    let deadline: chrono::DateTime<chrono::Utc> = deadline.into();
    Ok(deadline.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

fn conformance_pod_dry_run(options: ConformanceLaunchOptions) -> Result<()> {
    let (specification, _) = conformance_pod_specification(&options)?;
    let request = runpod_control::dry_run_schedule_pod(
        &specification,
        &termination_deadline(options.maximum_runtime_seconds)?,
        &options.api_key_environment,
    )?;
    println!("{}", serde_json::to_string_pretty(&request)?);
    Ok(())
}

async fn conformance_pod_create(options: ConformanceLaunchOptions) -> Result<()> {
    let output_path = options.out.as_ref().ok_or(RunpodCliError::Invalid(
        "conformance Pod create requires --out",
    ))?;
    let launch_path = options.launch_out.as_ref().ok_or(RunpodCliError::Invalid(
        "conformance Pod create requires --launch-out",
    ))?;
    let create_path = options.create_out.as_ref().ok_or(RunpodCliError::Invalid(
        "conformance Pod create requires --create-out",
    ))?;
    if launch_path == output_path || create_path == output_path || create_path == launch_path {
        return Err(RunpodCliError::Invalid(
            "schedule, launch, and supervised receipt paths must differ",
        ));
    }
    let mut output = ReservedJsonOutput::new(output_path)?;
    let mut launch_output = ReservedJsonOutput::new(launch_path)?;
    let mut create_output = ReservedJsonOutput::new(create_path)?;
    let (specification, candidate) = conformance_pod_specification(&options)?;
    let declared_staged_input_bytes = checked_object_bytes(
        conformance_input_objects(&options.candidate, &candidate)?
            .into_iter()
            .chain(std::iter::once(object_for_file(
                &options.candidate.join("candidate.json"),
                "candidate.json",
            )?)),
    )?;
    let client = runpod_control::RunpodClient::from_environment(
        &options.api_key_environment,
        runpod_control::RunpodClientLimits::default(),
    )?;
    let s3 = runpod_s3::RunpodS3Client::from_environment(
        &specification.network_volume.id,
        &specification.network_volume.data_center_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    let started = Instant::now();
    let started_at_ms = unix_time_ms();
    let requested_terminate_after = termination_deadline(options.maximum_runtime_seconds)?;
    let scheduled = client
        .schedule_pod_with_termination(&specification, &requested_terminate_after)
        .await?;
    if let Err(error) = create_output.write(&scheduled) {
        let cleanup = if client.delete_pod(&scheduled.pod_id).await.is_ok() {
            "succeeded"
        } else {
            "failed"
        };
        return Err(RunpodCliError::StateWrite {
            reason: error.to_string(),
            cleanup,
        });
    }
    let schedule_receipt_sha256 = object_for_file(create_path, "schedule-receipt.json")?.sha256;
    let pod = client
        .admit_scheduled_pod(&scheduled, &specification)
        .await?;
    if let Err(error) = launch_output.write(&pod) {
        let cleanup = if client.delete_pod(&pod.id).await.is_ok() {
            "succeeded"
        } else {
            "failed"
        };
        return Err(RunpodCliError::StateWrite {
            reason: error.to_string(),
            cleanup,
        });
    }
    let launch_receipt_sha256 = object_for_file(launch_path, "launch-receipt.json")?.sha256;
    let observation_staged_bytes =
        match stage_conformance_observation(&options, &candidate, &pod).await {
            Ok(bytes) => bytes,
            Err(error) => {
                let cleanup = if client.delete_pod(&pod.id).await.is_ok() {
                    "succeeded"
                } else {
                    "failed"
                };
                return Err(RunpodCliError::ObservationStage {
                    reason: error.to_string(),
                    cleanup,
                });
            }
        };
    let receipt = supervise_pod(
        &client,
        &s3,
        &pod,
        SupervisionRequest {
            started,
            started_at_ms,
            run_prefix: SafeRelativePath::new(options.run_prefix.clone())?,
            completion_key: SafeRelativePath::new(format!(
                "conformance/results/{}.json",
                options.run_id
            ))?,
            maximum_hourly_price: options.maximum_hourly_price,
            maximum_runtime_seconds: options.maximum_runtime_seconds,
            maximum_total_compute_usd: options.maximum_total_compute_usd,
            declared_staged_input_bytes,
            observation_staged_bytes,
            requested_terminate_after,
            schedule_receipt_sha256,
            launch_receipt_sha256,
        },
    )
    .await;
    output.write(&receipt)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if receipt.outcome == "completed" && receipt.pod_termination == "succeeded" {
        Ok(())
    } else {
        Err(RunpodCliError::SupervisedRun(receipt.outcome))
    }
}

fn conformance_pod_specification(
    options: &ConformanceLaunchOptions,
) -> Result<(runpod_control::PodCreateSpec, RunpodTeiConformanceCandidate)> {
    let candidate = load_conformance_candidate(&options.candidate)?;
    validate_watchdog_limits(
        options.maximum_hourly_price,
        options.maximum_runtime_seconds,
        options.maximum_total_compute_usd,
    )?;
    let volume: runpod_control::NetworkVolume = read_json(&options.volume)?;
    let expected_image = format!(
        "{}@{}",
        candidate.executor_image.repository, candidate.executor_image.digest
    );
    if options.image != expected_image {
        return Err(RunpodCliError::Invalid(
            "Pod image does not match the conformance candidate",
        ));
    }
    let run_prefix = SafeRelativePath::new(options.run_prefix.clone())?;
    SafeRelativePath::new(format!("conformance/results/{}.json", options.run_id))?;
    let root = format!("{}/{}", runpod_control::WORKSPACE_MOUNT, run_prefix);
    let specification = runpod_control::PodCreateSpec {
        name: options.name.clone(),
        image: options.image.clone(),
        gpu_type_id: options.gpu_type_id.clone(),
        network_volume: volume,
        worker_binary: candidate.execution.worker_binary.clone(),
        worker_arguments: vec![
            "conformance".into(),
            "--root".into(),
            root,
            "--candidate".into(),
            "candidate.json".into(),
            "--run-id".into(),
            options.run_id.clone(),
            "--observation".into(),
            format!("runtime/conformance/{}/observation.json", options.run_id),
            "--observation-wait-seconds".into(),
            "300".into(),
            "--health-wait-seconds".into(),
            "3600".into(),
        ],
        container_disk_gb: options.container_disk_gb,
        maximum_adjusted_hourly_price: options.maximum_hourly_price,
    };
    Ok((specification, candidate))
}

async fn stage_conformance_observation(
    options: &ConformanceLaunchOptions,
    candidate: &RunpodTeiConformanceCandidate,
    pod: &runpod_control::Pod,
) -> Result<u64> {
    let expected_accelerator = RunpodAcceleratorIdentity {
        provider: candidate.accelerator.provider.clone(),
        model: candidate.accelerator.gpu_model_id.clone(),
        architecture: candidate.accelerator.architecture_image_class.clone(),
        compute_capability: candidate.accelerator.compute_capability.clone(),
        count: candidate.accelerator.gpu_count,
    };
    let accelerator = returned_accelerator(&expected_accelerator, &options.gpu_type_id, pod)?;
    let key = SafeRelativePath::new(format!(
        "runtime/conformance/{}/observation.json",
        options.run_id
    ))?;
    let observation = WorkerObservation {
        schema_version: "livefire.rag.runpod-worker-observation/1",
        machine: RunpodMachineIdentity {
            pod_id: pod.id.clone(),
            machine_id: pod.machine.id.clone(),
        },
        accelerator,
    };
    let temporary = tempfile::tempdir()?;
    let path = key.join_to(temporary.path());
    fs::create_dir_all(
        path.parent()
            .ok_or(RunpodCliError::Invalid("conformance observation parent"))?,
    )?;
    write_canonical_json(&path, &observation)?;
    let object = object_for_file(&path, key.as_str())?;
    let manifest = runpod_s3::RunpodS3Manifest::new(
        SafeRelativePath::new(options.run_prefix.clone())?,
        [object.clone()],
    )?;
    let s3 = runpod_s3::RunpodS3Client::from_environment(
        &pod.network_volume.id,
        &pod.network_volume.data_center_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    s3.put_object(&manifest, &object, temporary.path()).await?;
    Ok(object.bytes)
}

fn build_bundle(options: BuildBundleOptions) -> Result<()> {
    if options.out.exists() || options.workers == 0 {
        return Err(RunpodCliError::Invalid(
            "output already exists or worker count is zero",
        ));
    }
    let prepared = portable::load_prepared(&options.prepared)
        .map_err(|error| RunpodCliError::Portable(error.to_string()))?;
    let plan = portable::load_embedding_plan_v2(&options.plan)
        .map_err(|error| RunpodCliError::Portable(error.to_string()))?;
    plan.validate_manifest_binding(&prepared)?;

    let policy_bytes = read_regular_file(&options.embedding_policy)?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let compact = policy.embedding_profile(&policy_bytes)?;
    if plan.embedding_profile.component.id != compact.id
        || plan.embedding_profile.component.version != compact.version
        || plan.embedding_profile.component.sha256.as_str() != compact.sha256
        || plan.embedding_profile.model_artifact.id != policy.model_artifact_set.id
        || plan.embedding_profile.model_artifact.version != policy.model_artifact_set.version
        || plan.embedding_profile.model_artifact.sha256.as_str() != policy.model_artifact_set.sha256
    {
        return Err(RunpodCliError::Invalid(
            "embedding policy does not match the v2 plan",
        ));
    }

    let model_manifest_bytes = read_regular_file(&options.model_manifest)?;
    let model_manifest = parse_tei_model_artifact_set_v1(&model_manifest_bytes)?;
    if model_manifest.sha256()? != policy.model_artifact_set.sha256 {
        return Err(RunpodCliError::Invalid(
            "model manifest does not match the embedding policy",
        ));
    }
    let tokenizer_bytes = read_regular_file(&options.tokenizer)?;
    let tokenizer_digest = digest(&tokenizer_bytes);
    if tokenizer_digest.as_str() != policy.executable_tokenizer.object.sha256
        || u64::try_from(tokenizer_bytes.len()).ok()
            != Some(policy.executable_tokenizer.object.bytes)
        || plan.executable_tokenizer.artifact.sha256 != tokenizer_digest
    {
        return Err(RunpodCliError::Invalid(
            "tokenizer bytes do not match the plan and embedding policy",
        ));
    }
    let fixture_bytes = read_regular_file(&options.conformance_fixture)?;
    if digest(&fixture_bytes).as_str() != policy.conformance.fixture.sha256
        || u64::try_from(fixture_bytes.len()).ok() != Some(policy.conformance.fixture.bytes)
    {
        return Err(RunpodCliError::Invalid(
            "conformance fixture does not match the embedding policy",
        ));
    }
    let query_plan_bytes = read_regular_file(&options.query_plan)?;
    query_vector_plan_queries(&query_plan_bytes)?;

    let execution: RunpodExecutionIdentity = read_json(&options.execution)?;
    validate_execution_identity(&execution, &plan.embedding_profile.component, &policy)?;
    let worker_bytes = read_regular_file(&options.worker_binary)?;
    let worker_digest = digest(&worker_bytes);
    if execution.worker_binary.sha256 != worker_digest {
        return Err(RunpodCliError::Invalid(
            "worker binary does not match the execution identity",
        ));
    }
    let image_build_receipt: RunpodExecutorImageBuildReceipt =
        read_json(&options.executor_image_build)?;
    image_build_receipt.validate()?;
    if image_build_receipt.component_sha256.as_str() != policy.executor_image_build.sha256
        || execution.executor_image_build.sha256 != image_build_receipt.component_sha256
        || !same_json(&image_build_receipt.executor_image, &policy.executor_image)?
        || !same_json(&image_build_receipt.tei_base_image, &policy.tei_image)?
        || image_build_receipt.worker_binary.component_sha256 != worker_digest
        || image_build_receipt.worker_binary.object.sha256 != worker_digest
        || image_build_receipt.worker_binary.object.bytes
            != u64::try_from(worker_bytes.len()).unwrap_or(u64::MAX)
    {
        return Err(RunpodCliError::Invalid(
            "executor image build receipt differs from the policy or worker binary",
        ));
    }

    let parent = options
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = Builder::new()
        .prefix(".runpod-bundle-")
        .tempdir_in(parent)?;
    let root = staging.path();

    let prepared_manifest = staged_component(
        root,
        &options.prepared.join("manifest.json"),
        "input/prepared/manifest.json",
        prepared.component_sha256.clone(),
    )?;
    let plan_file = if options.plan.is_dir() {
        options.plan.join("plan.json")
    } else {
        options.plan.clone()
    };
    let plan_root = plan_file
        .parent()
        .ok_or(RunpodCliError::Invalid("plan parent is absent"))?;
    let embedding_plan = staged_component(
        root,
        &plan_file,
        "input/plan/plan.json",
        plan.component_sha256.clone(),
    )?;
    let document_token_counts = staged_object(
        root,
        &plan.document_token_counts_object.path.join_to(plan_root),
        "input/plan/document-token-counts.u32le",
    )?;
    let embedding_profile = staged_component(
        root,
        &options.embedding_policy,
        "input/profile/embedding-policy.json",
        plan.embedding_profile.component.sha256.clone(),
    )?;
    let executor_image_build = staged_component(
        root,
        &options.executor_image_build,
        "input/profile/executor-image-build-receipt.json",
        image_build_receipt.component_sha256.clone(),
    )?;
    let executable_tokenizer = staged_component(
        root,
        &options.tokenizer,
        "input/tokenizer/tokenizer.json",
        plan.executable_tokenizer.artifact.sha256.clone(),
    )?;
    let conformance_fixture = staged_object(
        root,
        &options.conformance_fixture,
        "input/profile/conformance-fixture.json",
    )?;
    let query_plan = staged_object(root, &options.query_plan, "input/query/queries.jsonl")?;
    let worker_binary = staged_component(
        root,
        &options.worker_binary,
        image_build_receipt.worker_binary.object.path.as_str(),
        worker_digest,
    )?;
    let model_manifest_artifact = staged_component(
        root,
        &options.model_manifest,
        "input/model/manifest.json",
        plan.embedding_profile.model_artifact.sha256.clone(),
    )?;

    let mut model_objects = Vec::with_capacity(model_manifest.objects.len());
    for object in &model_manifest.objects {
        let staged = staged_object(
            root,
            &rag_pipeline::resolve_existing_artifact(
                &options.model_root,
                &SafeRelativePath::new(object.path.clone())?,
            )?,
            &format!("input/model/files/{}", object.path),
        )?;
        if staged.bytes != object.bytes || staged.sha256.as_str() != object.sha256 {
            return Err(RunpodCliError::Invalid(
                "model file does not match the model manifest",
            ));
        }
        model_objects.push(staged);
    }

    let mut prepared_documents = Vec::with_capacity(prepared.documents.len());
    for document in &prepared.documents {
        let source = document.object.path.join_to(&options.prepared);
        let object = staged_object(
            root,
            &source,
            &format!("input/prepared/{}", document.object.path),
        )?;
        if object.bytes != document.object.bytes || object.sha256 != document.object.sha256 {
            return Err(RunpodCliError::Invalid(
                "prepared document file does not match its manifest",
            ));
        }
        prepared_documents.push(CloudPreparedDocumentArtifact {
            prepared_path: document.object.path.clone(),
            object,
        });
    }

    let artifacts = RunpodBundleArtifacts {
        prepared_manifest,
        embedding_plan,
        document_token_counts,
        embedding_profile,
        executor_image_build,
        executable_tokenizer,
        conformance_fixture,
        query_plan,
        worker_binary,
        model_manifest: model_manifest_artifact,
        model_objects,
        prepared_documents,
    };
    let bundle =
        build_runpod_embedding_bundle(&prepared, &plan, artifacts, execution, options.workers)?;
    write_canonical_json(&root.join(BUNDLE_FILE), &bundle)?;
    bundle.validate_input_files(root, &prepared, &plan)?;
    let staged_root = staging.keep();
    publish_without_overwrite(&staged_root, &options.out)?;
    println!("{}", serde_json::to_string_pretty(&bundle)?);
    Ok(())
}

fn load_and_validate_bundle(root: &Path) -> Result<RunpodEmbeddingBundle> {
    let bundle: RunpodEmbeddingBundle = read_json(&root.join(BUNDLE_FILE))?;
    let prepared: rag_pipeline::PreparedCorpusManifest =
        read_json(&root.join(bundle.artifacts.prepared_manifest.object.key.as_str()))?;
    let plan: rag_pipeline::EmbeddingPlanV2 =
        read_json(&root.join(bundle.artifacts.embedding_plan.object.key.as_str()))?;
    plan.validate_manifest_binding(&prepared)?;
    bundle.validate_input_files(root, &prepared, &plan)?;
    let plan_path = bundle.artifacts.embedding_plan.object.key.join_to(root);
    let plan_root = plan_path
        .parent()
        .ok_or(RunpodCliError::Invalid("staged plan parent is absent"))?;
    plan.read_document_token_counts(plan_root)?;
    let policy_path = bundle.artifacts.embedding_profile.object.key.join_to(root);
    let policy_bytes = read_regular_file(&policy_path)?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let model_manifest_path = bundle.artifacts.model_manifest.object.key.join_to(root);
    let model_manifest_bytes = read_regular_file(&model_manifest_path)?;
    let model_manifest = parse_tei_model_artifact_set_v1(&model_manifest_bytes)?;
    if model_manifest.sha256()? != policy.model_artifact_set.sha256 {
        return Err(RunpodCliError::Invalid(
            "model manifest differs from the embedding policy",
        ));
    }
    let fixture = object_for_file(
        &bundle.artifacts.conformance_fixture.key.join_to(root),
        bundle.artifacts.conformance_fixture.key.as_str(),
    )?;
    if fixture != bundle.artifacts.conformance_fixture
        || fixture.bytes != policy.conformance.fixture.bytes
        || fixture.sha256.as_str() != policy.conformance.fixture.sha256
    {
        return Err(RunpodCliError::Invalid(
            "conformance fixture differs from the embedding policy",
        ));
    }
    let query_plan_bytes = read_regular_file(&bundle.artifacts.query_plan.key.join_to(root))?;
    query_vector_plan_queries(&query_plan_bytes)?;
    let tokenizer = object_for_file(
        &bundle
            .artifacts
            .executable_tokenizer
            .object
            .key
            .join_to(root),
        bundle.artifacts.executable_tokenizer.object.key.as_str(),
    )?;
    if tokenizer.bytes != policy.executable_tokenizer.object.bytes
        || tokenizer.sha256.as_str() != policy.executable_tokenizer.object.sha256
    {
        return Err(RunpodCliError::Invalid(
            "executable tokenizer differs from the embedding policy",
        ));
    }
    if bundle.artifacts.model_objects.len() != model_manifest.objects.len() {
        return Err(RunpodCliError::Invalid(
            "model object coverage differs from the complete model manifest",
        ));
    }
    for (declared, expected) in bundle
        .artifacts
        .model_objects
        .iter()
        .zip(model_manifest.objects.iter())
    {
        if !declared
            .key
            .as_str()
            .ends_with(&format!("/{}", expected.path))
            || declared.bytes != expected.bytes
            || declared.sha256.as_str() != expected.sha256
        {
            return Err(RunpodCliError::Invalid(
                "model objects differ from the complete model manifest",
            ));
        }
    }
    Ok(bundle)
}

async fn stage(options: StageOptions) -> Result<()> {
    let bundle = load_and_validate_bundle(&options.bundle)?;
    let run_prefix = SafeRelativePath::new(options.run_prefix)?;
    let mut objects = bundle_input_objects(&bundle);
    objects.push(staged_bundle_object(&options.bundle)?);
    let manifest = runpod_s3::RunpodS3Manifest::new(run_prefix, objects.clone())?;
    let client = runpod_s3::RunpodS3Client::from_environment(
        &options.network_volume_id,
        &options.datacenter_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    for object in &objects {
        match client.head_object(&manifest, object).await? {
            runpod_s3::HeadObjectState::Present => {}
            runpod_s3::HeadObjectState::Missing => {
                client
                    .put_object(&manifest, object, &options.bundle)
                    .await?;
            }
        }
    }
    println!(
        "staged {} exact objects under the requested run prefix",
        objects.len()
    );
    Ok(())
}

async fn fetch(options: FetchOptions) -> Result<()> {
    refuse_existing(&options.out)?;
    let bundle = load_and_validate_bundle(&options.bundle)?;
    let run_prefix = SafeRelativePath::new(options.run_prefix)?;
    let client = runpod_s3::RunpodS3Client::from_environment(
        &options.network_volume_id,
        &options.datacenter_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    let parent = options
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = Builder::new().prefix(".runpod-fetch-").tempdir_in(parent)?;
    let embeddings_root = staging.path().join("embeddings");
    let evidence_root = staging.path().join("evidence");
    fs::create_dir(&embeddings_root)?;
    fs::create_dir(&evidence_root)?;
    let plan: rag_pipeline::EmbeddingPlanV2 = read_json(
        &bundle
            .artifacts
            .embedding_plan
            .object
            .key
            .join_to(&options.bundle),
    )?;
    let mut markers = Vec::with_capacity(bundle.assignments.len());
    for assignment in &bundle.assignments {
        let (marker, _) = client
            .get_completion_marker(
                run_prefix.clone(),
                &bundle,
                &assignment.worker_id,
                &evidence_root,
            )
            .await?;
        if marker.outcome != WorkerAttemptOutcome::Completed {
            return Err(RunpodCliError::Invalid(
                "deterministic completion marker is not completed",
            ));
        }
        markers.push(marker);
    }
    let output_objects = markers
        .iter()
        .flat_map(|marker| {
            marker.outputs.iter().flat_map(|output| {
                [
                    output.result.clone(),
                    output.receipt.clone(),
                    output.report.clone(),
                ]
            })
        })
        .collect::<Vec<_>>();
    let output_manifest =
        runpod_s3::RunpodS3Manifest::new(run_prefix.clone(), output_objects.clone())?;
    for object in &output_objects {
        client
            .get_worker_output(&output_manifest, object, &embeddings_root)
            .await?;
    }
    let query_vector_objects = query_vector_objects(&markers);
    let query_vector_manifest =
        runpod_s3::RunpodS3Manifest::new(run_prefix, query_vector_objects.clone())?;
    for object in &query_vector_objects {
        client
            .get_worker_output(&query_vector_manifest, object, staging.path())
            .await?;
    }
    let report = build_runpod_run_report(&bundle, &markers)?;
    write_canonical_json(&evidence_root.join("run-report.json"), &report)?;
    let receipt = build_fetch_receipt(&bundle, &markers, &evidence_root)?;
    write_canonical_json(&evidence_root.join("fetch-receipt.json"), &receipt)?;
    verify_fetched_values(FetchedVerification {
        bundle: &bundle,
        plan: &plan,
        bundle_root: &options.bundle,
        fetched_root: staging.path(),
        embeddings_root: &embeddings_root,
        evidence_root: &evidence_root,
        markers: &markers,
        report: &report,
        receipt: &receipt,
    })?;
    let staged_root = staging.keep();
    publish_without_overwrite(&staged_root, &options.out)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn verify_fetched(bundle_root: &Path, fetched_root: &Path) -> Result<()> {
    let bundle = load_and_validate_bundle(bundle_root)?;
    let plan: rag_pipeline::EmbeddingPlanV2 = read_json(
        &bundle
            .artifacts
            .embedding_plan
            .object
            .key
            .join_to(bundle_root),
    )?;
    let embeddings_root = fetched_root.join("embeddings");
    let evidence_root = fetched_root.join("evidence");
    let report: RunpodRunReport = read_json(&evidence_root.join("run-report.json"))?;
    let receipt: RunpodFetchReceipt = read_json(&evidence_root.join("fetch-receipt.json"))?;
    let mut markers = Vec::with_capacity(bundle.assignments.len());
    for assignment in &bundle.assignments {
        let key = format!("attempts/{}/completed.json", assignment.worker_id);
        let marker: RunpodWorkerAttemptMarker = read_json(&evidence_root.join(key))?;
        markers.push(marker);
    }
    verify_fetched_values(FetchedVerification {
        bundle: &bundle,
        plan: &plan,
        bundle_root,
        fetched_root,
        embeddings_root: &embeddings_root,
        evidence_root: &evidence_root,
        markers: &markers,
        report: &report,
        receipt: &receipt,
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

struct FetchedVerification<'a> {
    bundle: &'a RunpodEmbeddingBundle,
    plan: &'a rag_pipeline::EmbeddingPlanV2,
    bundle_root: &'a Path,
    fetched_root: &'a Path,
    embeddings_root: &'a Path,
    evidence_root: &'a Path,
    markers: &'a [RunpodWorkerAttemptMarker],
    report: &'a RunpodRunReport,
    receipt: &'a RunpodFetchReceipt,
}

fn verify_fetched_values(values: FetchedVerification<'_>) -> Result<()> {
    let FetchedVerification {
        bundle,
        plan,
        bundle_root,
        fetched_root,
        embeddings_root,
        evidence_root,
        markers,
        report,
        receipt,
    } = values;
    report.validate_against(bundle, markers)?;
    let rebuilt = build_runpod_run_report(bundle, markers)?;
    if &rebuilt != report {
        return Err(RunpodCliError::Invalid(
            "run report is not the canonical report for these markers",
        ));
    }
    let expected_receipt = build_fetch_receipt(bundle, markers, evidence_root)?;
    if &expected_receipt != receipt {
        return Err(RunpodCliError::Invalid(
            "fetch receipt differs from the exact downloaded byte counts",
        ));
    }
    for marker in markers {
        marker.validate_against(bundle)?;
        let canonical = marker.canonical_object()?;
        verify_local_object(evidence_root, &canonical)?;
        for output in &marker.outputs {
            for object in [&output.result, &output.receipt, &output.report] {
                verify_local_object(embeddings_root, object)?;
            }
        }
        if let Some(output) = &marker.query_vector_set {
            for object in [&output.manifest, &output.query_plan, &output.vectors] {
                verify_local_object(fetched_root, object)?;
            }
        }
    }
    verify_query_vector_set(bundle, bundle_root, fetched_root, markers)?;
    portable::validate_unfinalized_embedding_artifacts(embeddings_root, plan)
        .map_err(|error| RunpodCliError::Portable(error.to_string()))?;
    Ok(())
}

fn verify_query_vector_set(
    bundle: &RunpodEmbeddingBundle,
    bundle_root: &Path,
    fetched_root: &Path,
    markers: &[RunpodWorkerAttemptMarker],
) -> Result<()> {
    let marker = markers
        .iter()
        .find(|marker| marker.worker_id == bundle.query_vector_output.worker_id)
        .ok_or(RunpodCliError::Invalid(
            "query-vector worker marker is absent",
        ))?;
    let output = marker
        .query_vector_set
        .as_ref()
        .ok_or(RunpodCliError::Invalid(
            "query-vector output is absent from worker marker",
        ))?;
    let policy_path = bundle
        .artifacts
        .embedding_profile
        .object
        .key
        .join_to(bundle_root);
    let policy_bytes = read_regular_file(&policy_path)?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let compact = policy.embedding_profile(&policy_bytes)?;
    let query_plan_path = bundle.artifacts.query_plan.key.join_to(bundle_root);
    let sealed = SealedQueryVectorSet::open(
        &fetched_root.join("query-vectors"),
        &bundle.execution.embedding_profile,
        &bundle.execution.returned_model,
        compact.dimensions,
        &compact.normalization,
        Some(&query_plan_path),
    )?;
    if sealed.manifest.component_sha256 != output.component_sha256 {
        return Err(RunpodCliError::Invalid(
            "query-vector manifest differs from worker marker",
        ));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunpodFetchReceipt {
    schema_version: String,
    bundle_sha256: Digest,
    downloaded_completion_marker_bytes: u64,
    downloaded_embedding_output_bytes: u64,
    downloaded_query_vector_output_bytes: u64,
    downloaded_total_bytes: u64,
    generated_run_report_bytes: u64,
}

fn output_objects(markers: &[RunpodWorkerAttemptMarker]) -> Vec<CloudObjectRef> {
    markers
        .iter()
        .flat_map(|marker| {
            marker.outputs.iter().flat_map(|output| {
                [
                    output.result.clone(),
                    output.receipt.clone(),
                    output.report.clone(),
                ]
            })
        })
        .collect()
}

fn query_vector_objects(markers: &[RunpodWorkerAttemptMarker]) -> Vec<CloudObjectRef> {
    markers
        .iter()
        .filter_map(|marker| marker.query_vector_set.as_ref())
        .flat_map(|output| {
            [
                output.manifest.clone(),
                output.query_plan.clone(),
                output.vectors.clone(),
            ]
        })
        .collect()
}

fn build_fetch_receipt(
    bundle: &RunpodEmbeddingBundle,
    markers: &[RunpodWorkerAttemptMarker],
    evidence_root: &Path,
) -> Result<RunpodFetchReceipt> {
    let marker_bytes = checked_object_bytes(
        markers
            .iter()
            .map(RunpodWorkerAttemptMarker::canonical_object)
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )?;
    let output_bytes = checked_object_bytes(output_objects(markers))?;
    let query_vector_bytes = checked_object_bytes(query_vector_objects(markers))?;
    let downloaded_total_bytes = marker_bytes
        .checked_add(output_bytes)
        .and_then(|bytes| bytes.checked_add(query_vector_bytes))
        .ok_or(RunpodCliError::Invalid("downloaded byte count overflowed"))?;
    let generated_run_report_bytes = fs::metadata(evidence_root.join("run-report.json"))?.len();
    Ok(RunpodFetchReceipt {
        schema_version: "livefire.rag.runpod-fetch-receipt/1".into(),
        bundle_sha256: bundle.component_sha256.clone(),
        downloaded_completion_marker_bytes: marker_bytes,
        downloaded_embedding_output_bytes: output_bytes,
        downloaded_query_vector_output_bytes: query_vector_bytes,
        downloaded_total_bytes,
        generated_run_report_bytes,
    })
}

fn verify_local_object(root: &Path, object: &CloudObjectRef) -> Result<()> {
    let path = rag_pipeline::resolve_existing_artifact(root, &object.key)?;
    let observed = object_for_file(&path, object.key.as_str())?;
    if &observed != object {
        return Err(RunpodCliError::Invalid(
            "downloaded object differs from its completion marker",
        ));
    }
    Ok(())
}

async fn run_volume(command: VolumeCommand) -> Result<()> {
    match command {
        VolumeCommand::Create(options) => {
            let mut output = ReservedJsonOutput::new(&options.out)?;
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            let volume = client
                .create_network_volume(&runpod_control::NetworkVolumeCreateSpec {
                    name: options.name,
                    size: options.size_gb,
                    data_center_id: options.datacenter_id,
                })
                .await?;
            if let Err(error) = output.write(&volume) {
                let cleanup = if client.delete_network_volume(&volume.id).await.is_ok() {
                    "succeeded"
                } else {
                    "failed"
                };
                return Err(RunpodCliError::StateWrite {
                    reason: error.to_string(),
                    cleanup,
                });
            }
            println!("{}", serde_json::to_string_pretty(&volume)?);
            Ok(())
        }
        VolumeCommand::Status(options) => {
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            let volume = client.get_network_volume(&options.id).await?;
            println!("{}", serde_json::to_string_pretty(&volume)?);
            Ok(())
        }
        VolumeCommand::Terminate(options) => {
            require_termination_confirmation(&options.id, &options.confirm_terminate)?;
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            client.delete_network_volume(&options.id).await?;
            println!("terminated network volume {}", options.id);
            Ok(())
        }
    }
}

async fn run_pod(command: PodCommand) -> Result<()> {
    match command {
        PodCommand::DryRun(options) => {
            let specification = pod_specification(&options)?;
            let request = runpod_control::dry_run_schedule_pod(
                &specification,
                &termination_deadline(options.maximum_runtime_seconds)?,
                &options.api_key_environment,
            )?;
            println!("{}", serde_json::to_string_pretty(&request)?);
            Ok(())
        }
        PodCommand::Create(options) => {
            let output_path = options
                .out
                .as_ref()
                .ok_or(RunpodCliError::Invalid("Pod create requires --out"))?;
            let launch_path = options
                .launch_out
                .as_ref()
                .ok_or(RunpodCliError::Invalid("Pod create requires --launch-out"))?;
            let create_path = options
                .create_out
                .as_ref()
                .ok_or(RunpodCliError::Invalid("Pod create requires --create-out"))?;
            if launch_path == output_path
                || create_path == output_path
                || create_path == launch_path
            {
                return Err(RunpodCliError::Invalid(
                    "schedule, launch, and supervised receipt paths must differ",
                ));
            }
            let mut output = ReservedJsonOutput::new(output_path)?;
            let mut launch_output = ReservedJsonOutput::new(launch_path)?;
            let mut create_output = ReservedJsonOutput::new(create_path)?;
            let specification = pod_specification(&options)?;
            let bundle = load_and_validate_bundle(&options.bundle)?;
            let declared_staged_input_bytes = checked_object_bytes(
                bundle_input_objects(&bundle)
                    .into_iter()
                    .chain(std::iter::once(staged_bundle_object(&options.bundle)?)),
            )?;
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            let s3 = runpod_s3::RunpodS3Client::from_environment(
                &specification.network_volume.id,
                &specification.network_volume.data_center_id,
                &options.access_key_environment,
                &options.secret_key_environment,
                runpod_s3::RunpodS3Limits::default(),
            )?;
            let started = Instant::now();
            let started_at_ms = unix_time_ms();
            let requested_terminate_after = termination_deadline(options.maximum_runtime_seconds)?;
            let scheduled = client
                .schedule_pod_with_termination(&specification, &requested_terminate_after)
                .await?;
            if let Err(error) = create_output.write(&scheduled) {
                let cleanup = if client.delete_pod(&scheduled.pod_id).await.is_ok() {
                    "succeeded"
                } else {
                    "failed"
                };
                return Err(RunpodCliError::StateWrite {
                    reason: error.to_string(),
                    cleanup,
                });
            }
            let schedule_receipt_sha256 =
                object_for_file(create_path, "schedule-receipt.json")?.sha256;
            let pod = client
                .admit_scheduled_pod(&scheduled, &specification)
                .await?;
            if let Err(error) = launch_output.write(&pod) {
                let cleanup = if client.delete_pod(&pod.id).await.is_ok() {
                    "succeeded"
                } else {
                    "failed"
                };
                return Err(RunpodCliError::StateWrite {
                    reason: error.to_string(),
                    cleanup,
                });
            }
            let launch_receipt_sha256 = object_for_file(launch_path, "launch-receipt.json")?.sha256;
            let observation_staged_bytes = match stage_worker_observation(&options, &pod).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    let cleanup = if client.delete_pod(&pod.id).await.is_ok() {
                        "succeeded"
                    } else {
                        "failed"
                    };
                    return Err(RunpodCliError::ObservationStage {
                        reason: error.to_string(),
                        cleanup,
                    });
                }
            };
            let receipt = supervise_pod(
                &client,
                &s3,
                &pod,
                SupervisionRequest {
                    started,
                    started_at_ms,
                    run_prefix: SafeRelativePath::new(options.run_prefix.clone())?,
                    completion_key: SafeRelativePath::new(format!(
                        "attempts/{}/completed.json",
                        options.worker_id
                    ))?,
                    maximum_hourly_price: options.maximum_hourly_price,
                    maximum_runtime_seconds: options.maximum_runtime_seconds,
                    maximum_total_compute_usd: options.maximum_total_compute_usd,
                    declared_staged_input_bytes,
                    observation_staged_bytes,
                    requested_terminate_after,
                    schedule_receipt_sha256,
                    launch_receipt_sha256,
                },
            )
            .await;
            output.write(&receipt)?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            if receipt.outcome == "completed" && receipt.pod_termination == "succeeded" {
                Ok(())
            } else {
                Err(RunpodCliError::SupervisedRun(receipt.outcome))
            }
        }
        PodCommand::Status(options) => {
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            let pod = client.get_pod(&options.id).await?;
            println!("{}", serde_json::to_string_pretty(&pod)?);
            Ok(())
        }
        PodCommand::Terminate(options) => {
            require_termination_confirmation(&options.id, &options.confirm_terminate)?;
            let client = runpod_control::RunpodClient::from_environment(
                &options.api_key_environment,
                runpod_control::RunpodClientLimits::default(),
            )?;
            client.delete_pod(&options.id).await?;
            println!("terminated Pod {}", options.id);
            Ok(())
        }
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerObservation {
    schema_version: &'static str,
    machine: RunpodMachineIdentity,
    accelerator: RunpodAcceleratorIdentity,
}

async fn stage_worker_observation(
    options: &PodLaunchOptions,
    pod: &runpod_control::Pod,
) -> Result<u64> {
    let bundle = load_and_validate_bundle(&options.bundle)?;
    let accelerator =
        returned_accelerator(&bundle.execution.accelerator, &options.gpu_type_id, pod)?;
    let key = SafeRelativePath::new(format!(
        "runtime/{}/attempts/{}/observation.json",
        options.worker_id, options.attempt_id
    ))?;
    let observation = WorkerObservation {
        schema_version: "livefire.rag.runpod-worker-observation/1",
        machine: RunpodMachineIdentity {
            pod_id: pod.id.clone(),
            machine_id: pod.machine.id.clone(),
        },
        accelerator,
    };
    let temporary = tempfile::tempdir()?;
    let path = key.join_to(temporary.path());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_canonical_json(&path, &observation)?;
    let object = object_for_file(&path, key.as_str())?;
    let manifest = runpod_s3::RunpodS3Manifest::new(
        SafeRelativePath::new(options.run_prefix.clone())?,
        [object.clone()],
    )?;
    let s3 = runpod_s3::RunpodS3Client::from_environment(
        &pod.network_volume.id,
        &pod.network_volume.data_center_id,
        &options.access_key_environment,
        &options.secret_key_environment,
        runpod_s3::RunpodS3Limits::default(),
    )?;
    s3.put_object(&manifest, &object, temporary.path()).await?;
    Ok(object.bytes)
}

fn checked_object_bytes(objects: impl IntoIterator<Item = CloudObjectRef>) -> Result<u64> {
    objects.into_iter().try_fold(0_u64, |total, object| {
        total
            .checked_add(object.bytes)
            .ok_or(RunpodCliError::Invalid(
                "declared staged byte count overflowed",
            ))
    })
}

fn returned_accelerator(
    expected: &RunpodAcceleratorIdentity,
    requested_gpu_type_id: &str,
    pod: &runpod_control::Pod,
) -> Result<RunpodAcceleratorIdentity> {
    if pod.gpu.id != requested_gpu_type_id
        || pod.gpu.display_name != expected.model
        || pod.gpu.count != expected.count
    {
        return Err(RunpodCliError::Invalid(
            "returned Pod GPU type, display name, or count differs from the sealed accelerator",
        ));
    }
    Ok(RunpodAcceleratorIdentity {
        provider: expected.provider.clone(),
        model: pod.gpu.display_name.clone(),
        architecture: expected.architecture.clone(),
        compute_capability: expected.compute_capability.clone(),
        count: pod.gpu.count,
    })
}

fn pod_specification(options: &PodLaunchOptions) -> Result<runpod_control::PodCreateSpec> {
    let bundle = load_and_validate_bundle(&options.bundle)?;
    validate_watchdog_limits(
        options.maximum_hourly_price,
        options.maximum_runtime_seconds,
        options.maximum_total_compute_usd,
    )?;
    let assignment = bundle
        .assignments
        .iter()
        .find(|assignment| assignment.worker_id == options.worker_id)
        .ok_or(RunpodCliError::Invalid(
            "worker ID is not assigned by the bundle",
        ))?;
    if options.attempt_number == 0 || options.attempt_id.is_empty() {
        return Err(RunpodCliError::Invalid("attempt identity is invalid"));
    }
    let run_prefix = SafeRelativePath::new(options.run_prefix.clone())?;
    let volume: runpod_control::NetworkVolume = read_json(&options.volume)?;
    let policy_bytes = read_regular_file(
        &bundle
            .artifacts
            .embedding_profile
            .object
            .key
            .join_to(&options.bundle),
    )?;
    let policy = parse_tei_checkpoint_profile_v3(&policy_bytes)?;
    let expected_image = format!(
        "{}@{}",
        policy.executor_image.repository, policy.executor_image.digest
    );
    if options.image != expected_image {
        return Err(RunpodCliError::Invalid(
            "Pod image does not match the measured embedding policy",
        ));
    }
    let root = format!("{}/{}", runpod_control::WORKSPACE_MOUNT, run_prefix);
    let observation = format!(
        "runtime/{}/attempts/{}/observation.json",
        assignment.worker_id, options.attempt_id
    );
    Ok(runpod_control::PodCreateSpec {
        name: options.name.clone(),
        image: options.image.clone(),
        gpu_type_id: options.gpu_type_id.clone(),
        network_volume: volume,
        worker_binary: bundle.execution.worker_binary.clone(),
        worker_arguments: vec![
            "run".into(),
            "--root".into(),
            root,
            "--bundle".into(),
            BUNDLE_FILE.into(),
            "--worker-id".into(),
            assignment.worker_id.clone(),
            "--attempt-id".into(),
            options.attempt_id.clone(),
            "--attempt-number".into(),
            options.attempt_number.to_string(),
            "--observation".into(),
            observation,
            "--observation-wait-seconds".into(),
            "300".into(),
        ],
        container_disk_gb: options.container_disk_gb,
        maximum_adjusted_hourly_price: options.maximum_hourly_price,
    })
}

fn require_termination_confirmation(id: &str, confirmation: &str) -> Result<()> {
    if id != confirmation {
        return Err(RunpodCliError::Invalid(
            "termination confirmation must exactly equal the requested ID",
        ));
    }
    Ok(())
}

fn refuse_existing(path: &Path) -> Result<()> {
    if path.exists() {
        return Err(RunpodCliError::Invalid("output path already exists"));
    }
    Ok(())
}

struct ReservedJsonOutput {
    path: PathBuf,
    file: Option<File>,
}

impl ReservedJsonOutput {
    fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
        })
    }

    fn write(&mut self, value: &impl serde::Serialize) -> Result<()> {
        use std::io::Write;

        let bytes = rag_pipeline::canonical_json_bytes(value)?;
        let file = self
            .file
            .as_mut()
            .ok_or(RunpodCliError::Invalid("output was already finalized"))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        self.file = None;
        Ok(())
    }
}

impl Drop for ReservedJsonOutput {
    fn drop(&mut self) {
        if self.file.is_some() {
            self.file.take();
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn _write_new_json_for_test(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let mut output = ReservedJsonOutput::new(path)?;
    output.write(value)
}

fn bundle_input_objects(bundle: &RunpodEmbeddingBundle) -> Vec<CloudObjectRef> {
    let artifacts = &bundle.artifacts;
    let mut objects = vec![
        artifacts.prepared_manifest.object.clone(),
        artifacts.embedding_plan.object.clone(),
        artifacts.document_token_counts.clone(),
        artifacts.embedding_profile.object.clone(),
        artifacts.executor_image_build.object.clone(),
        artifacts.executable_tokenizer.object.clone(),
        artifacts.conformance_fixture.clone(),
        artifacts.query_plan.clone(),
        artifacts.worker_binary.object.clone(),
        artifacts.model_manifest.object.clone(),
    ];
    objects.extend(artifacts.model_objects.iter().cloned());
    objects.extend(
        artifacts
            .prepared_documents
            .iter()
            .map(|document| document.object.clone()),
    );
    objects
}

fn staged_bundle_object(root: &Path) -> Result<CloudObjectRef> {
    object_for_file(&root.join(BUNDLE_FILE), BUNDLE_FILE)
}

fn validate_execution_identity(
    execution: &RunpodExecutionIdentity,
    profile_component: &ComponentRef,
    policy: &rag_embedding::TeiCheckpointProfileV3,
) -> Result<()> {
    let expected_accelerator = RunpodAcceleratorIdentity {
        provider: policy.accelerator.provider.clone(),
        model: policy.accelerator.gpu_model_id.clone(),
        architecture: policy.accelerator.architecture_image_class.clone(),
        compute_capability: policy.accelerator.compute_capability.clone(),
        count: policy.accelerator.gpu_count,
    };
    if execution.embedding_profile != *profile_component
        || execution.model_artifact.id != policy.model_artifact_set.id
        || execution.model_artifact.version != policy.model_artifact_set.version
        || execution.model_artifact.sha256.as_str() != policy.model_artifact_set.sha256
        || execution.executor_image.id != policy.executor_image.component.id
        || execution.executor_image.version != policy.executor_image.component.version
        || execution.executor_image.sha256.as_str() != policy.executor_image.component.sha256
        || execution.executor_image_build.id != policy.executor_image_build.id
        || execution.executor_image_build.version != policy.executor_image_build.version
        || execution.executor_image_build.sha256.as_str() != policy.executor_image_build.sha256
        || execution.runtime.id != policy.runtime.id
        || execution.runtime.version != policy.runtime.version
        || execution.runtime.sha256.as_str() != policy.runtime.sha256
        || execution.accelerator != expected_accelerator
        || execution.returned_model != policy.conformance.returned_model
    {
        return Err(RunpodCliError::Invalid(
            "execution identity does not match the measured embedding policy",
        ));
    }
    Ok(())
}

fn staged_component(
    root: &Path,
    source: &Path,
    key: &str,
    component_sha256: Digest,
) -> Result<CloudComponentArtifact> {
    Ok(CloudComponentArtifact {
        component_sha256,
        object: staged_object(root, source, key)?,
    })
}

fn staged_object(root: &Path, source: &Path, key: &str) -> Result<CloudObjectRef> {
    reject_symlink(source)?;
    let object = object_for_file(source, key)?;
    let destination = object.key.join_to(root);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    if destination.exists() {
        return Err(RunpodCliError::Invalid("duplicate staged object path"));
    }
    if fs::hard_link(source, &destination).is_err() {
        fs::copy(source, &destination)?;
        fs::set_permissions(&destination, fs::metadata(source)?.permissions())?;
    }
    Ok(object)
}

fn object_for_file(path: &Path, key: &str) -> Result<CloudObjectRef> {
    reject_symlink(path)?;
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(RunpodCliError::Invalid(
            "object is not a non-empty regular file",
        ));
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
    Ok(CloudObjectRef {
        key: SafeRelativePath::new(key)?,
        bytes: metadata.len(),
        sha256: Digest::new(format!("{:x}", hasher.finalize()))?,
    })
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>> {
    reject_symlink(path)?;
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(RunpodCliError::Invalid("input is not a regular file"));
    }
    Ok(fs::read(path)?)
}

fn reject_symlink(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(RunpodCliError::Invalid("symbolic-link inputs are refused"));
    }
    Ok(())
}

fn publish_without_overwrite(staged_root: &Path, destination: &Path) -> Result<()> {
    // Reserving the destination with create_dir is the no-overwrite step.
    // bundle.json moves last, so an interrupted publication is never accepted
    // as a complete bundle by the validation command.
    fs::create_dir(destination)?;
    let mut entries = fs::read_dir(staged_root)?.collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    entries.sort_by_key(|entry| entry.file_name() == BUNDLE_FILE);
    for entry in entries {
        fs::rename(entry.path(), destination.join(entry.file_name()))?;
    }
    fs::remove_dir(staged_root)?;
    Ok(())
}

fn digest(bytes: &[u8]) -> Digest {
    Digest::new(format!("{:x}", Sha256::digest(bytes))).expect("SHA-256 is a valid digest")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn nested_cli_names_the_runpod_actions_in_ordinary_language() {
        for arguments in [
            vec![
                "rag",
                "runpod",
                "executor-image",
                "validate",
                "--receipt",
                "executor-image-build.json",
                "--dockerfile",
                "Dockerfile",
                "--worker-binary",
                "rag-runpod-worker",
            ],
            vec!["rag", "runpod", "bundle", "validate", "--bundle", "sealed"],
            vec![
                "rag",
                "runpod",
                "verify",
                "--bundle",
                "sealed",
                "--fetched",
                "completed",
            ],
            vec![
                "rag",
                "runpod",
                "pod",
                "terminate",
                "--id",
                "pod-1",
                "--confirm-terminate",
                "pod-1",
            ],
            vec![
                "rag",
                "runpod",
                "conformance",
                "build",
                "--template",
                "candidate-template.json",
                "--input-root",
                "inputs",
                "--out",
                "candidate",
            ],
            vec![
                "rag",
                "runpod",
                "conformance",
                "seal",
                "--candidate",
                "candidate",
                "--first-result",
                "first.json",
                "--fresh-pod-replay-result",
                "replay.json",
                "--policy-draft",
                "draft.json",
                "--out",
                "policy.json",
            ],
            vec![
                "rag",
                "runpod",
                "storage-challenge",
                "dry-run",
                "--executor-image-build",
                "executor-image-build.json",
                "--volume",
                "volume.json",
                "--image",
                "ghcr.io/example/worker@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "--gpu-type-id",
                "NVIDIA A30",
                "--name",
                "storage-proof",
                "--run-prefix",
                "runs/storage-proof",
                "--maximum-hourly-price",
                "1.0",
                "--maximum-runtime-seconds",
                "300",
                "--maximum-total-compute-usd",
                "0.10",
            ],
        ] {
            crate::Cli::try_parse_from(arguments).unwrap();
        }
    }

    #[test]
    fn storage_challenge_schemas_are_closed_public_contracts() {
        for (text, id) in [
            (
                include_str!(
                    "../../rag-pipeline/schema/runpod-storage-challenge-response.v1.schema.json"
                ),
                "https://livefire.dev/rag/runpod-storage-challenge-response.v1.schema.json",
            ),
            (
                include_str!(
                    "../../rag-pipeline/schema/runpod-storage-challenge-receipt.v1.schema.json"
                ),
                "https://livefire.dev/rag/runpod-storage-challenge-receipt.v1.schema.json",
            ),
        ] {
            let schema: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(schema["$id"], id);
            assert_eq!(schema["additionalProperties"], false);
        }
    }

    #[test]
    fn storage_challenge_request_binds_fresh_bytes_and_has_no_credentials() {
        let root = tempfile::tempdir().unwrap();
        let dockerfile = root.path().join("Dockerfile");
        let worker = root.path().join("rag-runpod-worker");
        let receipt_path = root.path().join("executor-image-build.json");
        let volume_path = root.path().join("volume.json");
        fs::write(&dockerfile, b"FROM scratch\n").unwrap();
        fs::write(&worker, b"linux-amd64-worker").unwrap();
        let image = format!(
            "ghcr.io/example/livefire-rag-worker@sha256:{}",
            "a".repeat(64)
        );
        seal_executor_image(SealExecutorImageOptions {
            executor_image: image.clone(),
            executor_component_id: "livefire.rag.runpod-executor-image".into(),
            executor_version: "test-build".into(),
            tei_base_image: format!(
                "ghcr.io/huggingface/text-embeddings-inference@sha256:{}",
                "b".repeat(64)
            ),
            tei_base_component_id: "huggingface.text-embeddings-inference".into(),
            tei_base_version: "1.9.3".into(),
            dockerfile,
            worker_binary: worker,
            dockerfile_object_path: "container/Dockerfile".into(),
            worker_object_path: "bin/rag-runpod-worker".into(),
            out: receipt_path.clone(),
        })
        .unwrap();
        write_canonical_json(
            &volume_path,
            &runpod_control::NetworkVolume {
                id: "volume-1".into(),
                name: "storage-proof".into(),
                size: 10,
                data_center_id: "US-KS-2".into(),
            },
        )
        .unwrap();
        let options = StorageChallengeOptions {
            executor_image_build: receipt_path,
            volume: volume_path,
            image: image.clone(),
            gpu_type_id: "NVIDIA A30".into(),
            name: "storage-proof".into(),
            run_prefix: "runs/storage-proof".into(),
            container_disk_gb: 20,
            maximum_hourly_price: 1.0,
            maximum_runtime_seconds: 300,
            maximum_total_compute_usd: 0.1,
            api_key_environment: "RUNPOD_API_KEY".into(),
            access_key_environment: "RUNPOD_S3_ACCESS_KEY".into(),
            secret_key_environment: "RUNPOD_S3_SECRET_KEY".into(),
            out: None,
            launch_out: None,
            create_out: None,
        };
        let prepared = prepare_storage_challenge(&image).unwrap();
        let specification = storage_challenge_specification(&options, &prepared.challenge).unwrap();
        let request = runpod_control::dry_run_schedule_pod(
            &specification,
            "2026-08-17T00:05:00Z",
            "RUNPOD_API_KEY",
        )
        .unwrap();
        let rendered = serde_json::to_string(&request).unwrap();
        assert!(rendered.contains(prepared.challenge.sha256.as_str()));
        assert!(rendered.contains("storage-challenge"));
        assert!(!rendered.contains("secret"));
        assert!(rendered.contains("Bearer <redacted>"));
        assert!(!rendered.contains("runpod-secret-key-material"));
    }

    #[test]
    fn executor_image_receipt_is_derived_from_exact_local_files() {
        let root = tempfile::tempdir().unwrap();
        let dockerfile = root.path().join("Dockerfile");
        let worker = root.path().join("rag-runpod-worker");
        let receipt_path = root.path().join("receipt.json");
        fs::write(&dockerfile, b"FROM scratch\n").unwrap();
        fs::write(&worker, b"linux-amd64-worker").unwrap();
        let image_sha = "a".repeat(64);
        let base_sha = "b".repeat(64);
        seal_executor_image(SealExecutorImageOptions {
            executor_image: format!("ghcr.io/example/livefire-rag-worker@sha256:{image_sha}"),
            executor_component_id: "livefire.rag.runpod-executor-image".into(),
            executor_version: "test-build".into(),
            tei_base_image: format!(
                "ghcr.io/huggingface/text-embeddings-inference@sha256:{base_sha}"
            ),
            tei_base_component_id: "huggingface.text-embeddings-inference".into(),
            tei_base_version: "1.9.3".into(),
            dockerfile: dockerfile.clone(),
            worker_binary: worker.clone(),
            dockerfile_object_path: "container/Dockerfile".into(),
            worker_object_path: "bin/rag-runpod-worker".into(),
            out: receipt_path.clone(),
        })
        .unwrap();

        let receipt = validate_executor_image_files(&receipt_path, &dockerfile, &worker).unwrap();
        assert_eq!(
            receipt.worker_binary.component_sha256,
            receipt.worker_binary.object.sha256
        );
        fs::write(&worker, b"different-worker").unwrap();
        assert!(validate_executor_image_files(&receipt_path, &dockerfile, &worker).is_err());
    }

    #[test]
    fn executor_image_reference_requires_one_exact_digest() {
        assert!(
            parse_digest_pinned_image("ghcr.io/example/worker:latest", "image".into(), "1".into())
                .is_err()
        );
        assert!(
            parse_digest_pinned_image(
                &format!("ghcr.io/example/worker@sha256:{}", "A".repeat(64)),
                "image".into(),
                "1".into()
            )
            .is_err()
        );
    }

    #[test]
    fn local_publication_is_no_overwrite_and_bundle_marker_moves_last() {
        let root = tempfile::tempdir().unwrap();
        let staged = root.path().join("staged");
        fs::create_dir(&staged).unwrap();
        fs::create_dir(staged.join("input")).unwrap();
        fs::write(staged.join("input/object"), b"object").unwrap();
        fs::write(staged.join(BUNDLE_FILE), b"bundle").unwrap();
        let destination = root.path().join("published");
        publish_without_overwrite(&staged, &destination).unwrap();
        assert_eq!(fs::read(destination.join(BUNDLE_FILE)).unwrap(), b"bundle");
        assert_eq!(
            fs::read(destination.join("input/object")).unwrap(),
            b"object"
        );

        let another = root.path().join("another");
        fs::create_dir(&another).unwrap();
        fs::write(another.join(BUNDLE_FILE), b"changed").unwrap();
        assert!(publish_without_overwrite(&another, &destination).is_err());
        assert_eq!(fs::read(destination.join(BUNDLE_FILE)).unwrap(), b"bundle");
    }

    #[test]
    fn reserved_state_file_is_removed_on_failure_and_never_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("pod.json");
        {
            let _reservation = ReservedJsonOutput::new(&state).unwrap();
            assert!(ReservedJsonOutput::new(&state).is_err());
        }
        assert!(!state.exists());
        _write_new_json_for_test(&state, &serde_json::json!({"id":"pod-1"})).unwrap();
        assert!(ReservedJsonOutput::new(&state).is_err());
        assert_eq!(fs::read_to_string(state).unwrap(), r#"{"id":"pod-1"}"#);
    }

    #[test]
    fn termination_requires_the_same_exact_id_twice() {
        require_termination_confirmation("pod-1", "pod-1").unwrap();
        assert!(require_termination_confirmation("pod-1", "pod-2").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn downloaded_object_cannot_escape_through_a_symlinked_parent() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("object.json"), b"outside").unwrap();
        symlink(outside.path(), root.path().join("escaped")).unwrap();
        let object = CloudObjectRef {
            key: SafeRelativePath::new("escaped/object.json").unwrap(),
            bytes: 7,
            sha256: digest(b"outside"),
        };
        assert!(verify_local_object(root.path(), &object).is_err());
    }

    #[test]
    fn conformance_seal_limits_require_exact_policy_equality() {
        let measured = SealedExecutionLimits {
            maximum_client_batch_size: 8,
            maximum_batch_tokens: 65_536,
            maximum_concurrent_requests: 4,
            request_timeout_ms: 120_000,
            maximum_response_bytes: 1_048_576,
        };
        for drifted in [
            SealedExecutionLimits {
                maximum_client_batch_size: 7,
                ..measured
            },
            SealedExecutionLimits {
                maximum_batch_tokens: 65_535,
                ..measured
            },
            SealedExecutionLimits {
                maximum_concurrent_requests: 3,
                ..measured
            },
            SealedExecutionLimits {
                request_timeout_ms: 119_999,
                ..measured
            },
            SealedExecutionLimits {
                maximum_response_bytes: 1_048_575,
                ..measured
            },
        ] {
            assert_ne!(measured, drifted);
        }
    }

    #[test]
    fn matched_candidate_policy_rejects_resealed_draft_limit_drift() {
        let candidate: RunpodTeiConformanceCandidate = serde_json::from_str(include_str!(
            "../../../rust-fixtures/runpod/tei-conformance-candidate.v1.json"
        ))
        .unwrap();
        candidate.validate().unwrap();
        let policy_bytes = include_bytes!("../../../rust-fixtures/runpod/embedding-policy.v3.json");
        let policy = parse_tei_checkpoint_profile_v3(policy_bytes).unwrap();
        validate_sealed_policy_candidate_binding(&policy, &candidate).unwrap();

        let mut drifted: serde_json::Value = serde_json::from_slice(policy_bytes).unwrap();
        drifted["batching"]["maximum_batch_tokens"] = serde_json::json!(65_535);
        let drifted =
            parse_tei_checkpoint_profile_v3(&rag_pipeline::canonical_json_bytes(&drifted).unwrap())
                .unwrap();
        assert!(validate_sealed_policy_candidate_binding(&drifted, &candidate).is_err());
    }
}
