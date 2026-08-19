use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rag_runpod_worker::{
    ConformanceOptions, RunOptions, StorageChallengeOptions, conformance, prepare_runtime_storage,
    publish_storage_challenge_failure, run, storage_challenge, verify_storage_objects,
};

#[derive(Debug, Parser)]
#[command(name = "rag-runpod-worker")]
#[command(about = "Execute one sealed RunPod embedding assignment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Prove that the mounted run directory is writable after dropping privileges.
    StorageProbe {
        #[arg(long)]
        root: PathBuf,
        /// Verify a relative staged object as PATH BYTES SHA256 after dropping privileges.
        #[arg(
            long = "required-object",
            num_args = 3,
            value_names = ["PATH", "BYTES", "SHA256"]
        )]
        required_objects: Vec<String>,
    },
    /// Prove a host-to-mount-to-host round trip with one exact challenge.
    StorageChallenge {
        #[arg(long)]
        root: PathBuf,
        /// Repository of the exact executor image being challenged.
        #[arg(long)]
        executor_image_repository: String,
        /// sha256:<digest> identity of the exact executor image.
        #[arg(long)]
        executor_image_digest: String,
        #[arg(long)]
        challenge: String,
        #[arg(long)]
        challenge_bytes: u64,
        #[arg(long)]
        challenge_sha256: String,
        #[arg(long)]
        response: String,
        /// Prefix for a content-bound, code-suffixed failure receipt.
        #[arg(long)]
        failure_prefix: String,
        #[arg(long, default_value_t = 300)]
        wait_seconds: u64,
    },
    /// Measure a sealed TEI candidate before an embedding policy exists.
    Conformance {
        #[arg(long, default_value = "/workspace")]
        root: PathBuf,
        #[arg(long)]
        candidate: String,
        #[arg(long)]
        run_id: String,
        #[arg(long)]
        observation: String,
        #[arg(long, default_value_t = 300)]
        observation_wait_seconds: u64,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 3600)]
        health_wait_seconds: u64,
    },
    Run {
        #[arg(long, default_value = "/workspace")]
        root: PathBuf,
        #[arg(long)]
        bundle: String,
        #[arg(long)]
        worker_id: String,
        /// Number of consecutive token-balanced assignments to execute while
        /// the verified model remains loaded in this Pod.
        #[arg(long, default_value_t = 1)]
        assignment_count: u32,
        #[arg(long)]
        attempt_id: String,
        #[arg(long)]
        attempt_number: u32,
        #[arg(long)]
        observation: String,
        #[arg(long, default_value_t = 300)]
        observation_wait_seconds: u64,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        requests_in_flight: usize,
        #[arg(long, default_value_t = 3600)]
        health_wait_seconds: u64,
    },
}

#[tokio::main]
async fn main() {
    let Cli { command } = Cli::parse();
    let result = match command {
        Command::StorageProbe {
            root,
            required_objects,
        } => prepare_runtime_storage(&root).and_then(|root| {
            let (object_count, object_bytes) = verify_storage_objects(&root, &required_objects)?;
            println!(
                "{{\"schema_version\":\"livefire.rag.runpod-storage-probe/1\",\"status\":\"passed\",\"uid\":1000,\"gid\":1000,\"required_objects\":{object_count},\"read_bytes\":{object_bytes}}}"
            );
            Ok(())
        }),
        Command::StorageChallenge {
            root,
            executor_image_repository,
            executor_image_digest,
            challenge,
            challenge_bytes,
            challenge_sha256,
            response,
            failure_prefix,
            wait_seconds,
        } => {
            let executor_image = format!("{executor_image_repository}@{executor_image_digest}");
            let result = match prepare_runtime_storage(&root) {
                Ok(prepared_root) => {
                    storage_challenge(StorageChallengeOptions {
                        root: prepared_root,
                        executor_image: executor_image.clone(),
                        challenge: challenge.clone(),
                        challenge_bytes,
                        challenge_sha256: challenge_sha256.clone(),
                        response,
                        wait_seconds,
                    })
                    .await
                }
                Err(error) => Err(error),
            };
            if let Err(error) = &result {
                let _ = publish_storage_challenge_failure(
                    &root,
                    &failure_prefix,
                    executor_image,
                    challenge,
                    challenge_bytes,
                    challenge_sha256,
                    error.public_code(),
                );
            }
            result
        }
        Command::Conformance {
            root,
            candidate,
            run_id,
            observation,
            observation_wait_seconds,
            port,
            health_wait_seconds,
        } => match prepare_runtime_storage(&root) {
            Ok(root) => conformance(ConformanceOptions {
                root,
                candidate,
                run_id,
                observation,
                observation_wait_seconds,
                port,
                health_wait_seconds,
            })
            .await,
            Err(error) => Err(error),
        },
        Command::Run {
            root,
            bundle,
            worker_id,
            assignment_count,
            attempt_id,
            attempt_number,
            observation,
            observation_wait_seconds,
            port,
            batch_size,
            requests_in_flight,
            health_wait_seconds,
        } => match prepare_runtime_storage(&root) {
            Ok(root) => run(RunOptions {
                root,
                bundle,
                worker_id,
                assignment_count,
                attempt_id,
                attempt_number,
                observation,
                observation_wait_seconds,
                port,
                batch_size,
                requests_in_flight,
                health_wait_seconds,
            })
            .await,
            Err(error) => Err(error),
        },
    };
    if let Err(error) = result {
        eprintln!("runpod worker failed: {}", error.public_code());
        std::process::exit(1);
    }
}
