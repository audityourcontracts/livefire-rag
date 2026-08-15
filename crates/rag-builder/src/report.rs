use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservationStatus {
    Observed,
    Partial,
    CallerSupplied,
    NotMeasured,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GitState {
    pub status: ObservationStatus,
    pub commit: Option<String>,
    pub working_tree_dirty: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MachineContext {
    pub status: ObservationStatus,
    pub operating_system: Option<String>,
    pub operating_system_version: Option<String>,
    pub architecture: Option<String>,
    pub cpu_model: Option<String>,
    pub logical_cpu_count: Option<usize>,
    pub ram_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LmStudioContext {
    pub status: ObservationStatus,
    pub version: Option<String>,
    pub configured_model: String,
    pub returned_model: String,
    pub endpoint_kind: String,
    pub batch_size: Option<usize>,
    pub requests_in_flight: Option<usize>,
    pub cold_load_micros: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResourceUsage {
    pub status: ObservationStatus,
    pub rust_peak_rss_bytes: Option<u64>,
    pub lm_studio_peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TransportByteAccounting {
    pub status: ObservationStatus,
    pub request_body_bytes: Option<u64>,
    pub response_body_bytes: Option<u64>,
    pub submitted_text_bytes: Option<u64>,
    pub decoded_vector_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskArtifactSizes {
    pub status: ObservationStatus,
    pub vector_shard_bytes: Option<u64>,
    pub receipt_bytes: Option<u64>,
    pub task_report_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunArtifactSizes {
    pub status: ObservationStatus,
    pub prepared_corpus_bytes: Option<u64>,
    pub embedding_plan_bytes: Option<u64>,
    pub embedding_profile_bytes: Option<u64>,
    pub vector_shards_bytes: Option<u64>,
    pub receipts_bytes: Option<u64>,
    pub task_reports_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QueryArtifactSizes {
    pub status: ObservationStatus,
    pub index_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalRunContext {
    pub git: GitState,
    pub machine: MachineContext,
    pub resources: ResourceUsage,
}

impl LocalRunContext {
    pub(crate) fn observe() -> Self {
        Self {
            git: observe_git(),
            machine: observe_machine(),
            resources: observe_resources(),
        }
    }

    pub(crate) fn deterministic_test_vectors() -> Self {
        Self {
            git: GitState {
                status: ObservationStatus::NotMeasured,
                commit: None,
                working_tree_dirty: None,
            },
            machine: MachineContext {
                status: ObservationStatus::NotMeasured,
                operating_system: None,
                operating_system_version: None,
                architecture: None,
                cpu_model: None,
                logical_cpu_count: None,
                ram_bytes: None,
            },
            resources: ResourceUsage {
                status: ObservationStatus::NotMeasured,
                rust_peak_rss_bytes: None,
                lm_studio_peak_rss_bytes: None,
            },
        }
    }
}

impl LmStudioContext {
    pub(crate) fn derived_vectors(model: &str) -> Self {
        Self {
            status: ObservationStatus::NotMeasured,
            version: None,
            configured_model: model.to_owned(),
            returned_model: model.to_owned(),
            endpoint_kind: "local_vector_derivation_no_model_calls".into(),
            batch_size: None,
            requests_in_flight: None,
            cold_load_micros: None,
        }
    }
    pub(crate) fn deterministic_test_vectors(model: &str) -> Self {
        Self {
            status: ObservationStatus::NotMeasured,
            version: None,
            configured_model: model.to_owned(),
            returned_model: model.to_owned(),
            endpoint_kind: "deterministic_test_vectors_no_model_calls".into(),
            batch_size: None,
            requests_in_flight: None,
            cold_load_micros: None,
        }
    }
    pub(crate) fn embedding(
        configured_model: &str,
        returned_model: &str,
        batch_size: usize,
        requests_in_flight: usize,
    ) -> Self {
        Self::new(
            configured_model,
            returned_model,
            Some(batch_size),
            Some(requests_in_flight),
        )
    }

    pub(crate) fn query(configured_model: &str, returned_model: &str) -> Self {
        Self::new(configured_model, returned_model, None, None)
    }

    fn new(
        configured_model: &str,
        returned_model: &str,
        batch_size: Option<usize>,
        requests_in_flight: Option<usize>,
    ) -> Self {
        let version = nonempty_environment("LIVEFIRE_LM_STUDIO_VERSION");
        let cold_load_micros = unsigned_environment("LIVEFIRE_LM_STUDIO_COLD_LOAD_MICROS");
        let status = if version.is_some() || cold_load_micros.is_some() {
            ObservationStatus::CallerSupplied
        } else {
            ObservationStatus::Partial
        };
        Self {
            status,
            version,
            configured_model: configured_model.to_owned(),
            returned_model: returned_model.to_owned(),
            endpoint_kind: "local_openai_compatible".into(),
            batch_size,
            requests_in_flight,
            cold_load_micros,
        }
    }
}

pub(crate) fn task_artifact_sizes(vector: &Path, receipt: &Path) -> TaskArtifactSizes {
    let vector_shard_bytes = regular_file_size(vector);
    let receipt_bytes = regular_file_size(receipt);
    TaskArtifactSizes {
        status: if vector_shard_bytes.is_some() && receipt_bytes.is_some() {
            ObservationStatus::Partial
        } else {
            ObservationStatus::Unavailable
        },
        vector_shard_bytes,
        receipt_bytes,
        // A file cannot include its own final serialized byte size without a
        // recursive definition. The complete run summary measures it later.
        task_report_bytes: None,
    }
}

pub(crate) fn run_artifact_sizes(
    prepared: &Path,
    plan: &Path,
    profile: &Path,
    vector_shards: &[PathBuf],
    receipts: &[PathBuf],
    task_reports: &[PathBuf],
) -> RunArtifactSizes {
    let prepared_corpus_bytes = artifact_size(prepared);
    let embedding_plan_bytes = artifact_size(plan);
    let embedding_profile_bytes = regular_file_size(profile);
    let vector_shards_bytes = regular_files_size(vector_shards);
    let receipts_bytes = regular_files_size(receipts);
    let task_reports_bytes = regular_files_size(task_reports);
    let all_observed = [
        prepared_corpus_bytes,
        embedding_plan_bytes,
        embedding_profile_bytes,
        vector_shards_bytes,
        receipts_bytes,
        task_reports_bytes,
    ]
    .iter()
    .all(Option::is_some);
    RunArtifactSizes {
        status: if all_observed {
            ObservationStatus::Observed
        } else {
            ObservationStatus::Partial
        },
        prepared_corpus_bytes,
        embedding_plan_bytes,
        embedding_profile_bytes,
        vector_shards_bytes,
        receipts_bytes,
        task_reports_bytes,
    }
}

fn regular_files_size(paths: &[PathBuf]) -> Option<u64> {
    paths.iter().try_fold(0_u64, |total, path| {
        total.checked_add(regular_file_size(path)?)
    })
}

pub(crate) fn query_artifact_sizes(index: &Path) -> QueryArtifactSizes {
    let index_bytes = artifact_size(index);
    QueryArtifactSizes {
        status: if index_bytes.is_some() {
            ObservationStatus::Observed
        } else {
            ObservationStatus::Unavailable
        },
        index_bytes,
    }
}

fn observe_git() -> GitState {
    let commit = command_output("git", &["rev-parse", "HEAD"])
        .filter(|value| (40..=64).contains(&value.len()))
        .filter(|value| value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(|value| value.to_ascii_lowercase());
    let dirty_output = command_output("git", &["status", "--porcelain"]);
    let working_tree_dirty = dirty_output.as_ref().map(|value| !value.is_empty());
    let status = match (commit.is_some(), working_tree_dirty.is_some()) {
        (true, true) => ObservationStatus::Observed,
        (true, false) | (false, true) => ObservationStatus::Partial,
        (false, false) => ObservationStatus::Unavailable,
    };
    GitState {
        status,
        commit,
        working_tree_dirty,
    }
}

fn observe_machine() -> MachineContext {
    let operating_system = Some(std::env::consts::OS.to_owned());
    let architecture = Some(std::env::consts::ARCH.to_owned());
    let operating_system_version = command_output("uname", &["-sr"]);
    let logical_cpu_count = std::thread::available_parallelism().ok().map(usize::from);
    let cpu_model = if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .or_else(|| command_output("sysctl", &["-n", "hw.model"]))
    } else {
        linux_cpu_model()
    };
    let ram_bytes = if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "hw.memsize"]).and_then(|value| value.parse().ok())
    } else {
        linux_ram_bytes()
    };
    let status = if operating_system_version.is_some()
        && logical_cpu_count.is_some()
        && cpu_model.is_some()
        && ram_bytes.is_some()
    {
        ObservationStatus::Observed
    } else {
        ObservationStatus::Partial
    };
    MachineContext {
        status,
        operating_system,
        operating_system_version,
        architecture,
        cpu_model,
        logical_cpu_count,
        ram_bytes,
    }
}

fn observe_resources() -> ResourceUsage {
    let rust_peak_rss_bytes = unsigned_environment("LIVEFIRE_RUST_PEAK_RSS_BYTES");
    let lm_studio_peak_rss_bytes = unsigned_environment("LIVEFIRE_LM_STUDIO_PEAK_RSS_BYTES");
    ResourceUsage {
        status: if rust_peak_rss_bytes.is_some() || lm_studio_peak_rss_bytes.is_some() {
            ObservationStatus::CallerSupplied
        } else {
            ObservationStatus::NotMeasured
        },
        rust_peak_rss_bytes,
        lm_studio_peak_rss_bytes,
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    Some(value)
}

fn nonempty_environment(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn unsigned_environment(name: &str) -> Option<u64> {
    nonempty_environment(name).and_then(|value| value.parse().ok())
}

fn regular_file_size(path: &Path) -> Option<u64> {
    let metadata = fs::symlink_metadata(path).ok()?;
    (metadata.is_file() && !metadata.file_type().is_symlink()).then_some(metadata.len())
}

fn artifact_size(path: &Path) -> Option<u64> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if metadata.is_file() && !metadata.file_type().is_symlink() {
        Some(metadata.len())
    } else if metadata.is_dir() && !metadata.file_type().is_symlink() {
        directory_size(path)
    } else {
        None
    }
}

fn directory_size(root: &Path) -> Option<u64> {
    let metadata = fs::symlink_metadata(root).ok()?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return None;
    }
    let mut total = 0_u64;
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).ok()? {
            let entry = entry.ok()?;
            let metadata = entry.metadata().ok()?;
            let file_type = entry.file_type().ok()?;
            if file_type.is_symlink() {
                return None;
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                total = total.checked_add(metadata.len())?;
            } else {
                return None;
            }
        }
    }
    Some(total)
}

fn linux_cpu_model() -> Option<String> {
    let content = fs::read_to_string("/proc/cpuinfo").ok()?;
    content.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        matches!(name.trim(), "model name" | "Hardware")
            .then(|| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn linux_ram_bytes() -> Option<u64> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    let kibibytes = content.lines().find_map(|line| {
        let value = line.strip_prefix("MemTotal:")?.trim();
        value.strip_suffix(" kB")?.trim().parse::<u64>().ok()
    })?;
    kibibytes.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_measurements_are_null_and_explicit() {
        let context = LmStudioContext::query("model", "model");
        assert!(matches!(
            context.status,
            ObservationStatus::Partial | ObservationStatus::CallerSupplied
        ));
        if matches!(context.status, ObservationStatus::Partial) {
            assert!(context.version.is_none());
            assert!(context.cold_load_micros.is_none());
        }
        assert!(context.batch_size.is_none());
        assert!(context.requests_in_flight.is_none());
    }

    #[test]
    fn artifact_size_never_follows_a_symlink() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("data"), b"1234").unwrap();
        assert_eq!(directory_size(root.path()), Some(4));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path().join("data"), root.path().join("link")).unwrap();
            assert_eq!(directory_size(root.path()), None);
        }
    }
}
