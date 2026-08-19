//! Durable, content-bound progress and failure records for one RunPod worker attempt.

use serde::{Deserialize, Serialize};

use super::{Digest, PipelineError, Result, component_digest, require_safe_u64, require_text};

pub const RUNPOD_WORKER_RUNTIME_EVENT_SCHEMA: &str = "livefire.rag.runpod-worker-runtime-event/1";

pub const RUNPOD_WORKER_RUNTIME_PHASES: &[&str] = &[
    "worker_started",
    "control_objects_verified",
    "model_tree_verified",
    "observation_verified",
    "gpu_verified",
    "tei_started",
    "tei_healthy",
    "conformance_passed",
    "query_vectors_published",
    "first_task_published",
    "assignment_completed",
];

/// One immutable checkpoint or terminal failure from a worker attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodWorkerRuntimeEvent {
    pub schema_version: String,
    pub component_sha256: Digest,
    pub bundle_file_sha256: Digest,
    pub worker_id: String,
    pub attempt_id: String,
    pub attempt_number: u32,
    pub sequence: u32,
    pub phase: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
}

impl RunpodWorkerRuntimeEvent {
    pub fn progress(
        bundle_file_sha256: Digest,
        worker_id: String,
        attempt_id: String,
        attempt_number: u32,
        sequence: u32,
        phase: &str,
    ) -> Result<Self> {
        Self::new(
            bundle_file_sha256,
            worker_id,
            attempt_id,
            attempt_number,
            sequence,
            phase,
            "completed",
            None,
        )
    }

    pub fn failure(
        bundle_file_sha256: Digest,
        worker_id: String,
        attempt_id: String,
        attempt_number: u32,
        sequence: u32,
        phase: &str,
        failure_code: &str,
    ) -> Result<Self> {
        Self::new(
            bundle_file_sha256,
            worker_id,
            attempt_id,
            attempt_number,
            sequence,
            phase,
            "failed",
            Some(failure_code.to_owned()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        bundle_file_sha256: Digest,
        worker_id: String,
        attempt_id: String,
        attempt_number: u32,
        sequence: u32,
        phase: &str,
        status: &str,
        failure_code: Option<String>,
    ) -> Result<Self> {
        let mut event = Self {
            schema_version: RUNPOD_WORKER_RUNTIME_EVENT_SCHEMA.into(),
            component_sha256: super::digest_bytes(b"unsealed runpod worker runtime event"),
            bundle_file_sha256,
            worker_id,
            attempt_id,
            attempt_number,
            sequence,
            phase: phase.to_owned(),
            status: status.to_owned(),
            failure_code,
        };
        event.seal()?;
        Ok(event)
    }

    pub fn validate(&self) -> Result<()> {
        require_text(&self.worker_id)?;
        require_text(&self.attempt_id)?;
        require_safe_u64(u64::from(self.attempt_number))?;
        require_safe_u64(u64::from(self.sequence))?;
        let valid_identifier = |value: &str| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        };
        let failure_is_valid = self.failure_code.as_deref().is_some_and(valid_identifier);
        if self.schema_version != RUNPOD_WORKER_RUNTIME_EVENT_SCHEMA
            || self.attempt_number == 0
            || self.sequence == 0
            || !valid_identifier(&self.worker_id)
            || !valid_identifier(&self.attempt_id)
            || !RUNPOD_WORKER_RUNTIME_PHASES.contains(&self.phase.as_str())
            || !matches!(self.status.as_str(), "completed" | "failed")
            || (self.status == "completed" && self.failure_code.is_some())
            || (self.status == "failed" && !failure_is_valid)
            || self.component_sha256 != component_digest(self)?
        {
            return Err(PipelineError::Invalid("runpod worker runtime event"));
        }
        Ok(())
    }

    pub fn seal(&mut self) -> Result<()> {
        self.component_sha256 = component_digest(self)?;
        self.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_and_failure_are_sealed_and_closed() {
        let bundle = Digest::new("a".repeat(64)).unwrap();
        let progress = RunpodWorkerRuntimeEvent::progress(
            bundle.clone(),
            "worker-0000".into(),
            "attempt-1".into(),
            1,
            1,
            "worker_started",
        )
        .unwrap();
        progress.validate().unwrap();
        let failure = RunpodWorkerRuntimeEvent::failure(
            bundle,
            "worker-0000".into(),
            "attempt-1".into(),
            1,
            2,
            "control_objects_verified",
            "model_object_binding",
        )
        .unwrap();
        failure.validate().unwrap();
        let mut changed = failure;
        changed.failure_code = Some("contains space".into());
        changed.component_sha256 = component_digest(&changed).unwrap();
        assert!(changed.validate().is_err());
    }

    #[test]
    fn schema_is_closed() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/runpod-worker-runtime-event.v1.schema.json"
        ))
        .unwrap();
        assert_eq!(schema["additionalProperties"], false);
    }
}
