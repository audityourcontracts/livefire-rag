//! Small content-bound response used to prove mounted RunPod storage behavior.

use serde::{Deserialize, Serialize};

use super::{CloudObjectRef, PipelineError, Result};

pub const RUNPOD_STORAGE_CHALLENGE_RESPONSE_SCHEMA: &str =
    "livefire.rag.runpod-storage-challenge-response/1";

/// The exact bytes a storage-only worker publishes after reading the host's
/// challenge through the mounted network volume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunpodStorageChallengeResponse {
    pub schema_version: String,
    pub executor_image: String,
    pub challenge: CloudObjectRef,
    pub uid: u32,
    pub gid: u32,
    pub publication: String,
}

impl RunpodStorageChallengeResponse {
    pub fn new(executor_image: String, challenge: CloudObjectRef) -> Result<Self> {
        let response = Self {
            schema_version: RUNPOD_STORAGE_CHALLENGE_RESPONSE_SCHEMA.into(),
            executor_image,
            challenge,
            uid: 1000,
            gid: 1000,
            publication: "hard_link_no_overwrite".into(),
        };
        response.validate()?;
        Ok(response)
    }

    pub fn validate(&self) -> Result<()> {
        let Some((repository, digest)) = self.executor_image.split_once("@sha256:") else {
            return Err(PipelineError::Invalid("storage challenge executor image"));
        };
        if self.schema_version != RUNPOD_STORAGE_CHALLENGE_RESPONSE_SCHEMA
            || repository.is_empty()
            || repository.chars().any(char::is_whitespace)
            || digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.challenge.bytes == 0
            || self.challenge.bytes > 1024 * 1024
            || self.uid != 1000
            || self.gid != 1000
            || self.publication != "hard_link_no_overwrite"
        {
            return Err(PipelineError::Invalid("storage challenge response"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Digest, SafeRelativePath};

    #[test]
    fn response_binds_the_exact_image_and_challenge() {
        let response = RunpodStorageChallengeResponse::new(
            format!("ghcr.io/example/worker@sha256:{}", "a".repeat(64)),
            CloudObjectRef {
                key: SafeRelativePath::new("runtime/storage-challenge/challenge.bin").unwrap(),
                bytes: 32,
                sha256: Digest::new("b".repeat(64)).unwrap(),
            },
        )
        .unwrap();
        response.validate().unwrap();
        let mut changed = response.clone();
        changed.executor_image = "ghcr.io/example/worker:latest".into();
        assert!(changed.validate().is_err());
    }

    #[test]
    fn response_schema_is_closed_and_names_the_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/runpod-storage-challenge-response.v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["$id"],
            "https://livefire.dev/rag/runpod-storage-challenge-response.v1.schema.json"
        );
        assert_eq!(schema["additionalProperties"], false);
    }
}
