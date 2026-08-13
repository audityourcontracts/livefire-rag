//! Portable data contracts shared by the experimental RAG components.
//!
//! These types intentionally contain no filesystem, database, HTTP, or SDK
//! runtime dependencies so the same wire values can be used by a later WASI
//! provider.

#![forbid(unsafe_code)]

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A lowercase, 64-character SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256(String);

impl Sha256 {
    /// Parse and validate a canonical lowercase SHA-256 digest.
    pub fn new(value: impl Into<String>) -> Result<Self, ContractError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .as_bytes()
                .iter()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
        {
            return Err(ContractError::InvalidSha256);
        }
        Ok(Self(value))
    }

    /// Borrow the canonical hexadecimal representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the digest and return its canonical representation.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for Sha256 {
    type Err = ContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for Sha256 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// An event reference that remains meaningful only with its exact OCSF
/// snapshot and mapping identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    pub snapshot_sha256: Sha256,
    pub mapping_sha256: Sha256,
    pub event_id: String,
    pub support_ref: String,
}

/// Retrieval implementation selected for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Dense,
    Lexical,
    Fused,
}

/// Occurrence-level filters applied before a document becomes eligible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchFilter {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_time_start_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_time_end_ms: Option<u64>,
}

/// A portable search request used by CLI and provider adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub query: String,
    pub mode: SearchMode,
    pub limit: u32,
    #[serde(default)]
    pub filter: SearchFilter,
}

/// Scores for one returned document. Scores not produced by the selected
/// retriever remain absent rather than being assigned misleading zeroes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchScores {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused: Option<f32>,
}

/// A ranked semantic document plus the snapshot-scoped occurrences that made
/// it eligible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchHit {
    pub rank: u32,
    pub document_id: String,
    pub semantic_text: String,
    pub scores: SearchScores,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ContractError {
    #[error("SHA-256 digest must contain exactly 64 lowercase hexadecimal characters")]
    InvalidSha256,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_accepts_only_canonical_lowercase_hex() {
        assert!(Sha256::new("a".repeat(64)).is_ok());
        assert!(Sha256::new("A".repeat(64)).is_err());
        assert!(Sha256::new("a".repeat(63)).is_err());
        assert!(Sha256::new("z".repeat(64)).is_err());
    }

    #[test]
    fn evidence_reference_round_trips_without_losing_bindings() {
        let reference = EvidenceRef {
            snapshot_sha256: Sha256::new("a".repeat(64)).expect("digest"),
            mapping_sha256: Sha256::new("b".repeat(64)).expect("digest"),
            event_id: "evt_1".to_owned(),
            support_ref: "sup_1".to_owned(),
        };
        let encoded = serde_json::to_string(&reference).expect("serialize");
        let decoded: EvidenceRef = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, reference);
    }
}
