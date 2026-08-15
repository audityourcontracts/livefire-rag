//! Portable data contracts shared by the experimental RAG components.
//!
//! These types intentionally contain no filesystem, database, HTTP, or SDK
//! runtime dependencies so the same wire values can be used by a later WASI
//! provider.

#![forbid(unsafe_code)]

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DOCUMENT_INPUT_TOKEN: &str = "{semantic_text}";
pub const MAX_DOCUMENT_FORMAT_BYTES: usize = 4_096;
pub const MAX_FORMATTED_DOCUMENT_BYTES: usize = 1_048_576;

/// Apply the closed document-input format shared by planning and execution.
/// Length limits are UTF-8 bytes, not Unicode scalar-value counts.
pub fn format_document_input(
    document_format: &str,
    semantic_text: &str,
) -> Result<String, ContractError> {
    if semantic_text.is_empty()
        || document_format.is_empty()
        || document_format.len() > MAX_DOCUMENT_FORMAT_BYTES
        || document_format.matches(DOCUMENT_INPUT_TOKEN).count() != 1
    {
        return Err(ContractError::InvalidDocumentFormat);
    }
    let remaining = document_format.replace(DOCUMENT_INPUT_TOKEN, "");
    if remaining.contains('{') || remaining.contains('}') {
        return Err(ContractError::InvalidDocumentFormat);
    }
    let size = document_format
        .len()
        .checked_sub(DOCUMENT_INPUT_TOKEN.len())
        .and_then(|fixed| fixed.checked_add(semantic_text.len()))
        .ok_or(ContractError::InvalidDocumentFormat)?;
    if size > MAX_FORMATTED_DOCUMENT_BYTES {
        return Err(ContractError::InvalidDocumentFormat);
    }
    Ok(document_format.replace(DOCUMENT_INPUT_TOKEN, semantic_text))
}

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
    #[error("document input format is invalid or exceeds its UTF-8 byte limit")]
    InvalidDocumentFormat,
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

    #[test]
    fn document_formatter_is_closed_and_byte_bounded() {
        assert_eq!(
            format_document_input("passage: {semantic_text}", "évent").unwrap(),
            "passage: évent"
        );
        for format in [
            "no placeholder",
            "{semantic_text} {semantic_text}",
            "{semantic_text} {unknown}",
            "left {semantic_text} right }",
            "left { {semantic_text} right",
        ] {
            assert!(format_document_input(format, "text").is_err(), "{format}");
        }
        assert!(format_document_input(DOCUMENT_INPUT_TOKEN, "").is_err());
        let exact = "x".repeat(MAX_FORMATTED_DOCUMENT_BYTES);
        assert_eq!(
            format_document_input(DOCUMENT_INPUT_TOKEN, &exact)
                .unwrap()
                .len(),
            MAX_FORMATTED_DOCUMENT_BYTES
        );
        assert!(
            format_document_input(
                DOCUMENT_INPUT_TOKEN,
                &"x".repeat(MAX_FORMATTED_DOCUMENT_BYTES + 1)
            )
            .is_err()
        );
        let maximum_format = format!(
            "{}{DOCUMENT_INPUT_TOKEN}",
            "x".repeat(MAX_DOCUMENT_FORMAT_BYTES - DOCUMENT_INPUT_TOKEN.len())
        );
        assert!(format_document_input(&maximum_format, "x").is_ok());
        assert!(format_document_input(&format!("x{maximum_format}"), "x").is_err());
    }
}
