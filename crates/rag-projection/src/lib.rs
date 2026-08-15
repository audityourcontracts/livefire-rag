//! Scenario-blind projection of typed OCSF observations into retrievable
//! semantic documents and occurrence metadata.
//!
//! This crate deliberately does not know about queries, incidents, qrels, or
//! expected evidence. The builder supplies source-object/row pointer material
//! after this pure transformation.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub const PROJECTION_SCHEMA_VERSION: &str = "livefire.rag.evidence-projection/1";
pub const PROJECTION_POLICY_ID: &str = "livefire.rag.generic-evidence-projection-policy";
pub const PROJECTION_POLICY_VERSION: &str = "2";

const MAX_LEAVES: usize = 160;
const MAX_LIST_ITEMS: usize = 24;
const MAX_VALUE_CHARS: usize = 240;
const MAX_FACET_TEXT_CHARS: usize = 1_024;
const MAX_SEMANTIC_TEXT_CHARS: usize = 3_072;
const MAX_PROJECTION_SCALARS_SCANNED: usize = 1_024;
const MAX_EXACT_ATTRIBUTES: usize = 256;
const MAX_EXACT_SCALARS_SCANNED: usize = 512;
const MAX_EXACT_LIST_ITEMS: usize = 64;
const MAX_EXACT_STRING_UTF8_BYTES: usize = 1_024;
const MAX_EXACT_PATH_CHARS: usize = 1_024;
const MAX_JCS_SAFE_INTEGER: i64 = (1_i64 << 53) - 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentRef {
    pub id: String,
    pub version: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionContext {
    pub snapshot: ComponentRef,
    pub mapping_pack: ComponentRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionInput<'a> {
    pub relation_name: &'a str,
    pub event_id: &'a str,
    pub typed_event_json: &'a str,
    pub support_ref: &'a str,
    pub context: &'a ProjectionContext,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionOutput {
    pub document: Option<ProjectedDocument>,
    pub occurrence: ProjectedOccurrence,
}

/// The identity-bearing portion of projection needed during document census.
/// It deliberately excludes exact occurrence attributes, which are materialized
/// only after a representative builder has selected a document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedDocumentSummary {
    pub document: Option<ProjectedDocument>,
    pub event_time: Option<String>,
    pub event_time_availability: EventTimeAvailability,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedDocument {
    pub document_id: String,
    pub document_kind: DocumentKind,
    pub semantic_text: String,
    pub facets: SemanticFacets,
    pub semantic_group_sha256: String,
    pub snapshot: ComponentRef,
    pub mapping_pack: ComponentRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedOccurrence {
    pub schema_version: String,
    pub event_id: String,
    pub support_ref: String,
    pub relation_name: String,
    pub document_kind: DocumentKind,
    pub terminal_disposition: TerminalDisposition,
    pub disposition_reason: String,
    pub semantic_group_id: String,
    pub semantic_group_sha256: String,
    pub event_time: Option<String>,
    pub event_time_availability: EventTimeAvailability,
    pub exact_attributes: Vec<ExactAttribute>,
    pub exact_attribute_projection: ExactAttributeProjection,
    pub snapshot: ComponentRef,
    pub mapping_pack: ComponentRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    Activity,
    State,
    Detection,
    StructuredOnly,
}

impl DocumentKind {
    fn label(self) -> &'static str {
        match self {
            Self::Activity => "activity",
            Self::State => "state",
            Self::Detection => "detection",
            Self::StructuredOnly => "structured_only",
        }
    }

    fn semantic_label(self) -> &'static str {
        match self {
            Self::StructuredOnly => "structured only",
            _ => self.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalDisposition {
    DirectSemanticDocument,
    StructuredOnlyOccurrence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventTimeAvailability {
    Available,
    Missing,
    PresentUnparsed,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFacets {
    pub action: String,
    pub target: String,
    pub context: String,
    pub outcome: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactAttribute {
    pub namespace: String,
    pub path: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OmissionCount {
    pub reason: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactAttributeProjection {
    pub selected_count: usize,
    pub scalars_scanned: usize,
    pub known_omitted_scalar_count: usize,
    pub omitted_subtree_count: usize,
    pub omission_counts: Vec<OmissionCount>,
    pub scan_truncated: bool,
    pub source_hydration_required: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectionError {
    #[error("typed event is invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
}

struct ProjectionCore {
    typed_event: Value,
    parse_error: bool,
    document_kind: DocumentKind,
    semantic_group_id: String,
    semantic_group_sha256: String,
    terminal_disposition: TerminalDisposition,
    disposition_reason: &'static str,
    document: Option<ProjectedDocument>,
    event_time: Option<String>,
    event_time_availability: EventTimeAvailability,
}

#[derive(Clone, Debug)]
struct Leaf {
    path: String,
    value: Value,
}

#[derive(Default)]
struct FlattenState {
    scanned: usize,
    truncated: bool,
}

static LIST_INDEX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[\d+\]").unwrap());
static WORD_CHAR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\w$").unwrap());
static CAMEL_BOUNDARY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-z0-9])([A-Z])").unwrap());
static NON_KEY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-z0-9]+").unwrap());
static POSITIONAL_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(?:^|\.)unmapped\.\$token/\d+(?:\.|$)").unwrap());
static SECRET_ASSIGNMENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)(password|passwd|pwd|secret|token|api[-_]?key|access[-_]?key|authorization|cookie|private[-_]?key)(\s*(?:=|:)\s*|\s+)("[^"]*"|'[^']*'|[^\s,;]+)"#,
    )
    .unwrap()
});
static BEARER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+").unwrap());
static EMAIL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap());
static IPV4_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}").unwrap()
});
static UUID_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
        .unwrap()
});
static LONG_HEX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)[0-9a-f]{32,}").unwrap());
static CLOUD_IDENTIFIER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\barn:[a-z0-9-]+:[^\s,;]+").unwrap());
static ACCESS_KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b").unwrap());
static JWT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}").unwrap());
static MAC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}").unwrap());

pub fn project(input: ProjectionInput<'_>) -> Result<ProjectionOutput, ProjectionError> {
    let core = project_core(input.relation_name, input.typed_event_json, input.context);
    let (exact_attributes, mut exact_projection) = exact_attribute_subset(&core.typed_event);
    if core.parse_error {
        exact_projection.source_hydration_required = true;
    }
    let occurrence = ProjectedOccurrence {
        schema_version: PROJECTION_SCHEMA_VERSION.to_owned(),
        event_id: input.event_id.to_owned(),
        support_ref: input.support_ref.to_owned(),
        relation_name: input.relation_name.to_owned(),
        document_kind: core.document_kind,
        terminal_disposition: core.terminal_disposition,
        disposition_reason: core.disposition_reason.to_owned(),
        semantic_group_id: core.semantic_group_id,
        semantic_group_sha256: core.semantic_group_sha256,
        event_time: core.event_time,
        event_time_availability: core.event_time_availability,
        exact_attributes,
        exact_attribute_projection: exact_projection,
        snapshot: input.context.snapshot.clone(),
        mapping_pack: input.context.mapping_pack.clone(),
    };
    Ok(ProjectionOutput {
        document: core.document,
        occurrence,
    })
}

/// Project only document identity/content and event-time accounting. This uses
/// the same core as [`project`] and therefore cannot drift from occurrence
/// projection, but avoids walking and allocating exact attributes.
pub fn project_document_summary(
    relation_name: &str,
    typed_event_json: &str,
    context: &ProjectionContext,
) -> Result<ProjectedDocumentSummary, ProjectionError> {
    let core = project_core(relation_name, typed_event_json, context);
    Ok(ProjectedDocumentSummary {
        document: core.document,
        event_time: core.event_time,
        event_time_availability: core.event_time_availability,
    })
}

fn project_core(
    relation_name: &str,
    typed_event_json: &str,
    context: &ProjectionContext,
) -> ProjectionCore {
    // Projection is total over source rows. Malformed or non-object typed JSON
    // becomes a structured-only occurrence which the authoritative source can
    // still hydrate; it must not disappear because parsing failed.
    let parsed: Option<Value> = serde_json::from_str(typed_event_json).ok();
    let parse_error = parsed.as_ref().and_then(Value::as_object).is_none();
    let typed_event = parsed
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(Map::new()));
    let leaves = flattened_leaves(&typed_event);

    let document_kind = relation_kind(relation_name);
    let known_relation = document_kind.is_some();
    let document_kind = document_kind.unwrap_or(DocumentKind::StructuredOnly);
    let derivation_only = relation_name == "ocsf_ext_livefire_system_metric";
    let searchable = known_relation && !derivation_only && !parse_error;

    let semantic_leaves = semantic_entries(&leaves);
    let facets = SemanticFacets {
        action: facet_text(&semantic_leaves, Role::Action),
        target: facet_text(&semantic_leaves, Role::Target),
        context: facet_text(&semantic_leaves, Role::Context),
        outcome: facet_text(&semantic_leaves, Role::Outcome),
    };
    let semantic_text = compose_semantic_text(document_kind, relation_name, &facets, parse_error);

    let group_material = json!({
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "relation_name": relation_name,
        "document_kind": document_kind.label(),
        "action_text": facets.action,
        "target_text": facets.target,
        "context_text": facets.context,
        "outcome_text": facets.outcome,
        "semantic_leaves": semantic_leaves.iter()
            .filter(|leaf| !is_positional_raw_token(&leaf.path) && !is_identifier(&leaf.path) && !is_secret(&leaf.path))
            .map(|leaf| json!([leaf.path, semantic_value(&leaf.path, &leaf.value)]))
            .collect::<Vec<_>>(),
    });
    let semantic_group_sha256 = digest_value(&group_material);
    let semantic_group_id = format!("sha256:{semantic_group_sha256}");

    let (event_time, event_time_availability) = event_time(&leaves);
    let (terminal_disposition, disposition_reason) = if parse_error {
        (
            TerminalDisposition::StructuredOnlyOccurrence,
            "typed_event_unavailable",
        )
    } else if !known_relation {
        (
            TerminalDisposition::StructuredOnlyOccurrence,
            "unknown_typed_relation",
        )
    } else if derivation_only {
        (
            TerminalDisposition::StructuredOnlyOccurrence,
            "awaits_deterministic_window_derivation",
        )
    } else {
        (
            TerminalDisposition::DirectSemanticDocument,
            "projected_by_generic_typed_field_policy",
        )
    };

    let document = searchable.then(|| ProjectedDocument {
        document_id: semantic_group_id.clone(),
        document_kind,
        semantic_text,
        facets,
        semantic_group_sha256: semantic_group_sha256.clone(),
        snapshot: context.snapshot.clone(),
        mapping_pack: context.mapping_pack.clone(),
    });
    ProjectionCore {
        typed_event,
        parse_error,
        document_kind,
        semantic_group_id,
        semantic_group_sha256,
        terminal_disposition,
        disposition_reason,
        document,
        event_time,
        event_time_availability,
    }
}

/// Extract the normalized event-time state using the exact same bounded leaf
/// ordering as [`project`] without constructing semantic facets, exact
/// attributes, or content hashes. Builders use this for structurally
/// non-searchable metric rows so complete source accounting does not pay the
/// full semantic-projection cost.
#[must_use]
pub fn project_event_time(typed_event_json: &str) -> (Option<String>, EventTimeAvailability) {
    let parsed: Option<Value> = serde_json::from_str(typed_event_json).ok();
    let empty = Value::Object(Map::new());
    let typed_event = parsed
        .as_ref()
        .filter(|value| value.is_object())
        .unwrap_or(&empty);
    event_time(&flattened_leaves(typed_event))
}

fn flattened_leaves(typed_event: &Value) -> Vec<Leaf> {
    let mut flatten_state = FlattenState::default();
    let mut leaves = Vec::new();
    flatten_value(typed_event, "", &mut leaves, &mut flatten_state);
    leaves.sort_by_key(|leaf| (leaf_priority(&leaf.path), leaf.path.clone()));
    if leaves.len() > MAX_LEAVES {
        leaves.truncate(MAX_LEAVES);
    }
    leaves
}

pub fn relation_kind(relation: &str) -> Option<DocumentKind> {
    Some(match relation {
        "ocsf_cloud_resources_inventory_info"
        | "ocsf_ext_livefire_configuration_snapshot"
        | "ocsf_inventory_info"
        | "ocsf_user_inventory" => DocumentKind::State,
        "ocsf_detection_finding" => DocumentKind::Detection,
        "ocsf_ext_livefire_system_metric" => DocumentKind::StructuredOnly,
        "ocsf_api_activity"
        | "ocsf_application_lifecycle"
        | "ocsf_authentication"
        | "ocsf_datastore_activity"
        | "ocsf_dns_activity"
        | "ocsf_email_activity"
        | "ocsf_entity_management"
        | "ocsf_event_log_activity"
        | "ocsf_file_activity"
        | "ocsf_http_activity"
        | "ocsf_network_activity"
        | "ocsf_process_activity" => DocumentKind::Activity,
        _ => return None,
    })
}

fn flatten_value(value: &Value, path: &str, output: &mut Vec<Leaf>, state: &mut FlattenState) {
    if state.scanned >= MAX_PROJECTION_SCALARS_SCANNED {
        state.truncated = true;
        return;
    }
    match value {
        Value::Object(map) => {
            let mut children: Vec<_> = map.iter().collect();
            children.sort_by_key(|(key, _)| (subtree_priority(key), normalize_key(key)));
            for (key, child) in children {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                flatten_value(child, &child_path, output, state);
            }
            if map.is_empty() && !path.is_empty() {
                state.scanned += 1;
                output.push(Leaf {
                    path: path.to_owned(),
                    value: Value::Null,
                });
            }
        }
        Value::Array(items) => {
            if items.len() > MAX_LIST_ITEMS {
                state.truncated = true;
            }
            for (index, child) in items.iter().take(MAX_LIST_ITEMS).enumerate() {
                flatten_value(child, &format!("{path}[{index}]"), output, state);
            }
            if items.is_empty() && !path.is_empty() {
                state.scanned += 1;
                output.push(Leaf {
                    path: path.to_owned(),
                    value: Value::Null,
                });
            }
        }
        _ => {
            state.scanned += 1;
            output.push(Leaf {
                path: if path.is_empty() {
                    "value".to_owned()
                } else {
                    path.to_owned()
                },
                value: value.clone(),
            });
        }
    }
}

fn subtree_priority(key: &str) -> u8 {
    let normalized = normalize_key(key);
    let tokens = path_tokens(&normalized);
    if tokens.contains("unmapped") || tokens.contains("raw") {
        2
    } else if tokens
        .iter()
        .any(|token| PRIORITY_TOKENS.contains(&token.as_str()))
    {
        0
    } else {
        1
    }
}

fn leaf_priority(path: &str) -> u8 {
    let normalized = format!(".{}.", LIST_INDEX_RE.replace_all(path, "").to_lowercase());
    if is_time(path) || role(path) != Role::Context || is_free_text(path) {
        0
    } else if !normalized.contains(".unmapped.") && !is_positional_raw_token(path) {
        1
    } else {
        2
    }
}

fn semantic_entries(leaves: &[Leaf]) -> Vec<Leaf> {
    let typed_names: BTreeSet<String> = leaves
        .iter()
        .filter(|leaf| {
            let normalized = format!(
                ".{}.",
                LIST_INDEX_RE.replace_all(&leaf.path, "").to_lowercase()
            );
            !normalized.contains(".unmapped.")
                && !leaf.value.is_null()
                && !is_time(&leaf.path)
                && !is_volatile(&leaf.path)
                && !is_positional_raw_token(&leaf.path)
                && !is_semantic_noise(&leaf.path)
        })
        .map(|leaf| semantic_leaf_alias(&leaf_name(&leaf.path)).to_owned())
        .collect();
    leaves
        .iter()
        .filter(|leaf| {
            let normalized = format!(
                ".{}.",
                LIST_INDEX_RE.replace_all(&leaf.path, "").to_lowercase()
            );
            !leaf.value.is_null()
                && !is_time(&leaf.path)
                && !is_volatile(&leaf.path)
                && !is_positional_raw_token(&leaf.path)
                && !is_semantic_noise(&leaf.path)
                && (!is_identifier(&leaf.path)
                    || IDENTIFIER_PLACEHOLDERS.contains(&leaf_name(&leaf.path).as_str()))
                && !(normalized.contains(".unmapped.")
                    && typed_names.contains(semantic_leaf_alias(&leaf_name(&leaf.path))))
        })
        .cloned()
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Outcome,
    Action,
    Target,
    Context,
}

fn role(path: &str) -> Role {
    let tokens = path_tokens(path);
    if tokens
        .iter()
        .any(|token| OUTCOME_TOKENS.contains(&token.as_str()))
    {
        Role::Outcome
    } else if tokens
        .iter()
        .any(|token| ACTION_TOKENS.contains(&token.as_str()))
    {
        Role::Action
    } else if tokens
        .iter()
        .any(|token| TARGET_TOKENS.contains(&token.as_str()))
    {
        Role::Target
    } else {
        Role::Context
    }
}

fn facet_text(leaves: &[Leaf], wanted: Role) -> String {
    let fragments = leaves
        .iter()
        .filter(|leaf| role(&leaf.path) == wanted)
        .map(|leaf| {
            format!(
                "{}={}",
                label(&leaf.path),
                semantic_value(&leaf.path, &leaf.value)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    bounded(&fragments, MAX_FACET_TEXT_CHARS)
}

fn compose_semantic_text(
    kind: DocumentKind,
    relation: &str,
    facets: &SemanticFacets,
    unavailable: bool,
) -> String {
    let mut parts = vec![
        format!("kind: {}", kind.semantic_label()),
        format!("relation: {}", bounded(relation, 128)),
    ];
    for (name, text, budget) in [
        ("outcome", &facets.outcome, 768),
        ("action", &facets.action, 640),
        ("target", &facets.target, 640),
        ("context", &facets.context, 768),
    ] {
        if !text.is_empty() {
            parts.push(format!("{name}: {}", bounded(text, budget)));
        }
    }
    if unavailable {
        parts.push("content: unavailable typed event".to_owned());
    }
    let text = parts.join(" | ");
    debug_assert!(text.chars().count() <= MAX_SEMANTIC_TEXT_CHARS);
    text
}

fn semantic_value(path: &str, value: &Value) -> String {
    if is_secret(path) {
        return "<redacted:secret>".to_owned();
    }
    if is_identifier(path) {
        return format!("<redacted:{}>", leaf_name(path).replace('_', "-"));
    }
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => numeric_semantic_value(path, value),
        Value::String(value) => sanitize_free_text(value),
        _ => value.to_string(),
    }
}

fn numeric_semantic_value(path: &str, number: &serde_json::Number) -> String {
    let leaf_name = leaf_name(path);
    let leaf = semantic_leaf_alias(&leaf_name);
    let value = number.as_f64().unwrap_or(0.0);
    let tokens = path_tokens(path);
    let source_port = leaf == "src_port"
        || (leaf == "port" && (tokens.contains("src") || tokens.contains("source")));
    if source_port {
        if value.fract() != 0.0 || !(0.0..=65535.0).contains(&value) {
            return "<port:invalid>".to_owned();
        }
        let port = value as u64;
        return if port < 1024 {
            format!("<port:privileged:{port}>")
        } else if port < 49152 {
            "<port:registered>".to_owned()
        } else {
            "<port:dynamic>".to_owned()
        };
    }
    if SEMANTIC_UID_KEYS.contains(&leaf) || leaf == "dst_port" || leaf == "port" {
        return number.to_string();
    }
    if !is_quantity(path) {
        return number.to_string();
    }
    if value == 0.0 {
        return "<quantity:zero>".to_owned();
    }
    let sign = if value < 0.0 { "negative-" } else { "" };
    format!("<quantity:{sign}1e{}>", value.abs().log10().floor() as i32)
}

fn exact_attribute_subset(root: &Value) -> (Vec<ExactAttribute>, ExactAttributeProjection) {
    let mut attributes = Vec::new();
    let mut omissions = BTreeMap::<String, usize>::new();
    let mut scanned = 0_usize;
    let mut truncated = false;
    let mut omitted_subtrees = 0_usize;
    exact_visit(
        root,
        "",
        "",
        &mut attributes,
        &mut omissions,
        &mut scanned,
        &mut truncated,
        &mut omitted_subtrees,
    );
    attributes.sort_by(|a, b| a.path.cmp(&b.path));
    let omission_counts = omissions
        .into_iter()
        .map(|(reason, count)| OmissionCount { reason, count })
        .collect::<Vec<_>>();
    let known_omitted_scalar_count = omission_counts.iter().map(|row| row.count).sum();
    (
        attributes.clone(),
        ExactAttributeProjection {
            selected_count: attributes.len(),
            scalars_scanned: scanned,
            known_omitted_scalar_count,
            omitted_subtree_count: omitted_subtrees,
            omission_counts,
            scan_truncated: truncated,
            source_hydration_required: known_omitted_scalar_count > 0 || truncated,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn exact_visit(
    value: &Value,
    pointer: &str,
    semantic_path: &str,
    attributes: &mut Vec<ExactAttribute>,
    omissions: &mut BTreeMap<String, usize>,
    scanned: &mut usize,
    truncated: &mut bool,
    omitted_subtrees: &mut usize,
) {
    if *scanned >= MAX_EXACT_SCALARS_SCANNED {
        *truncated = true;
        *omitted_subtrees += 1;
        return;
    }
    match value {
        Value::Object(map) => {
            let mut rows: Vec<_> = map.iter().collect();
            rows.sort_by_key(|(key, _)| *key);
            for (key, child) in rows {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                let child_pointer = format!("{pointer}/{escaped}");
                let child_semantic = if semantic_path.is_empty() {
                    key.clone()
                } else {
                    format!("{semantic_path}.{key}")
                };
                exact_visit(
                    child,
                    &child_pointer,
                    &child_semantic,
                    attributes,
                    omissions,
                    scanned,
                    truncated,
                    omitted_subtrees,
                );
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().take(MAX_EXACT_LIST_ITEMS).enumerate() {
                exact_visit(
                    child,
                    &format!("{pointer}/{index}"),
                    &format!("{semantic_path}[{index}]"),
                    attributes,
                    omissions,
                    scanned,
                    truncated,
                    omitted_subtrees,
                );
            }
            if items.len() > MAX_EXACT_LIST_ITEMS {
                *truncated = true;
                *omitted_subtrees += items.len() - MAX_EXACT_LIST_ITEMS;
            }
        }
        scalar => {
            *scanned += 1;
            let omit = |reason: &str, omissions: &mut BTreeMap<String, usize>| {
                *omissions.entry(reason.to_owned()).or_default() += 1
            };
            if scalar.is_null() {
                omit("null_value", omissions);
                return;
            }
            if pointer.chars().count() > MAX_EXACT_PATH_CHARS {
                omit("oversize_path", omissions);
                return;
            }
            if is_secret(semantic_path) {
                omit("secret_field", omissions);
                return;
            }
            if is_free_text(semantic_path) {
                omit("free_text_field", omissions);
                return;
            }
            if let Value::String(text) = scalar {
                if text.len() > MAX_EXACT_STRING_UTF8_BYTES {
                    omit("oversize_string", omissions);
                    return;
                }
                if has_unsafe_credential_text(text) {
                    omit("unsafe_credential_value", omissions);
                    return;
                }
            }
            if let Value::Number(number) = scalar
                && let Some(integer) = number.as_i64()
                && !(-MAX_JCS_SAFE_INTEGER..=MAX_JCS_SAFE_INTEGER).contains(&integer)
            {
                omit("non_jcs_safe_integer", omissions);
                return;
            }
            if attributes.len() >= MAX_EXACT_ATTRIBUTES {
                omit("attribute_limit", omissions);
                return;
            }
            attributes.push(ExactAttribute {
                namespace: "ocsf".to_owned(),
                path: pointer.to_owned(),
                value: scalar.clone(),
            });
        }
    }
}

fn event_time(leaves: &[Leaf]) -> (Option<String>, EventTimeAvailability) {
    for name in [
        "time",
        "event_time",
        "timestamp",
        "observed_time",
        "start_time",
    ] {
        if let Some(leaf) = leaves
            .iter()
            .filter(|leaf| leaf_name(&leaf.path) == name)
            .min_by_key(|leaf| (leaf.path.matches('.').count(), leaf.path.clone()))
        {
            return normalize_time(&leaf.value);
        }
    }
    (None, EventTimeAvailability::Missing)
}

fn normalize_time(value: &Value) -> (Option<String>, EventTimeAvailability) {
    let numeric = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    if let Some(mut seconds) = numeric {
        let magnitude = seconds.abs();
        if magnitude >= 1e17 {
            seconds /= 1e9;
        } else if magnitude >= 1e14 {
            seconds /= 1e6;
        } else if magnitude >= 1e11 {
            seconds /= 1e3;
        }
        let whole = seconds.trunc() as i64;
        let nanos = ((seconds.fract().abs()) * 1e9).round() as u32;
        if let Some(stamp) = Utc.timestamp_opt(whole, nanos).single() {
            return (
                Some(stamp.to_rfc3339_opts(SecondsFormat::Millis, true)),
                EventTimeAvailability::Available,
            );
        }
    }
    if let Value::String(text) = value {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(text.trim()) {
            let utc = parsed.with_timezone(&Utc);
            let rendered = if utc.timestamp_subsec_nanos() == 0 {
                utc.to_rfc3339_opts(SecondsFormat::Secs, true)
            } else {
                utc.to_rfc3339()
            };
            return (Some(rendered), EventTimeAvailability::Available);
        }
        return (
            Some(bounded(text.trim(), MAX_VALUE_CHARS)),
            EventTimeAvailability::PresentUnparsed,
        );
    }
    (
        Some(value.to_string()),
        EventTimeAvailability::PresentUnparsed,
    )
}

fn digest_value(value: &Value) -> String {
    let canonical = serde_json_canonicalizer::to_vec(value).expect("projection values serialize");
    format!("{:x}", Sha256::digest(canonical))
}

fn replace_with_boundaries(
    text: &str,
    pattern: &Regex,
    replacement: &str,
    left_forbidden: fn(char) -> bool,
    right_forbidden: fn(char) -> bool,
) -> String {
    pattern
        .replace_all(text, |captures: &Captures<'_>| {
            let matched = captures.get(0).expect("whole regex match");
            let invalid_left = text[..matched.start()]
                .chars()
                .next_back()
                .is_some_and(left_forbidden);
            let invalid_right = text[matched.end()..]
                .chars()
                .next()
                .is_some_and(right_forbidden);
            if invalid_left || invalid_right {
                matched.as_str().to_owned()
            } else {
                replacement.to_owned()
            }
        })
        .into_owned()
}

fn is_regex_word(value: char) -> bool {
    let mut encoded = [0_u8; 4];
    WORD_CHAR_RE.is_match(value.encode_utf8(&mut encoded))
}

fn email_left_forbidden(value: char) -> bool {
    is_regex_word(value) || matches!(value, '.' | '+' | '-')
}

fn email_right_forbidden(value: char) -> bool {
    is_regex_word(value) || matches!(value, '.' | '-')
}

fn word_or_dot(value: char) -> bool {
    is_regex_word(value) || value == '.'
}

fn ascii_hex(value: char) -> bool {
    value.is_ascii_hexdigit()
}

fn jwt_character(value: char) -> bool {
    value.is_ascii_alphanumeric() || matches!(value, '_' | '-')
}

fn replace_in_band_identifiers(mut text: String, include_email_and_ip: bool) -> String {
    text = replace_with_boundaries(
        &text,
        &JWT_RE,
        "<redacted:jwt>",
        jwt_character,
        jwt_character,
    );
    text = replace_with_boundaries(
        &text,
        &MAC_RE,
        "<redacted:mac-address>",
        ascii_hex,
        ascii_hex,
    );
    if include_email_and_ip {
        text = replace_with_boundaries(
            &text,
            &EMAIL_RE,
            "<redacted:email-address>",
            email_left_forbidden,
            email_right_forbidden,
        );
        text = replace_with_boundaries(
            &text,
            &IPV4_RE,
            "<redacted:ip-address>",
            word_or_dot,
            word_or_dot,
        );
        text = replace_with_boundaries(&text, &UUID_RE, "<redacted:uuid>", ascii_hex, ascii_hex);
        text = replace_with_boundaries(
            &text,
            &LONG_HEX_RE,
            "<redacted:long-identifier>",
            ascii_hex,
            ascii_hex,
        );
    }
    text
}

fn sanitize_free_text(value: &str) -> String {
    let mut text = value.replace(['\0', '\r', '\n'], " ");
    text = SECRET_ASSIGNMENT_RE
        .replace_all(&text, "$1$2<redacted:secret>")
        .into_owned();
    text = BEARER_RE
        .replace_all(&text, "Bearer <redacted:secret>")
        .into_owned();
    text = ACCESS_KEY_RE
        .replace_all(&text, "<redacted:cloud-credential>")
        .into_owned();
    text = CLOUD_IDENTIFIER_RE
        .replace_all(&text, "<redacted:cloud-identifier>")
        .into_owned();
    text = replace_in_band_identifiers(text, true);
    bounded(
        &text.split_whitespace().collect::<Vec<_>>().join(" "),
        MAX_VALUE_CHARS,
    )
}

fn has_unsafe_credential_text(value: &str) -> bool {
    SECRET_ASSIGNMENT_RE.is_match(value)
        || BEARER_RE.is_match(value)
        || ACCESS_KEY_RE.is_match(value)
        || replace_with_boundaries(
            value,
            &JWT_RE,
            "<redacted:jwt>",
            jwt_character,
            jwt_character,
        ) != value
}

fn is_secret(path: &str) -> bool {
    let leaf = leaf_name(path);
    let tokens = path_tokens(path);
    SECRET_KEYS.contains(&leaf.as_str())
        || leaf.ends_with("_secret")
        || leaf.ends_with("_password")
        || leaf.ends_with("_token")
        || tokens
            .iter()
            .any(|token| SECRET_TOKENS.contains(&token.as_str()))
        || (["api", "key"].iter().all(|token| tokens.contains(*token)))
        || (["access", "key"]
            .iter()
            .all(|token| tokens.contains(*token)))
        || (["private", "key"]
            .iter()
            .all(|token| tokens.contains(*token)))
}

fn is_identifier(path: &str) -> bool {
    let leaf = leaf_name(path);
    if SEMANTIC_UID_KEYS.contains(&leaf.as_str()) {
        return false;
    }
    let tokens = path_tokens(path);
    IDENTIFIER_KEYS.contains(&leaf.as_str())
        || IDENTIFIER_ALIASES.contains(&leaf.as_str())
        || IDENTIFIER_SUFFIXES
            .iter()
            .any(|suffix| leaf.ends_with(suffix))
        || (leaf.ends_with("_name")
            && leaf
                .trim_end_matches("_name")
                .split('_')
                .any(|token| IDENTIFIER_PREFIXES.contains(&token)))
        || ((leaf == "name" || leaf == "value")
            && tokens
                .iter()
                .any(|token| IDENTIFIER_CONTAINERS.contains(&token.as_str())))
}

fn is_time(path: &str) -> bool {
    let leaf = leaf_name(path);
    TIME_KEYS.contains(&leaf.as_str())
        || leaf.ends_with("_time")
        || leaf.ends_with("_timestamp")
        || leaf.ends_with("_date")
}
fn is_volatile(path: &str) -> bool {
    VOLATILE_KEYS.contains(&leaf_name(path).as_str())
}
fn is_positional_raw_token(path: &str) -> bool {
    POSITIONAL_TOKEN_RE.is_match(path)
}
fn is_free_text(path: &str) -> bool {
    FREE_TEXT_KEYS.contains(&leaf_name(path).as_str()) || is_positional_raw_token(path)
}
fn is_quantity(path: &str) -> bool {
    let leaf = leaf_name(path);
    let tokens = path_tokens(path);
    QUANTITY_TOKENS.contains(&leaf.as_str())
        || tokens
            .iter()
            .any(|token| QUANTITY_TOKENS.contains(&token.as_str()))
}
fn is_semantic_noise(path: &str) -> bool {
    let normalized = LIST_INDEX_RE
        .replace_all(path, "")
        .split('.')
        .map(normalize_key)
        .collect::<Vec<_>>()
        .join(".");
    let leaf = leaf_name(path);
    format!(".{normalized}.").contains(".metadata.product.")
        || normalized.ends_with("metadata.version")
        || leaf == "support_ref"
        || leaf == "dest_content"
        || leaf == "src_content"
}
fn semantic_leaf_alias(leaf: &str) -> &str {
    match leaf {
        "dest_ip" => "dst_ip",
        "dest_mac" => "dst_mac",
        "dest_port" => "dst_port",
        "source_ip" => "src_ip",
        "source_mac" => "src_mac",
        "source_port" => "src_port",
        _ => leaf,
    }
}
fn leaf_name(path: &str) -> String {
    normalize_key(
        LIST_INDEX_RE
            .replace_all(path.rsplit('.').next().unwrap_or(path), "")
            .as_ref(),
    )
}
fn normalize_key(value: &str) -> String {
    // Braced capture references are required here: in Rust regex replacement
    // syntax, `$1_` is parsed as one capture name rather than capture 1 plus
    // an underscore. Without the braces, the character before each uppercase
    // boundary was dropped, so time and identifier suffixes went unrecognized.
    let camel = CAMEL_BOUNDARY_RE
        .replace_all(value, "${1}_${2}")
        .to_lowercase();
    NON_KEY_RE
        .replace_all(&camel, "_")
        .trim_matches('_')
        .to_owned()
}
fn path_tokens(path: &str) -> BTreeSet<String> {
    LIST_INDEX_RE
        .replace_all(path, "")
        .split('.')
        .flat_map(|part| {
            let normalized = normalize_key(part);
            let mut tokens = vec![normalized.clone()];
            tokens.extend(normalized.split('_').map(str::to_owned));
            tokens
        })
        .filter(|token| !token.is_empty())
        .collect()
}
fn label(path: &str) -> String {
    let parts = path
        .split('.')
        .filter(|part| *part != "ocsf")
        .collect::<Vec<_>>();
    parts[parts.len().saturating_sub(3)..].join(".")
}
fn bounded(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_owned()
    } else {
        let marker = "…[truncated]";
        value
            .chars()
            .take(limit - marker.chars().count())
            .collect::<String>()
            + marker
    }
}

const SECRET_KEYS: &[&str] = &[
    "api_key",
    "authorization",
    "access_token",
    "auth_token",
    "client_secret",
    "cookie",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
    "session_token",
    "refresh_token",
    "token",
];
const SECRET_TOKENS: &[&str] = &[
    "apikey",
    "passphrase",
    "passwd",
    "password",
    "secret",
    "token",
];
const IDENTIFIER_KEYS: &[&str] = &[
    "account",
    "actor",
    "actor_aliases",
    "address",
    "addresses",
    "credential_id",
    "device",
    "domain",
    "dst_ip",
    "email",
    "event_id",
    "host",
    "hostname",
    "identity",
    "identities",
    "interface",
    "message_trace_uid",
    "message_uid",
    "mac",
    "mac_address",
    "native_event_uid",
    "parent_process_uid",
    "principal",
    "record_uid",
    "recipients",
    "request_id",
    "support_ref",
    "resource",
    "resources",
    "sender",
    "session_id",
    "source_address",
    "src_ip",
    "src_mac",
    "subject",
    "target",
    "user",
    "pid",
    "ppid",
    "uid",
    "gid",
    "sid",
    "host_identifier",
    "auid",
    "euid",
    "egid",
    "ruid",
    "rgid",
    "uuid",
    "host_uuid",
    "dst_mac",
];
const IDENTIFIER_ALIASES: &[&str] = &[
    "account_name",
    "client_hostname",
    "computer_name",
    "destination_hostname",
    "source_hostname",
    "tenant_name",
    "user_name",
    "username",
    "workstation_name",
];
const IDENTIFIER_SUFFIXES: &[&str] = &[
    "_id",
    "_uid",
    "_pid",
    "_gid",
    "_sid",
    "_ref",
    "_sha256",
    "_hash",
    "_identifier",
    "_address",
    "_ip",
    "_mac",
    "_mac_address",
    "_arn",
    "_uuid",
];
const IDENTIFIER_PREFIXES: &[&str] = &[
    "account",
    "actor",
    "client",
    "computer",
    "destination",
    "device",
    "host",
    "identity",
    "principal",
    "recipient",
    "sender",
    "source",
    "tenant",
    "user",
    "username",
    "workstation",
];
const IDENTIFIER_CONTAINERS: &[&str] = &[
    "account",
    "actor",
    "actor_aliases",
    "address",
    "addresses",
    "bucket",
    "credential",
    "databucket",
    "device",
    "domain",
    "endpoint",
    "event",
    "host",
    "hostname",
    "identities",
    "identity",
    "interface",
    "ip",
    "principal",
    "recipient",
    "recipients",
    "record",
    "resource",
    "resources",
    "sender",
    "session",
    "user",
];
const IDENTIFIER_PLACEHOLDERS: &[&str] = &[
    "account",
    "actor",
    "address",
    "device",
    "domain",
    "dst_ip",
    "email",
    "host",
    "hostname",
    "principal",
    "recipient",
    "recipients",
    "resource",
    "resources",
    "sender",
    "source_address",
    "src_ip",
    "subject",
    "target",
    "user",
];
const SEMANTIC_UID_KEYS: &[&str] = &[
    "activity_id",
    "activity_name",
    "category_name",
    "category_uid",
    "class_name",
    "class_uid",
    "severity_id",
    "severity_name",
    "status_id",
    "status_code",
    "type_name",
    "type_uid",
];
const TIME_KEYS: &[&str] = &[
    "time",
    "event_time",
    "timestamp",
    "observed_time",
    "start_time",
    "end_time",
    "calendar_time",
    "atime",
    "ctime",
    "mtime",
    "uptime",
    "btime",
    "unix_time",
    "endtime",
    "starttime",
];
const VOLATILE_KEYS: &[&str] = &[
    "counter",
    "epoch",
    "line_number",
    "ordinal",
    "record_number",
    "sequence",
    "sequence_number",
];
const FREE_TEXT_KEYS: &[&str] = &[
    "body",
    "cmd_line",
    "command",
    "command_line",
    "content",
    "description",
    "details",
    "headers",
    "message",
    "payload",
    "query",
    "raw",
    "request",
    "script",
    "script_block",
    "stack_trace",
    "user_agent",
];
const ACTION_TOKENS: &[&str] = &[
    "action",
    "activity",
    "command",
    "command_line",
    "event_name",
    "method",
    "operation",
    "query_type",
    "request",
    "verb",
];
const TARGET_TOKENS: &[&str] = &[
    "application",
    "bucket",
    "databucket",
    "destination",
    "device",
    "domain",
    "dst_endpoint",
    "dst_ip",
    "file",
    "hostname",
    "object",
    "path",
    "process",
    "recipients",
    "resource",
    "service",
    "subject",
    "target",
    "user",
];
const OUTCOME_TOKENS: &[&str] = &[
    "blocked",
    "compliance",
    "disposition",
    "error",
    "finding",
    "from",
    "log_status",
    "outcome",
    "result",
    "severity",
    "state",
    "status",
    "status_code",
    "target_status_code",
    "to",
    "transition",
];
const QUANTITY_TOKENS: &[&str] = &[
    "bytes",
    "count",
    "duration",
    "length",
    "millis",
    "milliseconds",
    "packets",
    "rtt",
    "size",
    "time_taken",
    "total",
    "value",
];
const PRIORITY_TOKENS: &[&str] = &[
    "blocked",
    "compliance",
    "disposition",
    "error",
    "finding",
    "from",
    "log_status",
    "outcome",
    "result",
    "severity",
    "state",
    "status",
    "status_code",
    "target_status_code",
    "to",
    "transition",
    "action",
    "activity",
    "command",
    "command_line",
    "event_name",
    "method",
    "operation",
    "query_type",
    "request",
    "verb",
    "application",
    "bucket",
    "databucket",
    "destination",
    "device",
    "domain",
    "dst_endpoint",
    "dst_ip",
    "file",
    "hostname",
    "object",
    "path",
    "process",
    "recipients",
    "resource",
    "service",
    "subject",
    "target",
    "user",
    "body",
    "cmd_line",
    "content",
    "description",
    "details",
    "headers",
    "message",
    "payload",
    "raw",
    "script",
    "script_block",
    "stack_trace",
    "user_agent",
    "response",
];
