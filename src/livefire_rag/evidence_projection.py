"""Scenario-blind projection of typed OCSF rows into bounded semantic documents.

This module is deliberately data-policy only.  It does not know about hunts,
expected answers, indicators, or source products.  The projector is a total
function over snapshot rows: known typed relations produce a semantic document
and every other input receives an explicit ``structured_only`` disposition.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .canonical import canonical_json_bytes, component_ref, sha256_bytes


PROJECTION_SCHEMA_VERSION = "livefire.rag.evidence-projection/1"

# This is the closed typed relation set from the normalized snapshot contract.
# Relation names select only a generic document kind; field extraction is shared.
RELATION_DOCUMENT_KINDS: dict[str, str] = {
    "ocsf_api_activity": "activity",
    "ocsf_application_lifecycle": "activity",
    "ocsf_authentication": "activity",
    "ocsf_cloud_resources_inventory_info": "state",
    "ocsf_datastore_activity": "activity",
    "ocsf_detection_finding": "detection",
    "ocsf_dns_activity": "activity",
    "ocsf_email_activity": "activity",
    "ocsf_entity_management": "activity",
    "ocsf_event_log_activity": "activity",
    "ocsf_ext_livefire_configuration_snapshot": "state",
    "ocsf_ext_livefire_system_metric": "structured_only",
    "ocsf_file_activity": "activity",
    "ocsf_http_activity": "activity",
    "ocsf_inventory_info": "state",
    "ocsf_network_activity": "activity",
    "ocsf_process_activity": "activity",
    "ocsf_user_inventory": "state",
}
_DERIVATION_ONLY_RELATIONS = {"ocsf_ext_livefire_system_metric"}

MAX_LEAVES = 160
MAX_LIST_ITEMS = 24
MAX_VALUE_CHARS = 240
MAX_FACET_TEXT_CHARS = 1_024
MAX_SEMANTIC_TEXT_CHARS = 3_072
MAX_PROJECTION_SCALARS_SCANNED = 1_024
MAX_JCS_SAFE_INTEGER = (1 << 53) - 1
MAX_EXACT_ATTRIBUTES = 256
MAX_EXACT_SCALARS_SCANNED = 512
MAX_EXACT_LIST_ITEMS = 64
MAX_EXACT_STRING_UTF8_BYTES = 1_024
MAX_EXACT_PATH_CHARS = 1_024

_SECRET_KEYS = {
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
}
_IDENTIFIER_KEYS = {
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
}
_IDENTIFIER_CONTAINER_TOKENS = {
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
}
_SEMANTIC_IDENTIFIER_PLACEHOLDERS = {
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
}
_SEMANTIC_UID_KEYS = {
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
}
_TIME_KEYS = {
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
}
_VOLATILE_KEYS = {
    "counter",
    "epoch",
    "line_number",
    "ordinal",
    "record_number",
    "sequence",
    "sequence_number",
}
_FREE_TEXT_KEYS = {
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
}
_IDENTIFIER_SUFFIXES = (
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
)
_IDENTIFIER_NAME_PREFIX_TOKENS = {
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
}
_IDENTIFIER_ALIASES = {
    "account_name",
    "client_hostname",
    "computer_name",
    "destination_hostname",
    "source_hostname",
    "tenant_name",
    "user_name",
    "username",
    "workstation_name",
}
_SECRET_TOKENS = {
    "apikey",
    "passphrase",
    "passwd",
    "password",
    "secret",
    "token",
}

_ACTION_TOKENS = {
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
}
_TARGET_TOKENS = {
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
}
_OUTCOME_TOKENS = {
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
}

_ROLE_PRECEDENCE = ("outcome", "action", "target", "context")
_SEMANTIC_ROLE_BUDGETS = {
    "action": 640,
    "target": 640,
    "context": 768,
    "outcome": 768,
}
_PRIORITY_SUBTREE_TOKENS = (
    _OUTCOME_TOKENS
    | _ACTION_TOKENS
    | _TARGET_TOKENS
    | _FREE_TEXT_KEYS
    | {"state", "process", "request", "response", "finding", "file"}
)

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<name>password|passwd|pwd|secret|token|api[-_]?key|access[-_]?key|"
    r"authorization|cookie|private[-_]?key)(?P<sep>\s*(?:=|:)\s*|\s+)(?P<value>"
    r"\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_IPV4_RE = re.compile(
    r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
)
_UUID_RE = re.compile(
    r"(?i)(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])"
)
_LONG_HEX_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{32,}(?![0-9a-f])")
_CLOUD_IDENTIFIER_RE = re.compile(r"(?i)\barn:[a-z0-9-]+:[^\s,;]+")
_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_JWT_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")
_MAC_RE = re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])")
_LIST_INDEX_RE = re.compile(r"\[\d+\]")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_KEY_RE = re.compile(r"[^a-z0-9]+")
_POSITIONAL_TOKEN_RE = re.compile(r"(?:^|\.)unmapped\.\$token/\d+(?:\.|$)", re.IGNORECASE)

_QUANTITY_TOKENS = {
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
}
_SEMANTIC_LEAF_ALIASES = {
    "dest_ip": "dst_ip",
    "dest_mac": "dst_mac",
    "dest_port": "dst_port",
    "source_ip": "src_ip",
    "source_mac": "src_mac",
    "source_port": "src_port",
}


def projection_policy_material() -> dict[str, Any]:
    """Return the complete immutable policy material bound by projection packs."""

    return {
        "schema_version": "livefire.rag.generic-evidence-projection-policy/2",
        "relation_document_kinds": dict(sorted(RELATION_DOCUMENT_KINDS.items())),
        "derivation_only_relations": sorted(_DERIVATION_ONLY_RELATIONS),
        "bounds": {
            "max_leaves": MAX_LEAVES,
            "max_list_items": MAX_LIST_ITEMS,
            "max_value_chars": MAX_VALUE_CHARS,
            "max_facet_text_chars": MAX_FACET_TEXT_CHARS,
            "max_semantic_text_chars": MAX_SEMANTIC_TEXT_CHARS,
            "max_projection_scalars_scanned": MAX_PROJECTION_SCALARS_SCANNED,
            "max_exact_attributes": MAX_EXACT_ATTRIBUTES,
            "max_exact_scalars_scanned": MAX_EXACT_SCALARS_SCANNED,
            "max_exact_list_items": MAX_EXACT_LIST_ITEMS,
            "max_exact_string_utf8_bytes": MAX_EXACT_STRING_UTF8_BYTES,
            "max_exact_path_chars": MAX_EXACT_PATH_CHARS,
        },
        "identity_policy": {
            "semantic_group_excludes": [
                "event_id",
                "event_time",
                "support_ref",
                "typed_identifiers",
                "secret_values",
            ],
            "exact_attribute_contract": (
                "bounded_value_exact_typed_json_scalar_subset_hydrate_source_for_omissions"
            ),
            "exact_attribute_path": "rfc6901_json_pointer",
            "exact_attribute_inclusions": [
                "booleans",
                "jcs_safe_integers",
                "finite_numbers",
                "safe_bounded_strings",
            ],
            "exact_attribute_omissions": [
                "null_values",
                "secret_fields",
                "unsafe_credential_values",
                "free_text_fields",
                "oversize_strings",
                "oversize_paths",
                "non_finite_numbers",
                "non_jcs_safe_integers",
                "attribute_or_scan_bounds",
            ],
            "secret_values": "excluded_from_exact_attributes_and_sha256_digest_only_in_semantic_receipt",
            "field_name_normalization": "camel_case_and_punctuation_to_lower_snake_case",
            "identifier_suffixes": list(_IDENTIFIER_SUFFIXES),
            "identifier_leaf_names": sorted(_IDENTIFIER_KEYS),
            "identifier_aliases": sorted(_IDENTIFIER_ALIASES),
            "identifier_name_prefix_tokens": sorted(_IDENTIFIER_NAME_PREFIX_TOKENS),
            "identifier_container_tokens": sorted(_IDENTIFIER_CONTAINER_TOKENS),
            "semantic_uid_keys": sorted(_SEMANTIC_UID_KEYS),
            "key_and_path_normalization_patterns": {
                "list_index": _LIST_INDEX_RE.pattern,
                "camel_boundary": _CAMEL_BOUNDARY_RE.pattern,
                "non_key": _NON_KEY_RE.pattern,
                "positional_token": _POSITIONAL_TOKEN_RE.pattern,
            },
            "time_leaf_names": sorted(_TIME_KEYS),
            "volatile_leaf_names": sorted(_VOLATILE_KEYS),
            "free_text_leaf_names": sorted(_FREE_TEXT_KEYS),
            "excluded_path_families": ["unmapped.$token/<ordinal>"],
            "semantic_noise_paths": [
                "metadata.product.*",
                "metadata.version",
                "support_ref",
                "unmapped.dest_content",
                "unmapped.src_content",
            ],
            "quantity_normalization": "sign_and_base10_magnitude_bucket_except_semantic_ids_status_and_ports",
            "quantity_leaf_tokens": sorted(_QUANTITY_TOKENS),
            "semantic_null_values": "omitted",
            "semantic_identifier_placeholders": sorted(_SEMANTIC_IDENTIFIER_PLACEHOLDERS),
            "unmapped_duplicate_policy": "prefer_non_unmapped_typed_leaf",
            "semantic_leaf_aliases": dict(sorted(_SEMANTIC_LEAF_ALIASES.items())),
            "role_token_sets": {
                "action": sorted(_ACTION_TOKENS),
                "target": sorted(_TARGET_TOKENS),
                "outcome": sorted(_OUTCOME_TOKENS),
            },
            "role_precedence": list(_ROLE_PRECEDENCE),
            "semantic_role_budgets": dict(sorted(_SEMANTIC_ROLE_BUDGETS.items())),
            "projection_traversal": {
                "selection": "priority_typed_semantic_leaves_then_typed_context_then_unmapped",
                "subtree_priority_tokens": sorted(_PRIORITY_SUBTREE_TOKENS),
                "tie_break": "normalized_path_ascending",
            },
            "source_port_normalization": "privileged_registered_dynamic_bucket",
            "secret_leaf_names": sorted(_SECRET_KEYS),
            "secret_suffixes": ["_secret", "_password", "_token"],
            "secret_tokens": sorted(_SECRET_TOKENS),
            "secret_token_combinations": ["api+key", "access+key", "private+key"],
            "in_band_redaction_patterns": {
                "secret_assignment": _SECRET_ASSIGNMENT_RE.pattern,
                "bearer": _BEARER_RE.pattern,
                "email": _EMAIL_RE.pattern,
                "ipv4": _IPV4_RE.pattern,
                "uuid": _UUID_RE.pattern,
                "long_hex": _LONG_HEX_RE.pattern,
                "cloud_identifier": _CLOUD_IDENTIFIER_RE.pattern,
                "access_key": _ACCESS_KEY_RE.pattern,
                "jwt": _JWT_RE.pattern,
                "mac": _MAC_RE.pattern,
            },
        },
        "selection_policy": "closed_typed_relation_set_without_event-value_predicates",
        "unknown_or_unparsed_policy": "structured_only_occurrence",
    }


def projection_policy_ref() -> dict[str, str]:
    """Return the SDK component identity for the generic projection policy."""

    return component_ref(
        "livefire.rag.generic-evidence-projection-policy",
        "2",
        projection_policy_material(),
    )


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _tokens(path: str) -> set[str]:
    tokens: set[str] = set()
    for part in _LIST_INDEX_RE.sub("", path).split("."):
        normalized = _normalise_key(part)
        if normalized:
            tokens.add(normalized)
            tokens.update(token for token in normalized.split("_") if token)
    return tokens


def _normalise_key(value: str) -> str:
    camel_split = _CAMEL_BOUNDARY_RE.sub("_", value)
    return _NON_KEY_RE.sub("_", camel_split.lower()).strip("_")


def _leaf_name(path: str) -> str:
    return _normalise_key(_LIST_INDEX_RE.sub("", path.rsplit(".", 1)[-1]))


def _is_secret(path: str) -> bool:
    leaf = _leaf_name(path)
    tokens = _tokens(path)
    return (
        leaf in _SECRET_KEYS
        or leaf.endswith("_secret")
        or leaf.endswith("_password")
        or leaf.endswith("_token")
        or bool(tokens & _SECRET_TOKENS)
        or {"api", "key"} <= tokens
        or {"access", "key"} <= tokens
        or {"private", "key"} <= tokens
    )


def _is_identifier(path: str) -> bool:
    leaf = _leaf_name(path)
    if leaf in _SEMANTIC_UID_KEYS:
        return False
    path_tokens = _tokens(path)
    return (
        leaf in _IDENTIFIER_KEYS
        or leaf in _IDENTIFIER_ALIASES
        or leaf.endswith(_IDENTIFIER_SUFFIXES)
        or (
            leaf.endswith("_name")
            and bool(set(leaf.removesuffix("_name").split("_")) & _IDENTIFIER_NAME_PREFIX_TOKENS)
        )
        or (
            leaf in {"name", "value"}
            and bool(path_tokens & _IDENTIFIER_CONTAINER_TOKENS)
        )
    )


def _is_time(path: str) -> bool:
    leaf = _leaf_name(path)
    return (
        leaf in _TIME_KEYS
        or leaf.endswith("_time")
        or leaf.endswith("_timestamp")
        or leaf.endswith("_date")
    )


def _is_volatile(path: str) -> bool:
    return _leaf_name(path) in _VOLATILE_KEYS


def _is_positional_raw_token(path: str) -> bool:
    return _POSITIONAL_TOKEN_RE.search(path) is not None


def _is_free_text(path: str) -> bool:
    return _leaf_name(path) in _FREE_TEXT_KEYS or _is_positional_raw_token(path)


def _is_semantic_noise(path: str) -> bool:
    """Return true for producer/envelope or raw transport fields with no semantic role.

    This list is structural and source-value independent. It removes normalizer
    identity and opaque packet bytes while retaining typed and unmapped fields
    that can carry security behavior.
    """

    normalized = ".".join(_normalise_key(part) for part in _LIST_INDEX_RE.sub("", path).split("."))
    leaf = _leaf_name(path)
    return (
        ".metadata.product." in f".{normalized}."
        or normalized.endswith("metadata.version")
        or leaf == "support_ref"
        or leaf in {"dest_content", "src_content"}
    )


def _is_quantity(path: str) -> bool:
    leaf = _leaf_name(path)
    tokens = _tokens(path)
    return leaf in _QUANTITY_TOKENS or bool(tokens & _QUANTITY_TOKENS)


def _numeric_semantic_value(path: str, value: int | float) -> str:
    """Normalize quantities without erasing exact protocol/status semantics."""

    leaf = _SEMANTIC_LEAF_ALIASES.get(_leaf_name(path), _leaf_name(path))
    path_tokens = _tokens(path)
    source_port = leaf == "src_port" or (leaf == "port" and bool(path_tokens & {"src", "source"}))
    if source_port:
        numeric = float(value)
        if not numeric.is_integer() or numeric < 0 or numeric > 65535:
            return "<port:invalid>"
        port = int(numeric)
        if port < 1024:
            return f"<port:privileged:{port}>"
        if port < 49152:
            return "<port:registered>"
        return "<port:dynamic>"
    if leaf in _SEMANTIC_UID_KEYS or leaf == "dst_port" or leaf == "port":
        return str(value)
    if not _is_quantity(path):
        return str(value)
    numeric = float(value)
    if numeric == 0:
        return "<quantity:zero>"
    sign = "negative-" if numeric < 0 else ""
    magnitude = int(math.floor(math.log10(abs(numeric))))
    return f"<quantity:{sign}1e{magnitude}>"


def _semantic_entries(
    entries: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """Select behavior-bearing leaves with typed fields preferred to closure bags."""

    typed_leaf_names = {
        _SEMANTIC_LEAF_ALIASES.get(_leaf_name(path), _leaf_name(path))
        for path, value in entries
        if ".unmapped." not in f".{_LIST_INDEX_RE.sub('', path).lower()}."
        and value is not None
        and not _is_time(path)
        and not _is_volatile(path)
        and not _is_positional_raw_token(path)
        and not _is_semantic_noise(path)
    }
    selected: list[tuple[str, Any]] = []
    for path, value in entries:
        normalized = f".{_LIST_INDEX_RE.sub('', path).lower()}."
        if (
            value is None
            or _is_time(path)
            or _is_volatile(path)
            or _is_positional_raw_token(path)
            or _is_semantic_noise(path)
            or (
                _is_identifier(path)
                and _leaf_name(path) not in _SEMANTIC_IDENTIFIER_PLACEHOLDERS
            )
            or (
                ".unmapped." in normalized
                and _SEMANTIC_LEAF_ALIASES.get(_leaf_name(path), _leaf_name(path))
                in typed_leaf_names
            )
        ):
            continue
        selected.append((path, value))
    return selected


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "…[truncated]"
    return text[: limit - len(marker)] + marker


def _sanitize_free_text(value: str) -> str:
    """Redact common in-band secrets and identifiers without parsing a shell."""

    text = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('name')}{match.group('sep')}<redacted:secret>", text
    )
    text = _BEARER_RE.sub("Bearer <redacted:secret>", text)
    text = _ACCESS_KEY_RE.sub("<redacted:cloud-credential>", text)
    text = _CLOUD_IDENTIFIER_RE.sub("<redacted:cloud-identifier>", text)
    text = _JWT_RE.sub("<redacted:jwt>", text)
    text = _MAC_RE.sub("<redacted:mac-address>", text)
    text = _EMAIL_RE.sub("<redacted:email-address>", text)
    text = _IPV4_RE.sub("<redacted:ip-address>", text)
    text = _UUID_RE.sub("<redacted:uuid>", text)
    text = _LONG_HEX_RE.sub("<redacted:long-identifier>", text)
    return _bounded(" ".join(text.split()), MAX_VALUE_CHARS)


def _sanitize_structured_identifier(value: str) -> str:
    """Keep an exact identifier unless its value is itself credential material."""

    text = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('name')}{match.group('sep')}<redacted:secret>", text
    )
    text = _BEARER_RE.sub("Bearer <redacted:secret>", text)
    text = _ACCESS_KEY_RE.sub("<redacted:cloud-credential>", text)
    text = _JWT_RE.sub("<redacted:jwt>", text)
    text = _MAC_RE.sub("<redacted:mac-address>", text)
    return _bounded(" ".join(text.split()), MAX_VALUE_CHARS)


def _json_safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value if -MAX_JCS_SAFE_INTEGER <= value <= MAX_JCS_SAFE_INTEGER else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _flatten(
    value: Any,
    *,
    path: str = "",
    output: list[tuple[str, str | int | float | bool | None]] | None = None,
    state: dict[str, bool] | None = None,
) -> list[tuple[str, str | int | float | bool | None]]:
    """Collect bounded leaves, then reserve the output budget for typed semantics.

    Traversal visits behavior-bearing typed subtrees before generic context and
    visits ``unmapped`` closure bags last.  Selection then performs a second
    deterministic priority pass, so a large vendor bag cannot crowd a typed
    action, target, outcome, state, free-text command, or event time out of the
    semantic projection.
    """

    if output is not None or path:
        raise ValueError("_flatten is a root-only deterministic projection helper")
    state = state if state is not None else {"truncated": False}
    candidates: list[tuple[str, str | int | float | bool | None]] = []
    scanned = 0

    def subtree_order(item: tuple[Any, Any]) -> tuple[int, str]:
        key = str(item[0])
        normalized = _normalise_key(key)
        tokens = set(normalized.split("_")) | {normalized}
        if "unmapped" in tokens or "raw" in tokens:
            priority = 2
        elif tokens & _PRIORITY_SUBTREE_TOKENS:
            priority = 0
        else:
            priority = 1
        return priority, normalized

    def visit(node: Any, current_path: str) -> None:
        nonlocal scanned
        if scanned >= MAX_PROJECTION_SCALARS_SCANNED:
            state["truncated"] = True
            return
        if isinstance(node, Mapping):
            for original_key, child in sorted(node.items(), key=subtree_order):
                if scanned >= MAX_PROJECTION_SCALARS_SCANNED:
                    state["truncated"] = True
                    break
                key = str(original_key)
                child_path = f"{current_path}.{key}" if current_path else key
                visit(child, child_path)
            if not node and current_path:
                scanned += 1
                candidates.append((current_path, None))
            return
        if isinstance(node, (list, tuple)):
            if len(node) > MAX_LIST_ITEMS:
                state["truncated"] = True
            for index, item in enumerate(node[:MAX_LIST_ITEMS]):
                visit(item, f"{current_path}[{index}]")
            if not node and current_path:
                scanned += 1
                candidates.append((current_path, None))
            return
        scanned += 1
        candidates.append((current_path or "value", _json_safe_scalar(node)))

    def leaf_priority(item: tuple[str, Any]) -> tuple[int, str]:
        candidate_path, _ = item
        normalized = f".{_LIST_INDEX_RE.sub('', candidate_path).lower()}."
        if _is_time(candidate_path) or _role(candidate_path) != "context" or _is_free_text(candidate_path):
            priority = 0
        elif ".unmapped." not in normalized and not _is_positional_raw_token(candidate_path):
            priority = 1
        else:
            priority = 2
        return priority, candidate_path

    visit(value, "")
    candidates.sort(key=leaf_priority)
    if len(candidates) > MAX_LEAVES:
        state["truncated"] = True
    return candidates[:MAX_LEAVES]


def _redacted_structured_value(path: str, value: Any) -> Any:
    safe = _json_safe_scalar(value)
    if _is_secret(path):
        return {"classification": "secret", "sha256": _digest(safe)}
    if safe is None or isinstance(safe, (bool, int, float)):
        return safe
    # Identifiers remain exact, filterable typed metadata.  They are removed
    # from semantic text and semantic-group identity below, not destroyed.
    # Free-text sanitization still catches credentials embedded in the value.
    if _is_identifier(path):
        return _sanitize_structured_identifier(safe)
    return _sanitize_free_text(safe)


def _has_unsafe_credential_text(value: str) -> bool:
    return any(
        pattern.search(value) is not None
        for pattern in (_SECRET_ASSIGNMENT_RE, _BEARER_RE, _ACCESS_KEY_RE, _JWT_RE)
    )


def _exact_attribute_subset(value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select value-exact JSON scalars without reusing semantic normalization.

    Included values are copied directly from the parsed typed event.  Unsafe,
    non-portable, or bounded-out values are omitted rather than transformed;
    the returned metadata makes every such omission and scan bound explicit.
    """

    attributes: list[dict[str, Any]] = []
    omissions: Counter[str] = Counter()
    state = {"scalars_scanned": 0, "scan_truncated": False, "omitted_subtrees": 0}

    def omit(reason: str) -> None:
        omissions[reason] += 1

    def pointer_segment(value: Any) -> str:
        return str(value).replace("~", "~0").replace("/", "~1")

    def visit(node: Any, path: str, semantic_path: str) -> None:
        if state["scalars_scanned"] >= MAX_EXACT_SCALARS_SCANNED:
            state["scan_truncated"] = True
            state["omitted_subtrees"] += 1
            return
        if isinstance(node, Mapping):
            for original_key, child in sorted(node.items(), key=lambda item: str(item[0])):
                child_path = f"{path}/{pointer_segment(original_key)}"
                child_semantic_path = (
                    f"{semantic_path}.{original_key}" if semantic_path else str(original_key)
                )
                visit(child, child_path, child_semantic_path)
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node[:MAX_EXACT_LIST_ITEMS]):
                visit(child, f"{path}/{index}", f"{semantic_path}[{index}]")
            if len(node) > MAX_EXACT_LIST_ITEMS:
                state["scan_truncated"] = True
                state["omitted_subtrees"] += len(node) - MAX_EXACT_LIST_ITEMS
            return

        state["scalars_scanned"] += 1
        if node is None:
            omit("null_value")
            return
        if len(path) > MAX_EXACT_PATH_CHARS:
            omit("oversize_path")
            return
        if _is_secret(semantic_path):
            omit("secret_field")
            return
        if _is_free_text(semantic_path):
            omit("free_text_field")
            return
        if isinstance(node, bool):
            exact_value: str | int | float | bool = node
        elif isinstance(node, int):
            if not -MAX_JCS_SAFE_INTEGER <= node <= MAX_JCS_SAFE_INTEGER:
                omit("non_jcs_safe_integer")
                return
            exact_value = node
        elif isinstance(node, float):
            if not math.isfinite(node):
                omit("non_finite_number")
                return
            exact_value = node
        elif isinstance(node, str):
            if len(node.encode("utf-8")) > MAX_EXACT_STRING_UTF8_BYTES:
                omit("oversize_string")
                return
            if _has_unsafe_credential_text(node):
                omit("unsafe_credential_value")
                return
            exact_value = node
        else:
            omit("unsupported_scalar_type")
            return

        if len(attributes) >= MAX_EXACT_ATTRIBUTES:
            omit("attribute_limit")
            return
        attributes.append({"namespace": "ocsf", "path": path, "value": exact_value})

    visit(value, "", "")
    attributes.sort(key=lambda item: item["path"])
    omission_counts = [
        {"reason": reason, "count": count} for reason, count in sorted(omissions.items())
    ]
    omitted_known = sum(omissions.values())
    hydration_required = bool(omitted_known or state["scan_truncated"])
    metadata = {
        "contract": "bounded_value_exact_typed_json_scalar_subset",
        "selected_count": len(attributes),
        "scalars_scanned": state["scalars_scanned"],
        "known_omitted_scalar_count": omitted_known,
        "omitted_subtree_count": state["omitted_subtrees"],
        "omission_counts": omission_counts,
        "scan_truncated": state["scan_truncated"],
        "source_hydration_required": hydration_required,
        "limits": {
            "max_attributes": MAX_EXACT_ATTRIBUTES,
            "max_scalars_scanned": MAX_EXACT_SCALARS_SCANNED,
            "max_list_items": MAX_EXACT_LIST_ITEMS,
            "max_string_utf8_bytes": MAX_EXACT_STRING_UTF8_BYTES,
            "max_path_chars": MAX_EXACT_PATH_CHARS,
        },
    }
    return attributes, metadata


def _semantic_value(path: str, value: Any) -> str:
    safe = _json_safe_scalar(value)
    if _is_secret(path):
        return "<redacted:secret>"
    if _is_identifier(path):
        return f"<redacted:{_leaf_name(path).replace('_', '-')}>"
    if safe is None:
        return "null"
    if isinstance(safe, bool):
        return "true" if safe else "false"
    if isinstance(safe, (int, float)):
        return _numeric_semantic_value(path, safe)
    return _sanitize_free_text(safe)


def semantic_safe_value(path: str, value: Any) -> str:
    """Return the policy-bound identifier/secret-safe semantic scalar form."""

    return _semantic_value(path, value)


def _role(path: str) -> str:
    tokens = _tokens(path)
    if tokens & _OUTCOME_TOKENS:
        return "outcome"
    if tokens & _ACTION_TOKENS:
        return "action"
    if tokens & _TARGET_TOKENS:
        return "target"
    return "context"


def _label(path: str) -> str:
    parts = [part for part in path.split(".") if part not in {"ocsf"}]
    return ".".join(parts[-3:])


def _facet_text(entries: list[tuple[str, Any]], role: str) -> str:
    fragments = [
        f"{_label(path)}={_semantic_value(path, value)}"
        for path, value in entries
        if _role(path) == role
    ]
    return _bounded("; ".join(fragments), MAX_FACET_TEXT_CHARS)


def _compose_semantic_text(
    document_kind: str,
    relation: str,
    role_texts: Mapping[str, str],
    *,
    unavailable: bool,
) -> str:
    """Compose embedding text with an independent guaranteed budget per role."""

    fragments = [
        f"kind: {document_kind.replace('_', ' ')}",
        f"relation: {_bounded(relation, 128)}",
    ]
    for role in _ROLE_PRECEDENCE:
        text = role_texts.get(role, "")
        if text:
            fragments.append(f"{role}: {_bounded(text, _SEMANTIC_ROLE_BUDGETS[role])}")
    if unavailable:
        fragments.append("content: unavailable typed event")
    result = " | ".join(fragments)
    if len(result) > MAX_SEMANTIC_TEXT_CHARS:
        raise AssertionError("semantic role budgets exceed the total text contract")
    return result


def _find_leaf(entries: list[tuple[str, Any]], names: tuple[str, ...]) -> tuple[str, Any] | None:
    by_name: dict[str, list[tuple[str, Any]]] = {}
    for path, value in entries:
        by_name.setdefault(_leaf_name(path), []).append((path, value))
    for name in names:
        candidates = by_name.get(name, [])
        if candidates:
            return sorted(candidates, key=lambda item: (item[0].count("."), item[0]))[0]
    return None


def _normalise_time(value: Any) -> tuple[Any, str]:
    if isinstance(value, bool) or value is None:
        return value, "present_unparsed"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return str(value), "present_unparsed"
        numeric = float(value)
        magnitude = abs(numeric)
        divisor = 1.0
        if magnitude >= 1e17:
            divisor = 1_000_000_000.0
        elif magnitude >= 1e14:
            divisor = 1_000_000.0
        elif magnitude >= 1e11:
            divisor = 1_000.0
        try:
            stamp = datetime.fromtimestamp(numeric / divisor, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return value, "present_unparsed"
        return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"), "available"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "", "present_unparsed"
        try:
            return _normalise_time(float(text))
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return _bounded(text, MAX_VALUE_CHARS), "present_unparsed"
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "available"
        except ValueError:
            return _bounded(text, MAX_VALUE_CHARS), "present_unparsed"
    return str(value), "present_unparsed"


def _event_metadata(
    entries: list[tuple[str, Any]], parse_error: str | None, projection_truncated: bool
) -> dict[str, Any]:
    time_entry = _find_leaf(
        entries,
        ("time", "event_time", "timestamp", "observed_time", "start_time"),
    )
    if time_entry is None:
        event_time, availability, time_path = None, "missing", None
    else:
        time_path, raw_time = time_entry
        event_time, availability = _normalise_time(raw_time)

    metadata: dict[str, Any] = {
        "event_time": event_time,
        "event_time_availability": availability,
        "event_time_source_path": time_path,
        "projected_leaf_count": len(entries),
        "projection_leaf_limit": MAX_LEAVES,
        "projection_truncated": projection_truncated,
    }
    for output_name, names in (
        ("semantic_class", ("semantic_class", "class_name")),
        ("class_uid", ("class_uid",)),
        ("category_uid", ("category_uid",)),
        ("activity_id", ("activity_id",)),
        ("activity_name", ("activity_name",)),
        ("type_uid", ("type_uid",)),
        ("type_name", ("type_name",)),
    ):
        found = _find_leaf(entries, names)
        metadata[output_name] = _json_safe_scalar(found[1]) if found else None
    if parse_error is not None:
        metadata["input_error"] = parse_error
    return metadata


def _parse_typed_event(value: str | Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, UnicodeError) as error:
            return None, f"invalid_json:{error.__class__.__name__}"
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        return None, f"invalid_type:{type(value).__name__}"
    if not isinstance(parsed, dict):
        return None, "root_not_object"
    return parsed, None


def project_event(
    relation_name: str,
    event_id: str,
    typed_event_json: str | Mapping[str, Any],
    support_ref: str,
) -> dict[str, Any]:
    """Project one snapshot row and always return a terminal disposition.

    The semantic-group digest intentionally excludes event identity, support
    reference, and event time.  Equal semantic observations can therefore
    share one vector while retaining complete occurrence rows elsewhere.
    """

    relation = str(relation_name)
    row_event_id = str(event_id)
    row_support_ref = str(support_ref)
    typed_event, parse_error = _parse_typed_event(typed_event_json)
    flatten_state = {"truncated": False}
    entries = _flatten(typed_event or {}, state=flatten_state)
    known_relation = relation in RELATION_DOCUMENT_KINDS
    semantic_eligible = (
        known_relation
        and parse_error is None
        and relation not in _DERIVATION_ONLY_RELATIONS
    )
    document_kind = RELATION_DOCUMENT_KINDS.get(relation, "structured_only")

    semantic_entries = _semantic_entries(entries)
    action_text = _facet_text(semantic_entries, "action")
    target_text = _facet_text(semantic_entries, "target")
    context_text = _facet_text(semantic_entries, "context")
    outcome_text = _facet_text(semantic_entries, "outcome")
    semantic_text = _compose_semantic_text(
        document_kind,
        relation,
        {
            "action": action_text,
            "target": target_text,
            "context": context_text,
            "outcome": outcome_text,
        },
        unavailable=parse_error is not None,
    )

    structured_fields = {
        path: _redacted_structured_value(path, value) for path, value in entries
    }
    exact_attributes, exact_attribute_metadata = _exact_attribute_subset(typed_event or {})
    if parse_error is not None:
        # An empty exact subset is not evidence that an unavailable typed event
        # contained no filterable attributes. Preserve the known counts as
        # zero (the missing payload cannot be enumerated), but require source
        # hydration so callers never mistake this for a complete local view.
        exact_attribute_metadata = {
            **exact_attribute_metadata,
            "source_hydration_required": True,
        }
    event_metadata = _event_metadata(entries, parse_error, flatten_state["truncated"])
    terminal_disposition = (
        "direct_semantic_document" if semantic_eligible else "structured_only_occurrence"
    )
    if parse_error is not None:
        disposition_reason = "typed_event_unavailable"
    elif not known_relation:
        disposition_reason = "unknown_typed_relation"
    elif relation in _DERIVATION_ONLY_RELATIONS:
        disposition_reason = "awaits_deterministic_window_derivation"
    else:
        disposition_reason = "projected_by_generic_typed_field_policy"

    group_material = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "relation_name": relation,
        "document_kind": document_kind,
        "action_text": action_text,
        "target_text": target_text,
        "context_text": context_text,
        "outcome_text": outcome_text,
        "semantic_leaves": [
            [path, _semantic_value(path, value)]
            for path, value in semantic_entries
            if not _is_positional_raw_token(path)
            and not _is_identifier(path)
            and not _is_secret(path)
        ],
    }
    semantic_group_sha256 = _digest(group_material)
    result: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "terminal_disposition": terminal_disposition,
        "disposition_reason": disposition_reason,
        "relation_name": relation,
        "event_id": row_event_id,
        "support_ref": row_support_ref,
        "document_kind": document_kind,
        "semantic_text": semantic_text,
        "action_text": action_text,
        "target_text": target_text,
        "context_text": context_text,
        "outcome_text": outcome_text,
        "event_metadata": event_metadata,
        "structured_fields": structured_fields,
        "exact_attributes": exact_attributes,
        "exact_attribute_metadata": exact_attribute_metadata,
        "semantic_group_id": f"sha256:{semantic_group_sha256}",
        "semantic_group_sha256": semantic_group_sha256,
    }
    result["projection_sha256"] = _digest(result)
    return result


__all__ = [
    "PROJECTION_SCHEMA_VERSION",
    "RELATION_DOCUMENT_KINDS",
    "project_event",
    "projection_policy_material",
    "projection_policy_ref",
    "semantic_safe_value",
]
