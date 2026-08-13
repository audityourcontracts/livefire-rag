"""Deterministic derived-evidence overlay for a sealed projection pack.

The overlay is deliberately separate from the occurrence-complete projection
pack.  A source occurrence keeps its one terminal disposition while this pack
records a many-to-many membership relation to derived metric windows, network
windows, state transitions, and entity summaries.

The public builder admits only receipt-bound snapshot objects and a verified
base pack.  Fixture hooks are private and exist solely to exercise the same
derivation engine without manufacturing an SDK artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from .evidence_projection import RELATION_DOCUMENT_KINDS, semantic_safe_value


DERIVATION_SCHEMA_VERSION = "livefire.rag.evidence-derivation/1"
WINDOW_MILLIS = 300_000
MAX_DERIVED_NESTING_DEPTH = 8
MAX_DERIVED_CONTAINER_ITEMS = 32
MAX_DERIVED_SCALARS = 128
MAX_DERIVED_NODES = 256
DOCUMENTS_NAME = "documents.jsonl"
MEMBERSHIPS_NAME = "memberships.jsonl"
COVERAGE_NAME = "coverage-report.json"
LOCK_NAME = "objects.lock.json"
MANIFEST_NAME = "manifest.json"

METRIC_RELATIONS = frozenset({"ocsf_ext_livefire_system_metric"})
NETWORK_RELATIONS = frozenset(
    {"ocsf_network_activity", "ocsf_dns_activity", "ocsf_http_activity"}
)
STATE_RELATIONS = frozenset(
    {
        "ocsf_ext_livefire_configuration_snapshot",
        "ocsf_cloud_resources_inventory_info",
        "ocsf_inventory_info",
        "ocsf_user_inventory",
    }
)

_TIME_POINTERS = ("/header/time", "/ocsf/time", "/time", "/event_time")
_SOURCE_TYPE_POINTERS = (
    "/header/metadata/type",
    "/ocsf/metadata/type",
    "/metadata/type",
)
_CONFIG_SUBJECT_INSTANCE_POINTERS = (
    "/subject_instance_id",
    "/subject_instance_uid",
    "/subject_instance/id",
    "/subject_instance/uid",
    "/subject/id",
    "/subject/uid",
    "/resource/id",
    "/resource/uid",
)
_NETWORK_OPERATION_POINTERS = (
    "/action",
    "/method",
    "/query_type",
    "/activity_name",
    "/ocsf/activity_name",
)
_NETWORK_PROTOCOL_POINTERS = (
    "/protocol",
    "/protocol_stack",
    "/transport",
    "/ocsf/protocol_name",
    "/ocsf/app_name",
)
_NETWORK_OUTCOME_POINTERS = (
    "/status",
    "/status_code",
    "/rcode",
    "/response/status_code",
    "/ocsf/status",
)
_NETWORK_MEASURE_POINTERS = {
    "bytes": ("/bytes", "/traffic/bytes", "/ocsf/traffic/bytes"),
    "bytes_in": ("/bytes_in", "/traffic/bytes_in", "/ocsf/traffic/bytes_in"),
    "bytes_out": ("/bytes_out", "/traffic/bytes_out", "/ocsf/traffic/bytes_out"),
    "packets": ("/packets", "/traffic/packets", "/ocsf/traffic/packets"),
    "packets_in": ("/packets_in", "/traffic/packets_in", "/ocsf/traffic/packets_in"),
    "packets_out": ("/packets_out", "/traffic/packets_out", "/ocsf/traffic/packets_out"),
    "duration_millis": ("/duration_millis", "/duration", "/ocsf/duration"),
}
_VOLATILE_STATE_POINTERS = frozenset(
    {
        "/header/time",
        "/header/event_id",
        "/header/metadata/uid",
        "/ocsf/time",
        "/ocsf/metadata/uid",
        "/time",
        "/event_time",
        "/event_id",
        "/record_uid",
        "/native_event_uid",
        "/support_ref",
        "/ingest_time",
        "/ingestion_time",
    }
)
_SAFE_STATE_CATEGORIES = frozenset(
    {
        "active",
        "allowed",
        "blocked",
        "closed",
        "compliant",
        "created",
        "deleted",
        "denied",
        "disabled",
        "enabled",
        "failed",
        "inactive",
        "noncompliant",
        "open",
        "private",
        "public",
        "running",
        "stopped",
        "success",
        "unknown",
    }
)
_SAFE_GRAPH_TAXONOMY = {
    "entity_kind": frozenset(
        {
            "account", "application", "credential", "datastore", "device",
            "file", "identity", "network_endpoint", "process", "resource",
            "service", "session",
        }
    ),
    "participant_role": frozenset(
        {
            "actor", "device", "object", "observer", "owner", "principal",
            "recipient", "resource", "sender", "source", "subject", "target",
        }
    ),
    "relationship_kind": frozenset(
        {
            "acted_on", "authenticated_as", "authenticated_from",
            "communicated_with", "executed_on", "hosted_by", "member_of",
            "observed_on", "produced", "resolved_to", "same_session",
            "used_credential",
        }
    ),
}


class EvidenceDerivationError(RuntimeError):
    """Base class for derivation build and verification failures."""


class EvidenceDerivationCorrupt(EvidenceDerivationError):
    """A derivation overlay does not satisfy its immutable closure contract."""


@dataclass(frozen=True)
class AuxiliaryInput:
    relation: str
    path: Path
    sha256: str
    rows: int


def derivation_policy_material() -> dict[str, Any]:
    """Return the complete, scenario-blind v1 derivation policy."""

    return {
        "schema_version": "livefire.rag.evidence-derivation-policy/1",
        "policy_id": "livefire.rag.generic-evidence-derivation-policy",
        "version": "1",
        "input_contract": {
            "base_pack": "verified_immutable_projection_pack",
            "typed_relations": sorted(RELATION_DOCUMENT_KINDS),
            "graph_relations": ["entities", "participants", "relationships"],
            "selectors_permitted": False,
        },
        "scope": {
            "material": "sorted_unique_exact_participant_role_and_entity_id_pairs",
            "semantic_identity_exposure": False,
            "missing_scope": "explicit_ineligible_no_anonymous_merge",
        },
        "field_adapters": {
            "time_pointers": list(_TIME_POINTERS),
            "source_type_pointers": list(_SOURCE_TYPE_POINTERS),
            "configuration_subject_instance_pointers": list(
                _CONFIG_SUBJECT_INSTANCE_POINTERS
            ),
            "network_operation_pointers": list(_NETWORK_OPERATION_POINTERS),
            "network_protocol_pointers": list(_NETWORK_PROTOCOL_POINTERS),
            "network_outcome_pointers": list(_NETWORK_OUTCOME_POINTERS),
            "network_measure_pointers": {
                key: list(value) for key, value in sorted(_NETWORK_MEASURE_POINTERS.items())
            },
            "volatile_state_pointers": sorted(_VOLATILE_STATE_POINTERS),
        },
        "semantic_rendering": {
            "scalar_policy": "generic_evidence_projection_policy_semantic_safe_value",
            "numeric_policy": "sign_and_base10_magnitude_bucket",
            "safe_state_categories": sorted(_SAFE_STATE_CATEGORIES),
            "unknown_state_value": "typed_value_kind_only",
            "identifiers_in_text": False,
            "graph_taxonomy_allowlists": {
                key: sorted(value) for key, value in sorted(_SAFE_GRAPH_TAXONOMY.items())
            },
            "nested_value_bounds": {
                "maximum_depth": MAX_DERIVED_NESTING_DEPTH,
                "maximum_container_items": MAX_DERIVED_CONTAINER_ITEMS,
                "maximum_scalars": MAX_DERIVED_SCALARS,
                "maximum_visited_nodes": MAX_DERIVED_NODES,
                "over_bound": "omit_without_traversal",
            },
        },
        "metric_window": {
            "relations": sorted(METRIC_RELATIONS),
            "window_millis": WINDOW_MILLIS,
            "origin_epoch_millis": 0,
            "bounds": "start_inclusive_end_exclusive",
            "closure": "snapshot_sealed_observed",
            "completeness": "unknown_expected_cadence",
            "group_fields": ["relation", "source_type", "metric", "unit", "scope"],
            "aggregates": ["count", "minimum", "maximum", "integer_sum", "mean_fraction"],
        },
        "network_window": {
            "relations": sorted(NETWORK_RELATIONS),
            "window_millis": WINDOW_MILLIS,
            "origin_epoch_millis": 0,
            "bounds": "start_inclusive_end_exclusive",
            "closure": "snapshot_sealed_observed",
            "completeness": "unknown_expected_cadence",
            "group_fields": [
                "relation",
                "source_type",
                "operation_signature",
                "protocol_signature",
                "scope",
            ],
            "missing_measures": "counted_not_zero_filled",
        },
        "state_transition": {
            "relations": sorted(STATE_RELATIONS),
            "series_order": ["event_time_millis", "event_id"],
            "same_time_distinct_state": "ambiguous_no_transition",
            "first_observation": "no_predecessor",
            "equal_adjacent_state": "unchanged",
            "configuration_requires_explicit_stable_subject_instance": True,
            "missing_configuration_subject_instance": "ineligible",
            "closure": "snapshot_sealed_observed_history_outside_snapshot_unknown",
        },
        "entity": {
            "identity": "exact_entity_id",
            "membership": "sorted_unique_entity_occurrence_role",
            "semantic_fields": [
                "entity_kind",
                "participant_roles",
                "typed_relations",
                "relationship_kinds_and_directions",
                "neighbor_entity_kinds",
            ],
            "semantic_identifier_values_permitted": False,
            "orphan_behavior": "coverage_only_no_evidence_document",
        },
        "identity": {
            "canonicalization": "RFC8785",
            "hash": "sha256",
            "membership_set": "length_prefixed_sorted_unique_membership_tuples",
        },
        "leakage_controls": {
            "query_input_permitted": False,
            "labels_permitted": False,
            "qrels_permitted": False,
            "expected_evidence_permitted": False,
            "corpus_tuned_thresholds_permitted": False,
        },
    }


def derivation_policy_ref() -> dict[str, str]:
    return component_ref(
        "livefire.rag.generic-evidence-derivation-policy",
        "1",
        derivation_policy_material(),
    )


def derivation_manifest_identity(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256_omitting(dict(manifest), ("component", "sha256"))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_component(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a component reference")
    required = {"id", "version", "sha256"}
    allowed = required | {"uri"}
    if not required <= set(value) or set(value) - allowed:
        raise ValueError(f"{label} must be a closed component reference")
    if not all(isinstance(value.get(field), str) and value[field] for field in required):
        raise ValueError(f"{label} is invalid")
    if not _is_sha256(value["sha256"]):
        raise ValueError(f"{label}.sha256 is invalid")
    if "uri" in value and (not isinstance(value["uri"], str) or not value["uri"]):
        raise ValueError(f"{label}.uri is invalid")
    return dict(value)


def _parse_json(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def _pointer(value: Any, pointer: str) -> Any:
    current = value
    if not pointer.startswith("/"):
        raise ValueError("internal pointer is not RFC 6901")
    for encoded in pointer[1:].split("/"):
        segment = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit() and int(segment) < len(current):
            current = current[int(segment)]
        else:
            return None
    return current


def _first(value: Mapping[str, Any], pointers: Sequence[str]) -> Any:
    for pointer in pointers:
        selected = _pointer(value, pointer)
        if selected is not None:
            return selected
    return None


def _tag(value: Any) -> dict[str, Any]:
    return {"state": "absent"} if value is None else {"state": "present", "value": value}


def _event_time_millis(value: Mapping[str, Any]) -> tuple[int | None, str | None]:
    raw = _first(value, _TIME_POINTERS)
    if raw is None:
        return None, "missing_time"
    if isinstance(raw, bool):
        return None, "invalid_time"
    if isinstance(raw, int):
        candidate = raw
    elif isinstance(raw, float) and raw.is_integer():
        candidate = int(raw)
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lstrip("-").isdigit():
            candidate = int(stripped)
        else:
            try:
                parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return None, "invalid_time"
                candidate = int(parsed.timestamp() * 1000)
            except (ValueError, OverflowError):
                return None, "invalid_time"
    else:
        return None, "invalid_time"
    if candidate < 0:
        return None, "invalid_time"
    return candidate, None


def _iso_millis(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _safe_token(value: Any, *, path: str = "derived.context", maximum: int = 160) -> str:
    """Render nested derived material without publishing raw keys or values."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    state: dict[str, Any] = {
        "nodes": 0,
        "scalars": 0,
        "exhausted": False,
        "marker": "<omitted:node-bound>",
    }

    def render(child: Any, child_path: str, depth: int) -> str:
        if state["exhausted"]:
            return str(state["marker"])
        if state["nodes"] >= MAX_DERIVED_NODES:
            state["exhausted"] = True
            state["marker"] = "<omitted:node-bound>"
            return str(state["marker"])
        state["nodes"] += 1
        if depth > MAX_DERIVED_NESTING_DEPTH:
            return "<omitted:depth-bound>"
        if child is None:
            return "absent"
        if isinstance(child, Mapping):
            if len(child) > MAX_DERIVED_CONTAINER_ITEMS:
                return "<omitted:container-bound>"
            # Keys influence only the policy path. They are not rendered:
            # vendor-controlled keys can themselves contain secrets.
            parts = []
            for key, item in sorted(child.items(), key=lambda pair: str(pair[0])):
                parts.append(render(item, f"{child_path}.{key}", depth + 1))
                if state["exhausted"]:
                    break
            return "[" + ",".join(parts) + "]"
        if isinstance(child, (list, tuple)):
            if len(child) > MAX_DERIVED_CONTAINER_ITEMS:
                return "<omitted:container-bound>"
            parts = []
            for item in child:
                parts.append(render(item, f"{child_path}[]", depth + 1))
                if state["exhausted"]:
                    break
            return "[" + ",".join(parts) + "]"
        if state["scalars"] >= MAX_DERIVED_SCALARS:
            state["exhausted"] = True
            state["marker"] = "<omitted:scalar-bound>"
            return str(state["marker"])
        state["scalars"] += 1
        # All scalar types, including numeric identifiers and quantities, must
        # pass through the path-aware projection policy.
        return semantic_safe_value(child_path, child)

    text = render(value, path, 0)
    if len(text) > maximum:
        return text[: maximum - 1] + "…"
    return text


def _safe_graph_taxonomy(value: Any, family: str) -> str:
    """Expose only a closed generic graph taxonomy, never vendor values."""

    if isinstance(value, str) and value in _SAFE_GRAPH_TAXONOMY[family]:
        return value.replace("_", " ")
    return f"<redacted:{family.replace('_', '-')}>"


def _magnitude_bucket(value: int) -> str:
    if value == 0:
        return "zero"
    sign = "negative-" if value < 0 else ""
    return f"{sign}1e{len(str(abs(value))) - 1}"


def _state_semantic_category(value: Any) -> str:
    """Expose useful generic state categories without publishing identities."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not value.is_integer():
            return "numeric-state"
        return f"numeric-{_magnitude_bucket(int(value))}"
    if isinstance(value, str):
        normalized = "_".join(value.strip().lower().replace("-", " ").split())
        if normalized in _SAFE_STATE_CATEGORIES:
            return normalized
        return "string-state"
    if value is None:
        return "null-state"
    if isinstance(value, list):
        return "list-state"
    if isinstance(value, Mapping):
        return "object-state"
    return "other-state"


def _scope_material(scope_json: str | None) -> tuple[list[dict[str, str]] | None, str | None]:
    if not scope_json:
        return None, "missing_canonical_scope"
    try:
        value = json.loads(scope_json)
    except json.JSONDecodeError:
        return None, "invalid_canonical_scope"
    if not isinstance(value, list) or not value:
        return None, "missing_canonical_scope"
    result: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None, "invalid_canonical_scope"
        role, entity_id = item.get("role"), item.get("entity_id")
        if not isinstance(role, str) or not role or not isinstance(entity_id, str) or not entity_id:
            return None, "invalid_canonical_scope"
        result.add((role, entity_id))
    return [
        {"role": role, "entity_id": entity_id} for role, entity_id in sorted(result)
    ], None


def _membership_set_digest(members: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    prior: tuple[str, str] | None = None
    for occurrence_id, role in sorted(members):
        current = (occurrence_id, role)
        if prior is not None and current <= prior:
            raise EvidenceDerivationError("derivation membership tuples are not unique")
        prior = current
        payload = canonical_json_bytes([occurrence_id, role])
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _group_sha256(family: str, key: Any, policy: Mapping[str, str]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": DERIVATION_SCHEMA_VERSION,
                "family": family,
                "key": key,
                "policy": dict(policy),
            }
        )
    )


def _document(
    *,
    kind: str,
    relations: Sequence[str],
    semantic_text: str,
    facets: Mapping[str, Sequence[str]],
    group_key: Any,
    aggregate_material: Mapping[str, Any],
    members: Sequence[tuple[str, str]],
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    policy: Mapping[str, str],
    time_range: tuple[int, int] | None = None,
    closure_state: str = "snapshot_sealed_observed",
    completeness_state: str = "unknown_expected_coverage",
) -> dict[str, Any]:
    group_sha256 = _group_sha256(kind, group_key, policy)
    input_set_sha256 = _membership_set_digest(members)
    identity = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "source_snapshot": dict(source_snapshot),
        "base_projection_pack": dict(base_pack),
        "derivation_policy": dict(policy),
        "document_kind": kind,
        "group_sha256": group_sha256,
        "aggregate_material": dict(aggregate_material),
        "input_set_sha256": input_set_sha256,
    }
    document_id = "ddoc-" + sha256_bytes(canonical_json_bytes(identity))
    document: dict[str, Any] = {
        "schema_version": "livefire.rag.evidence-derived-document/1",
        "document_id": document_id,
        "document_sha256": "",
        "document_kind": kind,
        "representation": "derived",
        "searchable": True,
        "source_snapshot": dict(source_snapshot),
        "base_projection_pack": dict(base_pack),
        "derivation_policy": dict(policy),
        "relation_identities": [
            {"namespace": "ocsf", "relation": relation} for relation in sorted(set(relations))
        ],
        "semantic_projection": {
            "text": semantic_text[:3072],
            "facets": [
                {"name": name, "values": sorted(set(values))}
                for name, values in sorted(facets.items())
                if values
            ],
        },
        "derivation": {
            "group_sha256": group_sha256,
            "input_count": len(members),
            "input_set_sha256": input_set_sha256,
            "closure_state": closure_state,
            "completeness_state": completeness_state,
            "aggregate_material": dict(aggregate_material),
        },
        "occurrence_count": len({occurrence_id for occurrence_id, _ in members}),
    }
    if time_range is not None:
        document["time_range"] = {
            "start": _iso_millis(time_range[0]),
            "end": _iso_millis(time_range[1]),
            "bounds": "start_inclusive_end_exclusive",
        }
    document["document_sha256"] = canonical_sha256_omitting(document, ("document_sha256",))
    return document


def _membership(
    document_id: str,
    occurrence_id: str,
    input_role: str,
    policy: Mapping[str, str],
    entity_id: str | None = None,
) -> dict[str, Any]:
    material = {
        "schema_version": "livefire.rag.evidence-derivation-membership-identity/1",
        "derived_document_id": document_id,
        "occurrence_id": occurrence_id,
        "input_role": input_role,
        "derivation_policy": dict(policy),
    }
    if entity_id is not None:
        material["entity_id"] = entity_id
    membership_id = "dmem-" + sha256_bytes(canonical_json_bytes(material))
    row = {
        "schema_version": "livefire.rag.evidence-derivation-membership-row/1",
        "membership_id": membership_id,
        "membership_sha256": "",
        "derived_document_id": document_id,
        "occurrence_id": occurrence_id,
        "input_role": input_role,
        "derivation_policy": dict(policy),
    }
    if entity_id is not None:
        row["entity_id"] = entity_id
    row["membership_sha256"] = canonical_sha256_omitting(row, ("membership_sha256",))
    return row


def _state_without_volatility(value: Any, pointer: str = "") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in sorted(value.items()):
            encoded = str(key).replace("~", "~0").replace("/", "~1")
            child_pointer = f"{pointer}/{encoded}"
            if child_pointer in _VOLATILE_STATE_POINTERS:
                continue
            result[str(key)] = _state_without_volatility(child, child_pointer)
        return result
    if isinstance(value, list):
        return [
            _state_without_volatility(child, f"{pointer}/{index}")
            for index, child in enumerate(value)
        ]
    return value


def _state_signature(relation: str, value: Mapping[str, Any]) -> tuple[Any, str | None]:
    if relation == "ocsf_ext_livefire_configuration_snapshot":
        state = _pointer(value, "/state")
        if state is None:
            return None, "missing_state"
        return state, None
    return _state_without_volatility(value), None


def _state_discriminator(
    relation: str, value: Mapping[str, Any]
) -> tuple[Any | None, str | None]:
    if relation == "ocsf_ext_livefire_configuration_snapshot":
        subject_instance = _first(value, _CONFIG_SUBJECT_INSTANCE_POINTERS)
        if (
            isinstance(subject_instance, bool)
            or not isinstance(subject_instance, (str, int))
            or (isinstance(subject_instance, str) and not subject_instance.strip())
        ):
            return None, "missing_stable_subject_instance"
        return {
            "snapshot_kind": _tag(_pointer(value, "/snapshot_kind")),
            "subject_kind": _tag(_pointer(value, "/subject_kind")),
            "subject": _tag(_pointer(value, "/subject")),
            "subject_instance": _tag(subject_instance),
        }, None
    return {
        "class": _tag(
            _first(value, ("/semantic_class", "/ocsf/class_name", "/class_name"))
        ),
        "activity": _tag(
            _first(value, ("/activity_name", "/ocsf/activity_name", "/activity_id"))
        ),
    }, None


def _network_signature(value: Mapping[str, Any], pointers: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": pointer, **_tag(_pointer(value, pointer))} for pointer in pointers]


def _integer_measure(value: Mapping[str, Any], pointers: Sequence[str]) -> int | None:
    selected = _first(value, pointers)
    if isinstance(selected, bool):
        return None
    if isinstance(selected, int):
        return selected
    if isinstance(selected, float) and selected.is_integer():
        return int(selected)
    if isinstance(selected, str) and selected.strip().lstrip("-").isdigit():
        return int(selected.strip())
    return None


class _OutputWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.documents = (root / DOCUMENTS_NAME).open("wb")
        self.memberships = (root / MEMBERSHIPS_NAME).open("wb")
        self.document_count = 0
        self.membership_count = 0
        self.by_kind: Counter[str] = Counter()

    def add(
        self,
        document: Mapping[str, Any],
        members: Sequence[tuple[str, str]],
        policy: Mapping[str, str],
        *,
        entity_id: str | None = None,
    ) -> None:
        self.documents.write(canonical_json_bytes(dict(document), newline=True))
        self.document_count += 1
        self.by_kind[str(document["document_kind"])] += 1
        for occurrence_id, role in sorted(members):
            row = _membership(
                str(document["document_id"]),
                occurrence_id,
                role,
                policy,
                entity_id=entity_id,
            )
            self.memberships.write(canonical_json_bytes(row, newline=True))
            self.membership_count += 1

    def close(self) -> None:
        self.documents.close()
        self.memberships.close()


def _connect() -> Any:
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise EvidenceDerivationError(
            "DuckDB is required for evidence derivation; install livefire-rag[prototype]"
        ) from error
    return duckdb.connect()


def _create_fixture_tables(connection: Any) -> None:
    connection.execute("CREATE TABLE base_occurrences(event_id VARCHAR, occurrence_id VARCHAR, relation_name VARCHAR)")
    connection.execute("CREATE TABLE typed_events(event_id VARCHAR, relation_name VARCHAR, typed_event_json VARCHAR)")
    connection.execute("CREATE TABLE participants(event_id VARCHAR, entity_id VARCHAR, role VARCHAR, support_ref VARCHAR)")
    connection.execute("CREATE TABLE entities(entity_id VARCHAR, kind VARCHAR, display_name VARCHAR, canonical_value VARCHAR, support_ref VARCHAR)")
    connection.execute("CREATE TABLE relationships(relationship_id VARCHAR, kind VARCHAR, source_id VARCHAR, target_id VARCHAR, event_id VARCHAR, support_ref VARCHAR)")


def _insert_fixture_rows(
    connection: Any,
    *,
    typed_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    occurrences: Iterable[Mapping[str, Any]],
    participants: Iterable[Mapping[str, Any]],
    entities: Iterable[Mapping[str, Any]],
    relationships: Iterable[Mapping[str, Any]],
) -> None:
    occurrence_values = []
    for row in occurrences:
        event_id = row.get("event_id")
        if event_id is None and isinstance(row.get("source_pointer"), Mapping):
            event_id = row["source_pointer"].get("record_id")
        relation = row.get("relation_name")
        if relation is None and isinstance(row.get("relation_identity"), Mapping):
            relation = row["relation_identity"].get("relation")
        occurrence_values.append((event_id, row.get("occurrence_id"), relation))
    if occurrence_values:
        connection.executemany("INSERT INTO base_occurrences VALUES (?, ?, ?)", occurrence_values)
    typed_values = []
    for relation in sorted(typed_rows):
        for row in typed_rows[relation]:
            payload = row.get("typed_event_json")
            typed_values.append(
                (
                    row.get("event_id"),
                    relation,
                    payload if isinstance(payload, str) else canonical_json_bytes(payload).decode(),
                )
            )
    if typed_values:
        connection.executemany("INSERT INTO typed_events VALUES (?, ?, ?)", typed_values)
    participant_values = [
        (row.get("event_id"), row.get("entity_id"), row.get("role"), row.get("support_ref"))
        for row in participants
    ]
    if participant_values:
        connection.executemany(
            "INSERT INTO participants VALUES (?, ?, ?, ?)", participant_values
        )
    entity_values = [
            (
                row.get("entity_id"),
                row.get("kind"),
                row.get("display_name"),
                row.get("canonical_value"),
                row.get("support_ref"),
            )
            for row in entities
        ]
    if entity_values:
        connection.executemany(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)", entity_values
        )
    relationship_values = [
            (
                row.get("relationship_id"),
                row.get("kind"),
                row.get("source_id"),
                row.get("target_id"),
                row.get("event_id"),
                row.get("support_ref"),
            )
            for row in relationships
        ]
    if relationship_values:
        connection.executemany(
            "INSERT INTO relationships VALUES (?, ?, ?, ?, ?, ?)", relationship_values
        )


def _prepare_views(connection: Any) -> None:
    duplicate = connection.execute(
        "SELECT event_id FROM base_occurrences GROUP BY event_id HAVING count(*) <> 1 LIMIT 1"
    ).fetchone()
    if duplicate:
        raise EvidenceDerivationError("base occurrence event ids are not unique")
    mismatch = connection.execute(
        "SELECT t.event_id FROM typed_events t LEFT JOIN base_occurrences o USING(event_id) "
        "WHERE o.event_id IS NULL OR t.relation_name <> o.relation_name LIMIT 1"
    ).fetchone()
    if mismatch:
        raise EvidenceDerivationError("typed event does not resolve to its base occurrence")
    dangling_participant = connection.execute(
        "SELECT p.event_id FROM participants p LEFT JOIN base_occurrences o USING(event_id) "
        "WHERE o.event_id IS NULL LIMIT 1"
    ).fetchone()
    if dangling_participant:
        raise EvidenceDerivationError("participant references an absent base occurrence")
    dangling_entity = connection.execute(
        "SELECT p.entity_id FROM participants p LEFT JOIN entities e USING(entity_id) "
        "WHERE e.entity_id IS NULL LIMIT 1"
    ).fetchone()
    if dangling_entity:
        raise EvidenceDerivationError("participant references an absent entity")
    dangling_relationship = connection.execute(
        "SELECT r.relationship_id FROM relationships r "
        "LEFT JOIN entities s ON r.source_id=s.entity_id "
        "LEFT JOIN entities t ON r.target_id=t.entity_id "
        "LEFT JOIN base_occurrences o USING(event_id) "
        "WHERE s.entity_id IS NULL OR t.entity_id IS NULL OR o.event_id IS NULL LIMIT 1"
    ).fetchone()
    if dangling_relationship:
        raise EvidenceDerivationError("relationship references an absent entity or occurrence")
    connection.execute(
        "CREATE TEMP VIEW event_scopes AS "
        "SELECT event_id, to_json(list(struct_pack(role := role, entity_id := entity_id) "
        "ORDER BY role, entity_id)) AS scope_json FROM "
        "(SELECT DISTINCT event_id, role, entity_id FROM participants "
        "WHERE role IS NOT NULL AND role <> '' AND entity_id IS NOT NULL AND entity_id <> '') "
        "GROUP BY event_id"
    )
    connection.execute(
        "CREATE TEMP VIEW admitted_typed AS SELECT t.event_id, t.relation_name, "
        "t.typed_event_json, o.occurrence_id, s.scope_json "
        "FROM typed_events t JOIN base_occurrences o USING(event_id) "
        "LEFT JOIN event_scopes s USING(event_id)"
    )


def _iter_rows(cursor: Any, batch_size: int = 4096) -> Iterator[tuple[Any, ...]]:
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        yield from rows


def _derive_metric(
    connection: Any,
    writer: _OutputWriter,
    coverage: dict[str, Any],
    *,
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    policy: Mapping[str, str],
) -> None:
    relation = next(iter(METRIC_RELATIONS))
    cursor = connection.execute(
        "SELECT event_id, occurrence_id, typed_event_json, scope_json FROM admitted_typed "
        "WHERE relation_name=? ORDER BY event_id, occurrence_id",
        [relation],
    )
    groups: dict[bytes, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    applicable = 0
    eligible = 0
    for event_id, occurrence_id, payload, scope_json in _iter_rows(cursor):
        applicable += 1
        value = _parse_json(payload)
        if value is None:
            reasons["typed_event_unavailable"] += 1
            continue
        time_ms, reason = _event_time_millis(value)
        if reason:
            reasons[reason] += 1
            continue
        metric = _pointer(value, "/metric")
        raw_number = _pointer(value, "/value_milli")
        if not isinstance(metric, str) or not metric:
            reasons["missing_metric"] += 1
            continue
        if isinstance(raw_number, bool) or not isinstance(raw_number, int):
            reasons["invalid_value"] += 1
            continue
        scope, reason = _scope_material(scope_json)
        if reason:
            reasons[reason] += 1
            continue
        eligible += 1
        assert time_ms is not None and scope is not None
        start = (time_ms // WINDOW_MILLIS) * WINDOW_MILLIS
        key = {
            "relation": relation,
            "source_type": _tag(_first(value, _SOURCE_TYPE_POINTERS)),
            "metric": metric,
            "unit": _tag(_pointer(value, "/unit")),
            "scope": scope,
            "window_start_millis": start,
            "window_end_millis": start + WINDOW_MILLIS,
        }
        encoded = canonical_json_bytes(key)
        group = groups.setdefault(
            encoded,
            {"key": key, "values": [], "members": [], "relation": relation},
        )
        group["values"].append(raw_number)
        group["members"].append((occurrence_id, "sample"))
    for encoded in sorted(groups):
        group = groups[encoded]
        values = group["values"]
        key = group["key"]
        aggregate = {
            "sample_count": len(values),
            "minimum_value_milli": min(values),
            "maximum_value_milli": max(values),
            "sum_value_milli": str(sum(values)),
            "mean_value_milli": {"numerator": str(sum(values)), "denominator": len(values)},
        }
        semantic = (
            f"metric observation window | metric: {_safe_token(key['metric'], path='metric')} | "
            f"unit: {_safe_token(key['unit'].get('value'), path='unit')} | "
            f"samples: {_magnitude_bucket(len(values))} | "
            f"minimum magnitude: {_magnitude_bucket(min(values))} | "
            f"maximum magnitude: {_magnitude_bucket(max(values))}"
        )
        document = _document(
            kind="metric_window",
            relations=[relation],
            semantic_text=semantic,
            facets={
                "metric": [_safe_token(key["metric"], path="metric")],
                "unit": [_safe_token(key["unit"].get("value"), path="unit")],
            },
            group_key=key,
            aggregate_material=aggregate,
            members=group["members"],
            source_snapshot=source_snapshot,
            base_pack=base_pack,
            policy=policy,
            time_range=(key["window_start_millis"], key["window_end_millis"]),
            completeness_state="unknown_expected_cadence",
        )
        writer.add(document, group["members"], policy)
    coverage["families"]["metric_window"] = {
        "examined_source_record_count": coverage["closure"]["base_source_record_count"],
        "not_applicable_source_record_count": coverage["closure"]["base_source_record_count"] - applicable,
        "applicable_source_record_count": applicable,
        "eligible_source_record_count": eligible,
        "ineligible_source_record_count": applicable - eligible,
        "document_count": len(groups),
        "membership_count": eligible,
        "reason_counts": dict(sorted(reasons.items())),
    }


def _derive_network(
    connection: Any,
    writer: _OutputWriter,
    coverage: dict[str, Any],
    *,
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    policy: Mapping[str, str],
) -> None:
    placeholders = ",".join("?" for _ in NETWORK_RELATIONS)
    cursor = connection.execute(
        f"SELECT event_id, occurrence_id, relation_name, typed_event_json, scope_json "
        f"FROM admitted_typed WHERE relation_name IN ({placeholders}) ORDER BY relation_name,event_id,occurrence_id",
        sorted(NETWORK_RELATIONS),
    )
    groups: dict[bytes, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    by_relation: Counter[str] = Counter()
    eligible = 0
    for event_id, occurrence_id, relation, payload, scope_json in _iter_rows(cursor):
        by_relation[relation] += 1
        value = _parse_json(payload)
        if value is None:
            reasons["typed_event_unavailable"] += 1
            continue
        time_ms, reason = _event_time_millis(value)
        if reason:
            reasons[reason] += 1
            continue
        scope, reason = _scope_material(scope_json)
        if reason:
            reasons[reason] += 1
            continue
        eligible += 1
        assert time_ms is not None and scope is not None
        start = (time_ms // WINDOW_MILLIS) * WINDOW_MILLIS
        operation = _network_signature(value, _NETWORK_OPERATION_POINTERS)
        protocol = _network_signature(value, _NETWORK_PROTOCOL_POINTERS)
        key = {
            "relation": relation,
            "source_type": _tag(_first(value, _SOURCE_TYPE_POINTERS)),
            "operation_signature": operation,
            "protocol_signature": protocol,
            "scope": scope,
            "window_start_millis": start,
            "window_end_millis": start + WINDOW_MILLIS,
        }
        encoded = canonical_json_bytes(key)
        group = groups.setdefault(
            encoded,
            {
                "key": key,
                "members": [],
                "measures": {name: [] for name in _NETWORK_MEASURE_POINTERS},
                "missing": Counter(),
                "outcomes": Counter(),
                "relation": relation,
            },
        )
        group["members"].append((occurrence_id, "event"))
        for name, pointers in _NETWORK_MEASURE_POINTERS.items():
            measure = _integer_measure(value, pointers)
            if measure is None:
                group["missing"][name] += 1
            else:
                group["measures"][name].append(measure)
        outcome = _first(value, _NETWORK_OUTCOME_POINTERS)
        group["outcomes"][_safe_token(outcome, path="network.outcome")] += 1
    for encoded in sorted(groups):
        group = groups[encoded]
        key = group["key"]
        measures: dict[str, Any] = {}
        for name in sorted(_NETWORK_MEASURE_POINTERS):
            values = group["measures"][name]
            measures[name] = {
                "observed_count": len(values),
                "missing_count": group["missing"][name],
            }
            if values:
                measures[name].update(
                    {"minimum": min(values), "maximum": max(values), "sum": str(sum(values))}
                )
        aggregate = {
            "event_count": len(group["members"]),
            "measures": measures,
            "outcome_histogram": dict(sorted(group["outcomes"].items())),
        }
        operation_values = [
            _safe_token(item.get("value"), path=str(item["path"]))
            for item in key["operation_signature"]
            if item.get("state") == "present"
        ]
        protocol_values = [
            _safe_token(item.get("value"), path=str(item["path"]))
            for item in key["protocol_signature"]
            if item.get("state") == "present"
        ]
        outcome_values = [
            f"{value}:{_magnitude_bucket(count)}"
            for value, count in sorted(group["outcomes"].items())
        ]
        measure_values = [
            f"{name}:{_magnitude_bucket(int(details['sum']))}"
            for name, details in sorted(measures.items())
            if "sum" in details
        ]
        semantic = (
            f"network observation window | relation: {group['relation']} | "
            f"operation: {', '.join(operation_values) or 'absent'} | "
            f"protocol: {', '.join(protocol_values) or 'absent'} | "
            f"outcome: {', '.join(outcome_values) or 'absent'} | "
            f"measures: {', '.join(measure_values) or 'absent'} | "
            f"events: {_magnitude_bucket(len(group['members']))}"
        )
        document = _document(
            kind="network_window",
            relations=[group["relation"]],
            semantic_text=semantic,
            facets={
                "operation": operation_values,
                "protocol": protocol_values,
                "outcome": outcome_values,
                "measure": measure_values,
            },
            group_key=key,
            aggregate_material=aggregate,
            members=group["members"],
            source_snapshot=source_snapshot,
            base_pack=base_pack,
            policy=policy,
            time_range=(key["window_start_millis"], key["window_end_millis"]),
            completeness_state="unknown_expected_coverage",
        )
        writer.add(document, group["members"], policy)
    applicable = sum(by_relation.values())
    coverage["families"]["network_window"] = {
        "examined_source_record_count": coverage["closure"]["base_source_record_count"],
        "not_applicable_source_record_count": coverage["closure"]["base_source_record_count"] - applicable,
        "applicable_source_record_count": applicable,
        "eligible_source_record_count": eligible,
        "ineligible_source_record_count": applicable - eligible,
        "document_count": len(groups),
        "membership_count": eligible,
        "by_relation": dict(sorted(by_relation.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def _derive_transitions(
    connection: Any,
    writer: _OutputWriter,
    coverage: dict[str, Any],
    *,
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    policy: Mapping[str, str],
) -> None:
    placeholders = ",".join("?" for _ in STATE_RELATIONS)
    cursor = connection.execute(
        f"SELECT event_id, occurrence_id, relation_name, typed_event_json, scope_json "
        f"FROM admitted_typed WHERE relation_name IN ({placeholders}) ORDER BY relation_name,event_id,occurrence_id",
        sorted(STATE_RELATIONS),
    )
    series: dict[bytes, list[dict[str, Any]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    by_relation: Counter[str] = Counter()
    series_eligible = 0
    for event_id, occurrence_id, relation, payload, scope_json in _iter_rows(cursor):
        by_relation[relation] += 1
        value = _parse_json(payload)
        if value is None:
            reasons["typed_event_unavailable"] += 1
            continue
        time_ms, reason = _event_time_millis(value)
        if reason:
            reasons[reason] += 1
            continue
        scope, reason = _scope_material(scope_json)
        if reason:
            reasons[reason] += 1
            continue
        state, reason = _state_signature(relation, value)
        if reason:
            reasons[reason] += 1
            continue
        discriminator, reason = _state_discriminator(relation, value)
        if reason:
            reasons[reason] += 1
            continue
        series_eligible += 1
        key = {
            "relation": relation,
            "source_type": _tag(_first(value, _SOURCE_TYPE_POINTERS)),
            "discriminator": discriminator,
            "scope": scope,
        }
        signature = sha256_bytes(canonical_json_bytes(state))
        series[canonical_json_bytes(key)].append(
            {
                "key": key,
                "event_id": event_id,
                "occurrence_id": occurrence_id,
                "time": time_ms,
                "signature": signature,
                "state": state,
                "relation": relation,
            }
        )
    document_count = 0
    membership_count = 0
    for encoded in sorted(series):
        rows = sorted(series[encoded], key=lambda row: (row["time"], row["event_id"]))
        previous: dict[str, Any] | None = None
        offset = 0
        while offset < len(rows):
            end = offset + 1
            while end < len(rows) and rows[end]["time"] == rows[offset]["time"]:
                end += 1
            same_time = rows[offset:end]
            fingerprints = {row["signature"] for row in same_time}
            if len(fingerprints) > 1:
                outcomes["ambiguous_same_time"] += len(same_time)
                previous = None
                offset = end
                continue
            current = same_time[0]
            if len(same_time) > 1:
                outcomes["duplicate_same_state_same_time"] += len(same_time) - 1
            if previous is None:
                outcomes["no_predecessor"] += 1
            elif previous["signature"] == current["signature"]:
                outcomes["unchanged"] += 1
            else:
                members = [
                    (previous["occurrence_id"], "before"),
                    (current["occurrence_id"], "after"),
                ]
                key = {
                    **current["key"],
                    "before_time_millis": previous["time"],
                    "after_time_millis": current["time"],
                    "before_state_sha256": previous["signature"],
                    "after_state_sha256": current["signature"],
                }
                aggregate = {
                    "before_state_sha256": previous["signature"],
                    "after_state_sha256": current["signature"],
                    "before_time": _iso_millis(previous["time"]),
                    "after_time": _iso_millis(current["time"]),
                }
                before_category = _state_semantic_category(previous["state"])
                after_category = _state_semantic_category(current["state"])
                discriminator = current["key"]["discriminator"]
                subject_kind = "absent"
                if isinstance(discriminator, Mapping):
                    tagged_kind = discriminator.get("subject_kind")
                    if isinstance(tagged_kind, Mapping) and tagged_kind.get("state") == "present":
                        subject_kind = _safe_token(
                            tagged_kind.get("value"), path="subject_kind"
                        )
                semantic = (
                    f"state transition | relation: {current['relation']} | "
                    f"subject type: {subject_kind} | change: {before_category} to {after_category}"
                )
                document = _document(
                    kind="state_transition",
                    relations=[current["relation"]],
                    semantic_text=semantic,
                    facets={
                        "transition": ["state changed", f"{before_category} to {after_category}"],
                        "subject_kind": [subject_kind],
                    },
                    group_key=key,
                    aggregate_material=aggregate,
                    members=members,
                    source_snapshot=source_snapshot,
                    base_pack=base_pack,
                    policy=policy,
                    closure_state="snapshot_sealed_observed_history_outside_snapshot_unknown",
                    completeness_state="adjacent_observed_states_only",
                )
                writer.add(document, members, policy)
                document_count += 1
                membership_count += 2
                outcomes["transition"] += 1
            previous = current
            offset = end
    applicable = sum(by_relation.values())
    coverage["families"]["state_transition"] = {
        "examined_source_record_count": coverage["closure"]["base_source_record_count"],
        "not_applicable_source_record_count": coverage["closure"]["base_source_record_count"] - applicable,
        "applicable_source_record_count": applicable,
        "series_eligible_source_record_count": series_eligible,
        "ineligible_source_record_count": applicable - series_eligible,
        "document_count": document_count,
        "membership_count": membership_count,
        "series_count": len(series),
        "by_relation": dict(sorted(by_relation.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def _derive_entities(
    connection: Any,
    writer: _OutputWriter,
    coverage: dict[str, Any],
    *,
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    policy: Mapping[str, str],
) -> None:
    relationship_context: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    cursor = connection.execute(
        "SELECT r.source_id, r.target_id, r.kind, s.kind, t.kind FROM relationships r "
        "JOIN entities s ON r.source_id=s.entity_id JOIN entities t ON r.target_id=t.entity_id "
        "ORDER BY r.source_id,r.target_id,r.kind"
    )
    for source_id, target_id, kind, source_kind, target_kind in _iter_rows(cursor):
        relationship_context[source_id][("outbound", kind, target_kind)] += 1
        relationship_context[target_id][("inbound", kind, source_kind)] += 1

    cursor = connection.execute(
        "SELECT p.entity_id,e.kind,o.occurrence_id,o.relation_name,p.role "
        "FROM (SELECT DISTINCT event_id,entity_id,role FROM participants) p "
        "JOIN entities e USING(entity_id) JOIN base_occurrences o USING(event_id) "
        "ORDER BY p.entity_id,o.occurrence_id,p.role"
    )
    current_id: str | None = None
    current_kind: str | None = None
    members: list[tuple[str, str]] = []
    relations: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    document_count = 0
    membership_count = 0

    def finish() -> None:
        nonlocal current_id, current_kind, members, relations, roles
        nonlocal document_count, membership_count
        if current_id is None:
            return
        relationship_rows = relationship_context.get(current_id, Counter())
        relationship_summary = [
            {
                "direction": direction,
                "kind": kind,
                "neighbor_kind": neighbor_kind,
                "count": count,
            }
            for (direction, kind, neighbor_kind), count in sorted(relationship_rows.items())
        ]
        key = {"entity_id": current_id}
        aggregate = {
            "entity_kind": current_kind,
            "relation_counts": dict(sorted(relations.items())),
            "participant_role_counts": dict(sorted(roles.items())),
            "relationship_summary": relationship_summary,
        }
        relation_categories = [
            relation.removeprefix("ocsf_").replace("_", " ")
            for relation in sorted(relations)
        ]
        role_categories = sorted(
            {
                f"{_safe_graph_taxonomy(role, 'participant_role')}:{_magnitude_bucket(count)}"
                for role, count in roles.items()
            }
        )
        relationship_categories = sorted(
            {
                f"{row['direction']} "
                f"{_safe_graph_taxonomy(row['kind'], 'relationship_kind')} "
                f"{_safe_graph_taxonomy(row['neighbor_kind'], 'entity_kind')}"
                for row in relationship_summary
            }
        )
        semantic = (
            f"entity evidence summary | kind: {_safe_graph_taxonomy(current_kind, 'entity_kind')} | "
            f"event volume: {_magnitude_bucket(len({item[0] for item in members}))} | "
            f"roles: {', '.join(role_categories) or 'absent'} | "
            f"activity classes: {', '.join(relation_categories) or 'absent'} | "
            f"relationships: {', '.join(relationship_categories) or 'absent'}"
        )
        document = _document(
            kind="entity",
            relations=list(relations),
            semantic_text=semantic,
            facets={
                "entity_kind": [_safe_graph_taxonomy(current_kind, "entity_kind")],
                "participant_role": role_categories,
                "relation": relation_categories,
                "relationship_kind": relationship_categories,
            },
            group_key=key,
            aggregate_material=aggregate,
            members=members,
            source_snapshot=source_snapshot,
            base_pack=base_pack,
            policy=policy,
            completeness_state="complete_for_admitted_participant_graph",
        )
        writer.add(document, members, policy, entity_id=current_id)
        document_count += 1
        membership_count += len(members)
        current_id = None
        current_kind = None
        members = []
        relations = Counter()
        roles = Counter()

    for entity_id, entity_kind, occurrence_id, relation, role in _iter_rows(cursor):
        if current_id != entity_id:
            finish()
            current_id, current_kind = entity_id, entity_kind
        members.append((occurrence_id, role))
        relations[relation] += 1
        roles[role] += 1
    finish()

    graph_counts = {
        "entity_rows": int(connection.execute("SELECT count(*) FROM entities").fetchone()[0]),
        "participant_rows": int(connection.execute("SELECT count(*) FROM participants").fetchone()[0]),
        "relationship_rows": int(connection.execute("SELECT count(*) FROM relationships").fetchone()[0]),
    }
    orphan_count = int(
        connection.execute(
            "SELECT count(*) FROM entities e ANTI JOIN participants p USING(entity_id)"
        ).fetchone()[0]
    )
    event_without_participants = int(
        connection.execute(
            "SELECT count(*) FROM base_occurrences o ANTI JOIN participants p USING(event_id)"
        ).fetchone()[0]
    )
    coverage["families"]["entity"] = {
        "examined_source_record_count": coverage["closure"]["base_source_record_count"],
        "source_records_with_participants": coverage["closure"]["base_source_record_count"] - event_without_participants,
        "source_records_without_participants": event_without_participants,
        "document_count": document_count,
        "membership_count": membership_count,
        "orphan_entity_count": orphan_count,
        "graph_input_counts": graph_counts,
        "reason_counts": ({"orphan_entity_no_occurrence": orphan_count} if orphan_count else {}),
    }


def _build_from_connection(
    connection: Any,
    output_dir: Path,
    *,
    component_id: str,
    version: str,
    component_uri: str | None,
    source_snapshot: Mapping[str, str],
    base_pack: Mapping[str, str],
    auxiliary_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = derivation_policy_ref()
    _prepare_views(connection)
    base_count = int(connection.execute("SELECT count(*) FROM base_occurrences").fetchone()[0])
    typed_count = int(connection.execute("SELECT count(*) FROM typed_events").fetchone()[0])
    by_relation = {
        relation: int(count)
        for relation, count in connection.execute(
            "SELECT relation_name,count(*) FROM base_occurrences GROUP BY relation_name ORDER BY relation_name"
        ).fetchall()
    }
    missing_relations = set(RELATION_DOCUMENT_KINDS) - set(by_relation)
    if missing_relations and auxiliary_inputs:
        raise EvidenceDerivationError(
            f"receipt-bound base pack omits typed relations: {sorted(missing_relations)}"
        )
    coverage: dict[str, Any] = {
        "schema_version": "livefire.rag.evidence-derivation-coverage/1",
        "source_snapshot": dict(source_snapshot),
        "base_projection_pack": dict(base_pack),
        "derivation_policy": policy,
        "closure": {
            "base_source_record_count": base_count,
            "typed_derivation_input_count": typed_count,
            "unaccounted_typed_derivation_input_count": 0,
            "all_derivation_inputs_accounted": True,
        },
        "base_relation_counts": by_relation,
        "families": {},
    }
    writer = _OutputWriter(output_dir)
    try:
        _derive_metric(connection, writer, coverage, source_snapshot=source_snapshot, base_pack=base_pack, policy=policy)
        _derive_network(connection, writer, coverage, source_snapshot=source_snapshot, base_pack=base_pack, policy=policy)
        _derive_transitions(connection, writer, coverage, source_snapshot=source_snapshot, base_pack=base_pack, policy=policy)
        _derive_entities(connection, writer, coverage, source_snapshot=source_snapshot, base_pack=base_pack, policy=policy)
    finally:
        writer.close()
    coverage["closure"].update(
        {
            "derived_document_count": writer.document_count,
            "derivation_membership_count": writer.membership_count,
            "by_document_kind": dict(sorted(writer.by_kind.items())),
        }
    )
    write_canonical_json(output_dir / COVERAGE_NAME, coverage)
    artifacts = [
        artifact_ref(output_dir / DOCUMENTS_NAME, DOCUMENTS_NAME, "application/x-ndjson"),
        artifact_ref(output_dir / MEMBERSHIPS_NAME, MEMBERSHIPS_NAME, "application/x-ndjson"),
        artifact_ref(output_dir / COVERAGE_NAME, COVERAGE_NAME, "application/json"),
    ]
    artifacts.sort(key=lambda item: item["path"])
    write_canonical_json(
        output_dir / LOCK_NAME,
        {"schema_version": "livefire.object-lock/1", "objects": artifacts},
    )
    object_map = {Path(item["path"]).stem.replace("-", "_"): item for item in artifacts}
    component = {"id": component_id, "version": version, "sha256": "0" * 64}
    if component_uri is not None:
        component["uri"] = component_uri
    manifest = {
        "schema_version": "livefire.rag.evidence-derivation-pack/1",
        "component": component,
        "stage": "pre_embedding_derivation_overlay",
        "source_snapshot": dict(source_snapshot),
        "base_projection_pack": dict(base_pack),
        "derivation_policy": policy,
        "auxiliary_inputs": [dict(value) for value in sorted(auxiliary_inputs, key=lambda item: item["relation"])],
        "row_schemas": _derivation_row_schema_refs(),
        "physical_contract": {
            "documents_format": "canonical_jsonl",
            "memberships_format": "canonical_jsonl",
            "encoding": "utf-8",
            "line_termination": "lf",
            "document_order": "family_then_canonical_group_key",
            "membership_order": "document_emission_then_occurrence_id_then_input_role",
        },
        "objects": {
            "documents": next(item for item in artifacts if item["path"] == DOCUMENTS_NAME),
            "memberships": next(item for item in artifacts if item["path"] == MEMBERSHIPS_NAME),
            "coverage_report": next(item for item in artifacts if item["path"] == COVERAGE_NAME),
            "object_lock": artifact_ref(output_dir / LOCK_NAME, LOCK_NAME, "application/json"),
        },
        "closure": {
            "base_source_record_count": base_count,
            "derived_document_count": writer.document_count,
            "derivation_membership_count": writer.membership_count,
            "unresolved_membership_count": 0,
            "unaccounted_derivation_input_count": 0,
        },
    }
    manifest["component"]["sha256"] = derivation_manifest_identity(manifest)
    write_canonical_json(output_dir / MANIFEST_NAME, manifest)
    verify_evidence_derivation_pack(output_dir)
    return manifest


def _derivation_schema_root() -> Path:
    return Path(__file__).resolve().parents[2] / "specs"


def _derivation_row_schema_refs() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for logical, name in (
        ("derived_document", "evidence-derived-document.v1.schema.json"),
        ("derivation_membership", "evidence-derivation-membership-row.v1.schema.json"),
        ("derivation_coverage", "evidence-derivation-coverage.v1.schema.json"),
    ):
        path = _derivation_schema_root() / name
        schema = json.loads(path.read_text(encoding="utf-8"))
        result[logical] = component_ref(schema["$id"], "1", schema)
    return result


def _build_derivation_pack_for_test(
    output_dir: Path,
    *,
    typed_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    occurrences: Iterable[Mapping[str, Any]],
    participants: Iterable[Mapping[str, Any]],
    entities: Iterable[Mapping[str, Any]],
    relationships: Iterable[Mapping[str, Any]] = (),
    component_id: str,
    version: str,
    source_snapshot: Mapping[str, str],
    base_projection_pack: Mapping[str, str],
) -> dict[str, Any]:
    """Build an overlay from bounded fixtures using the production engine."""

    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite derivation pack: {out}")
    _validate_component(source_snapshot, "source_snapshot")
    _validate_component(base_projection_pack, "base_projection_pack")
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    connection = _connect()
    try:
        _create_fixture_tables(connection)
        _insert_fixture_rows(
            connection,
            typed_rows=typed_rows,
            occurrences=occurrences,
            participants=participants,
            entities=entities,
            relationships=relationships,
        )
        manifest = _build_from_connection(
            connection,
            staging,
            component_id=component_id,
            version=version,
            component_uri=None,
            source_snapshot=source_snapshot,
            base_pack=base_projection_pack,
            auxiliary_inputs=[],
        )
        os.rename(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()


def _load_receipt_inputs(snapshot_root: Path, receipt_path: Path) -> tuple[dict[str, str], list[AuxiliaryInput], dict[str, Path]]:
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    runnable = receipt.get("runnable_snapshot")
    component = runnable.get("component") if isinstance(runnable, Mapping) else None
    source_snapshot = _validate_component(component, "receipt runnable snapshot")
    objects = receipt.get("snapshot_manifest", {}).get("objects")
    if not isinstance(objects, list):
        raise EvidenceDerivationError("receipt lacks snapshot objects")
    required = set(RELATION_DOCUMENT_KINDS) | {"entities", "participants", "relationships"}
    found: dict[str, Mapping[str, Any]] = {}
    for row in objects:
        if isinstance(row, Mapping) and row.get("relation") in required:
            relation = str(row["relation"])
            if relation in found:
                raise EvidenceDerivationError(f"receipt duplicates relation: {relation}")
            found[relation] = row
    missing = required - set(found)
    if missing:
        raise EvidenceDerivationError(f"receipt omits derivation inputs: {sorted(missing)}")
    root = Path(snapshot_root).resolve()
    paths: dict[str, Path] = {}
    auxiliary: list[AuxiliaryInput] = []
    for relation in sorted(required):
        row = found[relation]
        relative = row.get("path")
        expected_sha = row.get("sha256")
        rows = row.get("rows")
        if not isinstance(relative, str) or not _is_sha256(expected_sha) or isinstance(rows, bool) or not isinstance(rows, int):
            raise EvidenceDerivationError(f"receipt relation metadata is invalid: {relation}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise EvidenceDerivationError(f"receipt path escapes snapshot: {relation}") from error
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise EvidenceDerivationError(f"receipt-bound object differs: {relation}")
        paths[relation] = path
        if relation in {"entities", "participants", "relationships"}:
            auxiliary.append(AuxiliaryInput(relation, path, expected_sha, rows))
    return source_snapshot, auxiliary, paths


def build_evidence_derivation_pack(
    output_dir: Path,
    *,
    snapshot_root: Path,
    receipt_path: Path,
    base_projection_pack: Path,
    component_id: str,
    version: str,
    component_uri: str | None = None,
) -> dict[str, Any]:
    """Build an immutable overlay from a receipt-fenced snapshot and base pack."""

    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite derivation pack: {out}")
    if not component_id or not version:
        raise ValueError("component_id and version must be non-empty")
    if component_uri is not None and not component_uri:
        raise ValueError("component_uri must be non-empty when supplied")
    source_snapshot, auxiliary, paths = _load_receipt_inputs(snapshot_root, receipt_path)
    # Verify the base component against the exact mounted snapshot rather than
    # trusting manifest bytes alone. Schema admission remains owned by the base
    # pack verifier/promoter; this call independently replays its source-bound
    # identity and projection closure.
    from .evidence_builder import _verify_evidence_pack
    from .evidence_projection import projection_policy_ref
    from .evidence_source import admit_typed_snapshot

    admitted = admit_typed_snapshot(Path(snapshot_root), Path(receipt_path))
    base_manifest = _verify_evidence_pack(
        Path(base_projection_pack),
        source_snapshot=admitted.component,
        relation_sources=admitted.relations,
        projection_policy=projection_policy_ref(),
        projector=None,
        trusted_builder=False,
    )
    base_component = _validate_component(base_manifest.get("component"), "base pack component")
    if base_manifest.get("source_snapshots") != [source_snapshot]:
        raise EvidenceDerivationError("base pack names a different source snapshot")
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    database = staging / "derivation.duckdb"
    connection = _connect()
    try:
        connection.execute(f"ATTACH '{str(database).replace("'", "''")}' AS staging")
        occurrence_path = str((Path(base_projection_pack) / "occurrences.jsonl").resolve())
        connection.execute(
            "CREATE TABLE base_occurrences AS SELECT "
            "json_extract_string(json,'$.source_pointer.record_id') AS event_id, "
            "json_extract_string(json,'$.occurrence_id') AS occurrence_id, "
            "json_extract_string(json,'$.relation_identity.relation') AS relation_name "
            "FROM read_ndjson_objects(?)",
            [occurrence_path],
        )
        typed_unions = []
        for relation in sorted(METRIC_RELATIONS | NETWORK_RELATIONS | STATE_RELATIONS):
            escaped = str(paths[relation]).replace("'", "''")
            typed_unions.append(
                f"SELECT event_id, '{relation}' AS relation_name, typed_event_json FROM read_parquet('{escaped}')"
            )
        connection.execute("CREATE TABLE typed_events AS " + " UNION ALL ".join(typed_unions))
        for name in ("participants", "entities", "relationships"):
            escaped = str(paths[name]).replace("'", "''")
            connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")
        manifest = _build_from_connection(
            connection,
            staging,
            component_id=component_id,
            version=version,
            component_uri=component_uri,
            source_snapshot=source_snapshot,
            base_pack=base_component,
            auxiliary_inputs=[
                {
                    "relation": item.relation,
                    "path": str(item.path.relative_to(Path(snapshot_root).resolve())),
                    "sha256": item.sha256,
                    "rows": item.rows,
                }
                for item in auxiliary
            ],
        )
        connection.close()
        connection = None
        database.unlink(missing_ok=True)
        os.rename(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceDerivationCorrupt(f"cannot read canonical JSON: {path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value, newline=True) != path.read_bytes():
        raise EvidenceDerivationCorrupt(f"object is not canonical JSON: {path}")
    return value


def verify_evidence_derivation_pack(root: Path) -> dict[str, Any]:
    """Verify artifact identity and internal document-membership closure."""

    pack = Path(root)
    manifest = _load_json(pack / MANIFEST_NAME)
    if manifest.get("schema_version") != "livefire.rag.evidence-derivation-pack/1":
        raise EvidenceDerivationCorrupt("unexpected derivation manifest schema")
    component = _validate_component(manifest.get("component"), "manifest component")
    if component["sha256"] != derivation_manifest_identity(manifest):
        raise EvidenceDerivationCorrupt("derivation manifest identity mismatch")
    if manifest.get("derivation_policy") != derivation_policy_ref():
        raise EvidenceDerivationCorrupt("derivation policy mismatch")
    lock = _load_json(pack / LOCK_NAME)
    expected_lock_rows = []
    for name, filename, media in (
        ("documents", DOCUMENTS_NAME, "application/x-ndjson"),
        ("memberships", MEMBERSHIPS_NAME, "application/x-ndjson"),
        ("coverage_report", COVERAGE_NAME, "application/json"),
    ):
        reference = artifact_ref(pack / filename, filename, media)
        if manifest.get("objects", {}).get(name) != reference:
            raise EvidenceDerivationCorrupt(f"manifest {name} artifact mismatch")
        expected_lock_rows.append(reference)
    expected_lock_rows.sort(key=lambda item: item["path"])
    if lock != {"schema_version": "livefire.object-lock/1", "objects": expected_lock_rows}:
        raise EvidenceDerivationCorrupt("derivation object lock mismatch")
    lock_ref = artifact_ref(pack / LOCK_NAME, LOCK_NAME, "application/json")
    if manifest.get("objects", {}).get("object_lock") != lock_ref:
        raise EvidenceDerivationCorrupt("manifest object-lock reference mismatch")

    documents: dict[str, dict[str, Any]] = {}
    with (pack / DOCUMENTS_NAME).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceDerivationCorrupt(f"invalid document JSONL line {line_number}") from error
            if canonical_json_bytes(value, newline=True) != line:
                raise EvidenceDerivationCorrupt(f"noncanonical document line {line_number}")
            document_id = value.get("document_id")
            if not isinstance(document_id, str) or document_id in documents:
                raise EvidenceDerivationCorrupt("duplicate or invalid derived document id")
            if value.get("document_sha256") != canonical_sha256_omitting(value, ("document_sha256",)):
                raise EvidenceDerivationCorrupt(f"derived document digest mismatch: {document_id}")
            if value.get("source_snapshot") != manifest.get("source_snapshot"):
                raise EvidenceDerivationCorrupt(f"derived document snapshot mismatch: {document_id}")
            if value.get("base_projection_pack") != manifest.get("base_projection_pack"):
                raise EvidenceDerivationCorrupt(f"derived document base-pack mismatch: {document_id}")
            if value.get("derivation_policy") != derivation_policy_ref():
                raise EvidenceDerivationCorrupt(f"derived document policy mismatch: {document_id}")
            derivation = value.get("derivation")
            if not isinstance(derivation, Mapping):
                raise EvidenceDerivationCorrupt(f"derived document lacks derivation material: {document_id}")
            identity = {
                "schema_version": DERIVATION_SCHEMA_VERSION,
                "source_snapshot": value["source_snapshot"],
                "base_projection_pack": value["base_projection_pack"],
                "derivation_policy": value["derivation_policy"],
                "document_kind": value.get("document_kind"),
                "group_sha256": derivation.get("group_sha256"),
                "aggregate_material": derivation.get("aggregate_material"),
                "input_set_sha256": derivation.get("input_set_sha256"),
            }
            if document_id != "ddoc-" + sha256_bytes(canonical_json_bytes(identity)):
                raise EvidenceDerivationCorrupt(f"derived document identity mismatch: {document_id}")
            documents[document_id] = value
    members_by_document: dict[str, list[tuple[str, str]]] = defaultdict(list)
    membership_count = 0
    with (pack / MEMBERSHIPS_NAME).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceDerivationCorrupt(f"invalid membership JSONL line {line_number}") from error
            if canonical_json_bytes(value, newline=True) != line:
                raise EvidenceDerivationCorrupt(f"noncanonical membership line {line_number}")
            document_id = value.get("derived_document_id")
            occurrence_id = value.get("occurrence_id")
            role = value.get("input_role")
            if document_id not in documents or not isinstance(occurrence_id, str) or not isinstance(role, str):
                raise EvidenceDerivationCorrupt("unresolved derivation membership")
            if value.get("derivation_policy") != derivation_policy_ref():
                raise EvidenceDerivationCorrupt("membership policy mismatch")
            entity_id = value.get("entity_id")
            if documents[document_id].get("document_kind") == "entity":
                if not isinstance(entity_id, str) or not entity_id:
                    raise EvidenceDerivationCorrupt(
                        "entity-document membership lacks canonical entity_id"
                    )
            elif entity_id is not None:
                raise EvidenceDerivationCorrupt(
                    "non-entity membership carries canonical entity_id"
                )
            if value.get("membership_sha256") != canonical_sha256_omitting(value, ("membership_sha256",)):
                raise EvidenceDerivationCorrupt("membership digest mismatch")
            identity = {
                "schema_version": "livefire.rag.evidence-derivation-membership-identity/1",
                "derived_document_id": document_id,
                "occurrence_id": occurrence_id,
                "input_role": role,
                "derivation_policy": derivation_policy_ref(),
            }
            if "entity_id" in value:
                identity["entity_id"] = value["entity_id"]
            if value.get("membership_id") != "dmem-" + sha256_bytes(canonical_json_bytes(identity)):
                raise EvidenceDerivationCorrupt("membership identity mismatch")
            members_by_document[document_id].append((occurrence_id, role))
            membership_count += 1
    for document_id, document in documents.items():
        members = members_by_document.get(document_id, [])
        if len(members) != len(set(members)):
            raise EvidenceDerivationCorrupt(f"duplicate membership for {document_id}")
        derivation = document.get("derivation", {})
        if derivation.get("input_count") != len(members):
            raise EvidenceDerivationCorrupt(f"input count mismatch for {document_id}")
        if derivation.get("input_set_sha256") != _membership_set_digest(members):
            raise EvidenceDerivationCorrupt(f"input-set digest mismatch for {document_id}")
        if document.get("occurrence_count") != len({item[0] for item in members}):
            raise EvidenceDerivationCorrupt(f"occurrence count mismatch for {document_id}")
    coverage = _load_json(pack / COVERAGE_NAME)
    if manifest.get("closure", {}).get("derived_document_count") != len(documents):
        raise EvidenceDerivationCorrupt("manifest derived-document closure mismatch")
    if manifest.get("closure", {}).get("derivation_membership_count") != membership_count:
        raise EvidenceDerivationCorrupt("manifest membership closure mismatch")
    if coverage.get("closure", {}).get("derived_document_count") != len(documents):
        raise EvidenceDerivationCorrupt("coverage derived-document closure mismatch")
    if coverage.get("closure", {}).get("derivation_membership_count") != membership_count:
        raise EvidenceDerivationCorrupt("coverage membership closure mismatch")
    base_count = manifest.get("closure", {}).get("base_source_record_count")
    relation_counts = coverage.get("base_relation_counts")
    families = coverage.get("families")
    if not isinstance(relation_counts, Mapping) or sum(relation_counts.values()) != base_count:
        raise EvidenceDerivationCorrupt("base relation-count closure mismatch")
    if not isinstance(families, Mapping):
        raise EvidenceDerivationCorrupt("derivation family coverage is unavailable")
    by_kind = Counter(document["document_kind"] for document in documents.values())
    member_by_kind = Counter()
    for document_id, members in members_by_document.items():
        member_by_kind[documents[document_id]["document_kind"]] += len(members)
    for family_name in ("metric_window", "network_window", "state_transition", "entity"):
        family = families.get(family_name)
        if not isinstance(family, Mapping):
            raise EvidenceDerivationCorrupt(f"missing family coverage: {family_name}")
        if family.get("document_count") != by_kind[family_name]:
            raise EvidenceDerivationCorrupt(f"family document closure mismatch: {family_name}")
        if family.get("membership_count") != member_by_kind[family_name]:
            raise EvidenceDerivationCorrupt(f"family membership closure mismatch: {family_name}")
    for family_name in ("metric_window", "network_window", "state_transition"):
        family = families[family_name]
        if family.get("applicable_source_record_count") + family.get(
            "not_applicable_source_record_count"
        ) != base_count:
            raise EvidenceDerivationCorrupt(f"family applicability closure mismatch: {family_name}")
    for family_name in ("metric_window", "network_window"):
        family = families[family_name]
        if family.get("eligible_source_record_count") + family.get(
            "ineligible_source_record_count"
        ) != family.get("applicable_source_record_count"):
            raise EvidenceDerivationCorrupt(f"family eligibility closure mismatch: {family_name}")
    entity = families["entity"]
    if entity.get("source_records_with_participants") + entity.get(
        "source_records_without_participants"
    ) != base_count:
        raise EvidenceDerivationCorrupt("entity event-coverage closure mismatch")
    return manifest


__all__ = [
    "DERIVATION_SCHEMA_VERSION",
    "EvidenceDerivationCorrupt",
    "EvidenceDerivationError",
    "build_evidence_derivation_pack",
    "derivation_manifest_identity",
    "derivation_policy_material",
    "derivation_policy_ref",
    "verify_evidence_derivation_pack",
]
