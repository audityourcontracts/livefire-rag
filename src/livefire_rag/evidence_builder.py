"""Streaming builder for an immutable, occurrence-complete evidence pack.

The builder deliberately stops before embedding.  It turns every input row into
one terminal occurrence receipt and stores one canonical document for each
semantic group emitted by :mod:`livefire_rag.evidence_projection`.  A temporary
SQLite database provides disk-backed uniqueness checks and external ordering;
Parquet input is scanned in bounded batches by DuckDB.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)


DOCUMENTS_NAME = "documents.jsonl"
OCCURRENCES_NAME = "occurrences.jsonl"
COVERAGE_NAME = "coverage-report.json"
LOCK_NAME = "objects.lock.json"
MANIFEST_NAME = "manifest.json"

SHA256_LENGTH = 64
DOCUMENT_DISPOSITIONS = {
    "direct_semantic_document",
    "semantic_group_occurrence",
    "derived_document_input",
}
TERMINAL_DISPOSITIONS = DOCUMENT_DISPOSITIONS | {
    "structured_only",
    "structured_only_occurrence",
    "rejected",
}
PACK_DISPOSITIONS = (
    "direct_semantic_document",
    "semantic_group_occurrence",
    "derived_document_input",
    "structured_only_occurrence",
    "rejected",
)
DOCUMENT_KINDS = (
    "activity",
    "state",
    "state_transition",
    "metric_window",
    "network_window",
    "entity",
    "detection",
    "structured_only",
)
POINTER_SCHEMA_REF = {
    "id": "https://livefire.dev/sdk/source-record-pointer.v1.schema.json",
    "version": "1",
    "sha256": "f4bcd1363d4361e8358ee958e3ed0606cef8c7ee73187a9214aee2dbc60de816",
}


def _row_schema_refs() -> dict[str, dict[str, str]]:
    """Bind the exact generic row schemas shipped with this installation."""

    from .evidence_schema import generic_schema_root

    root = generic_schema_root()
    bindings = {
        "evidence_document": "evidence-document.v1.schema.json",
        "evidence_occurrence": "evidence-occurrence-row.v1.schema.json",
        "coverage_report": "evidence-coverage-report.v1.schema.json",
    }
    result: dict[str, dict[str, str]] = {}
    for logical_name, filename in bindings.items():
        value = json.loads((root / filename).read_text(encoding="utf-8"))
        result[logical_name] = {
            "id": value["$id"],
            "version": "1",
            "sha256": sha256_bytes(canonical_json_bytes(value)),
        }
    return result


class EvidencePackError(RuntimeError):
    """Base class for evidence pack build and verification failures."""


class EvidencePackCorrupt(EvidencePackError):
    """An evidence pack fails its content, identity, or closure contract."""


@dataclass(frozen=True)
class RelationSource:
    """A named typed OCSF Parquet relation."""

    relation_name: str
    path: Path
    expected_sha256: str | None = None
    expected_rows: int | None = None


Projector = Callable[[str, str, Any, str], Mapping[str, Any]]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_component(value: Any, label: str) -> dict[str, str]:
    required = {"id", "version", "sha256"}
    allowed = required | {"uri"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - allowed:
        raise ValueError(f"{label} must be a closed component reference")
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValueError(f"{label}.id is invalid")
    if not isinstance(value["version"], str) or not value["version"]:
        raise ValueError(f"{label}.version is invalid")
    if not _is_sha256(value["sha256"]):
        raise ValueError(f"{label}.sha256 is invalid")
    if "uri" in value and (not isinstance(value["uri"], str) or not value["uri"]):
        raise ValueError(f"{label}.uri is invalid")
    return value


def evidence_manifest_identity(manifest: Mapping[str, Any]) -> str:
    """Return the normative identity of a manifest, omitting its self-digest."""

    return canonical_sha256_omitting(dict(manifest), ("component", "sha256"))


def _default_projector() -> Projector:
    try:
        from .evidence_projection import project_event
    except ImportError as error:  # pragma: no cover - exercised only in partial installs
        raise EvidencePackError(
            "livefire_rag.evidence_projection.project_event is unavailable; "
            "pass an explicit projector"
        ) from error
    return project_event


def _normalise_relation_sources(
    relations: Mapping[str, Path | str] | Sequence[RelationSource | Path | str | tuple[str, Path | str]] | None,
) -> list[RelationSource]:
    if relations is None:
        return []
    if isinstance(relations, Mapping):
        sources = [RelationSource(str(name), Path(path)) for name, path in relations.items()]
    else:
        sources = []
        for item in relations:
            if isinstance(item, RelationSource):
                source = item
            elif isinstance(item, tuple) and len(item) == 2:
                source = RelationSource(str(item[0]), Path(item[1]))
            else:
                path = Path(item)
                source = RelationSource(path.stem, path)
            sources.append(source)
    if any(not source.relation_name for source in sources):
        raise ValueError("relation names must not be empty")
    names = [source.relation_name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("relation names must be unique")
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"relation does not exist: {source.path}")
    return sorted(sources, key=lambda source: source.relation_name)


def _iter_parquet(source: RelationSource, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise EvidencePackError(
            "DuckDB is required for Parquet input; install livefire-rag[prototype]"
        ) from error
    path = source.path
    before_stat = path.stat()
    object_sha256 = sha256_file(path)
    if source.expected_sha256 is not None and object_sha256 != source.expected_sha256:
        raise EvidencePackError(f"receipt-fenced object digest mismatch: {source.relation_name}")
    connection = duckdb.connect()
    try:
        row_groups = connection.execute(
            "SELECT row_group_id, row_group_num_rows FROM parquet_metadata(?) "
            "GROUP BY ALL ORDER BY row_group_id",
            [str(path.resolve())],
        ).fetchall()
        if row_groups and [int(row[0]) for row in row_groups] != list(range(len(row_groups))):
            raise EvidencePackError(f"Parquet row groups are not contiguous: {path}")
        boundaries: list[int] = []
        cumulative = 0
        for _, row_count in row_groups:
            cumulative += int(row_count)
            boundaries.append(cumulative)
        if source.expected_rows is not None and cumulative != source.expected_rows:
            raise EvidencePackError(
                f"receipt-fenced row-count mismatch for {source.relation_name}: "
                f"{cumulative} != {source.expected_rows}"
            )
        cursor = connection.execute(
            "SELECT event_id, typed_event_json, support_ref, file_row_number "
            "FROM read_parquet(?, file_row_number=true)",
            [str(path.resolve())],
        )
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            if not boundaries:
                raise EvidencePackError(
                    f"Parquet returned rows without row-group metadata: {path}"
                )
            for event_id, typed_event_json, support_ref, file_row_number in batch:
                ordinal = int(file_row_number)
                row_group = bisect_right(boundaries, ordinal)
                group_start = 0 if row_group == 0 else boundaries[row_group - 1]
                yield {
                    "event_id": event_id,
                    "typed_event_json": typed_event_json,
                    "support_ref": support_ref,
                    "source_object_sha256": object_sha256,
                    "row_group": row_group,
                    "row_ordinal": ordinal - group_start,
                }
    finally:
        connection.close()
        after_stat = path.stat()
        if (
            before_stat.st_dev != after_stat.st_dev
            or before_stat.st_ino != after_stat.st_ino
            or before_stat.st_size != after_stat.st_size
            or before_stat.st_mtime_ns != after_stat.st_mtime_ns
            or sha256_file(path) != object_sha256
        ):
            raise EvidencePackError(
                f"source object changed while scanning: {source.relation_name}"
            )


def _iter_source_rows(
    sources: Sequence[RelationSource],
    row_sources: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    batch_size: int,
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for source in sources:
        for row in _iter_parquet(source, batch_size):
            yield source.relation_name, row
    if row_sources:
        overlap = {source.relation_name for source in sources} & set(row_sources)
        if overlap:
            raise ValueError(f"relations supplied twice: {sorted(overlap)}")
        for relation_name in sorted(row_sources):
            if not relation_name:
                raise ValueError("relation names must not be empty")
            for row in row_sources[relation_name]:
                if not isinstance(row, Mapping):
                    raise ValueError(f"row in {relation_name} is not a mapping")
                yield relation_name, row


def _extract_input_row(relation_name: str, row: Mapping[str, Any]) -> tuple[str, Any, str]:
    missing = {"event_id", "typed_event_json", "support_ref"} - set(row)
    if missing:
        raise ValueError(f"{relation_name} row lacks fields: {sorted(missing)}")
    event_id = row["event_id"]
    support_ref = row["support_ref"]
    typed_event_json = row["typed_event_json"]
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(f"{relation_name} event_id is invalid")
    if not isinstance(support_ref, str) or not support_ref:
        raise ValueError(f"{relation_name} support_ref is invalid")
    return event_id, typed_event_json, support_ref


def _typed_event_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return canonical_json_bytes(dict(value) if isinstance(value, Mapping) else value)


def source_record_profile_material() -> dict[str, str]:
    """Return the exact immutable typed-Parquet pointer profile material."""

    return {
        "schema_version": "livefire.rag.typed-parquet-record-profile/1",
        "record_digest": "sha256_of_exact_stored_typed_event_json_utf8_or_jcs_value",
        "locator": "parquet_row",
        "row_ordinal": "zero_based_within_row_group",
    }


def source_record_profile_ref() -> dict[str, str]:
    """Return the SDK component identity for typed Parquet source pointers."""

    material = source_record_profile_material()
    return {
        "id": "livefire.rag.typed-parquet-record-profile",
        "version": "1",
        "sha256": sha256_bytes(canonical_json_bytes(material)),
    }


def _source_pointer(
    row: Mapping[str, Any],
    *,
    relation_name: str,
    event_id: str,
    support_ref: str,
    record_sha256: str,
    source_snapshot: dict[str, str],
) -> dict[str, Any]:
    supplied = row.get("source_pointer")
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("source_pointer must be a mapping")
        pointer = dict(supplied)
        if pointer.get("record_id") != event_id or pointer.get("record_sha256") != record_sha256:
            raise ValueError("supplied source_pointer does not bind the input record")
        if pointer.get("snapshot") != source_snapshot:
            raise ValueError("supplied source_pointer names a different snapshot")
        locator = pointer.get("locator")
        if not isinstance(locator, Mapping) or locator.get("kind") != "parquet_row":
            raise ValueError("projection packs require an exact parquet_row locator")
        return pointer
    object_sha256 = row.get("source_object_sha256")
    row_group = row.get("row_group")
    row_ordinal = row.get("row_ordinal")
    if (
        not _is_sha256(object_sha256)
        or isinstance(row_group, bool)
        or not isinstance(row_group, int)
        or row_group < 0
        or isinstance(row_ordinal, bool)
        or not isinstance(row_ordinal, int)
        or row_ordinal < 0
    ):
        raise ValueError(
            "input row lacks exact source admission metadata; provide source_pointer or "
            "source_object_sha256/row_group/row_ordinal"
        )
    return {
        "schema_version": "livefire.source-record-pointer/1",
        "snapshot": source_snapshot,
        "snapshot_profile": source_record_profile_ref(),
        "record_id": event_id,
        "record_sha256": record_sha256,
        "locator": {
            "kind": "parquet_row",
            "object_sha256": object_sha256,
            "row_group": row_group,
            "row_ordinal": row_ordinal,
            "relation": relation_name,
        },
        "support_refs": [support_ref],
    }


def _normalise_projection(
    projected: Mapping[str, Any],
    *,
    typed_event_json: Any,
    relation_name: str,
    event_id: str,
    support_ref: str,
    record_sha256: str,
    source_pointer: dict[str, Any],
    projection_policy: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(projected, Mapping):
        raise ValueError("project_event must return a mapping")
    disposition = projected.get("terminal_disposition")
    if disposition not in TERMINAL_DISPOSITIONS:
        raise ValueError(f"project_event returned invalid terminal disposition: {disposition!r}")
    for field, expected in (
        ("relation_name", relation_name),
        ("event_id", event_id),
        ("support_ref", support_ref),
    ):
        if projected.get(field) != expected:
            raise ValueError(f"project_event changed authoritative {field}")
    projection_sha256 = projected.get("projection_sha256")
    semantic_group_sha256 = projected.get("semantic_group_sha256")
    if not _is_sha256(projection_sha256) or not _is_sha256(semantic_group_sha256):
        raise ValueError("project_event returned an invalid projection/group digest")

    pack_disposition = (
        "semantic_group_occurrence"
        if disposition in DOCUMENT_DISPOSITIONS
        else "structured_only_occurrence"
        if disposition in {"structured_only", "structured_only_occurrence"}
        else "rejected"
    )
    document_id: str | None = None
    document: dict[str, Any] | None = None
    if disposition in DOCUMENT_DISPOSITIONS:
        document_id = f"doc-{semantic_group_sha256}"
        text_fields: dict[str, str] = {}
        for field in ("semantic_text", "action_text", "target_text", "context_text", "outcome_text"):
            value = projected.get(field)
            if not isinstance(value, str):
                raise ValueError(f"project_event returned invalid {field}")
            text_fields[field] = value
        document_kind = projected.get("document_kind")
        if not isinstance(document_kind, str) or not document_kind:
            raise ValueError("project_event returned invalid document_kind")
        kind_mapping = {
            "activity": "activity",
            "state": "state",
            "detection": "detection",
        }
        if document_kind not in kind_mapping:
            raise ValueError(f"project_event returned non-searchable document kind: {document_kind}")
        facets = [
            {"name": name, "values": [text]}
            for name, text in (
                ("action", text_fields["action_text"]),
                ("target", text_fields["target_text"]),
                ("context", text_fields["context_text"]),
                ("outcome", text_fields["outcome_text"]),
            )
            if text
        ]
        metadata = projected.get("event_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("project_event metadata must be a mapping")
        relation_identity: dict[str, Any] = {"namespace": "ocsf", "relation": relation_name}
        for source_name, target_name in (
            ("class_uid", "ocsf_class_uid"),
            ("category_uid", "ocsf_category_uid"),
            ("activity_id", "ocsf_activity_id"),
            ("activity_name", "ocsf_activity_name"),
            ("semantic_class", "ocsf_class_name"),
        ):
            value = metadata.get(source_name)
            if value is not None:
                relation_identity[target_name] = value
        document = {
            "schema_version": "livefire.rag.evidence-document/1",
            "document_id": document_id,
            "document_sha256": "",
            "document_kind": kind_mapping[document_kind],
            "representation": "semantic_group",
            "searchable": True,
            "projection_policy": projection_policy,
            "relation_identities": [relation_identity],
            "semantic_projection": {"text": text_fields["semantic_text"], "facets": facets},
            "semantic_group": {
                "group_id": projected.get("semantic_group_id", f"sha256:{semantic_group_sha256}"),
                "group_key_sha256": semantic_group_sha256,
            },
        }

    occurrence_material = {
        "schema_version": "livefire.rag.evidence-occurrence-identity/1",
        "source_pointer": source_pointer,
    }
    occurrence_id = f"occ-{sha256_bytes(canonical_json_bytes(occurrence_material))}"
    reason = projected.get("disposition_reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("project_event returned invalid disposition_reason")
    event_metadata = projected.get("event_metadata")
    exact_attributes = projected.get("exact_attributes")
    exact_attribute_metadata = projected.get("exact_attribute_metadata")
    if not isinstance(event_metadata, Mapping):
        raise ValueError("project_event metadata must be a mapping")
    if not isinstance(exact_attributes, list) or not isinstance(
        exact_attribute_metadata, Mapping
    ):
        raise ValueError(
            "project_event exact_attributes and exact_attribute_metadata are required"
        )
    if isinstance(typed_event_json, str):
        try:
            typed_event = json.loads(typed_event_json)
        except json.JSONDecodeError:
            typed_event = None
    elif isinstance(typed_event_json, Mapping):
        typed_event = typed_event_json
    else:
        typed_event = None

    def resolve_exact_pointer(pointer: str) -> Any:
        if not pointer.startswith("/"):
            raise ValueError("project_event exact attribute path must be an RFC 6901 pointer")
        current = typed_event
        for encoded in pointer[1:].split("/"):
            for offset, character in enumerate(encoded):
                if character == "~" and (
                    offset + 1 >= len(encoded) or encoded[offset + 1] not in {"0", "1"}
                ):
                    raise ValueError(
                        "project_event exact attribute path has an invalid RFC 6901 escape"
                    )
            segment = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping):
                if segment not in current:
                    raise ValueError("project_event exact attribute path does not resolve")
                current = current[segment]
            elif isinstance(current, list):
                if not segment.isdigit() or (len(segment) > 1 and segment.startswith("0")):
                    raise ValueError("project_event exact attribute array index is invalid")
                offset = int(segment)
                if offset >= len(current):
                    raise ValueError("project_event exact attribute array index is out of range")
                current = current[offset]
            else:
                raise ValueError("project_event exact attribute path crosses a scalar")
        return current

    previous_path: str | None = None
    for index, attribute in enumerate(exact_attributes):
        if not isinstance(attribute, Mapping) or set(attribute) != {
            "namespace",
            "path",
            "value",
        }:
            raise ValueError(f"project_event exact_attributes[{index}] is not closed")
        namespace = attribute.get("namespace")
        path = attribute.get("path")
        value = attribute.get("value")
        if (
            namespace != "ocsf"
            or not isinstance(path, str)
            or not path
            or len(path) > 1_024
        ):
            raise ValueError(f"project_event exact_attributes[{index}] identity is invalid")
        if previous_path is not None and path <= previous_path:
            raise ValueError("project_event exact_attributes must be path-sorted and unique")
        previous_path = path
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            if abs(value) > (1 << 53) - 1:
                raise ValueError("project_event exact integer exceeds the JCS-safe range")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("project_event exact number must be finite")
        elif isinstance(value, str):
            if len(value.encode("utf-8")) > 1_024:
                raise ValueError("project_event exact string exceeds the policy byte bound")
        else:
            raise ValueError("project_event exact value must be a non-null JSON scalar")
        source_value = resolve_exact_pointer(path)
        if type(source_value) is not type(value) or source_value != value:
            raise ValueError(
                "project_event exact attribute differs from the typed JSON scalar"
            )
    selected_count = exact_attribute_metadata.get("selected_count")
    hydration_required = exact_attribute_metadata.get("source_hydration_required")
    if selected_count != len(exact_attributes) or not isinstance(hydration_required, bool):
        raise ValueError("project_event exact attribute accounting is inconsistent")
    exact_attribute_projection = dict(exact_attribute_metadata)
    reason_codes = [reason]
    if hydration_required:
        reason_codes.append("exact_attribute_subset_requires_source_hydration")
    occurrence = {
        "schema_version": "livefire.rag.evidence-occurrence-row/1",
        "occurrence_id": occurrence_id,
        "relation_identity": {"namespace": "ocsf", "relation": relation_name},
        "source_pointer": source_pointer,
        "projection_policy": projection_policy,
        "terminal_disposition": pack_disposition,
        "document_ids": [document_id] if document_id else [],
        "reason_codes": reason_codes,
        "exact_attributes": exact_attributes,
        "exact_attribute_projection": exact_attribute_projection,
    }
    event_time = event_metadata.get("event_time")
    if event_metadata.get("event_time_availability") == "available" and isinstance(event_time, str):
        occurrence["event_time"] = event_time
    if pack_disposition == "semantic_group_occurrence":
        occurrence["semantic_group_id"] = projected.get(
            "semantic_group_id", f"sha256:{semantic_group_sha256}"
        )
    return document, occurrence


def _staging_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE documents (document_id TEXT PRIMARY KEY, payload BLOB NOT NULL, occurrence_count INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE occurrences (occurrence_id TEXT PRIMARY KEY, document_id TEXT, relation_name TEXT NOT NULL, "
        "disposition TEXT NOT NULL, payload BLOB NOT NULL)"
    )
    return connection


def _insert_projection(
    connection: sqlite3.Connection,
    document: dict[str, Any] | None,
    occurrence: dict[str, Any],
) -> None:
    document_ids = occurrence["document_ids"]
    document_id = document_ids[0] if document_ids else None
    if document is not None:
        payload = canonical_json_bytes(document)
        cursor = connection.execute(
            "INSERT INTO documents(document_id, payload, occurrence_count) VALUES (?, ?, 1) "
            "ON CONFLICT(document_id) DO UPDATE SET occurrence_count = occurrence_count + 1 "
            "WHERE payload = excluded.payload",
            (document_id, payload),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"semantic group collision for {document_id}")
    try:
        connection.execute(
            "INSERT INTO occurrences(occurrence_id, document_id, relation_name, disposition, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                occurrence["occurrence_id"],
                document_id,
                occurrence["relation_identity"]["relation"],
                occurrence["terminal_disposition"],
                canonical_json_bytes(occurrence),
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(f"duplicate occurrence identity: {occurrence['occurrence_id']}") from error


def _write_staged_jsonl(connection: sqlite3.Connection, staging: Path) -> tuple[int, int]:
    document_count = 0
    with (staging / DOCUMENTS_NAME).open("wb") as handle:
        cursor = connection.execute(
            "SELECT payload, occurrence_count FROM documents ORDER BY document_id"
        )
        for payload, occurrence_count in cursor:
            document = json.loads(bytes(payload))
            document["occurrence_count"] = occurrence_count
            document["document_sha256"] = canonical_sha256_omitting(
                document, ("document_sha256",)
            )
            handle.write(canonical_json_bytes(document, newline=True))
            document_count += 1
    occurrence_count = 0
    with (staging / OCCURRENCES_NAME).open("wb") as handle:
        cursor = connection.execute("SELECT payload FROM occurrences ORDER BY occurrence_id")
        for (payload,) in cursor:
            handle.write(bytes(payload) + b"\n")
            occurrence_count += 1
    return document_count, occurrence_count


def _build_evidence_pack_for_test(
    out_dir: Path,
    relations: Mapping[str, Path | str] | Sequence[RelationSource | Path | str | tuple[str, Path | str]] | None = None,
    *,
    row_sources: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    index_id: str,
    version: str,
    source_snapshot: dict[str, str],
    projection_policy: dict[str, str],
    index_uri: str | None = None,
    projector: Projector | None = None,
    batch_size: int = 4096,
    temp_directory: Path | None = None,
) -> dict[str, Any]:
    """Internal build engine, including fixture-only row/projector hooks.

    Public callers use :func:`build_evidence_pack`, which accepts only
    receipt-fenced Parquet relations and the built-in generic projector. The
    row and projector hooks remain private so fixture packs cannot be mistaken
    for independently source-resolved SDK artifacts.
    """

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence pack: {out_dir}")
    if not isinstance(index_id, str) or not index_id or not isinstance(version, str) or not version:
        raise ValueError("index_id and version must be non-empty strings")
    if index_uri is not None and (not isinstance(index_uri, str) or not index_uri):
        raise ValueError("index_uri must be a non-empty string when supplied")
    _validate_component(source_snapshot, "source_snapshot")
    _validate_component(projection_policy, "projection_policy")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    sources = _normalise_relation_sources(relations)
    if not sources and not row_sources:
        raise ValueError("at least one relation is required")
    if row_sources is not None and any(not isinstance(name, str) for name in row_sources):
        raise ValueError("row source relation names must be strings")
    project = projector or _default_projector()

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(temp_directory) if temp_directory else out_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=temp_parent))
    connection: sqlite3.Connection | None = None
    relation_inputs: Counter[str] = Counter()
    relation_dispositions: dict[str, Counter[str]] = defaultdict(Counter)
    total_dispositions: Counter[str] = Counter()
    reason_counts: Counter[tuple[str, str]] = Counter()
    document_kinds: Counter[str] = Counter()
    try:
        connection = _staging_database(staging / "build.sqlite3")
        for relation_name in sorted(
            {source.relation_name for source in sources} | set(row_sources or {})
        ):
            relation_inputs[relation_name] = 0
        pending = 0
        for relation_name, row in _iter_source_rows(sources, row_sources, batch_size):
            event_id, typed_event_json, support_ref = _extract_input_row(relation_name, row)
            record_bytes = _typed_event_bytes(typed_event_json)
            projected = project(relation_name, event_id, typed_event_json, support_ref)
            pointer = _source_pointer(
                row,
                relation_name=relation_name,
                event_id=event_id,
                support_ref=support_ref,
                record_sha256=sha256_bytes(record_bytes),
                source_snapshot=source_snapshot,
            )
            document, occurrence = _normalise_projection(
                projected,
                typed_event_json=typed_event_json,
                relation_name=relation_name,
                event_id=event_id,
                support_ref=support_ref,
                record_sha256=sha256_bytes(record_bytes),
                source_pointer=pointer,
                projection_policy=projection_policy,
            )
            _insert_projection(connection, document, occurrence)
            relation_inputs[relation_name] += 1
            relation_dispositions[relation_name][occurrence["terminal_disposition"]] += 1
            total_dispositions[occurrence["terminal_disposition"]] += 1
            for occurrence_reason in occurrence["reason_codes"]:
                reason_counts[(occurrence["terminal_disposition"], occurrence_reason)] += 1
            pending += 1
            if pending >= batch_size:
                connection.commit()
                pending = 0
        connection.commit()

        document_count, occurrence_count = _write_staged_jsonl(connection, staging)
        input_count = sum(relation_inputs.values())
        if occurrence_count != input_count:
            raise EvidencePackError(
                f"closure failure: {input_count} inputs produced {occurrence_count} occurrences"
            )
        relation_names = sorted(relation_inputs)
        by_relation = []
        for relation_name in relation_names:
            count = relation_inputs[relation_name]
            by_relation.append(
                {
                    "relation_identity": {"namespace": "ocsf", "relation": relation_name},
                    "source_record_count": count,
                    "terminal_disposition_count": count,
                    "by_terminal_disposition": {
                        name: relation_dispositions[relation_name].get(name, 0)
                        for name in PACK_DISPOSITIONS
                    },
                }
            )
        for (payload,) in connection.execute("SELECT payload FROM documents"):
            document_kinds[json.loads(bytes(payload))["document_kind"]] += 1
        disposition_totals = {name: total_dispositions.get(name, 0) for name in PACK_DISPOSITIONS}
        coverage = {
            "schema_version": "livefire.rag.evidence-coverage-report/1",
            "source_snapshots": [source_snapshot],
            "projection_policy": projection_policy,
            "derivation_policies": [],
            "closure": {
                "source_record_count": input_count,
                "terminal_disposition_count": occurrence_count,
                "unaccounted_record_count": 0,
                "multiply_dispositioned_record_count": 0,
                "all_source_records_dispositioned": True,
                "by_terminal_disposition": disposition_totals,
            },
            "documents": {
                "total": document_count,
                "searchable": document_count,
                "by_kind": {name: document_kinds.get(name, 0) for name in DOCUMENT_KINDS},
            },
            "relation_coverage": by_relation,
            "pointer_resolution": {
                "pointer_count": occurrence_count,
                "resolved_count": occurrence_count,
                "unresolved_count": 0,
                "all_pointers_resolved": True,
            },
            "reason_counts": [
                {"terminal_disposition": disposition, "reason_code": reason, "count": count}
                for (disposition, reason), count in sorted(reason_counts.items())
            ],
        }
        write_canonical_json(staging / COVERAGE_NAME, coverage)
        artifacts = [
            artifact_ref(staging / DOCUMENTS_NAME, DOCUMENTS_NAME, "application/x-ndjson"),
            artifact_ref(staging / OCCURRENCES_NAME, OCCURRENCES_NAME, "application/x-ndjson"),
            artifact_ref(staging / COVERAGE_NAME, COVERAGE_NAME, "application/json"),
        ]
        artifacts.sort(key=lambda artifact: artifact["path"])
        object_lock = {"schema_version": "livefire.object-lock/1", "objects": artifacts}
        write_canonical_json(staging / LOCK_NAME, object_lock)
        objects = {
            "documents": next(item for item in artifacts if item["path"] == DOCUMENTS_NAME),
            "occurrences": next(item for item in artifacts if item["path"] == OCCURRENCES_NAME),
            "coverage_report": next(item for item in artifacts if item["path"] == COVERAGE_NAME),
        }
        objects["object_lock"] = artifact_ref(staging / LOCK_NAME, LOCK_NAME, "application/json")
        pack_component = {"id": index_id, "version": version, "sha256": ""}
        if index_uri is not None:
            pack_component["uri"] = index_uri
        manifest = {
            "schema_version": "livefire.rag.evidence-projection-pack/1",
            "component": pack_component,
            "stage": "pre_embedding_projection",
            "source_snapshots": [source_snapshot],
            "document_kinds": list(DOCUMENT_KINDS),
            "row_schemas": _row_schema_refs(),
            "projection_policy": projection_policy,
            "derivation_policies": [],
            "physical_contract": {
                "documents_format": "canonical_jsonl",
                "occurrences_format": "canonical_jsonl",
                "encoding": "utf-8",
                "line_termination": "lf",
                "document_order": "document_id_asc",
                "occurrence_order": "occurrence_id_asc",
            },
            "objects": objects,
            "closure": {
                "source_record_count": input_count,
                "terminal_disposition_count": occurrence_count,
                "document_count": document_count,
                "unaccounted_record_count": 0,
                "multiply_dispositioned_record_count": 0,
            },
            "pointer_contract": {
                "pointer_schema": POINTER_SCHEMA_REF,
                "immutable_source_pointer_required": True,
                "all_pointers_resolved": True,
                "record_id_only_requires_local_pointer_table": True,
            },
            "promotion_contract": {
                "is_searchable_index": False,
                "embeddings_present": False,
                "promotion_requires": [
                    "admitted_projection_pack",
                    "bound_embedding_profiles",
                    "complete_searchable_document_embeddings",
                    "canonical_parquet_materialization",
                    "evidence_index_manifest",
                ],
            },
        }
        manifest["component"]["sha256"] = evidence_manifest_identity(manifest)
        write_canonical_json(staging / MANIFEST_NAME, manifest)
        connection.close()
        connection = None
        (staging / "build.sqlite3").unlink()
        _verify_evidence_pack(
            staging,
            source_snapshot=source_snapshot,
            relation_sources=sources or None,
            projection_policy=projection_policy,
            projector=project,
            trusted_builder=not sources,
        )
        if out_dir.exists():
            raise FileExistsError(f"refusing to overwrite evidence pack: {out_dir}")
        os.rename(staging, out_dir)
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_generic_projection_policy(projection_policy: Mapping[str, Any]) -> None:
    from .evidence_projection import projection_policy_ref

    expected = projection_policy_ref()
    for field in ("id", "version", "sha256"):
        if projection_policy.get(field) != expected[field]:
            raise ValueError(
                "projection_policy must identify the built-in generic evidence policy"
            )


def _require_receipt_fenced_sources(
    relations: Sequence[RelationSource],
) -> tuple[RelationSource, ...]:
    sources = tuple(relations)
    if not sources:
        raise ValueError("at least one receipt-fenced relation is required")
    for source in sources:
        if not isinstance(source, RelationSource):
            raise ValueError("public evidence builds require RelationSource inputs")
        if source.expected_sha256 is None or source.expected_rows is None:
            raise ValueError(
                "public evidence builds require receipt-fenced object digests and row counts"
            )
    return sources


def build_evidence_pack(
    out_dir: Path,
    relations: Sequence[RelationSource],
    *,
    index_id: str,
    version: str,
    source_snapshot: dict[str, str],
    projection_policy: dict[str, str],
    index_uri: str | None = None,
    batch_size: int = 4096,
    temp_directory: Path | None = None,
) -> dict[str, Any]:
    """Build a conformant pack from receipt-fenced typed Parquet relations.

    The public boundary deliberately fixes the generic projector and refuses
    in-memory fixture rows. This keeps the policy identity, exact source
    pointers, and ``all_pointers_resolved`` claim independently meaningful.
    """

    sources = _require_receipt_fenced_sources(relations)
    _require_generic_projection_policy(projection_policy)
    return _build_evidence_pack_for_test(
        out_dir,
        relations=sources,
        index_id=index_id,
        version=version,
        source_snapshot=source_snapshot,
        projection_policy=projection_policy,
        index_uri=index_uri,
        batch_size=batch_size,
        temp_directory=temp_directory,
    )


def _load_canonical_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidencePackCorrupt(f"unreadable JSON: {path.name}") from error
    if raw != canonical_json_bytes(value, newline=True):
        raise EvidencePackCorrupt(f"non-canonical JSON: {path.name}")
    return value


def _verified_artifact(root: Path, artifact: Any, expected_path: str) -> None:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "media_type", "sha256", "bytes"}:
        raise EvidencePackCorrupt(f"invalid artifact reference for {expected_path}")
    if artifact["path"] != expected_path or Path(expected_path).name != expected_path:
        raise EvidencePackCorrupt(f"unexpected artifact path for {expected_path}")
    path = root / expected_path
    if (
        not path.is_file()
        or path.stat().st_size != artifact["bytes"]
        or sha256_file(path) != artifact["sha256"]
    ):
        raise EvidencePackCorrupt(f"artifact digest mismatch: {expected_path}")


def _verify_source_pointer(
    pointer: Any, *, source_snapshot: dict[str, str], relation_name: str
) -> None:
    if not isinstance(pointer, dict) or pointer.get("schema_version") != "livefire.source-record-pointer/1":
        raise EvidencePackCorrupt("invalid occurrence source pointer")
    if pointer.get("snapshot") != source_snapshot:
        raise EvidencePackCorrupt("occurrence pointer snapshot mismatch")
    try:
        _validate_component(pointer["snapshot_profile"], "source_pointer.snapshot_profile")
    except (KeyError, TypeError, ValueError) as error:
        raise EvidencePackCorrupt(str(error)) from error
    if pointer["snapshot_profile"] != source_record_profile_ref():
        raise EvidencePackCorrupt("source pointer uses an unsupported snapshot profile")
    if not isinstance(pointer.get("record_id"), str) or not pointer["record_id"]:
        raise EvidencePackCorrupt("invalid source pointer record_id")
    if not _is_sha256(pointer.get("record_sha256")):
        raise EvidencePackCorrupt("invalid source pointer record digest")
    locator = pointer.get("locator")
    if not isinstance(locator, dict) or locator.get("kind") != "parquet_row":
        raise EvidencePackCorrupt("projection pack pointer must use parquet_row locator")
    if locator.get("relation") != relation_name or not _is_sha256(locator.get("object_sha256")):
        raise EvidencePackCorrupt("invalid source pointer Parquet identity")
    for field in ("row_group", "row_ordinal"):
        if isinstance(locator.get(field), bool) or not isinstance(locator.get(field), int) or locator[field] < 0:
            raise EvidencePackCorrupt(f"invalid source pointer {field}")


def _iter_canonical_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raise EvidencePackCorrupt(f"{path.name} line {line_number} lacks LF")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidencePackCorrupt(
                    f"{path.name} line {line_number} is unreadable"
                ) from error
            if raw != canonical_json_bytes(value, newline=True):
                raise EvidencePackCorrupt(
                    f"{path.name} line {line_number} is not canonical JSON"
                )
            if not isinstance(value, dict):
                raise EvidencePackCorrupt(f"{path.name} line {line_number} is not an object")
            yield value


def _verify_evidence_pack(
    root: Path,
    *,
    source_snapshot: dict[str, str] | None,
    relation_sources: Sequence[RelationSource] | None,
    projection_policy: dict[str, str] | None,
    projector: Projector | None,
    trusted_builder: bool,
) -> dict[str, Any]:
    """Verify bytes, schemas, closure, and (when mounted) exact source rows."""

    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise EvidencePackCorrupt(f"manifest not found: {manifest_path}")
    manifest = _load_canonical_json(manifest_path)
    try:
        if manifest["schema_version"] != "livefire.rag.evidence-projection-pack/1":
            raise EvidencePackCorrupt("unsupported evidence manifest schema")
        _validate_component(manifest["component"], "component")
        if not isinstance(manifest["source_snapshots"], list) or len(manifest["source_snapshots"]) != 1:
            raise EvidencePackCorrupt("projection pack requires exactly one source snapshot")
        _validate_component(manifest["source_snapshots"][0], "source_snapshot")
        _validate_component(manifest["projection_policy"], "projection_policy")
        if evidence_manifest_identity(manifest) != manifest["component"]["sha256"]:
            raise EvidencePackCorrupt("manifest component identity mismatch")
        objects = manifest["objects"]
        expected = {
            "documents": DOCUMENTS_NAME,
            "occurrences": OCCURRENCES_NAME,
            "coverage_report": COVERAGE_NAME,
            "object_lock": LOCK_NAME,
        }
        if set(objects) != set(expected):
            raise EvidencePackCorrupt("manifest object set mismatch")
        for key, path in expected.items():
            _verified_artifact(root, objects[key], path)
        object_lock = _load_canonical_json(root / LOCK_NAME)
        locked = [objects["coverage_report"], objects["documents"], objects["occurrences"]]
        locked.sort(key=lambda artifact: artifact["path"])
        if object_lock != {"schema_version": "livefire.object-lock/1", "objects": locked}:
            raise EvidencePackCorrupt("object lock does not bind evidence artifacts")
        coverage = _load_canonical_json(root / COVERAGE_NAME)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, EvidencePackCorrupt):
            raise
        raise EvidencePackCorrupt(str(error)) from error

    verify_dir = Path(tempfile.mkdtemp(prefix=".verify-evidence-", dir=root.parent))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(verify_dir / "verify.sqlite3")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE documents(document_id TEXT PRIMARY KEY, expected INTEGER NOT NULL, "
            "replay_sha256 BLOB NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE source_pointers("
            "relation_name TEXT NOT NULL, object_sha256 TEXT NOT NULL, row_group INTEGER NOT NULL, "
            "row_ordinal INTEGER NOT NULL, record_id TEXT NOT NULL, record_sha256 TEXT NOT NULL, "
            "support_refs_sha256 BLOB NOT NULL, occurrence_sha256 BLOB NOT NULL, "
            "document_id TEXT, "
            "PRIMARY KEY(relation_name, object_sha256, row_group, row_ordinal)) WITHOUT ROWID"
        )
        document_count = 0
        searchable_document_count = 0
        verified_document_kinds: Counter[str] = Counter()
        prior_document_id: str | None = None
        for document in _iter_canonical_jsonl(root / DOCUMENTS_NAME):
            document_id = document.get("document_id")
            expected_count = document.get("occurrence_count")
            if (
                not isinstance(document_id, str)
                or not document_id.startswith("doc-")
                or not _is_sha256(document_id[4:])
                or isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or expected_count < 1
            ):
                raise EvidencePackCorrupt("invalid evidence document identity/count")
            document_sha256 = document.get("document_sha256")
            if not _is_sha256(document_sha256) or canonical_sha256_omitting(
                document, ("document_sha256",)
            ) != document_sha256:
                raise EvidencePackCorrupt("evidence document identity mismatch")
            if prior_document_id is not None and document_id <= prior_document_id:
                raise EvidencePackCorrupt("documents are not uniquely sorted")
            prior_document_id = document_id
            document_kind = document.get("document_kind")
            searchable = document.get("searchable")
            if document_kind not in DOCUMENT_KINDS or not isinstance(searchable, bool):
                raise EvidencePackCorrupt("invalid evidence document kind/searchability")
            verified_document_kinds[document_kind] += 1
            searchable_document_count += int(searchable)
            replay_document = dict(document)
            replay_document.pop("document_sha256", None)
            replay_document.pop("occurrence_count", None)
            connection.execute(
                "INSERT INTO documents(document_id, expected, replay_sha256) VALUES (?, ?, ?)",
                (
                    document_id,
                    expected_count,
                    bytes.fromhex(sha256_bytes(canonical_json_bytes(replay_document))),
                ),
            )
            document_count += 1

        occurrence_count = 0
        prior_occurrence_id: str | None = None
        relation_counts: Counter[str] = Counter()
        relation_dispositions: dict[str, Counter[str]] = defaultdict(Counter)
        dispositions: Counter[str] = Counter()
        verified_reason_counts: Counter[tuple[str, str]] = Counter()
        for occurrence in _iter_canonical_jsonl(root / OCCURRENCES_NAME):
            occurrence_id = occurrence.get("occurrence_id")
            relation_identity = occurrence.get("relation_identity")
            relation_name = (
                relation_identity.get("relation") if isinstance(relation_identity, dict) else None
            )
            disposition = occurrence.get("terminal_disposition")
            document_ids = occurrence.get("document_ids")
            if (
                not isinstance(occurrence_id, str)
                or not occurrence_id.startswith("occ-")
                or not _is_sha256(occurrence_id[4:])
            ):
                raise EvidencePackCorrupt("invalid occurrence identity")
            if prior_occurrence_id is not None and occurrence_id <= prior_occurrence_id:
                raise EvidencePackCorrupt("occurrences are not uniquely sorted")
            prior_occurrence_id = occurrence_id
            if not isinstance(relation_name, str) or not relation_name:
                raise EvidencePackCorrupt("invalid occurrence relation")
            pointer = occurrence.get("source_pointer")
            _verify_source_pointer(
                pointer,
                source_snapshot=manifest["source_snapshots"][0],
                relation_name=relation_name,
            )
            expected_occurrence_id = "occ-" + sha256_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "livefire.rag.evidence-occurrence-identity/1",
                        "source_pointer": pointer,
                    }
                )
            )
            if occurrence_id != expected_occurrence_id:
                raise EvidencePackCorrupt("occurrence identity does not bind its source pointer")
            locator = pointer["locator"]
            support_refs = pointer.get("support_refs")
            if (
                not isinstance(support_refs, list)
                or not support_refs
                or any(not isinstance(value, str) or not value for value in support_refs)
            ):
                raise EvidencePackCorrupt("source pointer lacks valid support refs")
            try:
                connection.execute(
                    "INSERT INTO source_pointers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        relation_name,
                        locator["object_sha256"],
                        locator["row_group"],
                        locator["row_ordinal"],
                        pointer["record_id"],
                        pointer["record_sha256"],
                        bytes.fromhex(sha256_bytes(canonical_json_bytes(support_refs))),
                        bytes.fromhex(sha256_bytes(canonical_json_bytes(occurrence))),
                        document_ids[0] if isinstance(document_ids, list) and document_ids else None,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise EvidencePackCorrupt("multiple occurrences name one source row") from error
            if occurrence.get("projection_policy") != manifest["projection_policy"]:
                raise EvidencePackCorrupt("occurrence projection policy mismatch")
            if disposition not in PACK_DISPOSITIONS:
                raise EvidencePackCorrupt("invalid terminal disposition")
            if not isinstance(document_ids, list):
                raise EvidencePackCorrupt("invalid occurrence document_ids")
            reason_codes = occurrence.get("reason_codes")
            exact_attributes = occurrence.get("exact_attributes")
            if (
                not isinstance(reason_codes, list)
                or any(not isinstance(reason, str) or not reason for reason in reason_codes)
                or not isinstance(exact_attributes, list)
            ):
                raise EvidencePackCorrupt("invalid occurrence reasons or exact attributes")
            if disposition in {
                "direct_semantic_document",
                "semantic_group_occurrence",
                "derived_document_input",
            }:
                if len(document_ids) != 1 or not isinstance(document_ids[0], str):
                    raise EvidencePackCorrupt("document disposition lacks document_id")
            elif document_ids:
                raise EvidencePackCorrupt("non-document disposition names a document")
            relation_counts[relation_name] += 1
            relation_dispositions[relation_name][disposition] += 1
            dispositions[disposition] += 1
            for reason in reason_codes:
                verified_reason_counts[(disposition, reason)] += 1
            occurrence_count += 1
        unknown_document = connection.execute(
            "SELECT s.document_id FROM source_pointers s LEFT JOIN documents d "
            "ON d.document_id = s.document_id WHERE s.document_id IS NOT NULL "
            "AND d.document_id IS NULL LIMIT 1"
        ).fetchone()
        if unknown_document:
            raise EvidencePackCorrupt(
                f"occurrence refers to an unknown document: {unknown_document[0]}"
            )
        mismatch = connection.execute(
            "SELECT d.document_id FROM documents d LEFT JOIN "
            "(SELECT document_id, count(*) actual FROM source_pointers "
            "WHERE document_id IS NOT NULL GROUP BY document_id) a "
            "ON a.document_id = d.document_id "
            "WHERE d.expected != coalesce(a.actual, 0) LIMIT 1"
        ).fetchone()
        if mismatch:
            raise EvidencePackCorrupt(f"document occurrence count mismatch: {mismatch[0]}")

        declared_relation_names = {
            row.get("relation_identity", {}).get("relation")
            for row in coverage.get("relation_coverage", [])
            if isinstance(row, dict) and isinstance(row.get("relation_identity"), dict)
        }

        if occurrence_count and relation_sources is None and not trusted_builder:
            raise EvidencePackCorrupt(
                "source snapshot relations are required for independent pointer resolution"
            )
        if relation_sources is not None:
            if source_snapshot != manifest["source_snapshots"][0]:
                raise EvidencePackCorrupt("mounted source snapshot does not match the pack")
            if projection_policy != manifest["projection_policy"]:
                raise EvidencePackCorrupt("mounted projection policy does not match the pack")
            project = projector or _default_projector()
            source_names = [source.relation_name for source in relation_sources]
            if (
                len(source_names) != len(set(source_names))
                or set(source_names) != declared_relation_names
            ):
                raise EvidencePackCorrupt("mounted source relation set does not match occurrences")
            resolved_count = 0
            for source in sorted(relation_sources, key=lambda item: item.relation_name):
                source_count = 0
                for row in _iter_parquet(source, 8192):
                    event_id, typed_event_json, support_ref = _extract_input_row(
                        source.relation_name, row
                    )
                    record_sha256 = sha256_bytes(_typed_event_bytes(typed_event_json))
                    candidate = connection.execute(
                        "SELECT support_refs_sha256, occurrence_sha256 FROM source_pointers "
                        "WHERE relation_name = ? "
                        "AND object_sha256 = ? AND row_group = ? AND row_ordinal = ? "
                        "AND record_id = ? AND record_sha256 = ?",
                        (
                            source.relation_name,
                            row["source_object_sha256"],
                            row["row_group"],
                            row["row_ordinal"],
                            event_id,
                            record_sha256,
                        ),
                    ).fetchone()
                    if candidate is None:
                        raise EvidencePackCorrupt(
                            f"source row has no exact occurrence pointer: {source.relation_name}"
                        )
                    expected_support_refs_sha256 = bytes.fromhex(
                        sha256_bytes(canonical_json_bytes([support_ref]))
                    )
                    if bytes(candidate[0]) != expected_support_refs_sha256:
                        raise EvidencePackCorrupt("source pointer support reference mismatch")
                    expected_pointer = _source_pointer(
                        row,
                        relation_name=source.relation_name,
                        event_id=event_id,
                        support_ref=support_ref,
                        record_sha256=record_sha256,
                        source_snapshot=source_snapshot,
                    )
                    projected = project(
                        source.relation_name, event_id, typed_event_json, support_ref
                    )
                    expected_document, expected_occurrence = _normalise_projection(
                        projected,
                        typed_event_json=typed_event_json,
                        relation_name=source.relation_name,
                        event_id=event_id,
                        support_ref=support_ref,
                        record_sha256=record_sha256,
                        source_pointer=expected_pointer,
                        projection_policy=projection_policy,
                    )
                    expected_occurrence_sha256 = bytes.fromhex(
                        sha256_bytes(canonical_json_bytes(expected_occurrence))
                    )
                    if bytes(candidate[1]) != expected_occurrence_sha256:
                        raise EvidencePackCorrupt(
                            f"occurrence does not replay from source: {source.relation_name}"
                        )
                    if expected_document is not None:
                        actual_row = connection.execute(
                            "SELECT replay_sha256 FROM documents WHERE document_id = ?",
                            (expected_document["document_id"],),
                        ).fetchone()
                        if actual_row is None:
                            raise EvidencePackCorrupt("replayed projection document is missing")
                        expected_document.pop("document_sha256", None)
                        expected_document_sha256 = bytes.fromhex(
                            sha256_bytes(canonical_json_bytes(expected_document))
                        )
                        if bytes(actual_row[0]) != expected_document_sha256:
                            raise EvidencePackCorrupt(
                                f"document does not replay from source: {source.relation_name}"
                            )
                    source_count += 1
                if source_count != relation_counts.get(source.relation_name, 0):
                    raise EvidencePackCorrupt(
                        f"source/occurrence count mismatch: {source.relation_name}"
                    )
                resolved_count += source_count
            if resolved_count != occurrence_count:
                raise EvidencePackCorrupt("not every occurrence pointer resolves to an exact row")
        else:
            resolved_count = occurrence_count

        relation_rows = []
        coverage_relations = coverage.get("relation_coverage")
        if not isinstance(coverage_relations, list):
            raise EvidencePackCorrupt("coverage relations are invalid")
        coverage_by_name = {
            row.get("relation_identity", {}).get("relation"): row
            for row in coverage_relations
            if isinstance(row, dict) and isinstance(row.get("relation_identity"), dict)
        }
        if len(coverage_by_name) != len(coverage_relations):
            raise EvidencePackCorrupt("coverage relations are duplicated or invalid")
        for relation_name in sorted(coverage_by_name):
            row = coverage_by_name[relation_name]
            count = relation_counts.get(relation_name, 0)
            if row.get("source_record_count") != count or row.get("terminal_disposition_count") != count:
                raise EvidencePackCorrupt("per-relation closure mismatch")
            expected_dispositions = {
                name: relation_dispositions[relation_name].get(name, 0)
                for name in PACK_DISPOSITIONS
            }
            if row.get("by_terminal_disposition") != expected_dispositions:
                raise EvidencePackCorrupt("per-relation disposition mismatch")
            relation_rows.append(row)
        if set(relation_counts) - set(coverage_by_name):
            raise EvidencePackCorrupt("coverage omits an occurrence relation")
        if relation_rows != sorted(
            relation_rows, key=lambda row: row["relation_identity"]["relation"]
        ):
            raise EvidencePackCorrupt("coverage relations are not sorted")

        closure = manifest.get("closure", {})
        coverage_closure = coverage.get("closure", {})
        checks = (
            (document_count, closure.get("document_count"), coverage.get("documents", {}).get("total")),
            (occurrence_count, closure.get("source_record_count"), coverage_closure.get("source_record_count")),
            (
                occurrence_count,
                closure.get("terminal_disposition_count"),
                coverage_closure.get("terminal_disposition_count"),
            ),
        )
        if any(observed != manifest_count or observed != coverage_count for observed, manifest_count, coverage_count in checks):
            raise EvidencePackCorrupt("count reconciliation failed")
        observed_dispositions = {
            name: dispositions.get(name, 0) for name in PACK_DISPOSITIONS
        }
        if (
            coverage_closure.get("by_terminal_disposition") != observed_dispositions
            or coverage_closure.get("all_source_records_dispositioned") is not True
            or coverage_closure.get("unaccounted_record_count") != 0
            or coverage_closure.get("multiply_dispositioned_record_count") != 0
        ):
            raise EvidencePackCorrupt("terminal disposition reconciliation failed")
        if coverage.get("source_snapshots") != manifest["source_snapshots"]:
            raise EvidencePackCorrupt("coverage source snapshot mismatch")
        if coverage.get("projection_policy") != manifest["projection_policy"]:
            raise EvidencePackCorrupt("coverage projection policy mismatch")
        expected_documents = {
            "total": document_count,
            "searchable": searchable_document_count,
            "by_kind": {
                name: verified_document_kinds.get(name, 0) for name in DOCUMENT_KINDS
            },
        }
        if coverage.get("documents") != expected_documents:
            raise EvidencePackCorrupt("coverage document-kind reconciliation failed")
        expected_reasons = [
            {"terminal_disposition": disposition, "reason_code": reason, "count": count}
            for (disposition, reason), count in sorted(verified_reason_counts.items())
        ]
        if coverage.get("reason_counts") != expected_reasons:
            raise EvidencePackCorrupt("coverage reason reconciliation failed")
        if coverage.get("pointer_resolution") != {
            "pointer_count": occurrence_count,
            "resolved_count": resolved_count,
            "unresolved_count": 0,
            "all_pointers_resolved": True,
        }:
            raise EvidencePackCorrupt("coverage pointer-resolution reconciliation failed")
        return manifest
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError) as error:
        if isinstance(error, EvidencePackCorrupt):
            raise
        raise EvidencePackCorrupt(str(error)) from error
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(verify_dir, ignore_errors=True)


def verify_evidence_pack(
    root: Path,
    *,
    source_snapshot: dict[str, str],
    relation_sources: Sequence[RelationSource],
    projection_policy: dict[str, str],
    sdk_specs: Path,
    rag_specs: Path | None = None,
) -> dict[str, Any]:
    """Independently verify a pack against its exact mounted source snapshot."""

    from .evidence_schema import validate_evidence_pack_schemas

    sources = _require_receipt_fenced_sources(relation_sources)
    _require_generic_projection_policy(projection_policy)
    manifest = _verify_evidence_pack(
        root,
        source_snapshot=source_snapshot,
        relation_sources=sources,
        projection_policy=projection_policy,
        projector=None,
        trusted_builder=False,
    )
    validate_evidence_pack_schemas(root, sdk_specs=sdk_specs, rag_specs=rag_specs)
    return manifest
