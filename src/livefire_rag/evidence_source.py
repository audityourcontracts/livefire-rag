"""Admission of a completed normalized snapshot for generic evidence projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_file
from .evidence_builder import RelationSource
from .evidence_projection import RELATION_DOCUMENT_KINDS


class SnapshotAdmissionError(RuntimeError):
    """The supplied normalized snapshot is incomplete or differs from its receipt."""


@dataclass(frozen=True)
class AdmittedTypedSnapshot:
    """A closed set of receipt-bound typed relations."""

    component: dict[str, str]
    relations: tuple[RelationSource, ...]
    expected_rows: dict[str, int]
    receipt_sha256: str


def _load_receipt(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotAdmissionError(f"cannot read snapshot receipt: {path}") from error
    if not isinstance(value, Mapping):
        raise SnapshotAdmissionError("snapshot receipt must be a JSON object")
    return value


def _parquet_row_count(path: Path) -> int:
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise SnapshotAdmissionError(
            "DuckDB is required for this historical test oracle; install livefire-rag[test]"
        ) from error
    connection = duckdb.connect()
    try:
        value = connection.execute(
            "SELECT coalesce(sum(row_group_num_rows), 0) FROM ("
            "SELECT row_group_id, row_group_num_rows FROM parquet_metadata(?) GROUP BY ALL)",
            [str(path.resolve())],
        ).fetchone()[0]
    finally:
        connection.close()
    return int(value)


def admit_typed_snapshot(
    snapshot_root: Path,
    receipt_path: Path,
    *,
    snapshot_id: str | None = None,
    snapshot_version: str | None = None,
    expected_relations: set[str] | None = None,
) -> AdmittedTypedSnapshot:
    """Resolve and verify every typed relation named by a completed build receipt.

    Selection is structural: every receipt object whose relation is in the closed
    typed-relation contract is required, and no scenario, event value, or expected
    answer participates in admission.
    """

    root = Path(snapshot_root).resolve()
    receipt_file = Path(receipt_path).resolve()
    if not root.is_dir():
        raise SnapshotAdmissionError(f"snapshot root does not exist: {root}")
    receipt = _load_receipt(receipt_file)
    logical_sha256 = receipt.get("output_logical_sha256")
    if not isinstance(logical_sha256, str) or len(logical_sha256) != 64:
        raise SnapshotAdmissionError("receipt lacks output_logical_sha256")
    try:
        int(logical_sha256, 16)
    except ValueError as error:
        raise SnapshotAdmissionError("receipt output_logical_sha256 is not hexadecimal") from error
    snapshot_manifest = receipt.get("snapshot_manifest")
    objects = snapshot_manifest.get("objects") if isinstance(snapshot_manifest, Mapping) else None
    if not isinstance(objects, list):
        raise SnapshotAdmissionError("receipt lacks snapshot_manifest.objects")
    if snapshot_manifest.get("logical_sha256") != logical_sha256:
        raise SnapshotAdmissionError(
            "snapshot manifest logical digest differs from output logical digest"
        )
    runnable = receipt.get("runnable_snapshot")
    component = runnable.get("component") if isinstance(runnable, Mapping) else None
    component_required = {"id", "version", "sha256"}
    component_allowed = component_required | {"uri"}
    if (
        not isinstance(component, Mapping)
        or not component_required <= set(component)
        or set(component) - component_allowed
        or not all(
            isinstance(component.get(name), str) and component[name]
            for name in component
        )
    ):
        raise SnapshotAdmissionError("receipt lacks a closed runnable_snapshot.component")
    component = dict(component)
    if component["sha256"] != logical_sha256:
        raise SnapshotAdmissionError("runnable snapshot digest differs from output logical digest")
    if snapshot_id is not None and component["id"] != snapshot_id:
        raise SnapshotAdmissionError("runnable snapshot id differs from the expected id")
    if snapshot_version is not None and component["version"] != snapshot_version:
        raise SnapshotAdmissionError("runnable snapshot version differs from the expected version")

    completeness = receipt.get("completeness_receipt")
    closure = receipt.get("closure")
    if not isinstance(completeness, Mapping) or not isinstance(closure, Mapping):
        raise SnapshotAdmissionError("receipt lacks completeness or closure material")
    dataset_sha256 = snapshot_manifest.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or runnable.get("dataset_sha256") != dataset_sha256
        or completeness.get("dataset_sha256") != dataset_sha256
        or completeness.get("normalized_snapshot_sha256") != logical_sha256
    ):
        raise SnapshotAdmissionError("snapshot dataset/completeness lineage does not reconcile")
    mapping_pack = runnable.get("mapping_pack")
    relation_contract = runnable.get("relation_contract")
    if (
        not isinstance(mapping_pack, Mapping)
        or mapping_pack.get("sha256") != snapshot_manifest.get("mapping_pack_sha256")
        or mapping_pack.get("sha256") != completeness.get("mapping_pack_sha256")
        or not isinstance(relation_contract, Mapping)
        or relation_contract.get("sha256") != snapshot_manifest.get("relation_contract_sha256")
        or relation_contract.get("sha256") != completeness.get("relation_contract_sha256")
    ):
        raise SnapshotAdmissionError("snapshot mapping/relation-contract lineage does not reconcile")
    normalized_events = runnable.get("normalized_events")
    completeness_metrics = completeness.get("metrics")
    if (
        isinstance(normalized_events, bool)
        or not isinstance(normalized_events, int)
        or normalized_events < 0
        or closure.get("event_rows") != normalized_events
        or closure.get("mapped_events") != normalized_events
        or not isinstance(completeness_metrics, Mapping)
        or completeness_metrics.get("normalized_events") != normalized_events
    ):
        raise SnapshotAdmissionError("snapshot normalized-event closure does not reconcile")
    for field in (
        "unresolved_provenance_fields",
        "provenance_digest_mismatches",
        "rejected_malformed_records",
        "unsupported_records",
    ):
        if closure.get(field) != 0:
            raise SnapshotAdmissionError(f"snapshot closure is not clean: {field}")
    source_rows = runnable.get("source_rows")
    if (
        isinstance(source_rows, bool)
        or not isinstance(source_rows, int)
        or source_rows < 0
        or closure.get("input_rows") != source_rows
        or closure.get("mapped_source_records") != source_rows
        or completeness_metrics.get("source_rows") != source_rows
        or completeness_metrics.get("mapped_source_records") != source_rows
    ):
        raise SnapshotAdmissionError("snapshot source-row closure does not reconcile")

    required = set(expected_relations or RELATION_DOCUMENT_KINDS)
    by_relation: dict[str, Mapping[str, Any]] = {}
    for value in objects:
        if not isinstance(value, Mapping):
            raise SnapshotAdmissionError("receipt contains a non-object snapshot entry")
        relation = value.get("relation")
        if relation not in required:
            continue
        if relation in by_relation:
            raise SnapshotAdmissionError(f"receipt duplicates typed relation: {relation}")
        by_relation[str(relation)] = value
    missing = required - set(by_relation)
    if missing:
        raise SnapshotAdmissionError(f"receipt omits typed relations: {sorted(missing)}")

    sources: list[RelationSource] = []
    expected_rows_by_relation: dict[str, int] = {}
    for relation in sorted(required):
        entry = by_relation[relation]
        relative_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        rows = entry.get("rows")
        if relative_path != f"semantic/{relation}.parquet":
            raise SnapshotAdmissionError(f"unexpected typed relation path for {relation}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise SnapshotAdmissionError(f"invalid receipt digest for {relation}")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise SnapshotAdmissionError(f"invalid receipt row count for {relation}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SnapshotAdmissionError(f"typed relation escapes snapshot root: {relation}") from error
        if not path.is_file():
            raise SnapshotAdmissionError(f"typed relation is missing: {relative_path}")
        if sha256_file(path) != expected_sha256:
            raise SnapshotAdmissionError(f"typed relation digest mismatch: {relation}")
        observed_rows = _parquet_row_count(path)
        if observed_rows != rows:
            raise SnapshotAdmissionError(
                f"typed relation row-count mismatch for {relation}: {observed_rows} != {rows}"
            )
        sources.append(
            RelationSource(
                relation,
                path,
                expected_sha256=expected_sha256,
                expected_rows=rows,
            )
        )
        expected_rows_by_relation[relation] = rows

    if sum(expected_rows_by_relation.values()) != normalized_events:
        raise SnapshotAdmissionError(
            "typed relation rows do not reconcile to runnable normalized-event count"
        )
    events_objects = [entry for entry in objects if entry.get("relation") == "events"]
    if len(events_objects) != 1 or events_objects[0].get("rows") != normalized_events:
        raise SnapshotAdmissionError("snapshot events object does not reconcile to typed relations")

    return AdmittedTypedSnapshot(
        component=component,
        relations=tuple(sources),
        expected_rows=expected_rows_by_relation,
        receipt_sha256=sha256_file(receipt_file),
    )


__all__ = [
    "AdmittedTypedSnapshot",
    "SnapshotAdmissionError",
    "admit_typed_snapshot",
]
