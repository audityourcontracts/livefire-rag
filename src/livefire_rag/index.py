"""Immutable semantic index pack builder, verifier, and exact search engine."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import artifact_ref, canonical_json_bytes, canonical_sha256_omitting, sha256_file, write_canonical_json
from .contracts import INDEX_FORMAT_REF, SHA256_RE, ContractError, parse_timestamp


MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
VECTORS_NAME = "vectors.f32"
LOCK_NAME = "objects.lock.json"


class IndexErrorBase(RuntimeError):
    code = "invalid_binding"


class IndexCorrupt(IndexErrorBase):
    code = "corrupt_artifact"


class IndexNotFound(IndexErrorBase):
    code = "not_found"


def _validate_component(value: Any, label: str) -> dict[str, str]:
    required = {"id", "version", "sha256"}
    allowed = required | {"uri"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - allowed:
        raise ValueError(f"{label} must be a closed component reference")
    if not all(isinstance(value[key], str) and value[key] for key in value):
        raise ValueError(f"{label} has invalid fields")
    if not SHA256_RE.fullmatch(value["sha256"]):
        raise ValueError(f"{label}.sha256 is invalid")
    return value


def _validate_pointer(pointer: Any) -> dict[str, Any]:
    if not isinstance(pointer, dict):
        raise ValueError("source_pointer must be an object")
    required = {"schema_version", "snapshot", "snapshot_profile", "record_id", "record_sha256", "locator"}
    allowed = required | {"support_refs", "native_locator_sha256"}
    if set(pointer) - allowed or not required <= set(pointer):
        raise ValueError("source_pointer has missing or unknown fields")
    if pointer["schema_version"] != "livefire.source-record-pointer/1":
        raise ValueError("source_pointer schema_version is invalid")
    _validate_component(pointer["snapshot"], "source_pointer.snapshot")
    _validate_component(pointer["snapshot_profile"], "source_pointer.snapshot_profile")
    if not isinstance(pointer["record_id"], str) or not pointer["record_id"]:
        raise ValueError("source_pointer.record_id is invalid")
    if not isinstance(pointer["record_sha256"], str) or not SHA256_RE.fullmatch(pointer["record_sha256"]):
        raise ValueError("source_pointer.record_sha256 is invalid")
    locator = pointer["locator"]
    if not isinstance(locator, dict):
        raise ValueError("source_pointer.locator is invalid")
    kind = locator.get("kind")
    if kind == "record_id_only":
        if set(locator) != {"kind"}:
            raise ValueError("record_id_only locator has unknown fields")
    elif kind == "jsonl_record":
        if set(locator) != {"kind", "object_sha256", "line_ordinal"}:
            raise ValueError("jsonl_record locator fields are invalid")
        if not isinstance(locator["object_sha256"], str) or not SHA256_RE.fullmatch(locator["object_sha256"]):
            raise ValueError("jsonl_record object_sha256 is invalid")
        if isinstance(locator["line_ordinal"], bool) or not isinstance(locator["line_ordinal"], int) or locator["line_ordinal"] < 0:
            raise ValueError("jsonl_record line_ordinal is invalid")
    elif kind == "parquet_row":
        required_locator = {"kind", "object_sha256", "row_group", "row_ordinal"}
        if not required_locator <= set(locator) or set(locator) - (required_locator | {"relation"}):
            raise ValueError("parquet_row locator fields are invalid")
        if not isinstance(locator["object_sha256"], str) or not SHA256_RE.fullmatch(locator["object_sha256"]):
            raise ValueError("parquet_row object_sha256 is invalid")
        for field in ("row_group", "row_ordinal"):
            if isinstance(locator[field], bool) or not isinstance(locator[field], int) or locator[field] < 0:
                raise ValueError(f"parquet_row {field} is invalid")
        if "relation" in locator and (not isinstance(locator["relation"], str) or not locator["relation"]):
            raise ValueError("parquet_row relation is invalid")
    elif kind == "keyed_object":
        if set(locator) != {"kind", "object_sha256", "key_sha256"}:
            raise ValueError("keyed_object locator fields are invalid")
        for field in ("object_sha256", "key_sha256"):
            if not isinstance(locator[field], str) or not SHA256_RE.fullmatch(locator[field]):
                raise ValueError(f"keyed_object {field} is invalid")
    else:
        raise ValueError("source_pointer.locator kind is invalid")
    if "support_refs" in pointer and (
        not isinstance(pointer["support_refs"], list)
        or any(not isinstance(item, str) or not item for item in pointer["support_refs"])
    ):
        raise ValueError("source_pointer.support_refs is invalid")
    if "native_locator_sha256" in pointer and (
        not isinstance(pointer["native_locator_sha256"], str)
        or not SHA256_RE.fullmatch(pointer["native_locator_sha256"])
    ):
        raise ValueError("source_pointer.native_locator_sha256 is invalid")
    return pointer


def validate_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("document must be an object")
    required = {
        "schema_version", "command_id", "event_time", "observation_kind", "shell_family",
        "semantic_text", "preview", "source_pointer"
    }
    allowed = required | {"principal_key", "host_id", "source_kind", "occurrences", "limitations"}
    if set(document) - allowed or not required <= set(document):
        raise ValueError(f"document has missing or unknown fields: {document.get('command_id', '<unknown>')}")
    if document["schema_version"] != "livefire.rag.semantic-document/1":
        raise ValueError("document schema_version is invalid")
    if not isinstance(document["command_id"], str) or not document["command_id"]:
        raise ValueError("command_id is invalid")
    parse_timestamp(document["event_time"], "event_time")
    if document["observation_kind"] not in {"process_command_line", "powershell_script_block", "cloud_api_action"}:
        raise ValueError("observation_kind is invalid")
    if document["shell_family"] not in {"powershell", "cmd", "posix_shell", "python", "cloud_cli", "direct_exec", "unknown"}:
        raise ValueError("shell_family is invalid")
    if not isinstance(document["semantic_text"], str) or not document["semantic_text"]:
        raise ValueError("semantic_text is invalid")
    if not isinstance(document["preview"], str) or len(document["preview"]) > 4096:
        raise ValueError("preview is invalid")
    if "principal_key" in document:
        principal = document["principal_key"]
        if not isinstance(principal, dict) or set(principal) != {"namespace", "id"}:
            raise ValueError("principal_key is invalid")
    _validate_pointer(document["source_pointer"])
    return document


def manifest_identity(manifest: dict[str, Any]) -> str:
    component = manifest.get("component")
    if not isinstance(component, dict):
        raise ValueError("manifest component is missing")
    return canonical_sha256_omitting(manifest, ("component", "sha256"))


def _write_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for document in documents:
            handle.write(canonical_json_bytes(document, newline=True))


def build_index(
    out_dir: Path,
    documents: Iterable[dict[str, Any]],
    vectors: np.ndarray,
    *,
    index_id: str,
    version: str,
    embedding_profile: dict[str, Any],
    source_snapshots: list[dict[str, str]],
    admission_status: str = "development_only",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    if admission_status != "development_only":
        raise ValueError("the standalone POC builder only emits development_only indexes")
    docs = [validate_document(document) for document in documents]
    array = np.asarray(vectors, dtype="<f4")
    if array.ndim != 2 or array.shape[0] != len(docs) or array.shape[1] < 1:
        raise ValueError("vectors shape does not match documents")
    order = sorted(range(len(docs)), key=lambda index: docs[index]["command_id"])
    docs = [docs[index] for index in order]
    array = array[np.asarray(order)]
    ids = [item["command_id"] for item in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate command_id")
    if not np.isfinite(array).all():
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= 0.0001):
        raise ValueError("vectors must be L2-normalized within 0.0001")
    for snapshot in source_snapshots:
        _validate_component(snapshot, "source_snapshot")
    declared_snapshots = {canonical_json_bytes(snapshot) for snapshot in source_snapshots}
    pointer_snapshots = {
        canonical_json_bytes(doc["source_pointer"]["snapshot"]) for doc in docs
    }
    if pointer_snapshots - declared_snapshots:
        raise ValueError("document pointer names an undeclared source snapshot")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite index path: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        documents_path = staging / DOCUMENTS_NAME
        vectors_path = staging / VECTORS_NAME
        _write_documents(documents_path, docs)
        vectors_path.write_bytes(array.tobytes(order="C"))
        objects = {
            "schema_version": "livefire.object-lock/1",
            "objects": [
                artifact_ref(documents_path, DOCUMENTS_NAME, "application/x-ndjson"),
                artifact_ref(vectors_path, VECTORS_NAME, "application/vnd.livefire.float32-vectors"),
            ],
        }
        lock_path = staging / LOCK_NAME
        write_canonical_json(lock_path, objects)
        manifest = {
            "schema_version": "livefire.rag.semantic-index/1",
            "admission_status": admission_status,
            "component": {"id": index_id, "version": version, "sha256": ""},
            "index_format": INDEX_FORMAT_REF,
            "source_snapshots": source_snapshots,
            "embedding_profile": embedding_profile,
            "documents_count": len(docs),
            "dimensions": int(array.shape[1]),
            "objects": {
                "documents": objects["objects"][0],
                "vectors": objects["objects"][1],
                "object_lock": artifact_ref(lock_path, LOCK_NAME, "application/json"),
            },
            "distance_contract": {
                "vector_element_type": "float32",
                "vector_normalization": "l2",
                "accumulation": "float64",
                "distance": "cosine",
                "distance_encoding": "round_half_even_distance_times_1000000",
                "tie_break": "distance_asc_command_id_asc",
            },
            "limitations": limitations or [],
        }
        manifest["component"]["sha256"] = manifest_identity(manifest)
        write_canonical_json(staging / MANIFEST_NAME, manifest)
        os.replace(staging, out_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass
class SemanticIndex:
    root: Path
    manifest: dict[str, Any]
    documents: list[dict[str, Any]]
    vectors: np.ndarray
    by_id: dict[str, int]

    @classmethod
    def open(cls, root: Path) -> "SemanticIndex":
        root = root.resolve()
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise IndexNotFound(f"index manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IndexCorrupt("index manifest is unreadable") from error
        try:
            if manifest.get("schema_version") != "livefire.rag.semantic-index/1":
                raise ValueError("unsupported manifest schema")
            if manifest.get("admission_status") != "development_only":
                raise ValueError("POC provider accepts development_only indexes only")
            if manifest_identity(manifest) != manifest["component"]["sha256"]:
                raise ValueError("manifest component identity mismatch")
            if manifest.get("index_format") != INDEX_FORMAT_REF:
                raise ValueError("index format mismatch")
            objects = manifest["objects"]
            for name, expected_path in (("documents", DOCUMENTS_NAME), ("vectors", VECTORS_NAME), ("object_lock", LOCK_NAME)):
                artifact = objects[name]
                if artifact["path"] != expected_path:
                    raise ValueError(f"unexpected {name} path")
                path = root / artifact["path"]
                if not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"{name} object digest mismatch")
            lock = json.loads((root / LOCK_NAME).read_text(encoding="utf-8"))
            if lock.get("schema_version") != "livefire.object-lock/1":
                raise ValueError("object lock schema mismatch")
            if lock.get("objects") != [objects["documents"], objects["vectors"]]:
                raise ValueError("object lock does not bind manifest objects")
            documents: list[dict[str, Any]] = []
            with (root / DOCUMENTS_NAME).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.endswith("\n"):
                        raise ValueError(f"documents line {line_number} lacks canonical LF")
                    raw = json.loads(line)
                    if canonical_json_bytes(raw, newline=True).decode("utf-8") != line:
                        raise ValueError(f"documents line {line_number} is not canonical JSON")
                    documents.append(validate_document(raw))
            if len(documents) != manifest["documents_count"]:
                raise ValueError("document count mismatch")
            source_snapshots = manifest["source_snapshots"]
            for snapshot in source_snapshots:
                _validate_component(snapshot, "manifest.source_snapshot")
            declared_snapshots = {
                canonical_json_bytes(snapshot) for snapshot in source_snapshots
            }
            for document in documents:
                snapshot = document["source_pointer"]["snapshot"]
                if canonical_json_bytes(snapshot) not in declared_snapshots:
                    raise ValueError("document pointer names an undeclared source snapshot")
            ids = [doc["command_id"] for doc in documents]
            if ids != sorted(ids) or len(ids) != len(set(ids)):
                raise ValueError("documents must have unique command IDs in ascending order")
            dimensions = manifest["dimensions"]
            raw_vectors = (root / VECTORS_NAME).read_bytes()
            expected_bytes = len(documents) * dimensions * 4
            if len(raw_vectors) != expected_bytes:
                raise ValueError("vector byte length mismatch")
            vectors = np.frombuffer(raw_vectors, dtype="<f4").reshape((len(documents), dimensions))
            if not np.isfinite(vectors).all():
                raise ValueError("vectors contain non-finite values")
            norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
            if not np.all(np.abs(norms - 1.0) <= 0.0001):
                raise ValueError("vectors are not L2 normalized")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise IndexCorrupt(str(error)) from error
        return cls(root, manifest, documents, vectors, {doc["command_id"]: i for i, doc in enumerate(documents)})

    def eligible_indices(self, time_range: dict[str, Any] | None, filters: dict[str, Any] | None) -> list[int]:
        filters = filters or {}
        start = parse_timestamp(time_range["start"], "time_range.start") if time_range else None
        end = parse_timestamp(time_range["end_exclusive"], "time_range.end_exclusive") if time_range else None
        principals = {(item["namespace"], item["id"]) for item in filters.get("principals", [])}
        hosts = set(filters.get("host_ids", []))
        shells = set(filters.get("shell_families", []))
        snapshots = set(filters.get("source_snapshot_ids", []))
        excluded = set(filters.get("exclude_command_ids", []))
        eligible = []
        for index, document in enumerate(self.documents):
            event_time = parse_timestamp(document["event_time"], "event_time")
            principal = document.get("principal_key")
            principal_tuple = (principal.get("namespace"), principal.get("id")) if principal else None
            if start is not None and not start <= event_time < end:
                continue
            if principals and principal_tuple not in principals:
                continue
            if hosts and document.get("host_id") not in hosts:
                continue
            if shells and document["shell_family"] not in shells:
                continue
            if snapshots and document["source_pointer"]["snapshot"]["id"] not in snapshots:
                continue
            if document["command_id"] in excluded:
                continue
            eligible.append(index)
        return eligible

    def exact_search(self, query_vector: np.ndarray, eligible: list[int], top_n: int) -> list[tuple[int, int]]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.manifest["dimensions"],) or not np.isfinite(query).all():
            raise ContractError("query embedding shape or values are invalid")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.0001:
            raise ContractError("query embedding must be L2 normalized")
        if not eligible:
            return []
        selected_vectors = self.vectors[np.asarray(eligible)].astype(np.float64)
        distances = 1.0 - selected_vectors @ query.astype(np.float64)
        ids = np.asarray([self.documents[index]["command_id"] for index in eligible])
        order = np.lexsort((ids, distances))[:top_n]
        return [
            (eligible[int(position)], int(np.rint(float(distances[int(position)]) * 1_000_000)))
            for position in order
        ]

    def pointer_output(self, tool: str, ranked: list[tuple[int, int]], eligible_count: int, top_n: int) -> dict[str, Any]:
        coverage = {
            "status": "complete",
            "indexed_commands": len(self.documents),
            "eligible_commands": eligible_count,
            "requested_top_n": top_n,
            "returned_count": len(ranked),
            "exhausted": len(ranked) == eligible_count,
        }
        if not ranked:
            return {
                "schema_version": "livefire.rag.semantic-result/1",
                "kind": "miss",
                "tool": tool,
                "index": self.manifest["component"],
                "reason": "no commands matched the closed filters and time range",
                "coverage": coverage,
            }
        pointers = []
        for rank, (index, distance) in enumerate(ranked, 1):
            document = self.documents[index]
            pointers.append(
                {
                    "rank": rank,
                    "command_id": document["command_id"],
                    "cosine_distance_millionths": distance,
                    "preview": document["preview"],
                    "source_ref": document["source_pointer"],
                    "metadata": {
                        "event_time": document["event_time"],
                        "shell_family": document["shell_family"],
                        **({"host_id": document["host_id"]} if document.get("host_id") else {}),
                        **({"principal_key": document["principal_key"]} if document.get("principal_key") else {}),
                        **({"source_kind": document["source_kind"]} if document.get("source_kind") else {}),
                    },
                }
            )
        return {
            "schema_version": "livefire.rag.semantic-result/1",
            "kind": "pointer",
            "tool": tool,
            "index": self.manifest["component"],
            "pointers": pointers,
            "coverage": coverage,
        }
