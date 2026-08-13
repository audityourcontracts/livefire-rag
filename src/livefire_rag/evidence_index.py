"""Promotion and exact dense search for generic evidence projection packs.

The canonical Parquet files use a deliberately small physical envelope.  Each
document and occurrence retains its canonical JSON payload while columns needed
for joins and closed filters are materialized beside it.  The payload is the
logical row validated against the public evidence schemas; helper columns are a
non-authoritative cache and are replayed by the verifier.

Promotion verifies and binds an existing projection pack.  It does not create
or claim an SDK admission receipt; admission remains a host responsibility.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from .evidence_builder import (
    DOCUMENT_KINDS,
    EvidencePackCorrupt,
    RelationSource,
    evidence_manifest_identity,
    source_record_profile_ref,
    verify_evidence_pack,
)
from .evidence_bundle import (
    INDEX_FORMAT_DESCRIPTOR,
    INDEX_FORMAT_REF,
    PHYSICAL_PROFILE_REF,
)
from .evidence_derivation import verify_evidence_derivation_pack
from .evidence_schema import generic_schema_root


MANIFEST_NAME = "manifest.json"
BASE_MANIFEST_NAME = "base-index-manifest.json"
FORMAT_DESCRIPTOR_NAME = "index-format-descriptor.json"
DOCUMENTS_NAME = "documents.parquet"
OCCURRENCES_NAME = "occurrences.parquet"
EMBEDDINGS_NAME = "embeddings.parquet"
DERIVATION_DOCUMENTS_NAME = "derivation-documents.parquet"
DERIVATION_MEMBERSHIPS_NAME = "derivation-memberships.parquet"
COVERAGE_NAME = "coverage-report.json"
LOCK_NAME = "objects.lock.json"
BUILD_REPORT_NAME = "build-report.json"
EMBEDDING_PROFILE_NAME = "embedding-profile.json"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
Embedder = Callable[[Sequence[str]], np.ndarray]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class EvidenceIndexError(RuntimeError):
    """Base class for generic evidence-index failures."""


class EvidenceIndexCorrupt(EvidenceIndexError):
    """A promoted index does not replay from its bound projection pack."""


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - optional dependency
        raise EvidenceIndexError(
            "DuckDB is required; install livefire-rag[prototype]"
        ) from error
    return duckdb


def _component(value: Any, label: str) -> dict[str, str]:
    required = {"id", "version", "sha256"}
    allowed = required | {"uri"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - allowed:
        raise ValueError(f"{label} must be a closed component reference")
    if any(not isinstance(value[key], str) or not value[key] for key in value):
        raise ValueError(f"{label} contains an invalid value")
    digest = value["sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label}.sha256 is invalid")
    return value


def _profile_ref(profile: Mapping[str, Any], profile_id: str, version: str) -> dict[str, str]:
    if not profile_id or not version:
        raise ValueError("embedding profile id and version must be non-empty")
    return component_ref(profile_id, version, dict(profile))


def _schema_ref(name: str) -> dict[str, str]:
    schema = json.loads((generic_schema_root() / name).read_text(encoding="utf-8"))
    return component_ref(schema["$id"], "1", schema)


def _canonical_lines(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raise EvidenceIndexCorrupt(f"{path.name}:{line_number} lacks LF")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidenceIndexCorrupt(f"{path.name}:{line_number} is invalid JSON") from error
            if not isinstance(value, dict) or raw != canonical_json_bytes(value, newline=True):
                raise EvidenceIndexCorrupt(f"{path.name}:{line_number} is not canonical JSON")
            yield raw[:-1].decode("utf-8"), value


def _omit_null_object_fields(value: Any, *, preserve_json_nulls: bool = False) -> Any:
    """Omit nullable STRUCT fields while preserving nulls inside JSON values."""

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_is_json = preserve_json_nulls or key == "aggregate_material"
            if child is None and not child_is_json:
                continue
            result[key] = _omit_null_object_fields(
                child, preserve_json_nulls=child_is_json
            )
        return result
    if isinstance(value, list):
        return [
            _omit_null_object_fields(child, preserve_json_nulls=preserve_json_nulls)
            for child in value
        ]
    return value


def _create_tables(connection: Any) -> None:
    connection.execute(
        "CREATE TABLE documents(document_id VARCHAR PRIMARY KEY, document_sha256 VARCHAR, "
        "document_kind VARCHAR, searchable BOOLEAN, occurrence_count BIGINT, semantic_text VARCHAR, "
        "payload_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE occurrences(occurrence_id VARCHAR PRIMARY KEY, document_id VARCHAR, "
        "event_time VARCHAR, relation_namespace VARCHAR, relation_name VARCHAR, "
        "source_snapshot_sha256 VARCHAR, payload_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE derivation_documents(document_id VARCHAR PRIMARY KEY, "
        "document_sha256 VARCHAR, document_kind VARCHAR, searchable BOOLEAN, "
        "occurrence_count BIGINT, semantic_text VARCHAR, payload_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE derivation_memberships(membership_id VARCHAR PRIMARY KEY, "
        "derived_document_id VARCHAR, occurrence_id VARCHAR, input_role VARCHAR, "
        "payload_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE embeddings(schema_version VARCHAR, document_id VARCHAR PRIMARY KEY, "
        "document_sha256 VARCHAR, purpose VARCHAR, "
        "embedding_profile STRUCT(id VARCHAR, \"version\" VARCHAR, sha256 VARCHAR), "
        "dimensions INTEGER, normalization VARCHAR, vector FLOAT[])"
    )


def _load_projection_tables(connection: Any, pack: Path, batch_size: int) -> tuple[int, int]:
    document_rows: list[tuple[Any, ...]] = []
    document_count = 0
    for payload, document in _canonical_lines(pack / "documents.jsonl"):
        projection = document.get("semantic_projection")
        text = projection.get("text") if isinstance(projection, dict) else None
        document_rows.append((
            document["document_id"], document["document_sha256"], document["document_kind"],
            document["searchable"], document["occurrence_count"], text, payload,
        ))
        if len(document_rows) >= batch_size:
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", document_rows)
            document_rows.clear()
        document_count += 1
    if document_rows:
        connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", document_rows)

    occurrence_rows: list[tuple[Any, ...]] = []
    occurrence_count = 0
    for payload, occurrence in _canonical_lines(pack / "occurrences.jsonl"):
        ids = occurrence["document_ids"]
        pointer = occurrence["source_pointer"]
        occurrence_rows.append((
            occurrence["occurrence_id"], ids[0] if ids else None,
            occurrence.get("event_time"), occurrence["relation_identity"]["namespace"],
            occurrence["relation_identity"]["relation"], pointer["snapshot"]["sha256"], payload,
        ))
        if len(occurrence_rows) >= batch_size:
            connection.executemany("INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?)", occurrence_rows)
            occurrence_rows.clear()
        occurrence_count += 1
    if occurrence_rows:
        connection.executemany("INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?)", occurrence_rows)
    return document_count, occurrence_count


def _load_derivation_tables(connection: Any, pack: Path, batch_size: int) -> tuple[int, int]:
    document_rows: list[tuple[Any, ...]] = []
    document_count = 0
    for payload, document in _canonical_lines(pack / "documents.jsonl"):
        projection = document.get("semantic_projection")
        text = projection.get("text") if isinstance(projection, dict) else None
        document_rows.append((
            document["document_id"], document["document_sha256"], document["document_kind"],
            document["searchable"], document["occurrence_count"], text, payload,
        ))
        if len(document_rows) >= batch_size:
            connection.executemany(
                "INSERT INTO derivation_documents VALUES (?, ?, ?, ?, ?, ?, ?)", document_rows
            )
            document_rows.clear()
        document_count += 1
    if document_rows:
        connection.executemany(
            "INSERT INTO derivation_documents VALUES (?, ?, ?, ?, ?, ?, ?)", document_rows
        )

    membership_rows: list[tuple[Any, ...]] = []
    membership_count = 0
    for payload, membership in _canonical_lines(pack / "memberships.jsonl"):
        membership_rows.append((
            membership["membership_id"], membership["derived_document_id"],
            membership["occurrence_id"], membership["input_role"], payload,
        ))
        if len(membership_rows) >= batch_size:
            connection.executemany(
                "INSERT INTO derivation_memberships VALUES (?, ?, ?, ?, ?)", membership_rows
            )
            membership_rows.clear()
        membership_count += 1
    if membership_rows:
        connection.executemany(
            "INSERT INTO derivation_memberships VALUES (?, ?, ?, ?, ?)", membership_rows
        )
    return document_count, membership_count


def _resume_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS vectors(cache_key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, "
        "vector BLOB NOT NULL) WITHOUT ROWID"
    )
    return connection


def _cache_key(profile: dict[str, str], document_id: str, document_sha256: str, text: str) -> str:
    return sha256_bytes(canonical_json_bytes({
        "schema_version": "livefire.rag.embedding-cache-key/1",
        "embedding_profile": profile,
        "document_id": document_id,
        "document_sha256": document_sha256,
        "semantic_text_sha256": sha256_bytes(text.encode("utf-8")),
    }))


def _validate_vectors(vectors: Any, rows: int, dimensions: int, tolerance: float) -> np.ndarray:
    array = np.asarray(vectors, dtype="<f4")
    if array.shape != (rows, dimensions) or not np.isfinite(array).all():
        raise EvidenceIndexError("embedding response has invalid shape or values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= tolerance):
        raise EvidenceIndexError("embedding response violates the bound L2 tolerance")
    return array


def _embed_documents(
    connection: Any,
    resume: sqlite3.Connection,
    embedder: Embedder,
    profile: Mapping[str, Any],
    profile_ref: dict[str, str],
    batch_size: int,
) -> int:
    dimensions = int(profile["dimensions"])
    tolerance = int(profile["output_processing"]["required_l2_norm_tolerance_millionths"]) / 1_000_000
    prefix = str(profile.get("document_prefix", ""))
    inserted = 0
    prior_document_id: str | None = None
    while True:
        if prior_document_id is None:
            source_rows = connection.execute(
                "SELECT document_id, document_sha256, semantic_text FROM embedding_source_documents "
                "WHERE searchable ORDER BY document_id LIMIT ?", [batch_size]
            ).fetchall()
        else:
            source_rows = connection.execute(
                "SELECT document_id, document_sha256, semantic_text FROM embedding_source_documents "
                "WHERE searchable AND document_id > ? ORDER BY document_id LIMIT ?",
                [prior_document_id, batch_size],
            ).fetchall()
        if not source_rows:
            break
        prior_document_id = source_rows[-1][0]
        ready: dict[int, np.ndarray] = {}
        misses: list[tuple[int, str, str]] = []
        for offset, (document_id, document_sha256, semantic_text) in enumerate(source_rows):
            if not isinstance(semantic_text, str) or not semantic_text:
                raise EvidenceIndexError(f"searchable document lacks semantic text: {document_id}")
            text = prefix + semantic_text
            # A byte count is a conservative upper bound for this GGUF
            # tokenizer family: accepting only byte_length <= context avoids
            # relying on the server's undocumented truncation behaviour.
            if len(text.encode("utf-8")) > int(profile["maximum_tokens"]):
                raise EvidenceIndexError(
                    f"document exceeds conservative embedding context bound: {document_id}"
                )
            key = _cache_key(profile_ref, document_id, document_sha256, text)
            cached = resume.execute(
                "SELECT dimensions, vector FROM vectors WHERE cache_key = ?", (key,)
            ).fetchone()
            if cached is None:
                misses.append((offset, key, text))
            else:
                if cached[0] != dimensions or len(cached[1]) != dimensions * 4:
                    raise EvidenceIndexError("resumable embedding cache is corrupt")
                ready[offset] = np.frombuffer(cached[1], dtype="<f4").copy()
        if misses:
            embedded = _validate_vectors(
                embedder([row[2] for row in misses]), len(misses), dimensions, tolerance
            )
            for vector, (offset, key, _) in zip(embedded, misses, strict=True):
                ready[offset] = vector
                resume.execute(
                    "INSERT INTO vectors(cache_key, dimensions, vector) VALUES (?, ?, ?)",
                    (key, dimensions, vector.astype("<f4", copy=False).tobytes()),
                )
            resume.commit()
        ordered = np.stack([ready[index] for index in range(len(source_rows))])
        _validate_vectors(ordered, len(source_rows), dimensions, tolerance)
        connection.executemany(
            "INSERT INTO embeddings VALUES ('livefire.rag.evidence-embedding-row/1', ?, ?, "
            "'semantic_search', ?, ?, 'l2', ?)",
            [
                (document_id, document_sha256, profile_ref, dimensions, vector.tolist())
                for (document_id, document_sha256, _), vector in zip(source_rows, ordered, strict=True)
            ],
        )
        inserted += len(source_rows)
    return inserted


class _LoopbackEmbedder:
    def __init__(self, endpoint: str, model: str, timeout_seconds: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, *, body: bytes | None = None) -> tuple[bytes, Any]:
        request = urllib.request.Request(
            self.endpoint + path, data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
            return raw, json.loads(raw)
        except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceIndexError("embedding endpoint failed") from error

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        _, payload = self._request(
            "/v1/embeddings",
            body=json.dumps(
                {"model": self.model, "input": list(texts)}, separators=(",", ":")
            ).encode(),
        )
        data = sorted(payload.get("data", []), key=lambda row: row.get("index", -1))
        if [row.get("index") for row in data] != list(range(len(texts))):
            raise EvidenceIndexError("embedding response does not preserve complete input order")
        return np.asarray([row["embedding"] for row in data], dtype=np.float32)

    def preflight(
        self, profile: Mapping[str, Any], fixture_path: Path
    ) -> dict[str, Any]:
        raw_fixture = fixture_path.read_bytes()
        try:
            fixture = json.loads(raw_fixture)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceIndexError("embedding conformance fixture is invalid JSON") from error
        conformance = profile["conformance"]
        if sha256_bytes(raw_fixture) != conformance["fixture_sha256"]:
            raise EvidenceIndexError("embedding conformance fixture digest mismatch")
        request_body = fixture.get("request")
        if not isinstance(request_body, dict) or request_body.get("model") != self.model:
            raise EvidenceIndexError("embedding conformance fixture model mismatch")
        inputs = request_body.get("input")
        expected_query = profile["query_composition"].format(
            query_instruction=profile["query_instruction"], query="find encoded powershell"
        )
        if not isinstance(inputs, list) or len(inputs) != 2 or inputs[1] != expected_query:
            raise EvidenceIndexError("embedding conformance fixture generic query mismatch")
        if any(len(value.encode("utf-8")) > profile["maximum_tokens"] for value in inputs):
            raise EvidenceIndexError("embedding conformance input exceeds context bound")
        request_bytes = json.dumps(request_body, separators=(",", ":")).encode()
        vectors = None
        for _ in range(2):
            raw_response, payload = self._request("/v1/embeddings", body=request_bytes)
            vectors = _validate_vectors(
                [
                    row["embedding"]
                    for row in sorted(payload.get("data", []), key=lambda row: row["index"])
                ],
                2, int(profile["dimensions"]),
                int(profile["output_processing"]["required_l2_norm_tolerance_millionths"])
                / 1_000_000,
            )
            try:
                normalized = subprocess.run(
                    ["jq", "-c", "[.data | sort_by(.index)[] | .embedding]"],
                    input=raw_response, check=True, capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise EvidenceIndexError(
                    "jq is required for bound conformance normalization"
                ) from error
            if sha256_bytes(normalized) != conformance["normalized_output_sha256"]:
                raise EvidenceIndexError("embedding conformance output digest mismatch")

        _, models_payload = self._request("/api/v1/models")
        models = models_payload.get("models", [])
        matches = [row for row in models if row.get("key") == self.model]
        if len(matches) != 1:
            raise EvidenceIndexError("declared embedding model is not uniquely exposed")
        model = matches[0]
        instances = [row for row in model.get("loaded_instances", []) if row.get("id") == self.model]
        if len(instances) != 1:
            raise EvidenceIndexError("declared embedding model is not uniquely loaded")
        expected_quantization = str(profile["quantization"]).upper()
        exposed_quantization = str((model.get("quantization") or {}).get("name", "")).upper()
        if (
            model.get("type") != "embedding"
            or model.get("format") != "gguf"
            or exposed_quantization != expected_quantization
            or model.get("size_bytes") != profile["model_objects"][0]["bytes"]
            or instances[0].get("config", {}).get("context_length") != profile["maximum_tokens"]
        ):
            raise EvidenceIndexError("LM Studio exposed model/load identity mismatches profile")
        if profile.get("admission_status") != "development_only":
            raise EvidenceIndexError(
                "LM Studio API cannot prove artifact/runtime/tokenizer identity for production admission"
            )
        return {
            "schema_version": "livefire.rag.embedding-execution-preflight/1",
            "conformance": {
                "fixture_sha256": conformance["fixture_sha256"],
                "normalized_output_sha256": conformance["normalized_output_sha256"],
                "repeatability_probes_in_build": 2,
                "dimensions": int(vectors.shape[1]),
            },
            "overlength_verification": {
                "strategy": "utf8_byte_count_conservative_token_upper_bound",
                "maximum_tokens": profile["maximum_tokens"],
                "server_truncation_relied_upon": False,
            },
            "verified_exposed_properties": {
                "api_model_key": self.model, "type": model["type"],
                "format": model["format"], "quantization": exposed_quantization,
                "model_size_bytes": model["size_bytes"],
                "loaded_context_length": instances[0]["config"]["context_length"],
            },
            "unverifiable_via_lmstudio_api": [
                "model_artifact_sha256", "model_repository_revision", "tokenizer_sha256",
                "pooling", "inference_engine_sha256", "runtime_sha256",
            ],
        }


def loopback_embedder(endpoint: str, model: str, *, timeout_seconds: float = 300.0) -> Embedder:
    """Return an OpenAI-compatible local batch embedding adapter."""

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("embedding endpoint must be loopback HTTP")

    return _LoopbackEmbedder(endpoint, model, timeout_seconds)


_COMPONENT_STRUCTURE = {
    "id": "VARCHAR", "version": "VARCHAR", "sha256": "VARCHAR", "uri": "VARCHAR",
}
_RELATION_STRUCTURE = {
    "namespace": "VARCHAR", "relation": "VARCHAR", "schema_version": "VARCHAR",
    "ocsf_category_uid": "UBIGINT", "ocsf_category_name": "VARCHAR",
    "ocsf_class_uid": "UBIGINT", "ocsf_class_name": "VARCHAR",
    "ocsf_activity_id": "UBIGINT", "ocsf_activity_name": "VARCHAR",
}
_FACET_STRUCTURE = {"name": "VARCHAR", "values": ["VARCHAR"]}
_EXACT_ATTRIBUTE_STRUCTURE = {
    "namespace": "VARCHAR", "path": "VARCHAR", "value": "JSON",
}
_LOGICAL_STRUCTURES: dict[str, dict[str, Any]] = {
    "canonical_documents": {
        "schema_version": "VARCHAR", "document_id": "VARCHAR",
        "document_sha256": "VARCHAR", "document_kind": "VARCHAR",
        "representation": "VARCHAR", "searchable": "BOOLEAN",
        "projection_policy": _COMPONENT_STRUCTURE,
        "derivation_policy": _COMPONENT_STRUCTURE,
        "relation_identities": [_RELATION_STRUCTURE],
        "time_range": {"start": "VARCHAR", "end_exclusive": "VARCHAR"},
        "semantic_projection": {"text": "VARCHAR", "facets": [_FACET_STRUCTURE]},
        "semantic_group": {"group_id": "VARCHAR", "group_key_sha256": "VARCHAR"},
        "occurrence_count": "UBIGINT", "exact_attributes": [_EXACT_ATTRIBUTE_STRUCTURE],
    },
    "canonical_occurrences": {
        "schema_version": "VARCHAR", "occurrence_id": "VARCHAR", "event_time": "VARCHAR",
        "relation_identity": _RELATION_STRUCTURE,
        "source_pointer": {
            "schema_version": "VARCHAR", "snapshot": _COMPONENT_STRUCTURE,
            "snapshot_profile": _COMPONENT_STRUCTURE, "record_id": "VARCHAR",
            "record_sha256": "VARCHAR",
            "locator": {
                "kind": "VARCHAR", "object_sha256": "VARCHAR", "relation": "VARCHAR",
                "row_group": "UBIGINT", "row_ordinal": "UBIGINT",
                "line_ordinal": "UBIGINT", "key_sha256": "VARCHAR",
            },
            "support_refs": ["VARCHAR"], "native_locator_sha256": "VARCHAR",
        },
        "projection_policy": _COMPONENT_STRUCTURE, "terminal_disposition": "VARCHAR",
        "document_ids": ["VARCHAR"], "semantic_group_id": "VARCHAR",
        "reason_codes": ["VARCHAR"], "exact_attributes": [_EXACT_ATTRIBUTE_STRUCTURE],
        "exact_attribute_projection": {
            "contract": "VARCHAR", "selected_count": "UBIGINT",
            "scalars_scanned": "UBIGINT", "known_omitted_scalar_count": "UBIGINT",
            "omitted_subtree_count": "UBIGINT",
            "omission_counts": [{"reason": "VARCHAR", "count": "UBIGINT"}],
            "scan_truncated": "BOOLEAN", "source_hydration_required": "BOOLEAN",
            "limits": {
                "max_attributes": "UBIGINT", "max_scalars_scanned": "UBIGINT",
                "max_list_items": "UBIGINT", "max_string_utf8_bytes": "UBIGINT",
                "max_path_chars": "UBIGINT",
            },
        },
    },
    "canonical_derivation_documents": {
        "schema_version": "VARCHAR", "document_id": "VARCHAR",
        "document_sha256": "VARCHAR", "document_kind": "VARCHAR",
        "representation": "VARCHAR", "searchable": "BOOLEAN",
        "source_snapshot": _COMPONENT_STRUCTURE, "base_projection_pack": _COMPONENT_STRUCTURE,
        "derivation_policy": _COMPONENT_STRUCTURE,
        "relation_identities": [_RELATION_STRUCTURE],
        "time_range": {"start": "VARCHAR", "end": "VARCHAR", "bounds": "VARCHAR"},
        "semantic_projection": {"text": "VARCHAR", "facets": [_FACET_STRUCTURE]},
        "derivation": {
            "group_sha256": "VARCHAR", "input_count": "UBIGINT",
            "input_set_sha256": "VARCHAR", "closure_state": "VARCHAR",
            "completeness_state": "VARCHAR", "aggregate_material": "JSON",
        },
        "occurrence_count": "UBIGINT",
    },
    "canonical_derivation_memberships": {
        "schema_version": "VARCHAR", "membership_id": "VARCHAR",
        "membership_sha256": "VARCHAR", "derived_document_id": "VARCHAR",
        "occurrence_id": "VARCHAR", "input_role": "VARCHAR", "entity_id": "VARCHAR",
        "derivation_policy": _COMPONENT_STRUCTURE,
    },
}


def _materialize_logical_table(connection: Any, envelope: str, logical: str) -> None:
    """Reconstruct typed logical rows without corpus-sized schema aggregation."""

    row_count = connection.execute(f"SELECT count(*) FROM {envelope}").fetchone()[0]
    if row_count == 0:
        if logical not in {
            "canonical_derivation_documents", "canonical_derivation_memberships"
        }:
            raise EvidenceIndexError(f"cannot materialize an empty logical {logical} table")
    structure = json.dumps(_LOGICAL_STRUCTURES[logical], separators=(",", ":"))
    source = (
        f"SELECT json_transform(json(payload_json), ?) AS row_value FROM {envelope}"
        if row_count
        else "SELECT json_transform(json('{}'), ?) AS row_value"
    )
    connection.execute(
        f"CREATE TABLE {logical} AS SELECT row_value.* FROM ({source})"
        + (" WHERE FALSE" if not row_count else ""),
        [structure],
    )


def promote_evidence_pack(
    projection_pack: Path,
    out_dir: Path,
    *,
    relation_sources: Sequence[RelationSource],
    source_snapshot: dict[str, str],
    projection_policy: dict[str, str],
    sdk_specs: Path,
    embedding_profile: Mapping[str, Any],
    embedding_profile_id: str,
    embedding_profile_version: str,
    embedder: Embedder,
    embedding_conformance_fixture: Path,
    source_admission_receipt: dict[str, str],
    index_id: str,
    version: str,
    derivation_pack: Path | None = None,
    index_uri: str | None = None,
    resume_dir: Path | None = None,
    batch_size: int = 32,
    pilot_sample: Path | None = None,
) -> dict[str, Any]:
    """Promote a verified projection pack; never issue an SDK admission claim."""

    projection_pack = Path(projection_pack)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence index: {out_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _component(source_admission_receipt, "source_admission_receipt")
    profile = dict(embedding_profile)
    if profile.get("purpose") != "semantic_search" or profile.get("normalization") != "l2":
        raise ValueError("embedding profile must be L2 semantic_search")
    dimensions = profile.get("dimensions")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("embedding profile dimensions are invalid")
    profile_ref = _profile_ref(profile, embedding_profile_id, embedding_profile_version)
    maximum_batch_items = profile.get("batching", {}).get("maximum_batch_items")
    if (
        isinstance(maximum_batch_items, bool)
        or not isinstance(maximum_batch_items, int)
        or batch_size > maximum_batch_items
    ):
        raise ValueError("batch_size exceeds the embedding profile maximum")
    preflight = getattr(embedder, "preflight", None)
    if not callable(preflight):
        raise EvidenceIndexError("embedder lacks mandatory execution preflight")
    preflight_report = preflight(profile, Path(embedding_conformance_fixture))
    if not isinstance(preflight_report, dict):
        raise EvidenceIndexError("embedder preflight did not return a report")

    pilot_manifest: dict[str, Any] | None = None
    if pilot_sample is None:
        pack_manifest = verify_evidence_pack(
            projection_pack, source_snapshot=source_snapshot, relation_sources=relation_sources,
            projection_policy=projection_policy, sdk_specs=sdk_specs,
        )
        input_pack = projection_pack
    else:
        if derivation_pack is not None:
            raise EvidenceIndexError("pilot promotion does not support a derivation overlay")
        from .evidence_pilot import verify_evidence_pilot_sample

        pilot_manifest = verify_evidence_pilot_sample(
            Path(pilot_sample), projection_pack=projection_pack
        )
        pack_manifest = json.loads((projection_pack / MANIFEST_NAME).read_text(encoding="utf-8"))
        if pack_manifest.get("source_snapshots") != [source_snapshot]:
            raise EvidenceIndexError("pilot source snapshot binding mismatch")
        if pack_manifest.get("projection_policy") != projection_policy:
            raise EvidenceIndexError("pilot projection policy binding mismatch")
        input_pack = Path(pilot_sample)
    derivation_manifest: dict[str, Any] | None = None
    if derivation_pack is not None:
        derivation_pack = Path(derivation_pack)
        derivation_manifest = verify_evidence_derivation_pack(derivation_pack)
        if derivation_manifest.get("base_projection_pack") != pack_manifest["component"]:
            raise EvidenceIndexError("derivation pack does not bind the verified projection pack")
        if derivation_manifest.get("source_snapshot") != source_snapshot:
            raise EvidenceIndexError("derivation pack does not bind the verified source snapshot")
    duckdb = _duckdb()
    # These identities are shared with the standalone provider bundle.  The
    # promoter must emit the provider's exact format contract, not a private
    # look-alike, or the host must correctly reject the mounted index.
    physical_ref = dict(PHYSICAL_PROFILE_REF)
    descriptor = json.loads(json.dumps(INDEX_FORMAT_DESCRIPTOR))
    format_ref = dict(INDEX_FORMAT_REF)
    builder_ref = component_ref(
        "livefire.rag.evidence-index-promoter", "1",
        {"implementation": "livefire_rag.evidence_index", "contract": "1"},
    )
    record_identity_ref = component_ref(
        "livefire.rag.evidence-occurrence-pointer-policy", "1",
        {"identity": "unchanged_projection_pack_source_pointer", "contract": "1"},
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    resume_root = Path(resume_dir) if resume_dir else out_dir.parent / f".{out_dir.name}.resume"
    resume = _resume_database(resume_root / "embeddings.sqlite3")
    connection = duckdb.connect(str(staging / "promotion.duckdb"))
    try:
        _create_tables(connection)
        document_count, occurrence_count = _load_projection_tables(
            connection, input_pack, max(batch_size, 1024)
        )
        derived_document_count = 0
        derivation_membership_count = 0
        if derivation_pack is not None:
            derived_document_count, derivation_membership_count = _load_derivation_tables(
                connection, derivation_pack, max(batch_size, 1024)
            )
            if connection.execute(
                "SELECT count(*) FROM derivation_documents dd JOIN documents d USING(document_id)"
            ).fetchone()[0]:
                raise EvidenceIndexError("base and derived document identifiers collide")
            if connection.execute(
                "SELECT count(*) FROM derivation_memberships m "
                "LEFT JOIN derivation_documents d ON m.derived_document_id=d.document_id "
                "LEFT JOIN occurrences o USING(occurrence_id) "
                "WHERE d.document_id IS NULL OR o.occurrence_id IS NULL"
            ).fetchone()[0]:
                raise EvidenceIndexError("derivation membership is not closed over base occurrences")
        connection.execute(
            "CREATE VIEW embedding_source_documents AS "
            "SELECT document_id, document_sha256, searchable, semantic_text FROM documents "
            "UNION ALL SELECT document_id, document_sha256, searchable, semantic_text "
            "FROM derivation_documents"
        )
        embedded_count = _embed_documents(
            connection, resume, embedder, profile, profile_ref, batch_size
        )
        searchable_count = connection.execute(
            "SELECT count(*) FROM embedding_source_documents WHERE searchable"
        ).fetchone()[0]
        if searchable_count == 0:
            raise EvidenceIndexError("a searchable evidence index requires at least one searchable document")
        if embedded_count != searchable_count:
            raise EvidenceIndexError("embedding coverage is incomplete")
        _materialize_logical_table(connection, "documents", "canonical_documents")
        _materialize_logical_table(connection, "occurrences", "canonical_occurrences")
        tables = [
            ("canonical_documents", DOCUMENTS_NAME, "document_id"),
            ("canonical_occurrences", OCCURRENCES_NAME, "occurrence_id"),
            ("embeddings", EMBEDDINGS_NAME, "document_id"),
        ]
        if derivation_pack is not None:
            _materialize_logical_table(
                connection, "derivation_documents", "canonical_derivation_documents"
            )
            _materialize_logical_table(
                connection, "derivation_memberships", "canonical_derivation_memberships"
            )
            membership_columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info('canonical_derivation_memberships')"
                ).fetchall()
            }
            if "entity_id" not in membership_columns:
                connection.execute(
                    "ALTER TABLE canonical_derivation_memberships ADD COLUMN entity_id VARCHAR"
                )
            tables.extend([
                ("canonical_derivation_documents", DERIVATION_DOCUMENTS_NAME, "document_id"),
                ("canonical_derivation_memberships", DERIVATION_MEMBERSHIPS_NAME, "membership_id"),
            ])
        for table, filename, order in tables:
            connection.execute(
                f"COPY (SELECT * FROM {table} ORDER BY {order}) TO ? "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)",
                [str(staging / filename)],
            )
        connection.close()
        connection = None
        (staging / "promotion.duckdb").unlink()
        if pilot_manifest is None:
            shutil.copyfile(input_pack / "coverage-report.json", staging / COVERAGE_NAME)
        else:
            from .evidence_pilot import pilot_projection_coverage

            write_canonical_json(
                staging / COVERAGE_NAME,
                pilot_projection_coverage(input_pack, projection_pack),
            )
        write_canonical_json(staging / EMBEDDING_PROFILE_NAME, profile)

        data_artifacts = [
            artifact_ref(staging / DOCUMENTS_NAME, DOCUMENTS_NAME, PARQUET_MEDIA_TYPE),
            artifact_ref(staging / OCCURRENCES_NAME, OCCURRENCES_NAME, PARQUET_MEDIA_TYPE),
            artifact_ref(staging / EMBEDDINGS_NAME, EMBEDDINGS_NAME, PARQUET_MEDIA_TYPE),
            artifact_ref(staging / EMBEDDING_PROFILE_NAME, EMBEDDING_PROFILE_NAME, "application/json"),
            artifact_ref(staging / COVERAGE_NAME, COVERAGE_NAME, "application/json"),
        ]
        if derivation_pack is not None:
            data_artifacts.extend([
                artifact_ref(
                    staging / DERIVATION_DOCUMENTS_NAME,
                    DERIVATION_DOCUMENTS_NAME,
                    PARQUET_MEDIA_TYPE,
                ),
                artifact_ref(
                    staging / DERIVATION_MEMBERSHIPS_NAME,
                    DERIVATION_MEMBERSHIPS_NAME,
                    PARQUET_MEDIA_TYPE,
                ),
            ])
        base_manifest = {
            "schema_version": "livefire.index/1",
            "index_id": f"{index_id}.base", "index_version": version,
            "index_kind": "generic_security_evidence",
            "format": format_ref, "builder": builder_ref,
            "source_bindings": [{
                "source_snapshot": source_snapshot,
                "source_snapshot_profile": source_record_profile_ref(),
                "source_admission_receipt": source_admission_receipt,
                "record_identity_policy": record_identity_ref,
            }],
            "policies": {"projection": projection_policy, "embedding": profile_ref},
            "objects": data_artifacts,
            "source_pointer_table": next(
                row for row in data_artifacts if row["path"] == OCCURRENCES_NAME
            ),
            "coverage": {
                "source_records": occurrence_count,
                "indexed_documents": searchable_count,
                "excluded_records": occurrence_count - int(
                    json.loads((staging / COVERAGE_NAME).read_text())["closure"]
                    ["by_terminal_disposition"]["semantic_group_occurrence"]
                ),
                "reason_counts": {},
            },
            "query_time_contract": {
                "mode": "local_component", "network": ["loopback:openai-compatible-embeddings"],
                "secret_handles": [], "vendor_services": [],
                "required_local_components": [profile_ref],
            },
            "governance": {
                "inherits_source_confidentiality": True, "inherits_source_retention": True,
            },
        }
        write_canonical_json(staging / BASE_MANIFEST_NAME, base_manifest)
        base_ref = component_ref(f"{index_id}.base", version, base_manifest)
        write_canonical_json(staging / FORMAT_DESCRIPTOR_NAME, descriptor)
        report = {
            "schema_version": "livefire.rag.evidence-index-build-report/1",
            "admission_status": "not_sdk_admitted",
            "projection_pack": pack_manifest["component"],
            "documents": document_count, "derived_documents": derived_document_count,
            "occurrences": occurrence_count,
            "derivation_memberships": derivation_membership_count,
            "searchable_documents": searchable_count, "embeddings": embedded_count,
            "embedding_profile": profile_ref,
            "writer": {"implementation": "duckdb", "version": duckdb.__version__},
            "embedding_execution_preflight": preflight_report,
        }
        if pilot_manifest is not None:
            from .evidence_pilot import pilot_index_binding

            report["pilot_sample"] = pilot_index_binding(pilot_manifest)
        write_canonical_json(staging / BUILD_REPORT_NAME, report)
        data_artifacts.extend([
            artifact_ref(staging / BASE_MANIFEST_NAME, BASE_MANIFEST_NAME, "application/json"),
            artifact_ref(staging / FORMAT_DESCRIPTOR_NAME, FORMAT_DESCRIPTOR_NAME, "application/json"),
            artifact_ref(staging / BUILD_REPORT_NAME, BUILD_REPORT_NAME, "application/json"),
        ])
        data_artifacts.sort(key=lambda row: row["path"])
        write_canonical_json(staging / LOCK_NAME, {
            "schema_version": "livefire.object-lock/1", "objects": data_artifacts
        })
        role_by_path = {
            DOCUMENTS_NAME: "documents", OCCURRENCES_NAME: "occurrences",
            EMBEDDINGS_NAME: "embeddings", COVERAGE_NAME: "coverage_report",
            EMBEDDING_PROFILE_NAME: "embedding_profile",
            BASE_MANIFEST_NAME: "base_manifest", FORMAT_DESCRIPTOR_NAME: "format_descriptor",
            BUILD_REPORT_NAME: "build_report",
            DERIVATION_DOCUMENTS_NAME: "derivation_documents",
            DERIVATION_MEMBERSHIPS_NAME: "derivation_memberships",
        }
        objects = {role_by_path[row["path"]]: row for row in data_artifacts}
        objects["object_lock"] = artifact_ref(staging / LOCK_NAME, LOCK_NAME, "application/json")

        component = {"id": index_id, "version": version, "sha256": ""}
        if index_uri:
            component["uri"] = index_uri
        manifest = {
            "schema_version": "livefire.rag.evidence-index/1", "component": component,
            "projection_pack": pack_manifest["component"],
            "base_index_manifest": base_ref,
            "index_format_descriptor": format_ref, "physical_profile": physical_ref,
            "source_snapshots": pack_manifest["source_snapshots"],
            "document_kinds": list(DOCUMENT_KINDS),
            "row_schemas": {
                "evidence_document": _schema_ref("evidence-document.v1.schema.json"),
                "evidence_occurrence": _schema_ref("evidence-occurrence-row.v1.schema.json"),
                "evidence_embedding": _schema_ref("evidence-embedding-row.v1.schema.json"),
                "coverage_report": _schema_ref("evidence-coverage-report.v1.schema.json"),
            },
            "projection_policy": projection_policy,
            "derivation_policies": (
                [derivation_manifest["derivation_policy"]] if derivation_manifest else []
            ),
            "embedding_profiles": [profile_ref], "objects": objects,
            "coverage": {
                "source_record_count": occurrence_count,
                "terminal_disposition_count": occurrence_count,
                "document_count": document_count + derived_document_count,
                "searchable_document_count": searchable_count,
                "unaccounted_record_count": 0, "unresolved_pointer_count": 0,
            },
            "query_contract": {
                "canonical_format": "parquet", "source_filters_apply_to_occurrences": True,
                "semantic_groups_preserve_occurrences": True,
                "derived_caches_authoritative": False, "candidate_results_are_evidence": False,
                "tie_break": "ranking_score_desc_document_id_asc",
            },
        }
        if pilot_manifest is not None:
            from .evidence_pilot import pilot_index_binding

            manifest["pilot_sample"] = pilot_index_binding(pilot_manifest)
        if derivation_manifest is not None:
            manifest["derivation_packs"] = [derivation_manifest["component"]]
            manifest["row_schemas"].update({
                "derivation_document": _schema_ref("evidence-derived-document.v1.schema.json"),
                "derivation_membership": _schema_ref(
                    "evidence-derivation-membership-row.v1.schema.json"
                ),
            })
            manifest["coverage"].update({
                "derived_document_count": derived_document_count,
                "derivation_membership_count": derivation_membership_count,
            })
        manifest["component"]["sha256"] = canonical_sha256_omitting(
            manifest, ("component", "sha256")
        )
        write_canonical_json(staging / MANIFEST_NAME, manifest)
        verify_promoted_evidence_index(
            staging, projection_pack=(projection_pack if pilot_manifest is None else None),
            pilot_sample=(Path(pilot_sample) if pilot_manifest is not None else None),
            derivation_pack=derivation_pack,
            sdk_specs=sdk_specs,
        )
        if out_dir.exists():
            raise FileExistsError(f"refusing to overwrite evidence index: {out_dir}")
        os.rename(staging, out_dir)
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        resume.close()


def _verify_artifact(root: Path, ref: Any, name: str) -> None:
    if not isinstance(ref, dict) or ref.get("path") != name:
        raise EvidenceIndexCorrupt(f"invalid artifact reference: {name}")
    path = root / name
    if not path.is_file() or path.stat().st_size != ref.get("bytes") or sha256_file(path) != ref.get("sha256"):
        raise EvidenceIndexCorrupt(f"artifact digest mismatch: {name}")


def verify_promoted_evidence_index(
    root: Path, *, projection_pack: Path | None = None, sdk_specs: Path,
    derivation_pack: Path | None = None,
    pilot_sample: Path | None = None,
) -> dict[str, Any]:
    """Verify a sealed index locally, optionally replaying its source packs."""

    root = Path(root)
    projection_pack = Path(projection_pack) if projection_pack is not None else None
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    pilot_manifest: dict[str, Any] | None = None
    if pilot_sample is not None:
        from .evidence_pilot import pilot_index_binding, verify_evidence_pilot_sample

        pilot_manifest = verify_evidence_pilot_sample(Path(pilot_sample))
        if manifest.get("pilot_sample") != pilot_index_binding(pilot_manifest):
            raise EvidenceIndexCorrupt("pilot-sample binding mismatch")
        if manifest.get("projection_pack") != pilot_manifest.get("projection_pack"):
            raise EvidenceIndexCorrupt("pilot projection-pack binding mismatch")
    elif "pilot_sample" in manifest:
        # Standalone verification still validates the sealed index and typed
        # binding; replay of the optional sampling artifact is explicit.
        binding = manifest["pilot_sample"]
        if binding.get("scope_status") != "sample_only_not_corpus_coverage" or binding.get("corpus_miss_definitive") is not False:
            raise EvidenceIndexCorrupt("pilot scope binding is invalid")
    pack_manifest: dict[str, Any] | None = None
    if projection_pack is not None:
        pack_manifest = json.loads((projection_pack / "manifest.json").read_text(encoding="utf-8"))
        if evidence_manifest_identity(pack_manifest) != pack_manifest.get("component", {}).get("sha256"):
            raise EvidenceIndexCorrupt("bound projection-pack identity is invalid")
        if manifest.get("projection_pack") != pack_manifest.get("component"):
            raise EvidenceIndexCorrupt("projection-pack binding mismatch")
    derivation_manifest: dict[str, Any] | None = None
    if derivation_pack is not None:
        derivation_pack = Path(derivation_pack)
        derivation_manifest = verify_evidence_derivation_pack(derivation_pack)
        if manifest.get("derivation_packs") != [derivation_manifest["component"]]:
            raise EvidenceIndexCorrupt("derivation-pack binding mismatch")
        if pack_manifest is None:
            raise EvidenceIndexCorrupt("derivation replay requires projection-pack replay")
        if derivation_manifest.get("base_projection_pack") != pack_manifest.get("component"):
            raise EvidenceIndexCorrupt("derivation pack base binding mismatch")
        if manifest.get("derivation_policies") != [derivation_manifest["derivation_policy"]]:
            raise EvidenceIndexCorrupt("derivation policy binding mismatch")
    has_derivation = "derivation_documents" in manifest.get("objects", {})
    if has_derivation != ("derivation_memberships" in manifest.get("objects", {})):
        raise EvidenceIndexCorrupt("derivation object pair is incomplete")
    if derivation_pack is None and has_derivation and not manifest.get("derivation_packs"):
        raise EvidenceIndexCorrupt("derivation objects lack a bound derivation pack")
    if canonical_sha256_omitting(manifest, ("component", "sha256")) != manifest["component"]["sha256"]:
        raise EvidenceIndexCorrupt("evidence-index component identity mismatch")
    expected_objects = {
        "documents": DOCUMENTS_NAME, "occurrences": OCCURRENCES_NAME,
        "embeddings": EMBEDDINGS_NAME, "embedding_profile": EMBEDDING_PROFILE_NAME,
        "coverage_report": COVERAGE_NAME,
        "base_manifest": BASE_MANIFEST_NAME, "format_descriptor": FORMAT_DESCRIPTOR_NAME,
        "build_report": BUILD_REPORT_NAME,
        "object_lock": LOCK_NAME,
    }
    if has_derivation:
        expected_objects.update({
            "derivation_documents": DERIVATION_DOCUMENTS_NAME,
            "derivation_memberships": DERIVATION_MEMBERSHIPS_NAME,
        })
    if set(manifest.get("objects", {})) != set(expected_objects):
        raise EvidenceIndexCorrupt("evidence-index object set mismatch")
    for role, name in expected_objects.items():
        _verify_artifact(root, manifest["objects"][role], name)
    locked = [manifest["objects"][role] for role in expected_objects if role != "object_lock"]
    locked.sort(key=lambda row: row["path"])
    lock = json.loads((root / LOCK_NAME).read_text(encoding="utf-8"))
    if lock != {"schema_version": "livefire.object-lock/1", "objects": locked}:
        raise EvidenceIndexCorrupt("object lock mismatch")
    if projection_pack is not None and pilot_sample is None and (root / COVERAGE_NAME).read_bytes() != (projection_pack / "coverage-report.json").read_bytes():
        raise EvidenceIndexCorrupt("coverage report differs from projection pack")

    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        docs = str((root / DOCUMENTS_NAME).resolve())
        occs = str((root / OCCURRENCES_NAME).resolve())
        embs = str((root / EMBEDDINGS_NAME).resolve())
        document_count = connection.execute("SELECT count(*) FROM read_parquet(?)", [docs]).fetchone()[0]
        occurrence_count = connection.execute("SELECT count(*) FROM read_parquet(?)", [occs]).fetchone()[0]
        embedding_count = connection.execute("SELECT count(*) FROM read_parquet(?)", [embs]).fetchone()[0]
        derived_document_count = 0
        derivation_membership_count = 0
        derived_docs: str | None = None
        memberships: str | None = None
        if has_derivation:
            derived_docs = str((root / DERIVATION_DOCUMENTS_NAME).resolve())
            memberships = str((root / DERIVATION_MEMBERSHIPS_NAME).resolve())
            derived_document_count = connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [derived_docs]
            ).fetchone()[0]
            derivation_membership_count = connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [memberships]
            ).fetchone()[0]
        if (document_count, occurrence_count, embedding_count) != (
            manifest["coverage"]["document_count"] - derived_document_count,
            manifest["coverage"]["source_record_count"],
            manifest["coverage"]["searchable_document_count"],
        ):
            raise EvidenceIndexCorrupt("promoted row counts do not reconcile")
        if has_derivation and (
            derived_document_count != manifest["coverage"]["derived_document_count"]
            or derivation_membership_count
            != manifest["coverage"]["derivation_membership_count"]
        ):
            raise EvidenceIndexCorrupt("derivation row counts do not reconcile")
        document_union = "SELECT document_id, document_sha256, searchable FROM read_parquet(?)"
        union_parameters: list[Any] = [docs]
        if derived_docs is not None:
            document_union += " UNION ALL SELECT document_id, document_sha256, searchable FROM read_parquet(?)"
            union_parameters.append(derived_docs)
        violations = connection.execute(
            f"SELECT count(*) FROM ({document_union}) d FULL JOIN read_parquet(?) e USING(document_id) "
            "WHERE (d.searchable AND e.document_id IS NULL) OR (NOT d.searchable AND e.document_id IS NOT NULL) "
            "OR (e.document_id IS NOT NULL AND (d.document_sha256 != e.document_sha256 "
            "OR len(e.vector) != e.dimensions OR e.normalization != 'l2'))",
            [*union_parameters, embs],
        ).fetchone()[0]
        if violations:
            raise EvidenceIndexCorrupt("embedding/document coverage mismatch")
        cursor = connection.execute("SELECT vector FROM read_parquet(?) ORDER BY document_id", [embs])
        while True:
            rows = cursor.fetchmany(4096)
            if not rows:
                break
            array = np.asarray([row[0] for row in rows], dtype=np.float32)
            if not np.isfinite(array).all() or np.any(
                np.abs(np.linalg.norm(array.astype(np.float64), axis=1) - 1.0) > 0.0001
            ):
                raise EvidenceIndexCorrupt("embedding vector is not finite and L2 normalized")
        membership = connection.execute(
            "SELECT count(*) FROM read_parquet(?) o LEFT JOIN read_parquet(?) d "
            "ON list_extract(o.document_ids, 1)=d.document_id "
            "WHERE len(o.document_ids) > 0 AND d.document_id IS NULL",
            [occs, docs],
        ).fetchone()[0]
        if membership:
            raise EvidenceIndexCorrupt("occurrence references an unknown document")
        count_mismatch = connection.execute(
            "SELECT count(*) FROM read_parquet(?) d LEFT JOIN "
            "(SELECT list_extract(document_ids, 1) AS document_id, count(*) AS n "
            "FROM read_parquet(?) WHERE len(document_ids) > 0 GROUP BY document_id) o USING(document_id) "
            "WHERE d.occurrence_count != coalesce(o.n, 0)", [docs, occs],
        ).fetchone()[0]
        if count_mismatch:
            raise EvidenceIndexCorrupt("document occurrence counts do not reconcile")
        if memberships is not None and derived_docs is not None:
            derivation_violations = connection.execute(
                "SELECT count(*) FROM read_parquet(?) m "
                "LEFT JOIN read_parquet(?) d ON m.derived_document_id=d.document_id "
                "LEFT JOIN read_parquet(?) o USING(occurrence_id) "
                "WHERE d.document_id IS NULL OR o.occurrence_id IS NULL",
                [memberships, derived_docs, occs],
            ).fetchone()[0]
            if derivation_violations:
                raise EvidenceIndexCorrupt("derivation membership pointer closure failed")
            derivation_count_mismatch = connection.execute(
                "SELECT count(*) FROM read_parquet(?) d LEFT JOIN "
                "(SELECT derived_document_id AS document_id, count(DISTINCT occurrence_id) AS n "
                "FROM read_parquet(?) GROUP BY derived_document_id) m USING(document_id) "
                "WHERE d.occurrence_count != coalesce(m.n, 0)",
                [derived_docs, memberships],
            ).fetchone()[0]
            if derivation_count_mismatch:
                raise EvidenceIndexCorrupt("derived document occurrence counts do not reconcile")

        replay_pack = Path(pilot_sample) if pilot_sample is not None else projection_pack
        for table_path, jsonl_path, identity in (() if replay_pack is None else (
            (docs, replay_pack / "documents.jsonl", "document_id"),
            (occs, replay_pack / "occurrences.jsonl", "occurrence_id"),
        )):
            cursor = connection.execute(
                f"SELECT to_json(row_value) FROM read_parquet(?) AS row_value ORDER BY {identity}",
                [table_path],
            )
            with jsonl_path.open("rb") as source:
                while True:
                    rows = cursor.fetchmany(4096)
                    if not rows:
                        break
                    for (payload,) in rows:
                        source_row = json.loads(source.readline())
                        logical_row = _omit_null_object_fields(json.loads(payload))
                        if canonical_json_bytes(source_row) != canonical_json_bytes(logical_row):
                            raise EvidenceIndexCorrupt(f"{identity} payload differs from projection pack")
                if source.read(1):
                    raise EvidenceIndexCorrupt(f"{identity} projection pack has extra rows")
        if derivation_pack is not None:
            for table_path, jsonl_path, identity in (
                (derived_docs, derivation_pack / "documents.jsonl", "document_id"),
                (memberships, derivation_pack / "memberships.jsonl", "membership_id"),
            ):
                parquet_cursor = connection.execute(
                    f"SELECT to_json(row_value) FROM read_parquet(?) AS row_value ORDER BY {identity}",
                    [table_path],
                )
                source_connection = _duckdb().connect()
                try:
                    source_cursor = source_connection.execute(
                        "SELECT json::VARCHAR FROM read_ndjson_objects(?) "
                        f"ORDER BY json_extract_string(json, '$.{identity}')",
                        [str(jsonl_path.resolve())],
                    )
                    while parquet_rows := parquet_cursor.fetchmany(4096):
                        source_rows = source_cursor.fetchmany(len(parquet_rows))
                        if len(source_rows) != len(parquet_rows):
                            raise EvidenceIndexCorrupt(
                                f"{identity} derivation pack is truncated"
                            )
                        for (payload,), (source_payload,) in zip(
                            parquet_rows, source_rows, strict=True
                        ):
                            if canonical_json_bytes(
                                _omit_null_object_fields(json.loads(source_payload))
                            ) != canonical_json_bytes(
                                _omit_null_object_fields(json.loads(payload))
                            ):
                                raise EvidenceIndexCorrupt(
                                    f"{identity} payload differs from derivation pack"
                                )
                    if source_cursor.fetchone() is not None:
                        raise EvidenceIndexCorrupt(
                            f"{identity} derivation pack has extra rows"
                        )
                finally:
                    source_connection.close()
    finally:
        connection.close()

    # Validate the two manifests against the exact supplied SDK/RAG schemas.
    from .evidence_schema import _offline_registry
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Resource

    registry, schemas = _offline_registry(None, Path(sdk_specs))
    try:
        Draft202012Validator(
            schemas["evidence-index-manifest.v1.schema.json"], registry=registry,
            format_checker=FormatChecker(),
        ).validate(manifest)
        expected_row_schemas = {
            "evidence_document": _schema_ref("evidence-document.v1.schema.json"),
            "evidence_occurrence": _schema_ref("evidence-occurrence-row.v1.schema.json"),
            "evidence_embedding": _schema_ref("evidence-embedding-row.v1.schema.json"),
            "coverage_report": _schema_ref("evidence-coverage-report.v1.schema.json"),
        }
        if has_derivation:
            expected_row_schemas.update({
                "derivation_document": _schema_ref(
                    "evidence-derived-document.v1.schema.json"
                ),
                "derivation_membership": _schema_ref(
                    "evidence-derivation-membership-row.v1.schema.json"
                ),
            })
        if manifest.get("row_schemas") != expected_row_schemas:
            raise EvidenceIndexCorrupt("index row-schema bindings do not match")
        base = json.loads((root / BASE_MANIFEST_NAME).read_text(encoding="utf-8"))
        base_schema = json.loads(
            (Path(sdk_specs) / "index-manifest.v1.schema.json").read_text(encoding="utf-8")
        )
        registry = registry.with_resource(base_schema["$id"], Resource.from_contents(base_schema))
        Draft202012Validator(base_schema, registry=registry, format_checker=FormatChecker()).validate(base)
        if component_ref(f"{manifest['component']['id']}.base", manifest["component"]["version"], base) != manifest["base_index_manifest"]:
            raise EvidenceIndexCorrupt("base index manifest identity mismatch")
        descriptor = json.loads((root / FORMAT_DESCRIPTOR_NAME).read_text(encoding="utf-8"))
        descriptor_schema = json.loads(
            (Path(sdk_specs) / "index-format-descriptor.v1.schema.json").read_text(encoding="utf-8")
        )
        registry = registry.with_resource(
            descriptor_schema["$id"], Resource.from_contents(descriptor_schema)
        )
        Draft202012Validator(
            descriptor_schema, registry=registry, format_checker=FormatChecker()
        ).validate(descriptor)
        if descriptor != INDEX_FORMAT_DESCRIPTOR or descriptor.get("format") != manifest["index_format_descriptor"]:
            raise EvidenceIndexCorrupt("index format descriptor identity mismatch")
        if manifest.get("physical_profile") != PHYSICAL_PROFILE_REF:
            raise EvidenceIndexCorrupt("index physical profile binding mismatch")
        profile = json.loads((root / EMBEDDING_PROFILE_NAME).read_text(encoding="utf-8"))
        Draft202012Validator(
            schemas["embedding-policy.v1.schema.json"], registry=registry,
            format_checker=FormatChecker(),
        ).validate(profile)
        profile_ref = manifest["embedding_profiles"][0]
        if component_ref(profile_ref["id"], profile_ref["version"], profile) != profile_ref:
            raise EvidenceIndexCorrupt("embedding profile identity mismatch")
        embedding_validator = Draft202012Validator(
            schemas["evidence-embedding-row.v1.schema.json"], registry=registry,
            format_checker=FormatChecker(),
        )
        connection = _duckdb().connect()
        try:
            for filename, schema_name in (
                (DOCUMENTS_NAME, "evidence-document.v1.schema.json"),
                (OCCURRENCES_NAME, "evidence-occurrence-row.v1.schema.json"),
                *((
                    (DERIVATION_DOCUMENTS_NAME, "evidence-derived-document.v1.schema.json"),
                    (DERIVATION_MEMBERSHIPS_NAME, "evidence-derivation-membership-row.v1.schema.json"),
                ) if has_derivation else ()),
            ):
                logical_validator = Draft202012Validator(
                    schemas[schema_name], registry=registry, format_checker=FormatChecker()
                )
                logical_cursor = connection.execute(
                    "SELECT to_json(row_value) FROM read_parquet(?) AS row_value ORDER BY 1",
                    [str((root / filename).resolve())],
                )
                while True:
                    logical_rows = logical_cursor.fetchmany(1024)
                    if not logical_rows:
                        break
                    for (payload,) in logical_rows:
                        logical_validator.validate(_omit_null_object_fields(json.loads(payload)))
            cursor = connection.execute(
                "SELECT schema_version, document_id, document_sha256, purpose, embedding_profile, "
                "dimensions, normalization, vector FROM read_parquet(?) ORDER BY document_id",
                [str((root / EMBEDDINGS_NAME).resolve())],
            )
            while True:
                rows = cursor.fetchmany(1024)
                if not rows:
                    break
                for row in rows:
                    logical_embedding = {
                        "schema_version": row[0], "document_id": row[1],
                        "document_sha256": row[2], "purpose": row[3],
                        "embedding_profile": row[4], "dimensions": row[5],
                        "normalization": row[6], "vector": row[7],
                    }
                    embedding_validator.validate(logical_embedding)
                    if row[4] != profile_ref:
                        raise EvidenceIndexCorrupt("embedding row profile mismatch")
        finally:
            connection.close()
    except EvidenceIndexCorrupt:
        raise
    except Exception as error:
        raise EvidenceIndexCorrupt(f"index schema validation failed: {error}") from error
    return manifest


@dataclass
class EvidenceIndex:
    root: Path
    manifest: dict[str, Any]
    connection: Any
    profile: dict[str, Any]
    _closed: bool = False

    @property
    def component(self) -> dict[str, Any]:
        return self.manifest["component"]

    @property
    def source_snapshots(self) -> list[dict[str, Any]]:
        return self.manifest["source_snapshots"]

    @property
    def embedding_profile(self) -> dict[str, Any]:
        return self.manifest["embedding_profiles"][0]

    @classmethod
    def open(
        cls, root: Path, *, sdk_specs: Path,
        expected_format: Mapping[str, Any] | None = None,
        projection_pack: Path | None = None,
        derivation_pack: Path | None = None,
        replay_verify: bool = False,
    ) -> "EvidenceIndex":
        if replay_verify and projection_pack is None:
            raise ValueError("replay_verify requires projection_pack")
        if not replay_verify and (projection_pack is not None or derivation_pack is not None):
            raise ValueError("source packs are accepted only with replay_verify=True")
        manifest = verify_promoted_evidence_index(
            root,
            projection_pack=projection_pack if replay_verify else None,
            derivation_pack=derivation_pack if replay_verify else None,
            sdk_specs=sdk_specs,
        )
        if expected_format is not None and manifest["index_format_descriptor"] != dict(expected_format):
            raise EvidenceIndexCorrupt("evidence index format is incompatible")
        connection = _duckdb().connect()
        connection.read_parquet(str((Path(root) / DOCUMENTS_NAME).resolve())).create_view(
            "base_evidence_documents"
        )
        connection.read_parquet(str((Path(root) / OCCURRENCES_NAME).resolve())).create_view(
            "evidence_occurrences"
        )
        connection.read_parquet(str((Path(root) / EMBEDDINGS_NAME).resolve())).create_view(
            "evidence_embeddings"
        )
        if "derivation_documents" in manifest["objects"]:
            connection.read_parquet(
                str((Path(root) / DERIVATION_DOCUMENTS_NAME).resolve())
            ).create_view("evidence_derivation_documents")
            connection.read_parquet(
                str((Path(root) / DERIVATION_MEMBERSHIPS_NAME).resolve())
            ).create_view("evidence_derivation_memberships")
            connection.execute(
                "CREATE VIEW evidence_documents AS SELECT * FROM base_evidence_documents "
                "UNION ALL BY NAME SELECT * FROM evidence_derivation_documents"
            )
            connection.execute(
                "CREATE VIEW evidence_document_occurrences AS "
                "SELECT list_extract(document_ids, 1) AS document_id, occurrence_id "
                "FROM evidence_occurrences WHERE len(document_ids)>0 UNION ALL "
                "SELECT DISTINCT derived_document_id, occurrence_id "
                "FROM evidence_derivation_memberships"
            )
            membership_columns = {
                row[0]
                for row in connection.execute(
                    "DESCRIBE evidence_derivation_memberships"
                ).fetchall()
            }
            connection.execute(
                "CREATE VIEW evidence_occurrence_entity_ids AS "
                + (
                    "SELECT DISTINCT occurrence_id, entity_id FROM "
                    "evidence_derivation_memberships WHERE entity_id IS NOT NULL"
                    if "entity_id" in membership_columns
                    else "SELECT NULL::VARCHAR AS occurrence_id, NULL::VARCHAR AS entity_id WHERE FALSE"
                )
            )
        else:
            connection.execute(
                "CREATE VIEW evidence_documents AS SELECT * FROM base_evidence_documents"
            )
            connection.execute(
                "CREATE VIEW evidence_document_occurrences AS "
                "SELECT list_extract(document_ids, 1) AS document_id, occurrence_id "
                "FROM evidence_occurrences WHERE len(document_ids)>0"
            )
            connection.execute(
                "CREATE VIEW evidence_occurrence_entity_ids AS "
                "SELECT NULL::VARCHAR AS occurrence_id, NULL::VARCHAR AS entity_id WHERE FALSE"
            )
        profile = json.loads((Path(root) / EMBEDDING_PROFILE_NAME).read_text(encoding="utf-8"))
        return cls(Path(root), manifest, connection, profile)

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def prepare_eligible(self, request: Mapping[str, Any]) -> tuple[int, int]:
        """Materialize the occurrence-first closed universe in DuckDB.

        The temporary table contains occurrence rows, not merely document IDs,
        so every returned pointer is known to satisfy the complete filter set.
        """

        if self._closed:
            raise EvidenceIndexError("evidence index is closed")
        if request.get("schema_version") != "livefire.rag.evidence-search.input/1":
            raise ValueError("unsupported evidence.search request")
        top_n = request.get("top_n")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 1000:
            raise ValueError("top_n must be in [1,1000]")
        filters = request.get("filters") or {}
        if filters.get("entity_ids"):
            entity_filter_capable = bool(
                self.manifest.get("derivation_packs")
                and self.connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM evidence_occurrence_entity_ids)"
                ).fetchone()[0]
            )
            if not entity_filter_capable:
                raise ValueError(
                    "entity_ids is unavailable without an admitted entity-membership projection"
                )

        clauses = ["TRUE"]
        parameters: list[Any] = []
        if filters.get("entity_ids"):
            values = filters["entity_ids"]
            clauses.append(
                "EXISTS (SELECT 1 FROM evidence_occurrence_entity_ids ei "
                "WHERE ei.occurrence_id=o.occurrence_id AND ei.entity_id IN ("
                + ",".join("?" for _ in values) + "))"
            )
            parameters.extend(values)
        time_range = request.get("time_range")
        if time_range:
            clauses.extend([
                "o.event_time IS NOT NULL",
                "CAST(o.event_time AS TIMESTAMPTZ) >= CAST(? AS TIMESTAMPTZ)",
                "CAST(o.event_time AS TIMESTAMPTZ) < CAST(? AS TIMESTAMPTZ)",
            ])
            parameters.extend([time_range["start"], time_range["end_exclusive"]])
        if filters.get("relations"):
            relation_pairs = [(row["namespace"], row["relation"]) for row in filters["relations"]]
            clauses.append("(" + " OR ".join(
                "(o.relation_identity.namespace=? AND o.relation_identity.relation=?)" for _ in relation_pairs
            ) + ")")
            for pair in relation_pairs:
                parameters.extend(pair)
        if filters.get("source_snapshot_sha256"):
            values = filters["source_snapshot_sha256"]
            clauses.append("o.source_pointer.snapshot.sha256 IN (" + ",".join("?" for _ in values) + ")")
            parameters.extend(values)
        if filters.get("document_kinds"):
            values = filters["document_kinds"]
            clauses.append("d.document_kind IN (" + ",".join("?" for _ in values) + ")")
            parameters.extend(values)
        if filters.get("exclude_document_ids"):
            values = filters["exclude_document_ids"]
            clauses.append("m.document_id NOT IN (" + ",".join("?" for _ in values) + ")")
            parameters.extend(values)
        uid_fields = {
            "ocsf_category_uids": "ocsf_category_uid",
            "ocsf_class_uids": "ocsf_class_uid",
            "ocsf_activity_ids": "ocsf_activity_id",
        }
        for request_field, row_field in uid_fields.items():
            values = filters.get(request_field)
            if values:
                placeholders = ",".join("?" for _ in values)
                clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(to_json(d), '$.relation_identities') r "
                    f"WHERE CAST(json_extract(r.value, '$.{row_field}') AS BIGINT) IN ({placeholders}))"
                )
                parameters.extend(values)
        operators = {"eq": "=", "not_eq": "!=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
        for predicate in filters.get("attribute_predicates", []):
            operator = operators[predicate["operator"]]
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(to_json(o), '$.exact_attributes') a "
                "WHERE json_extract_string(a.value, '$.namespace')=? "
                "AND json_extract_string(a.value, '$.path')=? "
                f"AND json_extract(a.value, '$.value') {operator} to_json(?))"
            )
            parameters.extend([predicate["namespace"], predicate["path"], predicate["value"]])
        where = " AND ".join(clauses)
        self.connection.execute(
            "CREATE OR REPLACE TABLE evidence_search_eligible_occurrences AS "
            "SELECT m.document_id, o.* FROM evidence_document_occurrences m "
            "JOIN evidence_occurrences o USING(occurrence_id) "
            "JOIN evidence_documents d USING(document_id) "
            f"WHERE {where}", parameters,
        )
        return self.connection.execute(
            "SELECT count(*), count(DISTINCT document_id) "
            "FROM evidence_search_eligible_occurrences"
        ).fetchone()

    def __enter__(self) -> "EvidenceIndex":
        if self._closed:
            raise EvidenceIndexError("evidence index is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def search(
        self,
        request: Mapping[str, Any],
        query_vector: Sequence[float] | None,
        *,
        max_occurrences: int = 100,
    ) -> dict[str, Any]:
        """Search the sealed index with memory bounded independently of corpus size."""

        retrieval = request.get("retrieval") or {}
        methods = retrieval.get("methods")
        fusion = retrieval.get("fusion")
        if not isinstance(methods, list) or not methods or set(methods) - {"dense", "lexical"}:
            raise ValueError("retrieval.methods is invalid")
        if (len(methods) == 1 and fusion != "none") or (
            len(methods) == 2 and fusion != "reciprocal_rank"
        ):
            raise ValueError("retrieval fusion is incompatible with its methods")
        if max_occurrences < 1:
            raise ValueError("max_occurrences must be positive")
        top_n = request.get("top_n")
        eligible_occurrence_count, eligible_document_count = self.prepare_eligible(request)

        dense_query: np.ndarray | None = None
        if "dense" in methods:
            if query_vector is None:
                raise ValueError("dense retrieval requires a query embedding")
            dense_query = np.asarray(query_vector, dtype=np.float64)
            dimensions = int(self.profile["dimensions"])
            if dense_query.shape != (dimensions,) or not np.isfinite(dense_query).all():
                raise ValueError("query embedding shape or values are invalid")
            if abs(float(np.linalg.norm(dense_query)) - 1.0) > 0.0001:
                raise ValueError("query embedding must be L2 normalized")

        self.connection.execute(
            "CREATE OR REPLACE TABLE evidence_search_scores("
            "document_id VARCHAR PRIMARY KEY, dense_distance BIGINT, lexical_score BIGINT)"
        )
        document_query = (
            "SELECT d.document_id, d.semantic_projection.text, e.vector "
            "FROM evidence_documents d "
            "JOIN (SELECT DISTINCT document_id FROM evidence_search_eligible_occurrences) q USING(document_id) "
            "LEFT JOIN evidence_embeddings e USING(document_id) ORDER BY d.document_id"
        )
        query_terms = Counter(token.lower() for token in TOKEN_RE.findall(str(request.get("query", ""))))
        document_frequency: Counter[str] = Counter()
        total_length = 0
        if "lexical" in methods and eligible_document_count:
            cursor = self.connection.cursor().execute(document_query)
            while rows := cursor.fetchmany(4096):
                for _, text, _ in rows:
                    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
                    total_length += len(tokens)
                    present = set(tokens)
                    for term in query_terms:
                        if term in present:
                            document_frequency[term] += 1
        average_length = total_length / eligible_document_count if eligible_document_count else 0.0

        cursor = self.connection.cursor().execute(document_query)
        writer = self.connection.cursor()
        score_rows: list[tuple[str, int | None, int | None]] = []
        while rows := cursor.fetchmany(4096):
            for document_id, text, vector in rows:
                dense_distance = None
                if dense_query is not None:
                    if vector is None:
                        raise EvidenceIndexCorrupt("eligible searchable document lacks an embedding")
                    distance = 1.0 - float(np.asarray(vector, dtype=np.float64) @ dense_query)
                    dense_distance = min(2_000_000, max(0, int(np.rint(distance * 1_000_000))))
                lexical_score = None
                if "lexical" in methods and average_length:
                    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
                    frequencies = Counter(tokens)
                    score = 0.0
                    for term, query_frequency in query_terms.items():
                        frequency = frequencies[term]
                        if not frequency:
                            continue
                        df = document_frequency[term]
                        inverse = math.log(1.0 + (eligible_document_count - df + 0.5) / (df + 0.5))
                        denominator = frequency + 1.2 * (
                            1.0 - 0.75 + 0.75 * len(tokens) / average_length
                        )
                        score += query_frequency * inverse * frequency * 2.2 / denominator
                    if score > 0:
                        lexical_score = int(round(score * 1_000_000))
                score_rows.append((document_id, dense_distance, lexical_score))
            writer.executemany(
                "INSERT INTO evidence_search_scores VALUES (?, ?, ?)", score_rows
            )
            score_rows.clear()

        if methods == ["dense"]:
            ranking_sql = (
                "SELECT document_id, 2000000-dense_distance AS ranking_score, "
                "dense_distance, NULL::BIGINT, NULL::BIGINT FROM evidence_search_scores "
                "WHERE dense_distance IS NOT NULL ORDER BY dense_distance, document_id LIMIT ?"
            )
        elif methods == ["lexical"]:
            ranking_sql = (
                "SELECT document_id, lexical_score, NULL::BIGINT, lexical_score, NULL::BIGINT "
                "FROM evidence_search_scores WHERE lexical_score IS NOT NULL "
                "ORDER BY lexical_score DESC, document_id LIMIT ?"
            )
        else:
            ranking_sql = (
                "WITH ranks AS (SELECT *, "
                "CASE WHEN dense_distance IS NOT NULL THEN row_number() OVER "
                "(ORDER BY dense_distance, document_id) END AS dense_rank, "
                "CASE WHEN lexical_score IS NOT NULL THEN row_number() OVER "
                "(PARTITION BY lexical_score IS NULL ORDER BY lexical_score DESC, document_id) END AS lexical_rank "
                "FROM evidence_search_scores), fused AS (SELECT *, CAST(round(1000000.0 * ("
                "CASE WHEN dense_rank IS NULL THEN 0 ELSE 1.0/(60+dense_rank) END + "
                "CASE WHEN lexical_score IS NULL THEN 0 ELSE 1.0/(60+lexical_rank) END)) AS BIGINT) AS fused_score "
                "FROM ranks) SELECT document_id, fused_score, dense_distance, lexical_score, fused_score "
                "FROM fused WHERE dense_distance IS NOT NULL OR lexical_score IS NOT NULL "
                "ORDER BY fused_score DESC, document_id LIMIT ?"
            )
        ranked = self.connection.execute(ranking_sql, [top_n]).fetchall()
        return self._result(
            request, ranked, eligible_occurrence_count, eligible_document_count,
            max_occurrences=max_occurrences,
        )

    def _result(
        self,
        request: Mapping[str, Any],
        ranked: Sequence[Sequence[Any]],
        eligible_occurrence_count: int,
        eligible_document_count: int,
        *,
        max_occurrences: int,
    ) -> dict[str, Any]:
        top_n = request["top_n"]
        pilot_sample = self.manifest.get("pilot_sample")
        is_pilot_sample = bool(
            isinstance(pilot_sample, dict)
            and pilot_sample.get("scope_status") == "sample_only_not_corpus_coverage"
        )
        coverage_reasons = ["semantic_candidates_require_hydration"]
        if is_pilot_sample:
            coverage_reasons.append("pilot_sample_not_corpus_coverage")
        common = {
            "schema_version": "livefire.rag.evidence-search.output/1",
            "tool": "evidence.search", "index": self.manifest["component"],
            "source_snapshots": self.manifest["source_snapshots"],
            "query_sha256": sha256_bytes(canonical_json_bytes(dict(request))),
            "coverage": {
                "status": "partial" if is_pilot_sample else "complete",
                "indexed_documents": self.manifest["coverage"]["searchable_document_count"],
                "eligible_documents": eligible_document_count,
                "eligible_occurrences": eligible_occurrence_count,
                "definitive": False,
                "reason_codes": coverage_reasons,
            },
            "selection": {
                "requested_top_n": top_n, "returned_count": len(ranked),
                "eligible_count": eligible_document_count,
                "exhausted": len(ranked) == eligible_document_count,
                "deterministic": True,
                "tie_break": "ranking_score_desc_document_id_asc",
            },
        }
        if not ranked:
            return {**common, "kind": "miss", "miss": {
                "reason": "no_eligible_occurrences" if not eligible_document_count else "no_ranked_candidates",
                "message": (
                    "No semantic document within the sealed pilot sample satisfies the closed occurrence filters; "
                    "this is not a corpus-wide miss."
                    if is_pilot_sample
                    else "No semantic document satisfies the closed occurrence filters."
                ),
            }}

        candidates = []
        for rank, row in enumerate(ranked, 1):
            document_id, ranking_score, dense_distance, lexical_score, fused_score = row
            document_payload = self.connection.execute(
                "SELECT to_json(d) FROM evidence_documents d WHERE document_id=?",
                [document_id],
            ).fetchone()
            if document_payload is None:
                raise EvidenceIndexCorrupt("ranked document is absent")
            document = _omit_null_object_fields(json.loads(document_payload[0]))
            occurrence_rows = self.connection.execute(
                "SELECT to_json(o) FROM evidence_search_eligible_occurrences o "
                "WHERE o.document_id=? ORDER BY o.occurrence_id LIMIT ?",
                [document_id, max_occurrences],
            ).fetchall()
            matching_count = self.connection.execute(
                "SELECT count(*) FROM evidence_search_eligible_occurrences WHERE document_id=?",
                [document_id],
            ).fetchone()[0]
            source_occurrences = []
            for (payload,) in occurrence_rows:
                occurrence = _omit_null_object_fields(json.loads(payload))
                item = {
                    "occurrence_id": occurrence["occurrence_id"],
                    "relation_identity": occurrence["relation_identity"],
                    "source_pointer": occurrence["source_pointer"],
                }
                if occurrence.get("event_time") is not None:
                    item["event_time"] = occurrence["event_time"]
                source_occurrences.append(item)
            candidates.append({
                "rank": rank, "document_id": document_id,
                "document_sha256": document["document_sha256"],
                "document_kind": document["document_kind"],
                "preview": document["semantic_projection"]["text"][:4096],
                "scores": {
                    "ranking_score_millionths": ranking_score,
                    "dense_distance_millionths": dense_distance,
                    "lexical_score_millionths": lexical_score,
                    "fused_score_millionths": fused_score,
                },
                "matched_facets": [],
                "matching_occurrence_count": matching_count,
                "returned_occurrence_count": len(source_occurrences),
                "occurrences_exhausted": len(source_occurrences) == matching_count,
                "source_occurrences": source_occurrences,
            })
        return {**common, "kind": "pointer", "candidates": candidates}

    def search_dense(
        self, request: Mapping[str, Any], query_vector: Sequence[float], *, max_occurrences: int = 20
    ) -> dict[str, Any]:
        """Occurrence-first closed filtering followed by exact float64 cosine ranking."""

        retrieval = request.get("retrieval")
        if retrieval != {"methods": ["dense"], "fusion": "none"}:
            raise ValueError("this core implements dense retrieval only")
        return self.search(request, query_vector, max_occurrences=max_occurrences)


__all__ = [
    "EvidenceIndex", "EvidenceIndexCorrupt", "EvidenceIndexError",
    "loopback_embedder", "promote_evidence_pack", "verify_promoted_evidence_index",
]
