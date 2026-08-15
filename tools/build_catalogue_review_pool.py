#!/usr/bin/env python3
"""Build a reviewer-safe pool from one sealed catalogue batch-search run.

Only ``review-pool.jsonl`` and ``manifest.json`` are suitable for reviewers.
System names, search modes, ranks, scores, and index identities are retained
under ``audit/`` for later analysis and must not be given to reviewers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import rfc8785
from jsonschema import Draft202012Validator


REVIEW_POOL = "review-pool.jsonl"
MANIFEST = "manifest.json"
AUDIT_DIR = "audit"
SYSTEM_PROVENANCE = "system-provenance.jsonl"
CANDIDATE_UNIVERSE = "candidate-universe.json"
SNAPSHOT_VALIDATION = "snapshot-validation.json"
AUDIT_MANIFEST = "manifest.json"
MODES = ("lexical", "dense", "fused")
SHA256_ZERO = "0" * 64
MAX_SAFE_INTEGER = 9_007_199_254_740_991


class ReviewPoolError(ValueError):
    """The input artifacts do not close over one immutable review pool."""


def _reject_constant(value: str) -> None:
    raise ReviewPoolError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReviewPoolError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewPoolError(f"invalid JSON in {label}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    value = _decode_json(path.read_bytes(), str(path))
    if not isinstance(value, dict):
        raise ReviewPoolError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raise ReviewPoolError(f"{path}:{line_number} lacks a final LF")
            value = _decode_json(raw, f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ReviewPoolError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    try:
        encoded = rfc8785.dumps(value)
    except (TypeError, ValueError) as error:
        raise ReviewPoolError(f"value cannot be canonically encoded: {error}") from error
    return encoded + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_digest(value: dict[str, Any]) -> str:
    if "component_sha256" not in value:
        raise ReviewPoolError("component_sha256 is absent")
    material = dict(value)
    del material["component_sha256"]
    return _sha256_bytes(_canonical(material))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReviewPoolError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ReviewPoolError(f"{label} is invalid")
    return value


def _require_count(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_SAFE_INTEGER:
        raise ReviewPoolError(f"{label} is invalid")
    return value


def _component_ref(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"id", "version", "sha256"}:
        raise ReviewPoolError(f"{label} component reference is invalid")
    return {
        "id": _require_text(value["id"], f"{label} id", 512),
        "version": _require_text(value["version"], f"{label} version", 512),
        "sha256": _require_sha256(value["sha256"], f"{label} sha256"),
    }


def _safe_artifact(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str):
        raise ReviewPoolError(f"{label} path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in relative
        or ":" in relative
        or "\x00" in relative
    ):
        raise ReviewPoolError(f"{label} path is unsafe")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / Path(*pure.parts)).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ReviewPoolError(f"{label} path escapes its artifact root") from error
    if not resolved.is_file():
        raise ReviewPoolError(f"{label} is not a regular file")
    return resolved


def _validate_receipt(root: Path, receipt: Any, expected_path: str) -> tuple[Path, list[dict[str, Any]]]:
    if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256", "bytes", "rows"}:
        raise ReviewPoolError(f"{expected_path} receipt has the wrong fields")
    if receipt["path"] != expected_path:
        raise ReviewPoolError(f"{expected_path} receipt path mismatch")
    path = _safe_artifact(root, receipt["path"], expected_path)
    expected_bytes = _require_count(receipt["bytes"], f"{expected_path} bytes", positive=True)
    expected_rows = _require_count(receipt["rows"], f"{expected_path} rows", positive=True)
    expected_sha = _require_sha256(receipt["sha256"], f"{expected_path} sha256")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha:
        raise ReviewPoolError(f"{expected_path} byte receipt mismatch")
    rows = _read_jsonl(path)
    if len(rows) != expected_rows:
        raise ReviewPoolError(f"{expected_path} row receipt mismatch")
    return path, rows


def _validate_raw_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(run_dir / MANIFEST)
    required = {
        "schema_version", "component_sha256", "status", "catalogue_sha256",
        "embedding_profile", "requests", "results", "request_count", "result_count",
        "modes", "top_n_values", "relation_filters", "request_shapes", "model",
        "query_vectors", "rank_merge",
    }
    if set(manifest) != required:
        raise ReviewPoolError("raw batch manifest has the wrong fields")
    if manifest["schema_version"] != "livefire.rag.catalogue-batch-search-run/1" or manifest["status"] != "complete":
        raise ReviewPoolError("raw batch run is not complete")
    component = _require_sha256(manifest["component_sha256"], "raw batch component_sha256")
    if component != _component_digest(manifest):
        raise ReviewPoolError("raw batch component digest mismatch")
    _, requests = _validate_receipt(run_dir, manifest["requests"], "requests.jsonl")
    _, results = _validate_receipt(run_dir, manifest["results"], "results.jsonl")
    if (
        _require_count(manifest["request_count"], "request_count", positive=True) != len(requests)
        or _require_count(manifest["result_count"], "result_count", positive=True) != len(results)
        or len(requests) != len(results)
    ):
        raise ReviewPoolError("raw request/result count closure failed")
    for ordinal, row in enumerate(requests):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("mode"), str)
            or isinstance(row.get("top_n"), bool)
            or not isinstance(row.get("top_n"), int)
            or not isinstance(row.get("relations"), list)
            or any(not isinstance(relation, str) for relation in row.get("relations", []))
        ):
            raise ReviewPoolError(f"raw request {ordinal} has invalid field types")
    manifest_modes = manifest["modes"]
    observed_modes = {row.get("mode") for row in requests}
    mode_order = {mode: ordinal for ordinal, mode in enumerate(("dense", "lexical", "fused"))}
    if (
        not isinstance(manifest_modes, list)
        or any(not isinstance(mode, str) for mode in manifest_modes)
        or len(manifest_modes) != len(set(manifest_modes))
        or set(manifest_modes) != observed_modes
        or manifest_modes
        != sorted(observed_modes, key=lambda mode: mode_order.get(mode, len(mode_order)))
    ):
        raise ReviewPoolError("raw request mode closure failed")
    top_values = sorted({row.get("top_n") for row in requests})
    if any(value < 1 or value > 100 for value in top_values) or manifest["top_n_values"] != top_values:
        raise ReviewPoolError("raw request top_n closure failed")
    relation_filters = sorted({tuple(row.get("relations", [])) for row in requests})
    if manifest["relation_filters"] != [list(value) for value in relation_filters]:
        raise ReviewPoolError("raw request relation-filter closure failed")
    shape_counts: defaultdict[tuple[str, int, tuple[str, ...]], int] = defaultdict(int)
    for row in requests:
        shape_counts[(row["mode"], row["top_n"], tuple(row["relations"]))] += 1
    expected_shapes = [
        {"mode": mode, "top_n": top_n, "relations": list(relations), "rows": rows}
        for (mode, top_n, relations), rows in sorted(
            shape_counts.items(), key=lambda item: (mode_order[item[0][0]], item[0][1], item[0][2])
        )
    ]
    if manifest["request_shapes"] != expected_shapes:
        raise ReviewPoolError("raw request-shape closure failed")
    semantic_queries = {
        row["query"] for row in requests if row["mode"] in {"dense", "fused"}
    }
    model = manifest["model"]
    vectors = manifest["query_vectors"]
    if (
        not isinstance(model, dict)
        or set(model) != {"status", "configured_model", "returned_model", "calls"}
        or not isinstance(vectors, list)
    ):
        raise ReviewPoolError("raw model receipt is invalid")
    _component_ref(manifest["embedding_profile"], "raw embedding profile")
    _require_text(model["configured_model"], "configured model", 1024)
    _require_count(model["calls"], "model calls")
    if model["returned_model"] is not None:
        _require_text(model["returned_model"], "returned model", 1024)
    for vector in vectors:
        if (
            not isinstance(vector, dict)
            or set(vector) != {"composed_query_sha256", "vector_sha256", "dimensions"}
            or _require_count(vector["dimensions"], "query vector dimensions", positive=True) < 1
        ):
            raise ReviewPoolError("raw query-vector receipt is invalid")
        _require_sha256(vector["composed_query_sha256"], "composed query sha256")
        _require_sha256(vector["vector_sha256"], "query vector sha256")
    if any(
        left["composed_query_sha256"] >= right["composed_query_sha256"]
        for left, right in zip(vectors, vectors[1:], strict=False)
    ):
        raise ReviewPoolError("raw query-vector receipts are not strictly sorted and unique")
    if semantic_queries:
        if (
            model["status"] != "used"
            or model["configured_model"] != model["returned_model"]
            or model["calls"] != len(semantic_queries)
            or len(vectors) != len(semantic_queries)
        ):
            raise ReviewPoolError("raw semantic model-call closure failed")
    elif (
        model
        != {
            "status": "not_used_all_lexical",
            "configured_model": model.get("configured_model"),
            "returned_model": None,
            "calls": 0,
        }
        or not isinstance(model.get("configured_model"), str)
        or not model["configured_model"]
        or vectors
    ):
        raise ReviewPoolError("raw lexical model-call closure failed")
    if manifest["rank_merge"] != {"policy": "reciprocal_rank_fusion_v1", "k": 60}:
        raise ReviewPoolError("raw batch rank-merge receipt is invalid")
    return manifest, requests, results


def _validate_fixture(path: Path) -> tuple[str, list[dict[str, str]]]:
    raw = path.read_bytes()
    fixture = _decode_json(raw, str(path))
    if not isinstance(fixture, dict) or fixture.get("schema_version") != "livefire.rag.generic-evidence-pilot-queries/1":
        raise ReviewPoolError("query fixture has the wrong schema version")
    rows = fixture.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ReviewPoolError("query fixture is empty")
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReviewPoolError(f"query fixture row {ordinal} is invalid")
        query_id = _require_text(row.get("query_id"), "fixture query_id", 128)
        query = _require_text(row.get("query"), "fixture query", 8192)
        if query_id in seen:
            raise ReviewPoolError("query fixture has duplicate query IDs")
        seen.add(query_id)
        queries.append({"query_id": query_id, "query": query})
    return _sha256_bytes(raw), queries


def _validate_request_result_closure(
    manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    results: list[dict[str, Any]],
    fixture_queries: list[dict[str, str]],
) -> None:
    fixture = {row["query_id"]: row["query"] for row in fixture_queries}
    expected_surfaces = {(query_id, mode) for query_id in fixture for mode in MODES}
    observed_surfaces: set[tuple[str, str]] = set()
    catalogue_sha = _require_sha256(manifest["catalogue_sha256"], "raw catalogue_sha256")
    for ordinal, (request, result) in enumerate(zip(requests, results, strict=True)):
        if set(request) != {"query_id", "query", "mode", "top_n", "relations"}:
            raise ReviewPoolError(f"request {ordinal} has the wrong fields")
        query_id = request["query_id"]
        mode = request["mode"]
        if query_id not in fixture or request["query"] != fixture[query_id] or mode not in MODES:
            raise ReviewPoolError(f"request {ordinal} is not bound to the frozen fixture")
        if request["relations"] != []:
            raise ReviewPoolError("review comparison requests must not use relation hints")
        _require_count(request["top_n"], "request top_n", positive=True)
        if (query_id, mode) in observed_surfaces:
            raise ReviewPoolError("duplicate query/mode request surface")
        observed_surfaces.add((query_id, mode))
        required_result = {
            "schema_version", "query_id", "catalogue_sha256", "query", "mode", "top_n",
            "relations", "rank_merge", "hits",
        }
        if not isinstance(result, dict) or set(result) != required_result:
            raise ReviewPoolError(f"result {ordinal} has the wrong fields")
        if (
            result["schema_version"] != "livefire.rag.catalogue-batch-search-result/1"
            or result["query_id"] != query_id
            or result["catalogue_sha256"] != catalogue_sha
            or result["query"] != request["query"]
            or result["mode"] != mode
            or result["top_n"] != request["top_n"]
            or result["relations"] != request["relations"]
            or result["rank_merge"] != "reciprocal_rank_fusion_v1"
            or not isinstance(result["hits"], list)
            or len(result["hits"]) > request["top_n"]
        ):
            raise ReviewPoolError(f"request/result ordered closure failed at row {ordinal}")
    if observed_surfaces != expected_surfaces:
        raise ReviewPoolError("raw run does not contain every frozen query in all three modes")


def _validate_catalogue(path: Path, raw_manifest: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    catalogue = _read_json(path)
    if catalogue.get("schema_version") != "livefire.rag.dataset-catalogue/1":
        raise ReviewPoolError("catalogue has the wrong schema version")
    component = _require_sha256(catalogue.get("component_sha256"), "catalogue component_sha256")
    if component != _component_digest(catalogue) or component != raw_manifest["catalogue_sha256"]:
        raise ReviewPoolError("catalogue component binding failed")
    catalogue_profile = _component_ref(
        catalogue.get("embedding_profile"), "catalogue embedding profile"
    )
    if catalogue_profile != _component_ref(
        raw_manifest["embedding_profile"], "raw embedding profile"
    ):
        raise ReviewPoolError("raw run uses a different embedding profile from the catalogue")
    datasets = catalogue.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ReviewPoolError("catalogue has no datasets")
    ids: set[str] = set()
    digests: set[str] = set()
    for entry in datasets:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset"), dict):
            raise ReviewPoolError("catalogue dataset entry is invalid")
        dataset_id = _require_text(entry["dataset"].get("id"), "catalogue dataset ID", 512)
        dataset_sha = _require_sha256(entry.get("dataset_sha256"), "catalogue dataset_sha256")
        if dataset_sha != _sha256_bytes(_canonical(entry["dataset"])):
            raise ReviewPoolError("catalogue dataset identity digest mismatch")
        if dataset_id in ids or dataset_sha in digests:
            raise ReviewPoolError("catalogue dataset identity is duplicated")
        ids.add(dataset_id)
        digests.add(dataset_sha)
        if _component_ref(
            entry.get("embedding_profile"), "dataset embedding profile"
        ) != catalogue_profile:
            raise ReviewPoolError("catalogue dataset uses a different embedding profile")
        _require_count(entry.get("searchable_document_count"), "searchable_document_count", positive=True)
        _require_count(entry.get("searchable_reference_count"), "searchable_reference_count", positive=True)
    return catalogue, path.parent.resolve(strict=True)


def _open_read_only_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1", uri=True)
    result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        connection.close()
        raise ReviewPoolError(f"SQLite integrity check failed: {path}")
    return connection


def _validate_object(index_root: Path, summary: Any, label: str) -> Path:
    if not isinstance(summary, dict):
        raise ReviewPoolError(f"{label} summary is invalid")
    path = _safe_artifact(index_root, summary.get("path"), label)
    expected_bytes = _require_count(summary.get("bytes"), f"{label} bytes", positive=True)
    expected_sha = _require_sha256(summary.get("sha256"), f"{label} sha256")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha:
        raise ReviewPoolError(f"{label} content receipt mismatch")
    return path


def _load_indexes(catalogue: dict[str, Any], catalogue_root: Path) -> dict[str, dict[str, Any]]:
    indexes: dict[str, dict[str, Any]] = {}
    for entry in catalogue["datasets"]:
        final_ref = entry.get("final_index")
        if not isinstance(final_ref, dict):
            raise ReviewPoolError("catalogue final index reference is invalid")
        index_path = _safe_artifact(catalogue_root, final_ref.get("path"), "final index manifest")
        index = _read_json(index_path)
        index_component = _require_sha256(index.get("component_sha256"), "index component_sha256")
        if index_component != _component_digest(index) or index_component != final_ref.get("sha256"):
            raise ReviewPoolError("final index component binding failed")
        build_scope = index.get("build_scope")
        complete = index.get("complete")
        if build_scope not in {"full", "sample"} or not isinstance(complete, bool):
            raise ReviewPoolError("final index build-scope declaration is invalid")
        if complete != (build_scope == "full"):
            raise ReviewPoolError("final index build-scope declaration is inconsistent")
        dataset = entry["dataset"]
        expected_source = {
            "snapshot_sha256": _require_sha256(
                dataset["source_snapshot"].get("sha256"), "dataset source snapshot sha256"
            ),
            "mapping_sha256": _require_sha256(
                dataset["mapping"].get("sha256"), "dataset mapping sha256"
            ),
        }
        if index.get("source") != expected_source:
            raise ReviewPoolError("final index source binding failed")
        provenance = index.get("pipeline_provenance")
        expected_provenance = {
            "dataset_sha256": entry["dataset_sha256"],
            "prepared_corpus_sha256": entry["prepared_corpus"]["sha256"],
            "embedding_plan_sha256": entry["embedding_plan"]["sha256"],
            "embedding_result_set_sha256": entry["embedding_result_set"]["sha256"],
        }
        if provenance != expected_provenance:
            raise ReviewPoolError("final index dataset binding failed")
        documents = _require_count(index.get("documents", {}).get("rows"), "index document rows", positive=True)
        occurrences = _require_count(index.get("occurrences", {}).get("rows"), "index occurrence rows", positive=True)
        if documents != entry["searchable_document_count"] or occurrences != entry["searchable_reference_count"]:
            raise ReviewPoolError("catalogue/index count binding failed")
        lexical = index.get("lexical")
        if not isinstance(lexical, dict) or lexical.get("schema") != "sqlite-inverted-bm25-v1":
            raise ReviewPoolError("review pooling requires the scalable SQLite lexical index")
        lookup = index.get("occurrence_lookup")
        lexical_path = _validate_object(index_path.parent, lexical, "lexical index")
        lookup_path = _validate_object(index_path.parent, lookup, "occurrence lookup")
        dataset_sha = entry["dataset_sha256"]
        indexes[dataset_sha] = {
            "entry": entry,
            "manifest": index,
            "lexical_path": lexical_path,
            "lookup_path": lookup_path,
        }
    return indexes


def _validate_index_embedding_profile(value: Any) -> dict[str, Any]:
    required = {"id", "version", "sha256", "model", "dimensions", "normalization"}
    optional = {"vector_derivation", "query_instruction", "query_composition"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or not set(value) <= required | optional
    ):
        raise ReviewPoolError("index embedding profile has the wrong fields")
    _require_text(value["id"], "index embedding profile id", 512)
    _require_text(value["version"], "index embedding profile version", 512)
    _require_sha256(value["sha256"], "index embedding profile sha256")
    _require_text(value["model"], "index embedding profile model", 1024)
    _require_count(value["dimensions"], "index embedding profile dimensions", positive=True)
    if value["normalization"] not in {"l2", "none"}:
        raise ReviewPoolError("index embedding profile normalization is invalid")
    instruction = value.get("query_instruction")
    composition = value.get("query_composition")
    if (instruction is None) != (composition is None):
        raise ReviewPoolError("index embedding query composition is incomplete")
    if instruction is not None:
        _require_text(instruction, "index query instruction", 8192)
        _require_text(composition, "index query composition", 1024)
        if (
            composition.count("{query_instruction}") != 1
            or composition.count("{query}") != 1
            or "{" in composition.replace("{query_instruction}", "").replace("{query}", "")
            or "}" in composition.replace("{query_instruction}", "").replace("{query}", "")
        ):
            raise ReviewPoolError("index embedding query composition is invalid")
    derivation = value.get("vector_derivation")
    if derivation is not None:
        if (
            not isinstance(derivation, dict)
            or set(derivation)
            != {"parent_embedding_profile_sha256", "parent_dimensions", "transformation"}
        ):
            raise ReviewPoolError("index vector derivation is invalid")
        _require_sha256(
            derivation["parent_embedding_profile_sha256"],
            "parent embedding profile sha256",
        )
        _require_count(derivation["parent_dimensions"], "parent dimensions", positive=True)
        _require_text(derivation["transformation"], "vector transformation", 512)
    return value


def _compose_query(profile: dict[str, Any], query: str) -> str:
    instruction = profile.get("query_instruction")
    composition = profile.get("query_composition")
    if instruction is None and composition is None:
        return query
    assert isinstance(instruction, str) and isinstance(composition, str)
    instruction_token = "{query_instruction}"
    query_token = "{query}"
    instruction_at = composition.index(instruction_token)
    query_at = composition.index(query_token)
    if instruction_at < query_at:
        return (
            composition[:instruction_at]
            + instruction
            + composition[instruction_at + len(instruction_token):query_at]
            + query
            + composition[query_at + len(query_token):]
        )
    return (
        composition[:query_at]
        + query
        + composition[query_at + len(query_token):instruction_at]
        + instruction
        + composition[instruction_at + len(instruction_token):]
    )


def _validate_embedding_closure(
    raw_manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    catalogue: dict[str, Any],
    indexes: dict[str, dict[str, Any]],
) -> None:
    required_ref = _component_ref(
        catalogue["embedding_profile"], "catalogue embedding profile"
    )
    first_profile: dict[str, Any] | None = None
    for loaded in indexes.values():
        profile = _validate_index_embedding_profile(
            loaded["manifest"].get("embedding_profile")
        )
        profile_ref = {
            "id": profile["id"], "version": profile["version"], "sha256": profile["sha256"]
        }
        if profile_ref != required_ref:
            raise ReviewPoolError("an admitted index uses a different embedding profile")
        vector_dimensions = _require_count(
            loaded["manifest"].get("vectors", {}).get("dimensions"),
            "index vector dimensions",
            positive=True,
        )
        if vector_dimensions != profile["dimensions"]:
            raise ReviewPoolError("index vectors use different dimensions from their profile")
        if first_profile is None:
            first_profile = profile
        elif profile != first_profile:
            raise ReviewPoolError("admitted indexes do not carry one exact embedding profile")
    if first_profile is None:
        raise ReviewPoolError("catalogue has no admitted index embedding profile")
    model = raw_manifest["model"]
    if (
        model["configured_model"] != first_profile["model"]
        or (model["status"] == "used" and model["returned_model"] != first_profile["model"])
    ):
        raise ReviewPoolError("raw model receipt does not match the admitted embedding profile")
    semantic_queries = {
        row["query"] for row in requests if row["mode"] in {"dense", "fused"}
    }
    expected_composed_digests = sorted(
        _sha256_bytes(_compose_query(first_profile, query).encode("utf-8"))
        for query in semantic_queries
    )
    vectors = raw_manifest["query_vectors"]
    if [row["composed_query_sha256"] for row in vectors] != expected_composed_digests:
        raise ReviewPoolError("raw query-vector receipts do not match the composed queries")
    if any(row["dimensions"] != first_profile["dimensions"] for row in vectors):
        raise ReviewPoolError("raw query-vector dimensions do not match the admitted profile")


def _candidate_universe(
    indexes: dict[str, dict[str, Any]], catalogue_sha256: str
) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    datasets: list[dict[str, Any]] = []
    for dataset_sha, loaded in sorted(indexes.items()):
        connection = _open_read_only_sqlite(loaded["lexical_path"])
        try:
            rows = connection.execute(
                "SELECT document_id FROM documents ORDER BY document_ordinal"
            )
            count = 0
            seen: set[str] = set()
            for (document_id,) in rows:
                _require_text(document_id, "indexed document_id", 512)
                if document_id in seen:
                    raise ReviewPoolError("indexed document occurs more than once")
                seen.add(document_id)
                digest.update(_canonical({"dataset_sha256": dataset_sha, "document_id": document_id}, newline=True))
                count += 1
            if count != loaded["entry"]["searchable_document_count"]:
                raise ReviewPoolError("candidate-universe document count mismatch")
            total += count
            datasets.append({
                "dataset_id": loaded["entry"]["dataset"]["id"],
                "dataset_sha256": dataset_sha,
                "document_count": count,
            })
        finally:
            connection.close()
    receipt = {
        "schema_version": "livefire.rag.catalogue-candidate-universe-receipt/1",
        "component_sha256": SHA256_ZERO,
        "catalogue_sha256": catalogue_sha256,
        "enumeration_key": ["dataset_sha256", "document_id"],
        "dataset_count": len(indexes),
        "document_count": total,
        "datasets": datasets,
        "candidate_universe_sha256": digest.hexdigest(),
    }
    receipt["component_sha256"] = _component_digest(receipt)
    return receipt


def _finite_number(value: Any, label: str, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReviewPoolError(f"{label} is invalid")
    return float(value)


def _exact_hit(
    hit: Any,
    indexes: dict[str, dict[str, Any]],
    connections: dict[str, tuple[sqlite3.Connection, sqlite3.Connection]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "rank", "reciprocal_rank_score", "dataset", "dataset_sha256", "index_sha256",
        "index_rank", "hit",
    }
    if not isinstance(hit, dict) or set(hit) != required or not isinstance(hit.get("hit"), dict):
        raise ReviewPoolError("catalogue hit has the wrong fields")
    dataset_sha = _require_sha256(hit["dataset_sha256"], "hit dataset_sha256")
    if dataset_sha not in indexes:
        raise ReviewPoolError("hit names a dataset outside the admitted catalogue")
    loaded = indexes[dataset_sha]
    entry = loaded["entry"]
    if hit["dataset"] != entry["dataset"] or hit["index_sha256"] != loaded["manifest"]["component_sha256"]:
        raise ReviewPoolError("hit dataset/index identity mismatch")
    _require_count(hit["rank"], "catalogue hit rank", positive=True)
    index_rank = _require_count(hit["index_rank"], "index hit rank", positive=True)
    reciprocal_rank_score = _finite_number(
        hit["reciprocal_rank_score"], "reciprocal-rank score"
    )
    if reciprocal_rank_score != 1.0 / (60 + index_rank):
        raise ReviewPoolError("catalogue reciprocal-rank score is inconsistent")
    detail = hit["hit"]
    detail_required = {
        "rank", "document_id", "semantic_text", "score", "dense_score", "lexical_score",
        "eligible_occurrence_count", "occurrences_exhausted", "occurrences",
    }
    if set(detail) != detail_required:
        raise ReviewPoolError("index hit has the wrong fields")
    document_id = _require_text(detail["document_id"], "hit document_id", 512)
    semantic_text = _require_text(detail["semantic_text"], "hit semantic_text", 1_048_576)
    if _require_count(detail["rank"], "hit rank", positive=True) != index_rank:
        raise ReviewPoolError("catalogue index rank does not match the underlying hit rank")
    _finite_number(detail["score"], "hit score")
    _finite_number(detail["dense_score"], "dense score", optional=True)
    _finite_number(detail["lexical_score"], "lexical score", optional=True)
    eligible_count = _require_count(detail["eligible_occurrence_count"], "eligible occurrence count")
    if not isinstance(detail["occurrences_exhausted"], bool) or not isinstance(detail["occurrences"], list):
        raise ReviewPoolError("hit occurrence summary is invalid")
    lexical, occurrence = connections[dataset_sha]
    stored = lexical.execute(
        "SELECT semantic_text FROM documents WHERE document_id = ?", (document_id,)
    ).fetchall()
    if stored != [(semantic_text,)]:
        raise ReviewPoolError("hit semantic text does not match the exact index document")
    stored_occurrences = occurrence.execute(
        "SELECT event_time_ms, relation, snapshot_sha256, mapping_sha256, event_id, support_ref "
        "FROM occurrences WHERE document_id = ? ORDER BY occurrence_id",
        (document_id,),
    ).fetchall()
    expected_occurrences = [
        {
            "event_time_ms": row[0], "relation": row[1], "snapshot_sha256": row[2],
            "mapping_sha256": row[3], "event_id": row[4], "support_ref": row[5],
        }
        for row in stored_occurrences[:50]
    ]
    if (
        eligible_count != len(stored_occurrences)
        or detail["occurrences"] != expected_occurrences
        or detail["occurrences_exhausted"] != (len(stored_occurrences) <= 50)
    ):
        raise ReviewPoolError("hit occurrence pointers do not match the exact index lookup")
    for pointer in expected_occurrences:
        _require_count(pointer["event_time_ms"], "event_time_ms") if pointer["event_time_ms"] is not None else None
        _require_text(pointer["relation"], "occurrence relation", 256)
        _require_sha256(pointer["snapshot_sha256"], "occurrence snapshot_sha256")
        _require_sha256(pointer["mapping_sha256"], "occurrence mapping_sha256")
        _require_text(pointer["event_id"], "occurrence event_id", 1024)
        _require_text(pointer["support_ref"], "occurrence support_ref", 1024)
    review_material = {
        "dataset_id": entry["dataset"]["id"],
        "document_id": document_id,
        "semantic_text": semantic_text,
        "eligible_occurrence_count": eligible_count,
        "occurrences": expected_occurrences,
    }
    private = {
        "dataset_sha256": dataset_sha,
        "index_sha256": hit["index_sha256"],
        "catalogue_rank": hit["rank"],
        "reciprocal_rank_score": hit["reciprocal_rank_score"],
        "index_rank": hit["index_rank"],
        "hit_rank": detail["rank"],
        "score": detail["score"],
        "dense_score": detail["dense_score"],
        "lexical_score": detail["lexical_score"],
        "occurrences_exhausted": detail["occurrences_exhausted"],
    }
    return review_material, private


def _validate_snapshot(
    snapshot_root: Path, review_rows: Iterable[dict[str, Any]]
) -> tuple[int, list[str]]:
    wanted = {
        (pointer["relation"], pointer["event_id"], pointer["support_ref"])
        for row in review_rows
        for pointer in row["occurrences"]
    }
    if not wanted:
        return 0, []
    try:
        import duckdb
    except ImportError as error:
        raise ReviewPoolError("snapshot validation needs the prototype/analysis DuckDB dependency") from error
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE wanted(relation VARCHAR, event_id VARCHAR, support_ref VARCHAR)")
        connection.executemany("INSERT INTO wanted VALUES (?, ?, ?)", sorted(wanted))
        for relation in sorted({row[0] for row in wanted}):
            if not relation or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in relation):
                raise ReviewPoolError("occurrence relation cannot name a typed Parquet file")
            parquet = (snapshot_root / "semantic" / f"{relation}.parquet").resolve(strict=True)
            try:
                parquet.relative_to(snapshot_root.resolve(strict=True))
            except ValueError as error:
                raise ReviewPoolError("typed Parquet path escapes the snapshot") from error
            missing = connection.execute(
                "SELECT w.event_id, w.support_ref FROM wanted w "
                "WHERE w.relation = ? AND NOT EXISTS ("
                "SELECT 1 FROM read_parquet(?) p "
                "WHERE p.event_id = w.event_id AND p.support_ref = w.support_ref) LIMIT 1",
                [relation, str(parquet)],
            ).fetchone()
            if missing is not None:
                raise ReviewPoolError(
                    f"returned pointer is absent from typed {relation} Parquet: {missing[0]}"
                )
    finally:
        connection.close()
    return len(wanted), sorted({row[0] for row in wanted})


def _snapshot_validation_receipt(
    snapshot_root: Path | None,
    review_rows: list[dict[str, Any]],
    expected_snapshot: dict[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": "livefire.rag.catalogue-review-snapshot-validation/1",
        "component_sha256": SHA256_ZERO,
        "status": "not_requested",
        "snapshot": expected_snapshot,
        "checked_unique_pointer_count": 0,
        "checked_relations": [],
        "build_receipt": None,
    }
    if snapshot_root is not None:
        root = Path(snapshot_root).resolve(strict=True)
        build_receipt_path = (root / "build-receipt.json").resolve(strict=True)
        try:
            build_receipt_path.relative_to(root)
        except ValueError as error:
            raise ReviewPoolError("snapshot build receipt escapes its root") from error
        build_receipt = _read_json(build_receipt_path)
        if build_receipt.get("runnable_snapshot", {}).get("component") != expected_snapshot:
            raise ReviewPoolError("snapshot build receipt does not match the catalogue")
        pointer_count, relations = _validate_snapshot(root, review_rows)
        receipt.update({
            "status": "exact_typed_parquet_membership_passed",
            "checked_unique_pointer_count": pointer_count,
            "checked_relations": relations,
            "build_receipt": {
                "path": "build-receipt.json",
                "bytes": build_receipt_path.stat().st_size,
                "sha256": _sha256_file(build_receipt_path),
            },
        })
    receipt["component_sha256"] = _component_digest(receipt)
    return receipt


def _candidate_id(fixture_sha: str, query_id: str, dataset_sha: str, document_id: str) -> str:
    material = {
        "domain": "livefire.rag.catalogue-review-candidate/1",
        "query_fixture_sha256": fixture_sha,
        "query_id": query_id,
        "dataset_sha256": dataset_sha,
        "document_id": document_id,
    }
    return f"candidate-{_sha256_bytes(_canonical(material))}"


def _pool_rows(
    results: list[dict[str, Any]],
    fixture_sha: str,
    indexes: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connections = {
        dataset_sha: (
            _open_read_only_sqlite(loaded["lexical_path"]),
            _open_read_only_sqlite(loaded["lookup_path"]),
        )
        for dataset_sha, loaded in indexes.items()
    }
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    provenance: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    queries: dict[str, str] = {}
    try:
        for result_ordinal, result in enumerate(results):
            query_id = result["query_id"]
            queries[query_id] = result["query"]
            seen_result: set[tuple[str, str]] = set()
            for expected_rank, hit in enumerate(result["hits"], 1):
                review_material, private = _exact_hit(hit, indexes, connections)
                if hit["rank"] != expected_rank:
                    raise ReviewPoolError("catalogue hit ranks are not ordered and contiguous")
                detail = hit["hit"]
                if (
                    result["mode"] == "dense"
                    and (detail["dense_score"] is None or detail["lexical_score"] is not None)
                ) or (
                    result["mode"] == "lexical"
                    and (detail["dense_score"] is not None or detail["lexical_score"] is None)
                ) or (
                    result["mode"] == "fused"
                    and detail["dense_score"] is None
                    and detail["lexical_score"] is None
                ):
                    raise ReviewPoolError("hit score fields do not match its search mode")
                result_key = (hit["dataset_sha256"], review_material["document_id"])
                if result_key in seen_result:
                    raise ReviewPoolError("one result contains the same catalogue document twice")
                seen_result.add(result_key)
                key = (query_id, *result_key)
                existing = pooled.get(key)
                if existing is not None and existing != review_material:
                    raise ReviewPoolError("duplicate pooled candidate has inconsistent reviewer content")
                pooled[key] = review_material
                provenance[key].append({
                    "mode": result["mode"], "result_ordinal": result_ordinal, **private,
                })
    finally:
        for lexical, occurrence in connections.values():
            lexical.close()
            occurrence.close()
    review_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for (query_id, dataset_sha, document_id), material in pooled.items():
        candidate_id = _candidate_id(fixture_sha, query_id, dataset_sha, document_id)
        review_rows.append({
            "schema_version": "livefire.rag.catalogue-review-pool-row/1",
            "candidate_id": candidate_id,
            "query_id": query_id,
            "query": queries[query_id],
            **material,
        })
        systems = sorted(provenance[(query_id, dataset_sha, document_id)], key=lambda row: row["result_ordinal"])
        private_rows.append({
            "schema_version": "livefire.rag.catalogue-review-system-provenance/1",
            "candidate_id": candidate_id,
            "query_id": query_id,
            "dataset_sha256": dataset_sha,
            "document_id": document_id,
            "systems": systems,
        })
    review_rows.sort(key=lambda row: row["candidate_id"])
    private_rows.sort(key=lambda row: row["candidate_id"])
    if not review_rows:
        raise ReviewPoolError("the raw run returned no review candidates")
    if len({row["candidate_id"] for row in review_rows}) != len(review_rows):
        raise ReviewPoolError("opaque candidate ID collision")
    return review_rows, private_rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int, str]:
    count = 0
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical(row, newline=True))
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count, path.stat().st_size, _sha256_file(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value, newline=True))
        handle.flush()
        os.fsync(handle.fileno())


def _validate_public_contracts(review_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    specs = Path(__file__).resolve().parents[1] / "specs"
    contracts = (
        (specs / "catalogue-review-pool-row.v1.schema.json", review_rows),
        (specs / "catalogue-review-pool-manifest.v1.schema.json", [manifest]),
    )
    for schema_path, values in contracts:
        schema = _read_json(schema_path)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for value in values:
            errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
            if errors:
                first = errors[0]
                location = "/".join(str(part) for part in first.absolute_path) or "<root>"
                raise ReviewPoolError(
                    f"public review schema violation at {location}: {first.message}"
                )


def build_review_pool(
    *, run_dir: Path, catalogue_path: Path, query_fixture: Path,
    out_dir: Path, snapshot_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a raw run and publish one blinded review directory atomically."""

    run_dir = Path(run_dir).resolve(strict=True)
    catalogue_path = Path(catalogue_path).resolve(strict=True)
    query_fixture = Path(query_fixture).resolve(strict=True)
    out_dir = Path(out_dir).absolute()
    if out_dir.exists():
        raise ReviewPoolError(f"refusing to overwrite existing output: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest, requests, results = _validate_raw_run(run_dir)
    fixture_sha, fixture_queries = _validate_fixture(query_fixture)
    _validate_request_result_closure(raw_manifest, requests, results, fixture_queries)
    catalogue, catalogue_root = _validate_catalogue(catalogue_path, raw_manifest)
    indexes = _load_indexes(catalogue, catalogue_root)
    _validate_embedding_closure(raw_manifest, requests, catalogue, indexes)
    universe = _candidate_universe(indexes, catalogue["component_sha256"])
    review_rows, private_rows = _pool_rows(results, fixture_sha, indexes)
    snapshot_validation = _snapshot_validation_receipt(
        snapshot_root, review_rows, catalogue["source_snapshot"]
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (staging / AUDIT_DIR).mkdir()
        pool_rows, pool_bytes, pool_sha = _write_jsonl(staging / REVIEW_POOL, review_rows)
        provenance_rows, provenance_bytes, provenance_sha = _write_jsonl(
            staging / AUDIT_DIR / SYSTEM_PROVENANCE, private_rows
        )
        _write_json(staging / AUDIT_DIR / CANDIDATE_UNIVERSE, universe)
        _write_json(staging / AUDIT_DIR / SNAPSHOT_VALIDATION, snapshot_validation)
        manifest = {
            "schema_version": "livefire.rag.catalogue-review-pool-manifest/1",
            "component_sha256": SHA256_ZERO,
            "status": "people_have_not_yet_marked_relevance",
            "system_labels_hidden": True,
            "query_fixture_sha256": fixture_sha,
            "raw_batch_run_sha256": raw_manifest["component_sha256"],
            "review_pool": {
                "path": REVIEW_POOL, "sha256": pool_sha, "bytes": pool_bytes, "rows": pool_rows,
            },
            "unique_query_count": len({row["query_id"] for row in review_rows}),
            "unique_candidate_count": len(review_rows),
        }
        manifest["component_sha256"] = _component_digest(manifest)
        _validate_public_contracts(review_rows, manifest)
        _write_json(staging / MANIFEST, manifest)
        audit_manifest = {
            "schema_version": "livefire.rag.catalogue-review-audit/1",
            "component_sha256": SHA256_ZERO,
            "review_pool_sha256": manifest["component_sha256"],
            "catalogue_component_sha256": catalogue["component_sha256"],
            "catalogue_file_sha256": _sha256_file(catalogue_path),
            "raw_batch_run_sha256": raw_manifest["component_sha256"],
            "query_fixture_sha256": fixture_sha,
            "system_provenance": {
                "path": SYSTEM_PROVENANCE, "sha256": provenance_sha,
                "bytes": provenance_bytes, "rows": provenance_rows,
            },
            "candidate_universe": {
                "path": CANDIDATE_UNIVERSE,
                "sha256": _sha256_file(staging / AUDIT_DIR / CANDIDATE_UNIVERSE),
                "bytes": (staging / AUDIT_DIR / CANDIDATE_UNIVERSE).stat().st_size,
                "component_sha256": universe["component_sha256"],
            },
            "snapshot_validation": {
                "path": SNAPSHOT_VALIDATION,
                "sha256": _sha256_file(staging / AUDIT_DIR / SNAPSHOT_VALIDATION),
                "bytes": (staging / AUDIT_DIR / SNAPSHOT_VALIDATION).stat().st_size,
                "component_sha256": snapshot_validation["component_sha256"],
                "status": snapshot_validation["status"],
                "checked_unique_pointer_count": snapshot_validation[
                    "checked_unique_pointer_count"
                ],
            },
        }
        audit_manifest["component_sha256"] = _component_digest(audit_manifest)
        _write_json(staging / AUDIT_DIR / AUDIT_MANIFEST, audit_manifest)
        if out_dir.exists():
            raise ReviewPoolError(f"refusing to overwrite existing output: {out_dir}")
        os.rename(staging, out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = build_review_pool(
        run_dir=args.run_dir,
        catalogue_path=args.catalogue,
        query_fixture=args.queries,
        out_dir=args.out,
        snapshot_root=args.snapshot_root,
    )
    print(json.dumps({
        "status": "published",
        "component_sha256": manifest["component_sha256"],
        "reviewer_files": [REVIEW_POOL, MANIFEST],
        "private_system_data": f"{AUDIT_DIR}/{SYSTEM_PROVENANCE}",
        "unique_candidate_count": manifest["unique_candidate_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
