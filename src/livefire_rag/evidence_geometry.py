"""Scenario-blind geometry diagnostics for a sealed pilot evidence index.

The analysis consumes only indexed document embeddings and the immutable pilot
selection ledger.  PCA is a visualization aid; every neighbor and isolation
measurement is computed in the original L2-normalized embedding space.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import shutil
import struct
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from .evidence_index import verify_promoted_evidence_index
from .evidence_pilot import pilot_index_binding, verify_evidence_pilot_sample


POLICY_NAME = "geometry-policy.json"
COORDINATES_NAME = "coordinates.parquet"
NEIGHBORS_NAME = "neighbors.parquet"
RELATION_SUMMARY_NAME = "relation-summary.json"
REPORT_NAME = "report.json"
PC12_NAME = "pca-pc1-pc2.png"
PC13_NAME = "pca-pc1-pc3.png"
LOCK_NAME = "objects.lock.json"
MANIFEST_NAME = "manifest.json"
KS = (10, 25, 50)
SCOPE_STATUS = "sample_only_not_corpus_coverage"
ADMISSION_STATUS = "local_evaluation_only_not_sdk_admitted"
CLAIM_STATUS = "geometric_isolation_not_maliciousness_or_retrieval_quality"
_METRIC_DECIMALS = 12
_COORDINATE_DECIMALS = 10


class EvidenceGeometryError(RuntimeError):
    """A sealed geometry report cannot be produced or verified faithfully."""


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - optional dependency
        raise EvidenceGeometryError(
            "DuckDB is required; install livefire-rag[prototype]"
        ) from error
    return duckdb


def _repository_policy() -> dict[str, Any]:
    name = "evidence-pilot-geometry-policy.v1.json"
    path = Path(__file__).resolve().parents[2] / "specs" / name
    if not path.is_file():
        path = Path(__file__).resolve().parent / "evidence_specs" / name
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceGeometryError("geometry policy is unreadable") from error
    if not isinstance(value, dict):
        raise EvidenceGeometryError("geometry policy is invalid")
    return value


def geometry_policy_ref(policy: Mapping[str, Any] | None = None) -> dict[str, str]:
    return component_ref(
        "livefire.rag.evidence-pilot-geometry-policy",
        "1",
        dict(policy) if policy is not None else _repository_policy(),
    )


def _rounded(value: float, places: int = _METRIC_DECIMALS) -> float:
    if not math.isfinite(value):
        raise EvidenceGeometryError("geometry result is not finite")
    rounded = round(float(value), places)
    return 0.0 if rounded == 0 else rounded


def _canonical_selection(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior = ""
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidenceGeometryError(
                    f"selection.jsonl:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict) or raw != canonical_json_bytes(row, newline=True):
                raise EvidenceGeometryError(
                    f"selection.jsonl:{line_number}: non-canonical JSON"
                )
            document_id = row.get("document_id")
            if not isinstance(document_id, str) or document_id <= prior:
                raise EvidenceGeometryError("pilot selection is not unique and sorted")
            prior = document_id
            rows.append(row)
    if not rows:
        raise EvidenceGeometryError("pilot selection is empty")
    return rows


def _load_inputs(
    index_root: Path, pilot_root: Path, sdk_specs: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], np.ndarray]:
    index_manifest = verify_promoted_evidence_index(
        index_root, pilot_sample=pilot_root, sdk_specs=sdk_specs
    )
    pilot_manifest = verify_evidence_pilot_sample(pilot_root, sdk_specs=sdk_specs)
    if index_manifest.get("pilot_sample") != pilot_index_binding(pilot_manifest):
        raise EvidenceGeometryError("index does not bind the supplied pilot sample")
    if "derivation_packs" in index_manifest:
        raise EvidenceGeometryError("pilot geometry does not accept an unselected overlay")

    selection_rows = _canonical_selection(pilot_root / "selection.jsonl")
    selection = {row["document_id"]: row for row in selection_rows}
    docs = str((index_root / "documents.parquet").resolve())
    embeddings = str((index_root / "embeddings.parquet").resolve())
    connection = _duckdb().connect()
    try:
        records = connection.execute(
            "SELECT e.document_id, e.document_sha256, e.dimensions, e.normalization, "
            "e.vector, d.document_sha256, d.document_kind, d.occurrence_count, "
            "to_json(d.relation_identities) "
            "FROM read_parquet(?) e JOIN read_parquet(?) d USING(document_id) "
            "ORDER BY e.document_id",
            [embeddings, docs],
        ).fetchall()
    finally:
        connection.close()
    if len(records) != len(selection_rows):
        raise EvidenceGeometryError("embedding and pilot-selection counts differ")

    metadata: list[dict[str, Any]] = []
    vectors: list[Sequence[float]] = []
    dimensions: int | None = None
    for record in records:
        (
            document_id, embedding_sha, row_dimensions, normalization, vector,
            document_sha, document_kind, occurrence_count, relation_json,
        ) = record
        selected = selection.get(document_id)
        if selected is None:
            raise EvidenceGeometryError("embedding exists outside the pilot selection")
        identities = json.loads(relation_json)
        if not isinstance(identities, list) or len(identities) != 1:
            raise EvidenceGeometryError("pilot document relation identity is not singular")
        relation = identities[0].get("relation")
        if relation != selected.get("relation"):
            raise EvidenceGeometryError("selection relation differs from indexed document")
        if embedding_sha != document_sha or normalization != "l2":
            raise EvidenceGeometryError("embedding/document binding is invalid")
        if dimensions is None:
            dimensions = int(row_dimensions)
        if row_dimensions != dimensions or len(vector) != dimensions:
            raise EvidenceGeometryError("embedding dimensions are inconsistent")
        weight = selected.get("sampling_weight")
        probability = selected.get("inclusion_probability")
        if (
            not isinstance(weight, dict) or not isinstance(probability, dict)
            or weight.get("numerator") != probability.get("denominator")
            or weight.get("denominator") != probability.get("numerator")
        ):
            raise EvidenceGeometryError("pilot inclusion and sampling weights disagree")
        occurrence_count = int(occurrence_count)
        metadata.append({
            "document_id": document_id,
            "document_sha256": document_sha,
            "document_kind": document_kind,
            "relation": relation,
            "occurrence_count": occurrence_count,
            "sampling_weight_numerator": int(weight["numerator"]),
            "sampling_weight_denominator": int(weight["denominator"]),
        })
        vectors.append(vector)
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise EvidenceGeometryError("embedding matrix is invalid")
    norms = np.linalg.norm(array, axis=1)
    if np.any(np.abs(norms - 1.0) > 0.0001):
        raise EvidenceGeometryError("embedding matrix is not L2 normalized")
    return index_manifest, pilot_manifest, metadata, array


def _deterministic_pca(
    vectors: np.ndarray, *, seed: int
) -> tuple[np.ndarray, list[float]]:
    """Return three visualization coordinates using a seeded, sign-fixed PCA."""

    sample_count, dimensions = vectors.shape
    coordinates = np.zeros((sample_count, 3), dtype=np.float64)
    rank = min(3, dimensions, max(0, sample_count - 1))
    if rank == 0:
        return coordinates, [0.0, 0.0, 0.0]
    centered = vectors - np.mean(vectors, axis=0, dtype=np.float64)
    target = min(dimensions, max(rank, rank + 8), max(1, sample_count - 1))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dimensions, target), dtype=np.float64)
    sample = centered @ omega
    for _ in range(2):
        basis, _ = np.linalg.qr(sample, mode="reduced")
        sample = centered @ (centered.T @ basis)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    reduced = basis.T @ centered
    _, singular, components = np.linalg.svd(reduced, full_matrices=False)
    components = components[:rank]
    singular = singular[:rank]
    for component in range(rank):
        loading = components[component]
        pivot = int(np.argmax(np.abs(loading)))
        if loading[pivot] < 0:
            components[component] *= -1
    coordinates[:, :rank] = centered @ components.T
    total_variance = float(np.sum(centered * centered))
    ratios = (
        [float(value * value / total_variance) for value in singular]
        if total_variance > 0 else [0.0] * rank
    )
    return coordinates, [_rounded(value) for value in ratios] + [0.0] * (3 - rank)


def _neighbor_geometry(
    metadata: Sequence[Mapping[str, Any]], vectors: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    count = len(metadata)
    document_ids = [str(row["document_id"]) for row in metadata]
    relations = [str(row["relation"]) for row in metadata]
    by_relation: dict[str, list[int]] = defaultdict(list)
    for index, relation in enumerate(relations):
        by_relation[relation].append(index)

    global_neighbor: list[int | None] = [None] * count
    global_distance: list[float | None] = [None] * count
    cross_neighbor: list[int | None] = [None] * count
    cross_distance: list[float | None] = [None] * count
    block_size = 256
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        similarities = vectors[start:stop] @ vectors.T
        for offset, index in enumerate(range(start, stop)):
            similarities[offset, index] = -np.inf
            if count > 1:
                nearest = int(np.argmax(similarities[offset]))
                global_neighbor[index] = nearest
                global_distance[index] = max(0.0, min(2.0, 1.0 - float(similarities[offset, nearest])))
            cross = similarities[offset].copy()
            cross[np.asarray([value == relations[index] for value in relations])] = -np.inf
            if np.isfinite(cross).any():
                nearest_cross = int(np.argmax(cross))
                cross_neighbor[index] = nearest_cross
                cross_distance[index] = max(0.0, min(2.0, 1.0 - float(cross[nearest_cross])))

    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    relation_stats: dict[tuple[str, int], tuple[float, float]] = {}
    for relation in sorted(by_relation):
        indices = by_relation[relation]
        group = vectors[indices]
        distances = 1.0 - np.clip(group @ group.T, -1.0, 1.0)
        np.fill_diagonal(distances, np.inf)
        local_neighbors: dict[int, list[int]] = {}
        for local_index, global_index in enumerate(indices):
            order = np.lexsort((np.asarray(indices), distances[local_index]))
            local_neighbors[global_index] = [indices[int(value)] for value in order if int(value) != local_index]
        for requested_k in KS:
            effective_k = min(requested_k, len(indices) - 1)
            values: list[float] = []
            for global_index in indices:
                neighbors = local_neighbors[global_index][:effective_k]
                neighbor_distances = [
                    max(0.0, min(2.0, 1.0 - float(vectors[global_index] @ vectors[item])))
                    for item in neighbors
                ]
                value = float(np.mean(neighbor_distances)) if neighbor_distances else 0.0
                values.append(value)
                reciprocal = (
                    sum(
                        global_index in local_neighbors[item][:effective_k]
                        for item in neighbors
                    ) / effective_k
                    if effective_k else 0.0
                )
                rows_by_key[(global_index, requested_k)] = {
                    "document_id": document_ids[global_index],
                    "relation": relation,
                    "requested_k": requested_k,
                    "effective_k": effective_k,
                    "neighbor_document_ids": [document_ids[item] for item in neighbors],
                    "neighbor_cosine_distances": [_rounded(value) for value in neighbor_distances],
                    "mean_cosine_distance": _rounded(value),
                    "kth_cosine_distance": (
                        _rounded(neighbor_distances[-1]) if neighbor_distances else None
                    ),
                    "reciprocal_neighbor_rate": _rounded(reciprocal),
                    "nearest_global_document_id": (
                        document_ids[global_neighbor[global_index]]
                        if global_neighbor[global_index] is not None else None
                    ),
                    "nearest_global_relation": (
                        relations[global_neighbor[global_index]]
                        if global_neighbor[global_index] is not None else None
                    ),
                    "nearest_global_cosine_distance": (
                        _rounded(global_distance[global_index])
                        if global_distance[global_index] is not None else None
                    ),
                    "nearest_cross_relation_document_id": (
                        document_ids[cross_neighbor[global_index]]
                        if cross_neighbor[global_index] is not None else None
                    ),
                    "nearest_cross_relation": (
                        relations[cross_neighbor[global_index]]
                        if cross_neighbor[global_index] is not None else None
                    ),
                    "nearest_cross_relation_cosine_distance": (
                        _rounded(cross_distance[global_index])
                        if cross_distance[global_index] is not None else None
                    ),
                }
            median = float(np.median(values))
            mad = float(np.median(np.abs(np.asarray(values) - median)))
            relation_stats[(relation, requested_k)] = (median, mad)

    rows: list[dict[str, Any]] = []
    for index in range(count):
        for requested_k in KS:
            row = rows_by_key[(index, requested_k)]
            median, mad = relation_stats[(relations[index], requested_k)]
            row["relation_median_cosine_distance"] = _rounded(median)
            row["relation_mad_cosine_distance"] = _rounded(mad)
            if row["effective_k"] == 0:
                row["robust_isolation_score"] = None
                row["score_status"] = "insufficient_relation_support"
            elif mad <= 1e-12:
                row["robust_isolation_score"] = None
                row["score_status"] = "zero_relation_mad"
            else:
                row["robust_isolation_score"] = _rounded(
                    0.6744897501960817
                    * (row["mean_cosine_distance"] - median) / mad
                )
                row["score_status"] = "available"
            rows.append(row)

    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    relation_totals: dict[str, int] = defaultdict(int)
    for index, relation in enumerate(relations):
        relation_totals[relation] += 1
        neighbor_index = global_neighbor[index]
        neighbor_relation = relations[neighbor_index] if neighbor_index is not None else "none"
        confusion_counts[(relation, neighbor_relation)] += 1
    confusion = [
        {
            "source_relation": source,
            "nearest_neighbor_relation": target,
            "document_count": value,
            "source_relation_fraction": _rounded(value / relation_totals[source]),
            "cross_relation": source != target,
        }
        for (source, target), value in sorted(confusion_counts.items())
    ]

    relation_summary: list[dict[str, Any]] = []
    for relation in sorted(by_relation):
        indices = by_relation[relation]
        for requested_k in KS:
            selected = [rows_by_key[(index, requested_k)] for index in indices]
            values = np.asarray([row["mean_cosine_distance"] for row in selected])
            relation_summary.append({
                "relation": relation,
                "requested_k": requested_k,
                "document_count": len(indices),
                "effective_k": min(requested_k, len(indices) - 1),
                "mean_isolation": _rounded(float(np.mean(values))),
                "median_isolation": _rounded(float(np.median(values))),
                "p90_isolation": _rounded(float(np.quantile(values, 0.9))),
                "p99_isolation": _rounded(float(np.quantile(values, 0.99))),
                "mean_reciprocal_neighbor_rate": _rounded(float(np.mean([
                    row["reciprocal_neighbor_rate"] for row in selected
                ]))),
                "available_robust_score_count": sum(
                    row.get("score_status") == "available" for row in selected
                ),
            })
    return rows, relation_summary, confusion


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        raise EvidenceGeometryError("sensitivity weights are invalid")
    return float(np.sum(values * weights) / total)


def _sensitivity(
    metadata: Sequence[Mapping[str, Any]], neighbor_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_document = {row["document_id"]: row for row in metadata}
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, str | None]] = [("all_relations", None)] + [
        ("relation", relation)
        for relation in sorted({str(row["relation"]) for row in metadata})
    ]
    for scope, relation in scopes:
        for requested_k in KS:
            rows = [
                row for row in neighbor_rows
                if row["requested_k"] == requested_k
                and (relation is None or row["relation"] == relation)
            ]
            values = np.asarray(
                [row["mean_cosine_distance"] for row in rows], dtype=np.float64
            )
            occurrences = np.asarray([
                by_document[row["document_id"]]["occurrence_count"] for row in rows
            ], dtype=np.float64)
            inclusion = np.asarray([
                by_document[row["document_id"]]["sampling_weight_numerator"]
                / by_document[row["document_id"]]["sampling_weight_denominator"]
                for row in rows
            ], dtype=np.float64)
            weights = {
                "document": np.ones(len(rows), dtype=np.float64),
                "occurrence_count": occurrences,
                "inverse_inclusion_probability": inclusion,
                "occurrence_count_times_inverse_inclusion_probability": occurrences * inclusion,
            }
            baseline = _weighted_mean(values, weights["document"])
            for name in (
                "document", "occurrence_count", "inverse_inclusion_probability",
                "occurrence_count_times_inverse_inclusion_probability",
            ):
                mean = _weighted_mean(values, weights[name])
                result = {
                    "scope": scope,
                    "requested_k": requested_k,
                    "weighting": name,
                    "weighted_mean_isolation": _rounded(mean),
                    "difference_from_document_mean": _rounded(mean - baseline),
                    "weight_sum": _rounded(float(np.sum(weights[name]))),
                }
                if relation is not None:
                    result["relation"] = relation
                output.append(result)
    return output


def _relation_color(relation: str) -> tuple[int, int, int]:
    hue = int(sha256_bytes(relation.encode("utf-8"))[:8], 16) / 0xFFFFFFFF
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.67, 0.78)
    return round(red * 255), round(green * 255), round(blue * 255)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _scatter_png(
    path: Path,
    coordinates: np.ndarray,
    relations: Sequence[str],
    marked: Sequence[bool],
    x_component: int,
    y_component: int,
) -> dict[str, list[float]]:
    width, height = 1200, 900
    left, right, top, bottom = 70, 30, 30, 60
    pixels = bytearray([255] * (width * height * 3))

    def point(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset:offset + 3] = bytes(color)

    for x in range(left, width - right):
        point(x, height - bottom, (70, 70, 70))
    for y in range(top, height - bottom + 1):
        point(left, y, (70, 70, 70))

    x_values = coordinates[:, x_component]
    y_values = coordinates[:, y_component]
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    x_padding = (x_max - x_min) * 0.04
    y_padding = (y_max - y_min) * 0.04
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    plot_width = width - left - right
    plot_height = height - top - bottom
    for index, relation in enumerate(relations):
        x = left + round((x_values[index] - x_min) / (x_max - x_min) * plot_width)
        y = top + round((y_max - y_values[index]) / (y_max - y_min) * plot_height)
        color = _relation_color(relation)
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            point(x + dx, y + dy, color)
        if marked[index]:
            for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
                point(x + dx, y + dy, (0, 0, 0))
    raw = b"".join(
        b"\x00" + bytes(pixels[row * width * 3:(row + 1) * width * 3])
        for row in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    return {
        "x": [_rounded(x_min, _COORDINATE_DECIMALS), _rounded(x_max, _COORDINATE_DECIMALS)],
        "y": [_rounded(y_min, _COORDINATE_DECIMALS), _rounded(y_max, _COORDINATE_DECIMALS)],
    }


def _write_parquet(
    staging: Path,
    metadata: Sequence[Mapping[str, Any]],
    coordinates: np.ndarray,
    neighbor_rows: Sequence[Mapping[str, Any]],
) -> None:
    connection = _duckdb().connect()
    try:
        connection.execute(
            "CREATE TABLE coordinates(document_id VARCHAR, relation VARCHAR, document_kind VARCHAR, "
            "occurrence_count BIGINT, sampling_weight_numerator BIGINT, "
            "sampling_weight_denominator BIGINT, pc1 DOUBLE, pc2 DOUBLE, pc3 DOUBLE, "
            "high_isolation_marker BOOLEAN)"
        )
        score_by_document: dict[str, list[float]] = defaultdict(list)
        for row in neighbor_rows:
            if row["robust_isolation_score"] is not None:
                score_by_document[row["document_id"]].append(row["robust_isolation_score"])
        coordinate_rows = []
        for index, row in enumerate(metadata):
            coordinate_rows.append((
                row["document_id"], row["relation"], row["document_kind"],
                row["occurrence_count"], row["sampling_weight_numerator"],
                row["sampling_weight_denominator"],
                _rounded(coordinates[index, 0], _COORDINATE_DECIMALS),
                _rounded(coordinates[index, 1], _COORDINATE_DECIMALS),
                _rounded(coordinates[index, 2], _COORDINATE_DECIMALS),
                max(score_by_document[row["document_id"]], default=-math.inf) >= 3.5,
            ))
        connection.executemany(
            "INSERT INTO coordinates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            coordinate_rows,
        )
        connection.execute(
            "CREATE TABLE neighbors(document_id VARCHAR, relation VARCHAR, requested_k INTEGER, "
            "effective_k INTEGER, neighbor_document_ids VARCHAR[], neighbor_cosine_distances DOUBLE[], "
            "mean_cosine_distance DOUBLE, kth_cosine_distance DOUBLE, reciprocal_neighbor_rate DOUBLE, "
            "relation_median_cosine_distance DOUBLE, relation_mad_cosine_distance DOUBLE, "
            "robust_isolation_score DOUBLE, score_status VARCHAR, nearest_global_document_id VARCHAR, "
            "nearest_global_relation VARCHAR, nearest_global_cosine_distance DOUBLE, "
            "nearest_cross_relation_document_id VARCHAR, nearest_cross_relation VARCHAR, "
            "nearest_cross_relation_cosine_distance DOUBLE)"
        )
        connection.executemany(
            "INSERT INTO neighbors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row[key] for key in (
                "document_id", "relation", "requested_k", "effective_k",
                "neighbor_document_ids", "neighbor_cosine_distances",
                "mean_cosine_distance", "kth_cosine_distance", "reciprocal_neighbor_rate",
                "relation_median_cosine_distance", "relation_mad_cosine_distance",
                "robust_isolation_score", "score_status", "nearest_global_document_id",
                "nearest_global_relation", "nearest_global_cosine_distance",
                "nearest_cross_relation_document_id", "nearest_cross_relation",
                "nearest_cross_relation_cosine_distance",
            )) for row in neighbor_rows],
        )
        connection.execute(
            "COPY (SELECT * FROM coordinates ORDER BY document_id) TO ? "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(staging / COORDINATES_NAME)],
        )
        connection.execute(
            "COPY (SELECT * FROM neighbors ORDER BY document_id, requested_k) TO ? "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(staging / NEIGHBORS_NAME)],
        )
    finally:
        connection.close()


def build_evidence_pilot_geometry(
    index_root: Path,
    pilot_root: Path,
    output_dir: Path,
    *,
    sdk_specs: Path,
    component_id: str,
    version: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Seal deterministic index-only PCA and original-space kNN diagnostics."""

    if not component_id or not version:
        raise ValueError("geometry component id and version must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed < 2**31:
        raise ValueError("geometry seed must be a signed 32-bit integer")
    index_root = Path(index_root).resolve()
    pilot_root = Path(pilot_root).resolve()
    output_dir = Path(output_dir).resolve()
    sdk_specs = Path(sdk_specs).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pilot geometry: {output_dir}")

    index_manifest, pilot_manifest, metadata, vectors = _load_inputs(
        index_root, pilot_root, sdk_specs
    )
    policy = _repository_policy()
    policy_ref = geometry_policy_ref(policy)
    selection_ref = pilot_manifest["objects"]["selection"]
    embeddings_ref = index_manifest["objects"]["embeddings"]
    seed_material = {
        "schema_version": "livefire.rag.evidence-pilot-geometry-seed/1",
        "index": index_manifest["component"],
        "embeddings": embeddings_ref,
        "pilot_sample": pilot_manifest["component"],
        "selection": selection_ref,
        "geometry_policy": policy_ref,
        "caller_seed": seed,
    }
    seed_sha256 = sha256_bytes(canonical_json_bytes(seed_material))
    derived_seed = int(seed_sha256[:16], 16)
    coordinates, variance = _deterministic_pca(vectors, seed=derived_seed)
    neighbor_rows, relation_summary, confusion = _neighbor_geometry(metadata, vectors)
    sensitivity = _sensitivity(metadata, neighbor_rows)
    marked_by_id = {
        row["document_id"]
        for row in neighbor_rows
        if row["robust_isolation_score"] is not None
        and row["robust_isolation_score"] >= 3.5
    }
    relations = [str(row["relation"]) for row in metadata]
    marked = [str(row["document_id"]) in marked_by_id for row in metadata]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        write_canonical_json(staging / POLICY_NAME, policy)
        _write_parquet(staging, metadata, coordinates, neighbor_rows)
        bounds12 = _scatter_png(staging / PC12_NAME, coordinates, relations, marked, 0, 1)
        bounds13 = _scatter_png(staging / PC13_NAME, coordinates, relations, marked, 0, 2)
        relation_colors = [
            {"relation": relation, "rgb_hex": "#%02x%02x%02x" % _relation_color(relation)}
            for relation in sorted(set(relations))
        ]
        summary = {
            "schema_version": "livefire.rag.evidence-pilot-relation-geometry-summary/1",
            "claim_status": CLAIM_STATUS,
            "relations": relation_summary,
            "cross_relation_nearest_neighbor_confusion": confusion,
            "aggregate_weight_sensitivity": sensitivity,
        }
        write_canonical_json(staging / RELATION_SUMMARY_NAME, summary)
        report = {
            "schema_version": "livefire.rag.evidence-pilot-geometry-report/1",
            "scope_status": SCOPE_STATUS,
            "admission_status": ADMISSION_STATUS,
            "claim_status": CLAIM_STATUS,
            "interpretation": {
                "isolation": "distance_from_same_relation_neighbors_in_this_embedding_corpus",
                "does_not_establish": [
                    "maliciousness", "security_relevance", "retrieval_quality", "evidence_truth"
                ],
            },
            "inputs": {
                "index": index_manifest["component"],
                "embeddings": embeddings_ref,
                "embedding_profile": index_manifest["embedding_profiles"][0],
                "pilot_sample": pilot_manifest["component"],
                "selection": selection_ref,
                "geometry_policy": policy_ref,
                "caller_seed": seed,
                "derived_seed_sha256": seed_sha256,
                "derived_seed_hex": seed_sha256[:16],
            },
            "population": {
                "documents": len(metadata),
                "occurrences": sum(int(row["occurrence_count"]) for row in metadata),
                "relations": len(set(relations)),
                "embedding_dimensions": int(vectors.shape[1]),
            },
            "pca": {
                "purpose": "visualization_only_not_neighbor_or_score_space",
                "explained_variance_ratio": variance,
                "pc1_pc2_bounds": bounds12,
                "pc1_pc3_bounds": bounds13,
                "relation_colors": relation_colors,
                "high_isolation_marker_document_count": len(marked_by_id),
            },
            "neighbor_geometry": {
                "space": "original_l2_embedding",
                "metric": "cosine_distance",
                "exact": True,
                "requested_k": list(KS),
                "tie_break": "cosine_distance_ascending_document_id_ascending",
                "per_document_rows": len(neighbor_rows),
            },
            "sensitivity": sensitivity,
        }
        write_canonical_json(staging / REPORT_NAME, report)
        artifacts = [
            artifact_ref(staging / name, name, media_type)
            for name, media_type in (
                (POLICY_NAME, "application/json"),
                (COORDINATES_NAME, "application/vnd.apache.parquet"),
                (NEIGHBORS_NAME, "application/vnd.apache.parquet"),
                (RELATION_SUMMARY_NAME, "application/json"),
                (REPORT_NAME, "application/json"),
                (PC12_NAME, "image/png"),
                (PC13_NAME, "image/png"),
            )
        ]
        artifacts.sort(key=lambda row: row["path"])
        write_canonical_json(staging / LOCK_NAME, {
            "schema_version": "livefire.object-lock/1", "objects": artifacts,
        })
        objects = {row["path"]: row for row in artifacts}
        objects[LOCK_NAME] = artifact_ref(
            staging / LOCK_NAME, LOCK_NAME, "application/vnd.livefire.object-lock+json"
        )
        component: dict[str, str] = {"id": component_id, "version": version, "sha256": ""}
        manifest = {
            "schema_version": "livefire.rag.evidence-pilot-geometry/1",
            "component": component,
            "scope_status": SCOPE_STATUS,
            "admission_status": ADMISSION_STATUS,
            "claim_status": CLAIM_STATUS,
            "index": index_manifest["component"],
            "pilot_sample": pilot_manifest["component"],
            "geometry_policy": policy_ref,
            "seed_sha256": seed_sha256,
            "objects": objects,
            "closure": {
                "selected_document_count": len(metadata),
                "coordinate_count": len(metadata),
                "neighbor_row_count": len(neighbor_rows),
                "all_selection_rows_joined": True,
                "all_embeddings_joined": True,
            },
        }
        manifest["component"]["sha256"] = canonical_sha256_omitting(
            manifest, ("component", "sha256")
        )
        write_canonical_json(staging / MANIFEST_NAME, manifest)
        verify_evidence_pilot_geometry(staging)
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_evidence_pilot_geometry(root: Path) -> dict[str, Any]:
    """Verify report identity, immutable objects, row counts, and claim boundary."""

    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceGeometryError("geometry manifest is unreadable") from error
    if canonical_sha256_omitting(manifest, ("component", "sha256")) != manifest.get("component", {}).get("sha256"):
        raise EvidenceGeometryError("geometry component identity mismatch")
    if (
        manifest.get("scope_status") != SCOPE_STATUS
        or manifest.get("admission_status") != ADMISSION_STATUS
        or manifest.get("claim_status") != CLAIM_STATUS
    ):
        raise EvidenceGeometryError("geometry scope or claim boundary is invalid")
    names = {
        POLICY_NAME, COORDINATES_NAME, NEIGHBORS_NAME, RELATION_SUMMARY_NAME,
        REPORT_NAME, PC12_NAME, PC13_NAME, LOCK_NAME,
    }
    if set(manifest.get("objects", {})) != names:
        raise EvidenceGeometryError("geometry object set mismatch")
    for name in names:
        ref = manifest["objects"][name]
        path = root / name
        if (
            ref.get("path") != name or not path.is_file()
            or ref.get("bytes") != path.stat().st_size
            or ref.get("sha256") != sha256_file(path)
        ):
            raise EvidenceGeometryError(f"geometry object mismatch: {name}")
    locked = [manifest["objects"][name] for name in names if name != LOCK_NAME]
    locked.sort(key=lambda row: row["path"])
    if json.loads((root / LOCK_NAME).read_text(encoding="utf-8")) != {
        "schema_version": "livefire.object-lock/1", "objects": locked,
    }:
        raise EvidenceGeometryError("geometry object lock mismatch")
    policy = json.loads((root / POLICY_NAME).read_text(encoding="utf-8"))
    if geometry_policy_ref(policy) != manifest.get("geometry_policy"):
        raise EvidenceGeometryError("geometry policy binding mismatch")
    report = json.loads((root / REPORT_NAME).read_text(encoding="utf-8"))
    if (
        report.get("claim_status") != CLAIM_STATUS
        or report.get("neighbor_geometry", {}).get("space") != "original_l2_embedding"
        or report.get("neighbor_geometry", {}).get("requested_k") != list(KS)
        or report.get("inputs", {}).get("derived_seed_sha256") != manifest.get("seed_sha256")
    ):
        raise EvidenceGeometryError("geometry report contract mismatch")
    connection = _duckdb().connect()
    try:
        coordinate_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str((root / COORDINATES_NAME).resolve())]
        ).fetchone()[0]
        neighbor_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str((root / NEIGHBORS_NAME).resolve())]
        ).fetchone()[0]
    finally:
        connection.close()
    closure = manifest.get("closure", {})
    if (
        coordinate_count != closure.get("coordinate_count")
        or neighbor_count != closure.get("neighbor_row_count")
        or neighbor_count != coordinate_count * len(KS)
    ):
        raise EvidenceGeometryError("geometry row counts do not close")
    if not (root / PC12_NAME).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") or not (
        root / PC13_NAME
    ).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise EvidenceGeometryError("geometry visualization is not PNG")
    return manifest


__all__ = [
    "CLAIM_STATUS", "EvidenceGeometryError", "build_evidence_pilot_geometry",
    "geometry_policy_ref", "verify_evidence_pilot_geometry",
]
