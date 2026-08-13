"""PCA visualization and original-space diagnostics for a Rust fast index."""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .index import FastIndex


def _pca(vectors: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    count, dimensions = vectors.shape
    output = np.zeros((count, 2), dtype=np.float64)
    rank = min(2, dimensions, max(0, count - 1))
    if rank == 0:
        return output, [0.0, 0.0]
    centered = vectors.astype(np.float64) - np.mean(vectors, axis=0, dtype=np.float64)
    target = min(dimensions, max(rank, rank + 8), max(1, count - 1))
    rng = np.random.default_rng(seed)
    sample = centered @ rng.standard_normal((dimensions, target))
    for _ in range(2):
        basis, _ = np.linalg.qr(sample, mode="reduced")
        sample = centered @ (centered.T @ basis)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    reduced = basis.T @ centered
    _, singular, components = np.linalg.svd(reduced, full_matrices=False)
    components = components[:rank]
    singular = singular[:rank]
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    output[:, :rank] = centered @ components.T
    total = float(np.sum(centered * centered))
    variance = [float(value * value / total) if total else 0.0 for value in singular]
    return output, variance + [0.0] * (2 - rank)


def _group(row: dict[str, Any]) -> str:
    for field in ("relation",):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    relations = row.get("relations_json")
    if isinstance(relations, str):
        try:
            values = json.loads(relations)
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list) and values and all(isinstance(value, str) for value in values):
            return "+".join(values)
    value = row.get("document_kind")
    if isinstance(value, str) and value:
        return value
    return "documents"


def _color(value: str) -> tuple[int, int, int]:
    hue = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.67, 0.78)
    return round(red * 255), round(green * 255), round(blue * 255)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_scatter(
    path: Path,
    coordinates: np.ndarray,
    groups: Sequence[str],
    marked: set[int],
) -> dict[str, list[float]]:
    width, height = 1200, 900
    left, right, top, bottom = 70, 30, 30, 60
    pixels = bytearray([255] * (width * height * 3))

    def point(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    for x in range(left, width - right):
        point(x, height - bottom, (70, 70, 70))
    for y in range(top, height - bottom + 1):
        point(left, y, (70, 70, 70))
    x_values, y_values = coordinates[:, 0], coordinates[:, 1]
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    x_pad, y_pad = (x_max - x_min) * 0.04, (y_max - y_min) * 0.04
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    plot_width, plot_height = width - left - right, height - top - bottom
    for index, group in enumerate(groups):
        x = left + round((x_values[index] - x_min) / (x_max - x_min) * plot_width)
        y = top + round((y_max - y_values[index]) / (y_max - y_min) * plot_height)
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            point(x + dx, y + dy, _color(group))
        if index in marked:
            for dx, dy in (
                (-4, 0), (4, 0), (0, -4), (0, 4),
                (-3, -3), (3, 3), (-3, 3), (3, -3),
            ):
                point(x + dx, y + dy, (0, 0, 0))
    raw = b"".join(
        b"\0" + bytes(pixels[row * width * 3 : (row + 1) * width * 3])
        for row in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    return {"x": [x_min, x_max], "y": [y_min, y_max]}


def write_pca_report(
    index_root: Path,
    output_dir: Path,
    *,
    seed: int = 0,
    mark_count: int = 12,
) -> dict[str, Any]:
    """Write a PCA PNG and JSON report without participating in retrieval."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(mark_count, bool) or not isinstance(mark_count, int) or mark_count < 0:
        raise ValueError("mark_count must be a non-negative integer")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis report: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        with FastIndex.open(index_root) as index:
            vectors = np.asarray(index.vectors, dtype=np.float32)
            coordinates, variance = _pca(vectors, seed)
            centroid = np.mean(vectors.astype(np.float64), axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            similarities = (
                vectors.astype(np.float64) @ (centroid / centroid_norm)
                if centroid_norm else np.zeros(len(vectors), dtype=np.float64)
            )
            distances = 1.0 - np.clip(similarities, -1.0, 1.0)
            marked_order = sorted(
                range(len(index.document_ids)),
                key=lambda item: (-float(distances[item]), index.document_ids[item]),
            )[: min(mark_count, len(index.document_ids))]
            marked = set(marked_order)
            groups = [_group(row) for row in index.metadata]
            bounds = _write_scatter(output_dir / "pca.png", coordinates, groups, marked)
            outliers = [
                {
                    "document_id": index.document_ids[item],
                    "vector_ordinal": item,
                    "group": groups[item],
                    "cosine_distance_from_corpus_centroid": round(float(distances[item]), 12),
                    "pc1": round(float(coordinates[item, 0]), 12),
                    "pc2": round(float(coordinates[item, 1]), 12),
                }
                for item in marked_order
            ]
            report = {
                "schema_version": "livefire.rag.fast-index-pca-report/1",
                "index": {
                    "snapshot_sha256": index.manifest["source"]["snapshot_sha256"],
                    "mapping_sha256": index.manifest["source"]["mapping_sha256"],
                    "document_order_sha256": index.header.document_order_sha256,
                },
                "population": {
                    "documents": index.header.count,
                    "dimensions": index.header.dimensions,
                    "groups": len(set(groups)),
                },
                "pca": {
                    "purpose": "visualization_only_not_retrieval_or_anomaly_space",
                    "seed": seed,
                    "explained_variance_ratio": [round(value, 12) for value in variance],
                    "bounds": bounds,
                    "image": "pca.png",
                },
                "markers": {
                    "metric": "original_embedding_space_cosine_distance_from_corpus_centroid",
                    "interpretation": "geometric isolation only; not maliciousness or relevance",
                    "requested_count": mark_count,
                    "marked_count": len(outliers),
                    "documents": outliers,
                },
                "group_colors": [
                    {"group": value, "rgb": "#%02x%02x%02x" % _color(value)}
                    for value in sorted(set(groups))
                ],
            }
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return report
    except BaseException:
        for path in output_dir.iterdir():
            path.unlink()
        output_dir.rmdir()
        raise


__all__ = ["write_pca_report"]
