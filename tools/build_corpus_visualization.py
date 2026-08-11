#!/usr/bin/env python3
"""Build a compact, privacy-safe PCA/isolation dataset for corpus inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from prototype_query_demo import ROOT, build_corpus, corpus_digest


KIND_LABELS = {
    "source_powershell_script_block": "Hydrated PowerShell 4104",
    "source_sysmon_process_command": "Hydrated Sysmon process",
    "source_bash_history": "Hydrated Bash history",
    "ocsf_process_command": "M21 typed process",
    "ocsf_api_activity": "M21 typed API",
}

ANCHORS = {
    "openbots:event_address:302:182806": "PowerShell logging bypass",
    "openbots:event_address:302:175924": "SYSTEM scheduled-task persistence",
    "openbots:event_address:302:167905": "Local service-account creation",
    "openbots:event_address:302:381380": "Address-range scanner",
    "openbots:event_address:302:410059": "Windows Firewall disabled",
    "openbots:event_address:306:560330": "Archive upload to S3",
    "m21:event:evt_f0ad7ba4f3ede5795614f1c2e911c5e4de0b378048b714f3bd13108215983169": "EC2 fleet launch denied",
    "m21:event:evt_680a612b911caed243856e58ddd85cb4fe6408f654cb2322e7d042808ba6eeb2": "IAM access-key creation denied",
    "m21:event:evt_4bf7559e5ed84dbfad3c4c36bae823b9cf98f91bbdd724598eb59771fcfe5344": "S3 bucket made public",
    "m21:event:evt_efeedcfa94bdeb97741ccbc1a97dc347ea1af04a4ab9c6c36f9908ab0d5549bb": "S3 bucket access tightened",
    "openbots:event_address:302:122607": "Encoded PowerShell process",
    "m21:event:evt_b7a17f5dfb21fcbac82eef3daa50d043721441e3ac72a1cfab8ba2f207f2c0f0": "S3 upload process projection",
}

PLOT_COLORS = {
    "source_powershell_script_block": "#7c3aed",
    "source_sysmon_process_command": "#2563eb",
    "source_bash_history": "#059669",
    "ocsf_process_command": "#d97706",
    "ocsf_api_activity": "#dc2626",
}


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values) * 100.0


def basename(value: object) -> str:
    if not value:
        return ""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def safe_summary(doc) -> str:
    if doc.kind == "ocsf_api_activity":
        service = str(doc.metadata.get("service") or "cloud API").split(".", 1)[0]
        operation = doc.metadata.get("operation") or "operation"
        status = doc.metadata.get("status") or "unknown status"
        return f"{service} {operation} — {status}"
    if doc.kind in {"source_sysmon_process_command", "ocsf_process_command"}:
        image = basename(doc.metadata.get("image"))
        parent = basename(doc.metadata.get("parent_image"))
        if image and parent:
            return f"{image} spawned by {parent}"
        if image:
            return image
        return "Process command"
    if doc.kind == "source_powershell_script_block":
        projection = doc.projection.lower()
        if "scriptblocklogging" in projection or "cachedgrouppolicysettings" in projection:
            return "PowerShell logging-policy modification"
        if "frombase64string" in projection and ("registry" in projection or "hklm:" in projection):
            return "PowerShell registry payload decode"
        if "downloadstring" in projection or "webclient" in projection:
            return "PowerShell remote-content retrieval"
        return "PowerShell script block"
    if doc.kind == "source_bash_history":
        projection = doc.projection.lower()
        if "s3-upload" in projection or "aws s3" in projection:
            return "Bash object-storage upload"
        if "curl " in projection or "wget " in projection:
            return "Bash network retrieval"
        if "tar " in projection:
            return "Bash archive operation"
        if "python " in projection:
            return "Bash Python command"
        return "Bash history command"
    return KIND_LABELS.get(doc.kind, doc.kind)


def all_locators(doc) -> set[str]:
    return {doc.locator, *doc.aliases}


def render_png(output: dict[str, object], path: Path) -> None:
    """Render a deterministic, redacted PCA overview as a standalone PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = output["points"]
    meta = output["meta"]
    anchor_order = {label: index + 1 for index, label in enumerate(ANCHORS.values())}
    anchors = sorted(
        (point for point in points if point["anchor"]),
        key=lambda point: anchor_order[point["summary"]],
    )
    isolation = sorted(
        points,
        key=lambda point: (-point["isolation"], point["id"]),
    )[:10]
    isolation_order = {point["id"]: index + 1 for index, point in enumerate(isolation)}

    figure = plt.figure(figsize=(18, 11), dpi=160, facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[3.7, 1.35],
        left=0.055,
        right=0.985,
        top=0.845,
        bottom=0.055,
        wspace=0.15,
        hspace=0.24,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    key_axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    panels = [("x12", "y12", "PC2"), ("x13", "y13", "PC3")]

    for axis, (x_field, y_field, y_label) in zip(axes, panels):
        for kind, label in KIND_LABELS.items():
            group = [point for point in points if point["kind"] == kind]
            axis.scatter(
                [point[x_field] for point in group],
                [point[y_field] for point in group],
                s=9,
                c=PLOT_COLORS[kind],
                alpha=0.24,
                linewidths=0,
                rasterized=True,
                label=label,
            )
        top_one_percent = [point for point in points if point["isolationPercentile"] >= 99.0]
        axis.scatter(
            [point[x_field] for point in top_one_percent],
            [point[y_field] for point in top_one_percent],
            s=45,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.8,
            alpha=0.9,
            label="Top 1% isolation",
        )
        axis.scatter(
            [point[x_field] for point in isolation],
            [point[y_field] for point in isolation],
            s=57,
            marker="D",
            facecolors="#111827",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        for point in isolation:
            axis.annotate(
                f"O{isolation_order[point['id']]}",
                (point[x_field], point[y_field]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="#111827",
                zorder=7,
            )
        for point in anchors:
            number = anchor_order[point["summary"]]
            axis.scatter(
                [point[x_field]],
                [point[y_field]],
                s=92,
                c=PLOT_COLORS[point["kind"]],
                edgecolors="#111827",
                linewidths=1.2,
                zorder=6,
            )
            axis.annotate(
                str(number),
                (point[x_field], point[y_field]),
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="white",
                zorder=8,
            )
        axis.set_xlabel("PC1 score")
        axis.set_ylabel(f"{y_label} score")
        variance_index = 1 if y_label == "PC2" else 2
        explained = (
            meta["explainedVariance"]["pc1"]
            + [meta["explainedVariance"]["pc2"], meta["explainedVariance"]["pc3"]][
                variance_index - 1
            ]
        )
        axis.set_title(f"PC1 × {y_label} · {explained * 100:.2f}% variance", loc="left", fontsize=12)
        axis.grid(True, color="#e5e7eb", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#9ca3af")
            spine.set_linewidth(0.7)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=PLOT_COLORS[kind], label=label)
        for kind, label in KIND_LABELS.items()
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor="#111827", label="Top 1% isolation ring"),
            Line2D([0], [0], marker="D", linestyle="", markersize=6, markerfacecolor="#111827", color="#111827", label="Top 10 isolation candidate"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    key_axes[0].axis("off")
    key_axes[0].set_title("Numbered investigation examples", loc="left", fontsize=11, fontweight="bold")
    anchor_lines = [
        f"{anchor_order[point['summary']]:>2}. {point['summary']}"
        for point in anchors
    ]
    midpoint = (len(anchor_lines) + 1) // 2
    for column, lines in enumerate((anchor_lines[:midpoint], anchor_lines[midpoint:])):
        key_axes[0].text(
            column * 0.5,
            0.88,
            "\n".join(lines),
            transform=key_axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.45,
            color="#111827",
        )

    key_axes[1].axis("off")
    key_axes[1].set_title("Top embedding-space isolation candidates", loc="left", fontsize=11, fontweight="bold")
    isolation_lines = []
    for point in isolation:
        summary = textwrap.shorten(point["summary"], width=43, placeholder="…")
        isolation_lines.append(
            f"O{isolation_order[point['id']]}  {point['isolationPercentile']:>6.2f}th  {summary}"
        )
    midpoint = (len(isolation_lines) + 1) // 2
    for column, lines in enumerate((isolation_lines[:midpoint], isolation_lines[midpoint:])):
        key_axes[1].text(
            column * 0.5,
            0.88,
            "\n".join(lines),
            transform=key_axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            family="monospace",
            linespacing=1.48,
            color="#111827",
        )

    figure.suptitle(
        "LiveFire RAG embedding corpus — PCA overview",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.055,
        0.945,
        (
            f"{meta['documents']:,} redacted semantic documents · {meta['dimensions']:,}-D Qwen3 embeddings · "
            f"corpus {meta['corpusDigest'][:12]}… · PCA seed 20260811"
        ),
        ha="left",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.text(
        0.055,
        0.018,
        "Exploratory projection only. Rings are embedding-space isolation in original cosine space—not maliciousness, anomaly probability, or evidence.",
        ha="left",
        fontsize=8.5,
        color="#4b5563",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        type=Path,
        default=ROOT / "reports/prototype-rag-demo/corpus-vectors.npy",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "reports/prototype-rag-demo/corpus-cache.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/prototype-rag-demo/corpus-visualization.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "tools/corpus-visualization.fragment.html",
    )
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--png-out", type=Path)
    args = parser.parse_args()

    corpus, corpus_stats, _ = build_corpus(duckdb.connect())
    vectors = np.load(args.vectors)
    cache = json.loads(args.cache.read_text())
    digest = corpus_digest(corpus)
    if cache.get("corpus_digest") != digest:
        raise SystemExit("vector cache corpus digest does not match rebuilt corpus")
    if vectors.shape != (len(corpus), 4096):
        raise SystemExit(f"unexpected vector shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise SystemExit("vectors contain NaN or infinity")

    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-5:
        raise SystemExit("vectors violate L2-normalization tolerance")

    # All isolation metrics are calculated in the original 4,096-D cosine
    # space. The 2-D/3-D PCA coordinates are display-only.
    neighbors = NearestNeighbors(n_neighbors=51, metric="cosine", algorithm="brute")
    distances, indices = neighbors.fit(vectors).kneighbors(vectors)
    nearest_dist = distances[:, 1]
    knn20_median = np.median(distances[:, 1:21], axis=1)
    knn10_median = np.median(distances[:, 1:11], axis=1)
    knn50_median = np.median(distances[:, 1:51], axis=1)

    lof = LocalOutlierFactor(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=-1)
    lof.fit_predict(vectors)
    lof_score = -lof.negative_outlier_factor_

    pca = PCA(n_components=100, svd_solver="randomized", random_state=20260811)
    projected = pca.fit_transform(vectors)
    reconstructed = pca.inverse_transform(projected)
    residual = np.linalg.norm(vectors - reconstructed, axis=1)

    knn20_pct = percentile(knn20_median)
    nearest_pct = percentile(nearest_dist)
    lof_pct = percentile(lof_score)
    residual_pct = percentile(residual)
    stable_pct_min = np.minimum.reduce(
        [percentile(knn10_median), knn20_pct, percentile(knn50_median)]
    )
    stable_consensus = (knn20_pct >= 99.0) & (
        (nearest_pct >= 98.0) | (lof_pct >= 98.0)
    )

    vector_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in vectors]
    exact_duplicate_vectors = len(vector_hashes) - len(set(vector_hashes))
    points = []
    found_anchors: set[str] = set()
    for i, doc in enumerate(corpus):
        anchor_label = None
        for locator in all_locators(doc):
            if locator in ANCHORS:
                anchor_label = ANCHORS[locator]
                found_anchors.add(locator)
                break
        point = {
            "id": doc.document_id,
            "kind": doc.kind,
            "kindLabel": KIND_LABELS[doc.kind],
            "summary": anchor_label or safe_summary(doc),
            "x12": round(float(projected[i, 0]), 6),
            "y12": round(float(projected[i, 1]), 6),
            "x13": round(float(projected[i, 0]), 6),
            "y13": round(float(projected[i, 2]), 6),
            "frequency": int(doc.occurrences),
            "nearestDistance": round(float(nearest_dist[i]), 6),
            "isolation": round(float(knn20_median[i]), 6),
            "isolationPercentile": round(float(knn20_pct[i]), 2),
            "lofPercentile": round(float(lof_pct[i]), 2),
            "residualPercentile": round(float(residual_pct[i]), 2),
            "stabilityFloor": round(float(stable_pct_min[i]), 2),
            "consensus": bool(stable_consensus[i]),
            "anchor": bool(anchor_label),
        }
        if anchor_label:
            point["locator"] = next(
                locator for locator in all_locators(doc) if locator in ANCHORS
            )
        points.append(point)

    output = {
        "meta": {
            "title": "LiveFire RAG corpus — PCA overview",
            "corpusDigest": digest,
            "documents": len(corpus),
            "dimensions": vectors.shape[1],
            "sourceObservations": sum(corpus_stats["input_rows"].values()),
            "exactDuplicateVectors": exact_duplicate_vectors,
            "reducer": "centered PCA, randomized SVD, seed 20260811",
            "explainedVariance": {
                "pc1": float(pca.explained_variance_ratio_[0]),
                "pc2": float(pca.explained_variance_ratio_[1]),
                "pc3": float(pca.explained_variance_ratio_[2]),
                "cumulative2": float(np.sum(pca.explained_variance_ratio_[:2])),
                "cumulative10": float(np.sum(pca.explained_variance_ratio_[:10])),
                "cumulative50": float(np.sum(pca.explained_variance_ratio_[:50])),
                "cumulative100": float(np.sum(pca.explained_variance_ratio_[:100])),
            },
            "isolationMetric": "median cosine distance to 20 nearest distinct semantic documents in original 4096-D space",
            "isolationDisplayThreshold": "top 1%; display budget, not an alert threshold",
            "anchorLocatorsFound": len(found_anchors),
            "normMin": float(norms.min()),
            "normMax": float(norms.max()),
        },
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    if args.html_out:
        template = args.template.read_text()
        if template.count("__CORPUS_DATA__") != 1:
            raise SystemExit("visualization template must have exactly one data placeholder")
        kind_keys = list(KIND_LABELS)
        kind_indexes = {kind: index for index, kind in enumerate(kind_keys)}
        compact_points = []
        for point in points:
            compact_points.append(
                [
                    point["id"],
                    kind_indexes[point["kind"]],
                    point["x12"],
                    point["y12"],
                    point["y13"],
                    point["frequency"],
                    point["isolationPercentile"],
                    point["lofPercentile"],
                    point["summary"]
                    if point["anchor"] or point["isolationPercentile"] >= 99.0
                    else "",
                    1 if point["anchor"] else 0,
                ]
            )
        compact_output = {
            "m": output["meta"],
            "kk": kind_keys,
            "kl": [KIND_LABELS[kind] for kind in kind_keys],
            "p": compact_points,
        }
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(
            template.replace(
                "__CORPUS_DATA__", json.dumps(compact_output, separators=(",", ":"))
            )
        )
    if args.png_out:
        render_png(output, args.png_out)
    print(json.dumps(output["meta"], indent=2))


if __name__ == "__main__":
    main()
