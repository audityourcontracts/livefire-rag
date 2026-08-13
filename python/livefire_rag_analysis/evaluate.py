"""Evaluate canonical retrieval runs, with relevance metrics when qrels exist."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class RetrievalEvaluationError(RuntimeError):
    """A retrieval run or qrel set is malformed."""


def _rows(value: Path | Iterable[Mapping[str, Any]], what: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        output: list[dict[str, Any]] = []
        try:
            with Path(value).open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise RetrievalEvaluationError(f"{what}:{number} is not an object")
                    output.append(parsed)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetrievalEvaluationError(f"{what} is unreadable") from error
        return output
    return [dict(row) for row in value]


def _run_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        query_id = row.get("query_id")
        document_id = _document_id(row)
        rank = row.get("rank")
        if (
            not isinstance(query_id, str) or not query_id
            or not isinstance(document_id, str) or not document_id
            or not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
            or (query_id, document_id) in seen
        ):
            raise RetrievalEvaluationError("retrieval run contains an invalid or duplicate row")
        score = row.get("score")
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise RetrievalEvaluationError("retrieval run contains a non-finite score")
        seen.add((query_id, document_id))
        normalized = dict(row)
        normalized["document_id"] = document_id
        grouped[query_id].append(normalized)
    if not grouped:
        raise RetrievalEvaluationError("retrieval run is empty")
    for query_id, group in grouped.items():
        group.sort(key=lambda row: (row["rank"], row["document_id"]))
        if [row["rank"] for row in group] != list(range(1, len(group) + 1)):
            raise RetrievalEvaluationError(f"retrieval ranks are not contiguous for {query_id}")
    return dict(sorted(grouped.items()))


def _qrel_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        query_id = row.get("query_id")
        document_id = _document_id(row)
        relevance = row.get("relevance", row.get("relevance_grade"))
        if (
            not isinstance(query_id, str) or not query_id
            or not isinstance(document_id, str) or not document_id
            or not isinstance(relevance, int) or isinstance(relevance, bool) or relevance < 0
            or document_id in grouped[query_id]
        ):
            raise RetrievalEvaluationError("qrels contain an invalid or duplicate row")
        grouped[query_id][document_id] = relevance
    return dict(grouped)


def _document_id(row: Mapping[str, Any]) -> Any:
    """Accept the fast-index name and the existing evaluator command alias."""

    document_id = row.get("document_id")
    command_id = row.get("command_id")
    if document_id is not None and command_id is not None and document_id != command_id:
        raise RetrievalEvaluationError("document_id and command_id aliases disagree")
    return document_id if document_id is not None else command_id


def _dcg(relevance: Sequence[int]) -> float:
    return sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))


def evaluate_retrieval_run(
    run: Path | Iterable[Mapping[str, Any]],
    *,
    qrels: Path | Iterable[Mapping[str, Any]] | None = None,
    cutoffs: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    """Return run diagnostics and qrel metrics without coupling to an index."""

    if (
        not cutoffs or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in cutoffs)
        or len(set(cutoffs)) != len(cutoffs)
    ):
        raise ValueError("cutoffs must be unique positive integers")
    cutoffs = tuple(sorted(cutoffs))
    groups = _run_groups(_rows(run, "run"))
    report: dict[str, Any] = {
        "schema_version": "livefire.rag.retrieval-run-evaluation/1",
        "queries": len(groups),
        "rows": sum(map(len, groups.values())),
        "cutoffs": list(cutoffs),
        "qrels_supplied": qrels is not None,
        "per_query": [],
    }
    if qrels is None:
        report["metrics_status"] = "unavailable_without_qrels"
        report["run_depth"] = {
            "minimum": min(map(len, groups.values())),
            "maximum": max(map(len, groups.values())),
            "mean": sum(map(len, groups.values())) / len(groups),
        }
        return report

    judgments = _qrel_groups(_rows(qrels, "qrels"))
    if set(groups) != set(judgments):
        raise RetrievalEvaluationError("run and qrel query sets differ")
    for query_id, ranking in groups.items():
        qrel = judgments[query_id]
        relevant = {document_id for document_id, value in qrel.items() if value > 0}
        if not relevant:
            raise RetrievalEvaluationError(f"qrels contain no relevant documents for {query_id}")
        ranked_ids = [row["document_id"] for row in ranking]
        ranked_relevance = [qrel.get(document_id, 0) for document_id in ranked_ids]
        ideal = sorted(qrel.values(), reverse=True)
        first = next((rank for rank, value in enumerate(ranked_relevance, 1) if value > 0), None)
        metrics: dict[str, Any] = {
            "query_id": query_id,
            "relevant_documents": len(relevant),
            "reciprocal_rank": 0.0 if first is None else 1.0 / first,
        }
        for cutoff in cutoffs:
            retrieved = ranked_ids[:cutoff]
            metrics[f"recall@{cutoff}"] = len(set(retrieved) & relevant) / len(relevant)
            dcg = _dcg(ranked_relevance[:cutoff])
            ideal_dcg = _dcg(ideal[:cutoff])
            metrics[f"ndcg@{cutoff}"] = dcg / ideal_dcg if ideal_dcg else 0.0
        report["per_query"].append(metrics)
    report["metrics_status"] = "available"
    report["macro"] = {
        name: sum(float(row[name]) for row in report["per_query"]) / len(report["per_query"])
        for name in ["reciprocal_rank"]
        + [metric for cutoff in cutoffs for metric in (f"recall@{cutoff}", f"ndcg@{cutoff}")]
    }
    return report


__all__ = ["RetrievalEvaluationError", "evaluate_retrieval_run"]
