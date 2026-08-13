"""Bounded, answer-neutral evaluation of a sealed pilot evidence index."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from .evidence_index import EvidenceIndex
from .evidence_service import EvidenceService


MODES = (
    ("lexical", {"methods": ["lexical"], "fusion": "none"}),
    ("dense", {"methods": ["dense"], "fusion": "none"}),
    ("fused", {"methods": ["dense", "lexical"], "fusion": "reciprocal_rank"}),
)
COMPARISONS = (
    ("dense", "lexical"),
    ("fused", "lexical"),
    ("fused", "dense"),
)
SCOPE_STATUS = "sample_only_not_corpus_coverage"
ADMISSION_STATUS = "local_evaluation_only_not_sdk_admitted"
FROZEN_QUERY_FIXTURE_SHA256 = (
    "3da177d46ffd87c5b284db1983828ce64d8d6c76999cf9729cef6d054706f456"
)


class EvidencePilotEvaluationError(RuntimeError):
    """The frozen pilot evaluation contract or result is invalid."""


def _load_fixture(path: Path) -> dict[str, Any]:
    if sha256_file(path) != FROZEN_QUERY_FIXTURE_SHA256:
        raise EvidencePilotEvaluationError(
            "pilot query fixture differs from the pre-ranking frozen digest"
        )
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidencePilotEvaluationError("pilot query fixture is unreadable") from error
    required = {
        "schema_version", "status", "scope", "authoring_constraints", "queries"
    }
    if not isinstance(fixture, dict) or set(fixture) != required:
        raise EvidencePilotEvaluationError("pilot query fixture has unknown or missing fields")
    if (
        fixture["schema_version"] != "livefire.rag.generic-evidence-pilot-queries/1"
        or fixture["status"] != "predeclared_before_pilot_ranking"
        or fixture["scope"] != SCOPE_STATUS
    ):
        raise EvidencePilotEvaluationError("pilot query fixture is not frozen and sample-scoped")
    expected_constraints = {
        "answer_neutral": True,
        "exact_event_identifiers_forbidden": True,
        "exact_hosts_principals_addresses_paths_and_resource_names_forbidden": True,
        "expected_relation_families_are_diagnostic_not_relevance_labels": True,
    }
    if fixture["authoring_constraints"] != expected_constraints:
        raise EvidencePilotEvaluationError("pilot query authoring constraints are not exact")
    queries = fixture["queries"]
    if not isinstance(queries, list) or not queries:
        raise EvidencePilotEvaluationError("pilot query fixture is empty")
    seen: set[str] = set()
    for row in queries:
        if not isinstance(row, dict) or set(row) != {
            "query_id", "query", "expected_relation_families"
        }:
            raise EvidencePilotEvaluationError("pilot query row has unknown or missing fields")
        query_id = row["query_id"]
        families = row["expected_relation_families"]
        if (
            not isinstance(query_id, str) or not query_id or query_id in seen
            or not isinstance(row["query"], str) or not row["query"]
            or not isinstance(families, list)
            or len(families) != len(set(families))
            or any(not isinstance(value, str) or not value for value in families)
        ):
            raise EvidencePilotEvaluationError("pilot query row is invalid or non-canonical")
        seen.add(query_id)
    return fixture


def _candidate_ids(output: Mapping[str, Any]) -> list[str]:
    return [row["document_id"] for row in output.get("candidates", [])]


def _comparison(
    query_id: str,
    left_mode: str,
    right_mode: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    top_n: int,
) -> dict[str, Any]:
    left_ids = _candidate_ids(left)
    right_ids = _candidate_ids(right)
    left_rank = {document_id: rank for rank, document_id in enumerate(left_ids, 1)}
    right_rank = {document_id: rank for rank, document_id in enumerate(right_ids, 1)}
    shared = sorted(set(left_ids) & set(right_ids))
    cutoffs = sorted({min(top_n, cutoff) for cutoff in (1, 5, 10, 20)})
    return {
        "schema_version": "livefire.rag.evidence-pilot-ranking-comparison/1",
        "query_id": query_id,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "requested_top_n": top_n,
        "left_returned_count": len(left_ids),
        "right_returned_count": len(right_ids),
        "shared_document_count": len(shared),
        "left_only_document_ids": sorted(set(left_ids) - set(right_ids)),
        "right_only_document_ids": sorted(set(right_ids) - set(left_ids)),
        "rank_deltas": [
            {
                "document_id": document_id,
                "left_rank": left_rank[document_id],
                "right_rank": right_rank[document_id],
                "right_minus_left": right_rank[document_id] - left_rank[document_id],
            }
            for document_id in shared
        ],
        "overlap_at_k": [
            {
                "k": cutoff,
                "intersection_count": len(
                    set(left_ids[:cutoff]) & set(right_ids[:cutoff])
                ),
                "union_count": len(set(left_ids[:cutoff]) | set(right_ids[:cutoff])),
            }
            for cutoff in cutoffs
        ],
    }


def _validate_pointer_closure(
    index: EvidenceIndex,
    output: Mapping[str, Any],
    cache: dict[str, dict[str, Any]],
) -> int:
    checked = 0
    for candidate in output.get("candidates", []):
        if [row["rank"] for row in output["candidates"]] != list(
            range(1, len(output["candidates"]) + 1)
        ):
            raise EvidencePilotEvaluationError("candidate ranks are not contiguous")
        for returned in candidate["source_occurrences"]:
            occurrence_id = returned["occurrence_id"]
            occurrence = cache.get(occurrence_id)
            if occurrence is None:
                row = index.connection.execute(
                    "SELECT to_json(o) FROM evidence_occurrences o WHERE occurrence_id=?",
                    [occurrence_id],
                ).fetchone()
                if row is None:
                    raise EvidencePilotEvaluationError(
                        f"returned occurrence is absent from index: {occurrence_id}"
                    )
                occurrence = json.loads(row[0])
                cache[occurrence_id] = occurrence
            if (
                candidate["document_id"] not in occurrence["document_ids"]
                or returned["source_pointer"] != occurrence["source_pointer"]
                or returned["relation_identity"] != occurrence["relation_identity"]
                or returned.get("event_time") != occurrence.get("event_time")
            ):
                raise EvidencePilotEvaluationError(
                    f"returned pointer is not closed over the indexed occurrence: {occurrence_id}"
                )
            checked += 1
    return checked


def run_evidence_pilot_evaluation(
    index_root: Path,
    query_fixture: Path,
    output_dir: Path,
    *,
    sdk_specs: Path,
    embed_query: Callable[[str, int], np.ndarray],
    component_id: str,
    version: str,
    top_n: int = 20,
    deadline_seconds: int = 300,
) -> dict[str, Any]:
    """Execute every frozen query in every fixed retrieval mode exactly once."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
        raise ValueError("pilot evaluation top_n must be between 1 and 100")
    if deadline_seconds < 1:
        raise ValueError("deadline_seconds must be positive")
    index_root = Path(index_root).resolve()
    query_fixture = Path(query_fixture).resolve()
    output_dir = Path(output_dir).resolve()
    sdk_specs = Path(sdk_specs).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pilot evaluation: {output_dir}")
    fixture = _load_fixture(query_fixture)

    # The complete plan and its digest exist before the first result is seen.
    plan_rows = [
        {
            "query_id": row["query_id"],
            "mode": mode,
            "request": {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": row["query"],
                "top_n": top_n,
                "retrieval": retrieval,
            },
            "expected_relation_families": row["expected_relation_families"],
        }
        for row in fixture["queries"]
        for mode, retrieval in MODES
    ]
    plan = {
        "schema_version": "livefire.rag.evidence-pilot-evaluation-plan/1",
        "status": "frozen_before_execution",
        "scope": SCOPE_STATUS,
        "query_fixture_sha256": sha256_file(query_fixture),
        "top_n": top_n,
        "modes": [mode for mode, _ in MODES],
        "runs": plan_rows,
    }
    plan_sha256 = sha256_bytes(canonical_json_bytes(plan))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copyfile(query_fixture, staging / "query-fixture.json")
        write_canonical_json(staging / "execution-plan.json", plan)
        rankings: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        pointer_cache: dict[str, dict[str, Any]] = {}
        pointer_count = 0
        with EvidenceIndex.open(index_root, sdk_specs=sdk_specs) as index:
            pilot = index.manifest.get("pilot_sample")
            if (
                not isinstance(pilot, dict)
                or pilot.get("scope_status") != SCOPE_STATUS
                or pilot.get("admission_status") != ADMISSION_STATUS
                or pilot.get("corpus_miss_definitive") is not False
            ):
                raise EvidencePilotEvaluationError(
                    "evaluation requires an explicitly non-admitted sealed pilot index"
                )
            service = EvidenceService(index, embed_query=embed_query, sdk_specs=sdk_specs)
            outputs: dict[tuple[str, str], dict[str, Any]] = {}
            for planned in plan_rows:
                output = service.search(
                    planned["request"],
                    int(time.time() * 1000) + deadline_seconds * 1000,
                )
                if (
                    output["coverage"]["status"] != "partial"
                    or "pilot_sample_not_corpus_coverage"
                    not in output["coverage"]["reason_codes"]
                    or (
                        output["kind"] == "miss"
                        and "not a corpus-wide miss" not in output["miss"]["message"]
                    )
                ):
                    raise EvidencePilotEvaluationError(
                        "retrieval output lost the pilot sample scope"
                    )
                pointer_count += _validate_pointer_closure(index, output, pointer_cache)
                observed = sorted({
                    occurrence["relation_identity"]["relation"]
                    for candidate in output.get("candidates", [])
                    for occurrence in candidate["source_occurrences"]
                })
                expected = planned["expected_relation_families"]
                first_expected_rank = next((
                    candidate["rank"]
                    for candidate in output.get("candidates", [])
                    if any(
                        occurrence["relation_identity"]["relation"] in expected
                        for occurrence in candidate["source_occurrences"]
                    )
                ), None)
                result = {
                    "schema_version": "livefire.rag.evidence-pilot-ranking/1",
                    "query_id": planned["query_id"], "mode": planned["mode"],
                    "request": planned["request"],
                    "expected_relation_families": expected,
                    "relation_family_diagnostic_only_not_relevance": {
                        "observed": observed,
                        "observed_expected_intersection": sorted(set(observed) & set(expected)),
                        "first_expected_family_rank": first_expected_rank,
                    },
                    "output": output,
                }
                rankings.append(result)
                outputs[(planned["query_id"], planned["mode"])] = output
            for query in fixture["queries"]:
                for left_mode, right_mode in COMPARISONS:
                    comparisons.append(_comparison(
                        query["query_id"], left_mode, right_mode,
                        outputs[(query["query_id"], left_mode)],
                        outputs[(query["query_id"], right_mode)], top_n,
                    ))
            index_component = index.component
            pilot_binding = pilot

        with (staging / "rankings.jsonl").open("wb") as handle:
            for row in rankings:
                handle.write(canonical_json_bytes(row, newline=True))
        with (staging / "comparisons.jsonl").open("wb") as handle:
            for row in comparisons:
                handle.write(canonical_json_bytes(row, newline=True))
        report = {
            "schema_version": "livefire.rag.evidence-pilot-evaluation-report/1",
            "admission_status": ADMISSION_STATUS,
            "scope_status": SCOPE_STATUS,
            "quality_claim_status": "diagnostic_only_no_qrels_no_retrieval_quality_claim",
            "index": index_component,
            "pilot_sample": pilot_binding,
            "query_fixture_sha256": sha256_file(query_fixture),
            "execution_plan_sha256": plan_sha256,
            "query_count": len(fixture["queries"]),
            "mode_count": len(MODES),
            "ranking_run_count": len(rankings),
            "comparison_count": len(comparisons),
            "returned_pointer_count": pointer_count,
            "returned_pointer_closure": True,
            "all_outputs_partial_sample_scope": True,
        }
        write_canonical_json(staging / "report.json", report)
        artifacts = [
            artifact_ref(staging / name, name, media_type)
            for name, media_type in (
                ("query-fixture.json", "application/json"),
                ("execution-plan.json", "application/json"),
                ("rankings.jsonl", "application/x-ndjson"),
                ("comparisons.jsonl", "application/x-ndjson"),
                ("report.json", "application/json"),
            )
        ]
        artifacts.sort(key=lambda row: row["path"])
        write_canonical_json(staging / "objects.lock.json", {
            "schema_version": "livefire.object-lock/1", "objects": artifacts,
        })
        objects = {row["path"]: row for row in artifacts}
        objects["objects.lock.json"] = artifact_ref(
            staging / "objects.lock.json", "objects.lock.json",
            "application/vnd.livefire.object-lock+json",
        )
        component: dict[str, Any] = {"id": component_id, "version": version, "sha256": ""}
        manifest = {
            "schema_version": "livefire.rag.evidence-pilot-evaluation/1",
            "component": component,
            "admission_status": ADMISSION_STATUS,
            "scope_status": SCOPE_STATUS,
            "index": index_component,
            "pilot_sample": pilot_binding,
            "objects": objects,
        }
        manifest["component"]["sha256"] = canonical_sha256_omitting(
            manifest, ("component", "sha256")
        )
        write_canonical_json(staging / "manifest.json", manifest)
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "EvidencePilotEvaluationError", "FROZEN_QUERY_FIXTURE_SHA256",
    "run_evidence_pilot_evaluation",
]
