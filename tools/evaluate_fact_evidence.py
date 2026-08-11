#!/usr/bin/env python3
"""Evaluate answer-neutral fact-to-evidence retrieval rankings.

The evaluator is deliberately independent of Livefire and model runtimes. It
consumes frozen query, qrel, hard-negative, and ranking artifacts and emits a
deterministic JSON report. Benchmark labels never participate in index build or
query execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RELEVANT_GRADE = 2
DEFAULT_K = 20
ELIGIBLE = {
    "direct_single",
    "direct_multi",
    "retrieve_then_compute",
    "exact_metadata",
    "eligible",
    "eligible_native",
    "eligible_external",
}
TERMINAL_ELIGIBILITY = {
    "eligible_native",
    "eligible_external",
    "external_source_unbound",
    "outside_index_domain",
    "not_retrieval_testable",
}
ELIGIBLE_INVENTORY = {"eligible_native", "eligible_external"}
REQUIRED_REAL_QUERY_SURFACES = {
    "analyst_question",
    "terse_soc",
    "entity_light_paraphrase",
}


class EvaluationError(ValueError):
    """Raised for malformed or internally inconsistent benchmark artifacts."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise EvaluationError(f"{path}: expected an array of objects")
        return value
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise EvaluationError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique(rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if None in key:
            raise EvaluationError(f"{label}: missing key field in {fields}: {row}")
        if key in seen:
            raise EvaluationError(f"{label}: duplicate key {key}")
        seen.add(key)


def require_unique_documents(
    rows: Iterable[dict[str, Any]],
    label: str,
    document_fields: tuple[str, ...],
) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        query_id = row.get("query_id")
        if query_id is None:
            raise EvaluationError(f"{label}: missing query_id: {row}")
        document_id = row_document_id(row, *document_fields)
        key = (str(query_id), document_id)
        if key in seen:
            raise EvaluationError(f"{label}: duplicate key {key}")
        seen.add(key)


def ranking_score_contract(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("score_kind") is not None:
        return str(row["score_kind"]), str(row.get("score_direction", ""))
    if ranking_distance(row) is not None:
        return "cosine_distance", "lower_is_better"
    if row.get("score") is not None:
        return "native_score", str(row.get("score_direction", ""))
    return "rank_only", "lower_rank_is_better"


def ranking_score(row: dict[str, Any]) -> float | None:
    distance = ranking_distance(row)
    if distance is not None:
        return distance
    if row.get("score") is not None:
        return float(row["score"])
    return None


def ranking_preference(row: dict[str, Any]) -> float | None:
    value = ranking_score(row)
    if value is None:
        return None
    _, direction = ranking_score_contract(row)
    return -value if direction == "lower_is_better" else value


def validate_rankings(
    rows: list[dict[str, Any]],
    label: str,
    known_query_ids: set[str] | None = None,
    strict_schema: bool = False,
) -> dict[str, str]:
    require_unique_documents(rows, label, ("document_id", "command_id"))
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contracts: set[tuple[str, str]] = set()
    system_ids: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if known_query_ids is not None and query_id not in known_query_ids:
            raise EvaluationError(f"{label}: ranking references unknown query {query_id}")
        if row.get("document_id") is not None and row.get("command_id") is not None:
            if str(row["document_id"]) != str(row["command_id"]):
                raise EvaluationError(f"{label}: conflicting document_id and command_id: {row}")
        if row.get("distance") is not None and row.get("distance_millionths") is not None:
            encoded = float(row["distance_millionths"]) / 1_000_000.0
            if not math.isclose(float(row["distance"]), encoded, abs_tol=0.5e-6):
                raise EvaluationError(f"{label}: conflicting distance aliases: {row}")
        if not isinstance(row.get("rank"), int) or row["rank"] < 1:
            raise EvaluationError(f"{label}: rank must be a positive integer: {row}")
        distance = ranking_distance(row)
        if distance is not None and (not isinstance(distance, (int, float)) or not math.isfinite(distance)):
            raise EvaluationError(f"{label}: distance must be finite: {row}")
        if distance is not None and not 0.0 <= distance <= 2.0:
            raise EvaluationError(f"{label}: cosine distance must be in [0,2]: {row}")
        value = ranking_score(row)
        if value is not None and not math.isfinite(value):
            raise EvaluationError(f"{label}: score must be finite: {row}")
        contract = ranking_score_contract(row)
        allowed_score_kinds = {
            "cosine_distance", "bm25", "exact_field_score", "reranker_score",
            "native_similarity", "native_score", "rank_only",
        }
        if contract[0] not in allowed_score_kinds:
            raise EvaluationError(f"{label}: unsupported score kind: {row}")
        if contract[1] not in {"lower_is_better", "higher_is_better", "lower_rank_is_better"}:
            raise EvaluationError(f"{label}: invalid score direction: {row}")
        if contract[0] == "cosine_distance" and contract[1] != "lower_is_better":
            raise EvaluationError(f"{label}: cosine distance must use lower_is_better")
        if strict_schema:
            normative_score_kinds = {
                "cosine_distance", "bm25", "exact_field_score", "reranker_score",
                "native_similarity",
            }
            if contract[0] not in normative_score_kinds:
                raise EvaluationError(f"{label}: non-normative score kind: {contract[0]}")
            required = {
                "schema_version", "system_id", "query_id", "command_id", "rank",
                "score_kind", "score_direction", "pointer_resolved", "filter_compliant",
            }
            missing = sorted(required - set(row))
            if missing:
                raise EvaluationError(f"{label}: missing normative fields {missing}: {row}")
            if row["schema_version"] != "livefire.rag.evidence-ranking-row/1":
                raise EvaluationError(f"{label}: unsupported schema_version: {row['schema_version']}")
            for field in ("system_id", "query_id", "command_id", "score_kind", "score_direction"):
                if not isinstance(row[field], str) or not row[field]:
                    raise EvaluationError(f"{label}: {field} must be a non-empty string: {row}")
            for field in ("pointer_resolved", "filter_compliant"):
                if type(row[field]) is not bool:
                    raise EvaluationError(f"{label}: {field} must be a boolean: {row}")
            if row.get("distance_millionths") is not None:
                if type(row["distance_millionths"]) is not int:
                    raise EvaluationError(f"{label}: distance_millionths must be an integer: {row}")
            if row.get("score") is not None:
                if isinstance(row["score"], bool) or not isinstance(row["score"], (int, float)):
                    raise EvaluationError(f"{label}: score must be numeric: {row}")
            if contract[0] == "cosine_distance" and distance is None:
                raise EvaluationError(f"{label}: cosine_distance requires distance_millionths")
            if contract[0] != "cosine_distance" and row.get("score") is None:
                raise EvaluationError(f"{label}: native score contract requires score")
            if distance is not None and row.get("score") is not None:
                raise EvaluationError(f"{label}: a ranking row cannot mix distance and native score")
        if row.get("system_id") is not None:
            system_ids.add(str(row["system_id"]))
        contracts.add(contract)
        by_query[query_id].append(row)
    if len(system_ids) > 1:
        raise EvaluationError(f"{label}: one ranking file must contain exactly one system_id")
    if len(contracts) > 1:
        raise EvaluationError(f"{label}: one ranking file must contain exactly one score contract")
    for query_id, query_rows in by_query.items():
        ordered = sorted(query_rows, key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in ordered]
        if ranks != list(range(1, len(ranks) + 1)):
            raise EvaluationError(f"{label}: ranks for {query_id} must be contiguous from 1")
        for previous, current in zip(ordered, ordered[1:]):
            previous_score = ranking_score(previous)
            current_score = ranking_score(current)
            if previous_score is None or current_score is None:
                continue
            _, direction = ranking_score_contract(previous)
            wrongly_ordered = (
                previous_score > current_score
                if direction == "lower_is_better"
                else previous_score < current_score
            )
            if wrongly_ordered:
                raise EvaluationError(f"{label}: scores are inconsistent with ranks for {query_id}")
            if previous_score == current_score:
                if ranking_document_id(previous) > ranking_document_id(current):
                    raise EvaluationError(
                        f"{label}: equal scores must use command-id ascending tie breaking for {query_id}"
                    )
    kind, direction = next(iter(contracts), ("rank_only", "lower_rank_is_better"))
    return {"score_kind": kind, "score_direction": direction}


def dcg(grades: list[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def row_document_id(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value)
    raise EvaluationError(f"missing document identifier ({', '.join(names)}): {row}")


def ranking_document_id(row: dict[str, Any]) -> str:
    return row_document_id(row, "document_id", "command_id")


def ranking_distance(row: dict[str, Any]) -> float | None:
    if row.get("distance") is not None:
        return float(row["distance"])
    if row.get("distance_millionths") is not None:
        return float(row["distance_millionths"]) / 1_000_000.0
    return None


def query_is_eligible(query: dict[str, Any]) -> bool:
    return query.get("eligibility") in ELIGIBLE and query.get("status", "active") == "active"


def load_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("atoms"), list):
        raise EvaluationError(f"{path}: inventory must be an object with an atoms array")
    if value.get("schema_version") != "livefire.rag.evidence-eligibility-ledger/1":
        raise EvaluationError(f"{path}: unsupported eligibility-ledger schema_version")
    require_unique(value["atoms"], ("atom_id",), "inventory atoms")
    declared_total = value.get("summary", {}).get("total_atoms")
    if declared_total is not None and int(declared_total) != len(value["atoms"]):
        raise EvaluationError(
            f"inventory total mismatch: summary declares {declared_total}, found {len(value['atoms'])}"
        )
    declared_by_benchmark = value.get("summary", {}).get("by_benchmark", {})
    if declared_by_benchmark:
        actual: dict[str, int] = defaultdict(int)
        for atom in value["atoms"]:
            actual[str(atom.get("benchmark_id", atom.get("domain", "unknown")))] += 1
        if dict(actual) != declared_by_benchmark:
            raise EvaluationError(
                f"inventory benchmark reconciliation mismatch: declared={declared_by_benchmark}, actual={dict(actual)}"
            )
    for atom in value["atoms"]:
        disposition = atom.get("eligibility")
        if disposition not in TERMINAL_ELIGIBILITY:
            raise EvaluationError(
                f"inventory atom {atom.get('atom_id')} lacks a sealed terminal eligibility disposition"
            )
        for field in ("cohort", "resampling_cluster_id", "incident_id"):
            if not isinstance(atom.get(field), str) or not atom[field]:
                raise EvaluationError(f"inventory atom {atom.get('atom_id')} must bind {field}")
    contract = value.get("suite_contract", "generic")
    if contract == "livefire-23-cloud-53-bots-v1":
        actual_cloud = sum(atom.get("cohort") == "cloud" for atom in value["atoms"])
        actual_bots = sum(atom.get("cohort") in {"bots_native", "external"} for atom in value["atoms"])
        actual_external = sum(atom.get("cohort") == "external" for atom in value["atoms"])
        if (len(value["atoms"]), actual_cloud, actual_bots, actual_external) != (76, 23, 53, 10):
            raise EvaluationError(
                "livefire-23-cloud-53-bots-v1 requires exactly 76 atoms: 23 cloud, "
                "53 BOTS including 10 external-enrichment atoms"
            )
    return value


def inventory_atom_is_eligible(atom: dict[str, Any]) -> bool:
    return atom.get("eligibility") in ELIGIBLE_INVENTORY


def validate_inventory_queries(
    inventory: dict[str, Any],
    queries: list[dict[str, Any]],
    minimum_surfaces: int,
) -> None:
    atoms = {str(row["atom_id"]): row for row in inventory["atoms"]}
    queries_by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        if query.get("fact_id") is None:
            raise EvaluationError(f"query must bind fact_id: {query}")
        fact_id = str(query["fact_id"])
        if fact_id not in atoms:
            raise EvaluationError(f"query references atom absent from inventory: {fact_id}")
        expected_eligible = inventory_atom_is_eligible(atoms[fact_id])
        if query_is_eligible(query) != expected_eligible:
            raise EvaluationError(
                f"query/inventory eligibility mismatch for {fact_id}: "
                f"query={query.get('eligibility')}, inventory={atoms[fact_id].get('eligibility')}"
            )
        for field in ("cohort", "resampling_cluster_id", "incident_id"):
            if query.get(field) != atoms[fact_id].get(field):
                raise EvaluationError(f"query/inventory {field} mismatch for {fact_id}")
        queries_by_fact[fact_id].append(query)
    eligible_atoms = {fact_id for fact_id, atom in atoms.items() if inventory_atom_is_eligible(atom)}
    missing = sorted(eligible_atoms - set(queries_by_fact))
    if missing:
        raise EvaluationError(f"eligible inventory atoms have no queries: {missing}")
    for fact_id in sorted(eligible_atoms):
        surfaces = queries_by_fact[fact_id]
        if len(surfaces) < minimum_surfaces:
            raise EvaluationError(
                f"eligible fact {fact_id} has {len(surfaces)} query surfaces; requires {minimum_surfaces}"
            )
        surface_names = [row.get("surface", row["query_id"]) for row in surfaces]
        if len(surface_names) != len(set(surface_names)):
            raise EvaluationError(f"eligible fact {fact_id} has duplicate query surfaces")
        if inventory.get("suite_contract") == "livefire-23-cloud-53-bots-v1":
            actual_surfaces = set(surface_names)
            if actual_surfaces != REQUIRED_REAL_QUERY_SURFACES:
                raise EvaluationError(
                    f"eligible fact {fact_id} must have exactly the real-suite surfaces "
                    f"{sorted(REQUIRED_REAL_QUERY_SURFACES)}; found {sorted(actual_surfaces)}"
                )
        clusters = {
            str(row.get("resampling_cluster_id", row.get("incident_id", fact_id)))
            for row in surfaces
        }
        if len(clusters) != 1:
            raise EvaluationError(f"eligible fact {fact_id} spans multiple resampling clusters")
        cohorts = {str(row.get("cohort", "unclassified")) for row in surfaces}
        if len(cohorts) != 1 or "unclassified" in cohorts:
            raise EvaluationError(f"eligible fact {fact_id} must bind exactly one cohort")
    incident_clusters: dict[str, set[str]] = defaultdict(set)
    for atom in atoms.values():
        incident_clusters[str(atom["incident_id"])].add(str(atom["resampling_cluster_id"]))
    split_incidents = sorted(incident for incident, clusters in incident_clusters.items() if len(clusters) > 1)
    if split_incidents:
        raise EvaluationError(f"inventory incidents span multiple resampling clusters: {split_incidents}")


def validate_benchmark_rows(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
) -> None:
    def require_fields(row: dict[str, Any], fields: set[str], label: str) -> None:
        missing = sorted(fields - set(row))
        if missing:
            raise EvaluationError(f"{label} missing required fields {missing}: {row}")

    query_required = {
        "schema_version", "query_id", "fact_id", "domain", "cohort",
        "resampling_cluster_id", "incident_id", "scope_ids", "split", "surface",
        "eligibility", "status", "query_text", "query_text_sha256", "source_fact_sha256",
    }
    qrel_required = {
        "schema_version", "qrel_id", "query_id", "fact_id", "command_id",
        "projection_sha256", "source_pointer", "relation", "relevance_grade",
        "eligibility", "status", "judgment_method", "judgment_sha256",
    }
    negative_required = {
        "schema_version", "negative_id", "query_id", "fact_id", "positive_command_id",
        "positive_projection_sha256", "negative_command_id", "negative_projection_sha256",
        "negative_source_pointer", "negative_class", "control_tier", "eligibility",
        "status", "selection_mode", "matching_policy", "stratum_sha256",
        "matching_features_sha256", "matched_fields", "selection_seed", "selection_rank",
        "sampling_weight_millionths", "judgment_sha256",
    }
    query_by_id = {str(row["query_id"]): row for row in queries}
    for query in queries:
        require_fields(query, query_required, "query")
        if query.get("schema_version") != "livefire.rag.evidence-query-row/1":
            raise EvaluationError(f"query has unsupported schema_version: {query}")
        for field in ("query_id", "fact_id", "query_text", "query_text_sha256"):
            if not isinstance(query.get(field), str) or not query[field]:
                raise EvaluationError(f"query {field} must be a non-empty string: {query}")
        actual_query_sha = hashlib.sha256(query["query_text"].encode("utf-8")).hexdigest()
        if query["query_text_sha256"] != actual_query_sha:
            raise EvaluationError(f"query_text_sha256 mismatch for {query['query_id']}")
        if query_is_eligible(query):
            expected = query.get("expected_top_k_cardinality")
            if type(expected) is not int:
                raise EvaluationError(
                    f"active query {query['query_id']} must declare integer expected_top_k_cardinality"
                )
            receipt_sha = query.get("candidate_universe_receipt_sha256")
            if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
                raise EvaluationError(
                    f"active query {query['query_id']} must bind candidate_universe_receipt_sha256"
                )

    qrel_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for qrel in qrels:
        require_fields(qrel, qrel_required, "qrel")
        if qrel.get("schema_version") != "livefire.rag.evidence-qrel-row/1":
            raise EvaluationError(f"qrel has unsupported schema_version: {qrel}")
        query_id = str(qrel.get("query_id", ""))
        command_id = str(qrel.get("command_id", ""))
        query = query_by_id.get(query_id)
        if query is None:
            raise EvaluationError(f"qrel references unknown query {query_id}")
        if qrel.get("fact_id") != query.get("fact_id"):
            raise EvaluationError(f"qrel/query fact_id mismatch for {query_id}:{command_id}")
        grade = qrel.get("relevance_grade")
        if type(grade) is not int or not 0 <= grade <= 3:
            raise EvaluationError(f"qrel relevance_grade must be an integer in [0,3]: {qrel}")
        relation_grades = {
            "direct_support": 3,
            "corroborating": 2,
            "contextual": 1,
            "contradictory": 0,
        }
        if qrel.get("relation") not in relation_grades or grade != relation_grades[qrel["relation"]]:
            raise EvaluationError(f"qrel relation/relevance_grade mismatch: {qrel}")
        if qrel.get("status") not in {"adjudicated", "provisional", "excluded"}:
            raise EvaluationError(f"qrel has invalid status: {qrel}")
        if qrel.get("eligibility") not in {
            "eligible", "duplicate_projection", "outside_index_domain",
            "unresolved_pointer", "excluded",
        }:
            raise EvaluationError(f"qrel has invalid eligibility: {qrel}")
        pointer = qrel.get("source_pointer")
        if not isinstance(pointer, dict) or pointer.get("record_id") != command_id:
            raise EvaluationError(f"qrel source pointer does not resolve command_id: {qrel}")
        qrel_by_pair[(query_id, command_id)] = qrel

    declared_negative_queries: set[str] = set()
    for negative in hard_negatives:
        require_fields(negative, negative_required, "hard negative")
        if negative.get("schema_version") != "livefire.rag.evidence-hard-negative-row/1":
            raise EvaluationError(f"hard negative has unsupported schema_version: {negative}")
        query_id = str(negative.get("query_id", ""))
        query = query_by_id.get(query_id)
        if query is None:
            raise EvaluationError(f"hard negative references unknown query {query_id}")
        if negative.get("fact_id") != query.get("fact_id"):
            raise EvaluationError(f"hard-negative/query fact_id mismatch for {query_id}")
        positive_id = str(negative.get("positive_command_id", ""))
        negative_id = str(negative.get("negative_command_id", ""))
        if not positive_id or not negative_id or positive_id == negative_id:
            raise EvaluationError(f"hard negative must bind distinct non-empty command IDs: {negative}")
        if negative.get("status") not in {"adjudicated", "provisional", "rejected"}:
            raise EvaluationError(f"hard negative has invalid status: {negative}")
        if negative.get("eligibility") not in {
            "eligible", "hidden_positive", "duplicate_positive", "unresolved_pointer",
            "unmatched", "excluded",
        }:
            raise EvaluationError(f"hard negative has invalid eligibility: {negative}")
        if negative.get("selection_mode") != "metadata_only_vector_blind":
            raise EvaluationError(f"hard negative violates vector-blind selection: {negative}")
        positive_qrel = qrel_by_pair.get((query_id, positive_id))
        negative_qrel = qrel_by_pair.get((query_id, negative_id))
        if positive_qrel is None or negative_qrel is None:
            raise EvaluationError(f"hard-negative pair lacks adjudicated qrels: {negative}")
        if negative.get("positive_projection_sha256") != positive_qrel.get("projection_sha256"):
            raise EvaluationError(f"hard-negative positive projection mismatch: {negative}")
        if negative.get("negative_projection_sha256") != negative_qrel.get("projection_sha256"):
            raise EvaluationError(f"hard-negative negative projection mismatch: {negative}")
        if negative.get("positive_projection_sha256") == negative.get("negative_projection_sha256"):
            raise EvaluationError(f"hard-negative projections must be distinct: {negative}")
        pointer = negative.get("negative_source_pointer")
        if not isinstance(pointer, dict) or pointer.get("record_id") != negative_id:
            raise EvaluationError(f"hard-negative source pointer mismatch: {negative}")
        if pointer.get("snapshot") != negative_qrel["source_pointer"].get("snapshot"):
            raise EvaluationError(f"hard-negative/qrel snapshot mismatch: {negative}")
        if pointer.get("snapshot_profile") != negative_qrel["source_pointer"].get("snapshot_profile"):
            raise EvaluationError(f"hard-negative/qrel snapshot profile mismatch: {negative}")
        if negative.get("status") == "adjudicated" and negative.get("eligibility") == "eligible":
            declared_negative_queries.add(query_id)
    active_query_ids = {query_id for query_id, query in query_by_id.items() if query_is_eligible(query)}
    missing_negative_queries = sorted(active_query_ids - declared_negative_queries)
    if missing_negative_queries:
        raise EvaluationError(
            "every active query surface requires an adjudicated matched hard negative; "
            f"missing={missing_negative_queries}"
        )


def validate_ranking_cardinality(
    queries: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    label: str,
    k: int,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rankings:
        counts[str(row["query_id"])] += 1
    for query in queries:
        if not query_is_eligible(query):
            continue
        query_id = str(query["query_id"])
        expected = query.get("expected_top_k_cardinality")
        if not isinstance(expected, int) or not 0 <= expected <= k:
            raise EvaluationError(
                f"{label}: active query {query_id} must declare expected_top_k_cardinality in [0,{k}]"
            )
        actual = counts.get(query_id, 0)
        if actual != expected:
            raise EvaluationError(
                f"{label}: query {query_id} returned {actual} rows; expected exactly {expected}"
            )


def validate_candidate_universes(
    queries: list[dict[str, Any]],
    universes: list[dict[str, Any]],
    k: int,
) -> None:
    require_unique(universes, ("query_id",), "candidate universes")
    by_query = {str(row["query_id"]): row for row in universes}
    active = {str(row["query_id"]): row for row in queries if query_is_eligible(row)}
    if set(by_query) != set(active):
        raise EvaluationError(
            "candidate-universe receipts must cover exactly the active queries: "
            f"missing={sorted(set(active) - set(by_query))}, extra={sorted(set(by_query) - set(active))}"
        )
    indexes: set[str] = set()
    required = {
        "schema_version", "query_id", "index", "filter_sha256",
        "candidate_document_count", "candidate_document_ids_sha256",
        "source_pointer_membership_sha256", "expected_top_k_cardinality",
        "computation_policy", "receipt_sha256",
    }
    for query_id, universe in by_query.items():
        missing = sorted(required - set(universe))
        if missing:
            raise EvaluationError(f"candidate universe {query_id} missing fields {missing}")
        if universe["schema_version"] != "livefire.rag.evidence-candidate-universe-row/1":
            raise EvaluationError(f"candidate universe {query_id} has unsupported schema_version")
        candidate_count = universe["candidate_document_count"]
        expected = universe["expected_top_k_cardinality"]
        if type(candidate_count) is not int or candidate_count < 0:
            raise EvaluationError(f"candidate universe {query_id} has invalid candidate_document_count")
        for digest_field in ("candidate_document_ids_sha256", "source_pointer_membership_sha256"):
            digest = universe[digest_field]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise EvaluationError(f"candidate universe {query_id} has invalid {digest_field}")
        if type(expected) is not int or expected != min(k, candidate_count):
            raise EvaluationError(
                f"candidate universe {query_id} expected_top_k_cardinality must equal min({k}, candidate_document_count)"
            )
        query = active[query_id]
        if query.get("expected_top_k_cardinality") != expected:
            raise EvaluationError(f"candidate universe/query cardinality mismatch for {query_id}")
        if query.get("candidate_universe_receipt_sha256") != universe.get("receipt_sha256"):
            raise EvaluationError(f"candidate universe/query receipt mismatch for {query_id}")
        indexes.add(json.dumps(universe["index"], sort_keys=True, separators=(",", ":")))
    if len(indexes) != 1:
        raise EvaluationError("all candidate-universe receipts must bind the same index")


def evaluate_rankings(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    k: int,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_ids = {str(row["query_id"]) for row in queries}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    qrel_groups_by_query: dict[str, dict[str, str]] = defaultdict(dict)
    for row in qrels:
        if row.get("status", "adjudicated") != "adjudicated":
            continue
        if row.get("eligibility", "eligible") not in {"eligible", "duplicate_projection"}:
            continue
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            raise EvaluationError(f"qrel references unknown query {query_id}")
        document_id = row_document_id(row, "document_id", "command_id")
        grade = int(row.get("relevance", row.get("relevance_grade", -1)))
        if not 0 <= grade <= 3:
            raise EvaluationError(f"qrel relevance must be in [0,3]: {row}")
        qrels_by_query[query_id][document_id] = grade
        qrel_groups_by_query[query_id][document_id] = str(
            row.get("evidence_group_id", row.get("duplicate_group_sha256", document_id))
        )
    group_grades: dict[tuple[str, str], set[int]] = defaultdict(set)
    for query_id, documents in qrels_by_query.items():
        for document_id, grade in documents.items():
            group_id = qrel_groups_by_query[query_id][document_id]
            group_grades[(query_id, group_id)].add(grade)
    inconsistent_groups = [key for key, grades in group_grades.items() if len(grades) > 1]
    if inconsistent_groups:
        raise EvaluationError(
            f"qrels assign inconsistent grades within evidence groups: {inconsistent_groups}"
        )
    negative_pairs_by_query: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for row in hard_negatives:
        if row.get("status", "adjudicated") != "adjudicated":
            continue
        if row.get("eligibility", "eligible") != "eligible":
            continue
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            raise EvaluationError(f"hard negative references unknown query {query_id}")
        document_id = row_document_id(row, "document_id", "negative_command_id")
        if document_id not in qrels_by_query.get(query_id, {}):
            raise EvaluationError(
                f"hard negative must have an adjudicated qrel for {query_id}: {document_id}"
            )
        if qrels_by_query.get(query_id, {}).get(document_id, 0) >= RELEVANT_GRADE:
            raise EvaluationError(f"hard negative is also relevant for {query_id}: {document_id}")
        positive_id = str(row["positive_command_id"]) if row.get("positive_command_id") else None
        if positive_id is not None and qrels_by_query.get(query_id, {}).get(positive_id, 0) < RELEVANT_GRADE:
            raise EvaluationError(
                f"hard negative positive_command_id is not a relevant qrel for {query_id}: {positive_id}"
            )
        negative_pairs_by_query[query_id].append((positive_id, document_id))
    ranking_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rankings:
        ranking_by_query[str(row["query_id"])].append(row)
    for rows in ranking_by_query.values():
        rows.sort(key=lambda item: int(item["rank"]))

    per_query: list[dict[str, Any]] = []
    eligible_queries = [row for row in queries if query_is_eligible(row)]
    mapped_fact_ids: set[str] = set()
    evaluated_queries = 0
    successful_queries = 0
    pointer_violations = 0
    filter_violations = 0

    for query in queries:
        query_id = str(query["query_id"])
        fact_id = str(query.get("fact_id", query_id))
        eligibility = str(query.get("eligibility", "needs_review"))
        relevant_documents = {
            document_id: grade
            for document_id, grade in qrels_by_query.get(query_id, {}).items()
            if grade >= RELEVANT_GRADE
        }
        relevant: dict[str, int] = {}
        for document_id, grade in relevant_documents.items():
            group_id = qrel_groups_by_query[query_id][document_id]
            relevant[group_id] = max(grade, relevant.get(group_id, 0))
        graded_groups: dict[str, int] = {}
        for document_id, grade in qrels_by_query.get(query_id, {}).items():
            if grade <= 0:
                continue
            group_id = qrel_groups_by_query[query_id][document_id]
            graded_groups[group_id] = max(grade, graded_groups.get(group_id, 0))
        if query_is_eligible(query) and relevant:
            mapped_fact_ids.add(fact_id)
        ranked = ranking_by_query.get(query_id, [])
        pointer_violations += sum(not bool(row.get("pointer_resolved", False)) for row in ranked)
        filter_violations += sum(not bool(row.get("filter_compliant", False)) for row in ranked)
        if not query_is_eligible(query):
            per_query.append(
                {
                    "query_id": query_id,
                    "fact_id": fact_id,
                    "domain": query.get("domain"),
                    "cohort": query.get("cohort"),
                    "resampling_cluster_id": query.get(
                        "resampling_cluster_id", query.get("incident_id", fact_id)
                    ),
                    "eligibility": eligibility,
                    "status": "excluded",
                    "relevant_documents": len(relevant),
                }
            )
            continue

        evaluated_queries += 1
        if not relevant or not ranked:
            per_query.append(
                {
                    "query_id": query_id,
                    "fact_id": fact_id,
                    "domain": query.get("domain"),
                    "cohort": query.get("cohort"),
                    "resampling_cluster_id": query.get(
                        "resampling_cluster_id", query.get("incident_id", fact_id)
                    ),
                    "eligibility": eligibility,
                    "status": "failed",
                    "failure_reason": "missing_relevant_qrels" if not relevant else "missing_ranking",
                    "relevant_documents": len(relevant),
                    f"ndcg_at_{k}": 0.0,
                    f"recall_at_{k}": 0.0,
                    f"mrr_at_{k}": 0.0,
                    "first_relevant_rank": None,
                    "hard_negatives_in_top_k": 0,
                    "best_hard_negative_rank": None,
                    "hard_negative_triplet_accuracy": None,
                    "hard_negative_triplet_coverage": 0.0,
                    "median_hard_negative_margin": None,
                }
            )
            continue

        successful_queries += 1
        top = ranked[:k]
        unjudged = [
            ranking_document_id(row)
            for row in top
            if ranking_document_id(row) not in qrels_by_query.get(query_id, {})
        ]
        if unjudged:
            raise EvaluationError(f"query {query_id} has unjudged top-{k} documents: {unjudged}")
        seen_groups: set[str] = set()
        grades: list[int] = []
        retrieved_groups: set[str] = set()
        for row in top:
            document_id = ranking_document_id(row)
            grade = qrels_by_query.get(query_id, {}).get(document_id, 0)
            group_id = qrel_groups_by_query.get(query_id, {}).get(document_id, document_id)
            if group_id in seen_groups:
                grades.append(0)
                continue
            seen_groups.add(group_id)
            grades.append(grade)
            if grade >= RELEVANT_GRADE:
                retrieved_groups.add(group_id)
        ideal = sorted(graded_groups.values(), reverse=True)
        ideal_dcg = dcg(ideal, k)
        ndcg = dcg(grades, k) / ideal_dcg if ideal_dcg else 0.0
        recall = len(retrieved_groups) / len(relevant)
        first_rank = next(
            (
                int(row["rank"])
                for row in top
                if qrel_groups_by_query.get(query_id, {}).get(
                    ranking_document_id(row), ranking_document_id(row)
                )
                in relevant
            ),
            None,
        )
        mrr = 1.0 / first_rank if first_rank is not None else 0.0

        ranked_preferences = {
            ranking_document_id(row): float(ranking_preference(row))
            for row in ranked
            if ranking_preference(row) is not None
        }
        positive_group_preferences: dict[str, float] = {}
        for document_id in relevant_documents:
            if document_id not in ranked_preferences:
                continue
            group_id = qrel_groups_by_query[query_id][document_id]
            positive_group_preferences[group_id] = max(
                ranked_preferences[document_id],
                positive_group_preferences.get(group_id, -math.inf),
            )
        triplet_margins: list[float] = []
        declared_triplets = 0
        observed_triplets = 0
        for positive_id, negative_id in negative_pairs_by_query.get(query_id, []):
            declared_triplets += 1 if positive_id is not None else len(relevant)
            if negative_id not in ranked_preferences:
                continue
            if positive_id is not None:
                if positive_id in ranked_preferences:
                    triplet_margins.append(
                        ranked_preferences[positive_id] - ranked_preferences[negative_id]
                    )
                    observed_triplets += 1
                continue
            for positive in positive_group_preferences.values():
                triplet_margins.append(positive - ranked_preferences[negative_id])
                observed_triplets += 1
        triplet_accuracy = (
            sum(float(margin > 0) for margin in triplet_margins) / declared_triplets
            if declared_triplets
            else None
        )
        triplet_coverage = observed_triplets / declared_triplets if declared_triplets else None
        margin = statistics.median(triplet_margins) if triplet_margins else None
        negative_documents = {
            negative_id for _, negative_id in negative_pairs_by_query.get(query_id, [])
        }
        hard_negative_ranks = [
            int(row["rank"])
            for row in top
            if ranking_document_id(row) in negative_documents
        ]

        per_query.append(
            {
                "query_id": query_id,
                "fact_id": fact_id,
                "domain": query.get("domain"),
                "cohort": query.get("cohort"),
                "resampling_cluster_id": query.get(
                    "resampling_cluster_id", query.get("incident_id", fact_id)
                ),
                "eligibility": eligibility,
                "status": "evaluated",
                "relevant_documents": len(relevant),
                f"ndcg_at_{k}": ndcg,
                f"recall_at_{k}": recall,
                f"mrr_at_{k}": mrr,
                "first_relevant_rank": first_rank,
                "hard_negatives_in_top_k": len(hard_negative_ranks),
                "best_hard_negative_rank": min(hard_negative_ranks) if hard_negative_ranks else None,
                "hard_negative_triplet_accuracy": triplet_accuracy,
                "hard_negative_triplet_coverage": triplet_coverage,
                "median_hard_negative_margin": margin,
            }
        )

    scored = [row for row in per_query if row["status"] in {"evaluated", "failed"}]
    per_fact: list[dict[str, Any]] = []
    evaluated_by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        evaluated_by_fact[str(row["fact_id"])].append(row)
    for fact_id, rows in sorted(evaluated_by_fact.items()):
        triplets = [
            float(row["hard_negative_triplet_accuracy"])
            for row in rows
            if row["hard_negative_triplet_accuracy"] is not None
        ]
        margins = [
            float(row["median_hard_negative_margin"])
            for row in rows
            if row["median_hard_negative_margin"] is not None
        ]
        triplet_coverages = [
            float(row["hard_negative_triplet_coverage"])
            for row in rows
            if row["hard_negative_triplet_coverage"] is not None
        ]
        per_fact.append(
            {
                "fact_id": fact_id,
                "domain": rows[0].get("domain"),
                "cohort": rows[0].get("cohort"),
                "resampling_cluster_id": rows[0].get("resampling_cluster_id", fact_id),
                "query_surfaces": len(rows),
                f"ndcg_at_{k}": mean(float(row[f"ndcg_at_{k}"]) for row in rows),
                f"recall_at_{k}": mean(float(row[f"recall_at_{k}"]) for row in rows),
                f"mrr_at_{k}": mean(float(row[f"mrr_at_{k}"]) for row in rows),
                "cohort_covered_at_k": all(float(row[f"recall_at_{k}"]) > 0 for row in rows),
                "hard_negative_triplet_accuracy": mean(triplets) if triplets else None,
                "hard_negative_triplet_coverage": mean(triplet_coverages) if triplet_coverages else None,
                "median_hard_negative_margin": statistics.median(margins) if margins else None,
            }
        )
    triplet_values = [
        float(row["hard_negative_triplet_accuracy"])
        for row in per_fact
        if row["hard_negative_triplet_accuracy"] is not None
    ]
    margin_values = [
        float(row["median_hard_negative_margin"])
        for row in per_fact
        if row["median_hard_negative_margin"] is not None
    ]
    triplet_coverage_values = [
        float(row["hard_negative_triplet_coverage"])
        for row in per_fact
        if row["hard_negative_triplet_coverage"] is not None
    ]
    macro = {
        f"ndcg_at_{k}": mean(float(row[f"ndcg_at_{k}"]) for row in per_fact),
        f"recall_at_{k}": mean(float(row[f"recall_at_{k}"]) for row in per_fact),
        f"mrr_at_{k}": mean(float(row[f"mrr_at_{k}"]) for row in per_fact),
        f"cohort_coverage_at_{k}": mean(float(row["cohort_covered_at_k"]) for row in per_fact),
        "hard_negative_triplet_accuracy": mean(triplet_values) if triplet_values else None,
        "hard_negative_triplet_coverage": mean(triplet_coverage_values) if triplet_coverage_values else None,
        "median_hard_negative_margin": statistics.median(margin_values) if margin_values else None,
    }
    per_cohort: dict[str, Any] = {}
    cohort_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_fact:
        cohort_rows[str(row.get("cohort", "unclassified"))].append(row)
    for cohort, rows in sorted(cohort_rows.items()):
        per_cohort[cohort] = {
            "facts": len(rows),
            "resampling_clusters": len({str(row["resampling_cluster_id"]) for row in rows}),
            "inference_status": (
                "inferential"
                if len({str(row["resampling_cluster_id"]) for row in rows}) >= 2
                else "descriptive"
            ),
            f"ndcg_at_{k}": mean(float(row[f"ndcg_at_{k}"]) for row in rows),
            f"recall_at_{k}": mean(float(row[f"recall_at_{k}"]) for row in rows),
            f"cohort_coverage_at_{k}": mean(float(row["cohort_covered_at_k"]) for row in rows),
        }
    if inventory is None:
        fact_ids = {str(row.get("fact_id", row["query_id"])) for row in queries}
        eligible_fact_ids = {str(row.get("fact_id", row["query_id"])) for row in eligible_queries}
    else:
        fact_ids = {str(row["atom_id"]) for row in inventory["atoms"]}
        eligible_fact_ids = {
            str(row["atom_id"]) for row in inventory["atoms"] if inventory_atom_is_eligible(row)
        }
    evaluated_fact_ids = {str(row["fact_id"]) for row in scored}
    coverage = {
        "facts_total": len(fact_ids),
        "queries_total": len(queries),
        "eligible_query_surfaces": len(eligible_queries),
        "eligible_facts": len(eligible_fact_ids),
        "mapped_facts": len(mapped_fact_ids),
        "evaluated_facts": len(evaluated_fact_ids),
        "evaluated_query_surfaces": evaluated_queries,
        "successful_query_surfaces": successful_queries,
        "query_surfaces_with_declared_hard_negatives": len(
            set(negative_pairs_by_query) & {str(row["query_id"]) for row in eligible_queries}
        ),
        "eligible_mapping_rate": len(mapped_fact_ids) / len(eligible_fact_ids) if eligible_fact_ids else 0.0,
        "eligible_evaluation_rate": len(evaluated_fact_ids) / len(eligible_fact_ids) if eligible_fact_ids else 0.0,
        "eligible_query_execution_rate": successful_queries / len(eligible_queries) if eligible_queries else 0.0,
        "hard_negative_declaration_rate": (
            len(set(negative_pairs_by_query) & {str(row["query_id"]) for row in eligible_queries})
            / len(eligible_queries)
            if eligible_queries
            else 0.0
        ),
    }
    return {
        "coverage": coverage,
        "correctness": {
            "unresolved_pointers": pointer_violations,
            "filter_violations": filter_violations,
        },
        "macro": macro,
        "per_cohort": per_cohort,
        "per_fact": per_fact,
        "per_query": per_query,
    }


def paired_bootstrap(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_rows = {
        row["fact_id"]: row
        for row in candidate["per_fact"]
        if row.get(metric) is not None
    }
    baseline_rows = {
        row["fact_id"]: row
        for row in baseline["per_fact"]
        if row.get(metric) is not None
    }
    fact_ids = sorted(set(candidate_rows) & set(baseline_rows))
    deltas_by_cluster: dict[str, list[float]] = defaultdict(list)
    for fact_id in fact_ids:
        candidate_row = candidate_rows[fact_id]
        baseline_row = baseline_rows[fact_id]
        candidate_cluster = str(candidate_row["resampling_cluster_id"])
        if candidate_cluster != str(baseline_row["resampling_cluster_id"]):
            raise EvaluationError(f"candidate/baseline cluster mismatch for fact {fact_id}")
        deltas_by_cluster[candidate_cluster].append(
            float(candidate_row[metric]) - float(baseline_row[metric])
        )
    deltas = [value for values in deltas_by_cluster.values() for value in values]
    if not deltas:
        return {"paired_facts": 0, "delta": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    clusters = sorted(deltas_by_cluster)
    bootstrap: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(clusters) for _ in clusters]
        bootstrap.append(mean(value for cluster in sampled for value in deltas_by_cluster[cluster]))
    return {
        "paired_facts": len(deltas),
        "delta": mean(deltas),
        "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "bootstrap_samples": samples,
        "resampling_unit": "cluster",
        "seed": seed,
    }


def attach_uncertainty(result: dict[str, Any], k: int, samples: int, seed: int) -> None:
    rows = result["per_fact"]
    rows_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[str(row["resampling_cluster_id"])].append(row)
    specifications = [
        (f"ndcg_at_{k}", mean),
        (f"recall_at_{k}", mean),
        (f"mrr_at_{k}", mean),
        ("cohort_covered_at_k", mean),
        ("hard_negative_triplet_accuracy", mean),
        ("hard_negative_triplet_coverage", mean),
        ("median_hard_negative_margin", statistics.median),
    ]
    rng = random.Random(seed)
    uncertainty: dict[str, Any] = {}
    for metric, statistic in specifications:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        clusters = sorted(rows_by_cluster)
        bootstrap: list[float] = []
        for _ in range(samples):
            sampled = [rng.choice(clusters) for _ in clusters]
            sampled_values = [
                float(row[metric])
                for cluster in sampled
                for row in rows_by_cluster[cluster]
                if row.get(metric) is not None
            ]
            if sampled_values:
                bootstrap.append(statistic(sampled_values))
        uncertainty[metric] = {
            "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
            "bootstrap_samples": samples,
            "resampling_unit": "cluster",
            "seed": seed,
            "resampling_clusters": len(clusters),
        }
    result["uncertainty"] = uncertainty
    for cohort, cohort_result in result.get("per_cohort", {}).items():
        cohort_rows = [row for row in rows if str(row.get("cohort")) == cohort]
        rows_by_cohort_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cohort_rows:
            rows_by_cohort_cluster[str(row["resampling_cluster_id"])].append(row)
        clusters = sorted(rows_by_cohort_cluster)
        cohort_uncertainty: dict[str, Any] = {}
        for metric, statistic in (
            (f"ndcg_at_{k}", mean),
            (f"recall_at_{k}", mean),
            ("cohort_covered_at_k", mean),
        ):
            bootstrap: list[float] = []
            for _ in range(samples):
                sampled = [rng.choice(clusters) for _ in clusters]
                sampled_values = [
                    float(row[metric])
                    for cluster in sampled
                    for row in rows_by_cohort_cluster[cluster]
                ]
                bootstrap.append(statistic(sampled_values))
            cohort_uncertainty[metric] = {
                "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
                "bootstrap_samples": samples,
                "resampling_unit": "cluster",
                "seed": seed,
                "resampling_clusters": len(clusters),
            }
        cohort_result["uncertainty"] = cohort_uncertainty


def compare_runs(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    k: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = [
        f"ndcg_at_{k}",
        f"recall_at_{k}",
        f"mrr_at_{k}",
        "cohort_covered_at_k",
        "hard_negative_triplet_accuracy",
        "hard_negative_triplet_coverage",
    ]
    comparison = {
        metric: paired_bootstrap(candidate, baseline, metric, samples, seed + offset)
        for offset, metric in enumerate(metrics)
    }
    for metric in ("median_hard_negative_margin",):
        candidate_value = candidate["macro"].get(metric)
        baseline_value = baseline["macro"].get(metric)
        comparison[metric] = {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "delta": (
                float(candidate_value) - float(baseline_value)
                if candidate_value is not None and baseline_value is not None
                else None
            ),
        }
    return comparison


def promotion_decision(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    gates: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append({"name": name, "passed": passed, "observed": observed, "required": required})

    check("baseline_present", baseline is not None and comparison is not None, baseline is not None, True)

    correctness = candidate["correctness"]
    if gates["require_zero_correctness_errors"]:
        observed = correctness["unresolved_pointers"] + correctness["filter_violations"]
        check("zero_correctness_errors", observed == 0, observed, 0)
    evaluation_rate = candidate["coverage"]["eligible_evaluation_rate"]
    check(
        "minimum_eligible_evaluation_rate",
        evaluation_rate >= gates["minimum_eligible_evaluation_rate"],
        evaluation_rate,
        gates["minimum_eligible_evaluation_rate"],
    )
    execution_rate = candidate["coverage"]["eligible_query_execution_rate"]
    check(
        "minimum_eligible_query_execution_rate",
        execution_rate >= gates["minimum_eligible_query_execution_rate"],
        execution_rate,
        gates["minimum_eligible_query_execution_rate"],
    )
    declaration_rate = candidate["coverage"]["hard_negative_declaration_rate"]
    check(
        "minimum_hard_negative_declaration_rate",
        declaration_rate >= gates["minimum_hard_negative_declaration_rate"],
        declaration_rate,
        gates["minimum_hard_negative_declaration_rate"],
    )
    for metric, gate_name in (
        (f"ndcg_at_{k}", "minimum_macro_ndcg"),
        (f"recall_at_{k}", "minimum_macro_recall"),
        (f"cohort_coverage_at_{k}", "minimum_cohort_coverage"),
        ("hard_negative_triplet_accuracy", "minimum_hard_negative_triplet_accuracy"),
        ("hard_negative_triplet_coverage", "minimum_hard_negative_triplet_coverage"),
    ):
        observed = candidate["macro"].get(metric)
        required = gates[gate_name]
        check(
            gate_name,
            observed is not None and float(observed) >= required,
            observed,
            required,
        )
    for cohort, values in sorted(candidate.get("per_cohort", {}).items()):
        observed = values.get(f"cohort_coverage_at_{k}")
        required = gates["minimum_each_cohort_coverage"]
        check(
            f"minimum_cohort_coverage:{cohort}",
            observed is not None and float(observed) >= required,
            observed,
            required,
        )
    margin_ci = candidate.get("uncertainty", {}).get("median_hard_negative_margin", {}).get("ci95")
    margin_lower = margin_ci[0] if margin_ci else None
    check(
        "positive_hard_negative_margin_ci_lower",
        margin_lower is not None and margin_lower > gates["minimum_hard_negative_margin_ci_lower"],
        margin_lower,
        f"> {gates['minimum_hard_negative_margin_ci_lower']}",
    )
    if baseline is not None and comparison is not None:
        primary = comparison[f"ndcg_at_{k}"]
        recall = comparison[f"recall_at_{k}"]
        triplet_delta = comparison["hard_negative_triplet_accuracy"]["delta"]
        check(
            "primary_ndcg_non_inferiority",
            primary["ci95"][0] >= gates["primary_ndcg_non_inferiority_margin"],
            primary["ci95"][0],
            gates["primary_ndcg_non_inferiority_margin"],
        )
        check(
            "minimum_primary_ndcg_point_delta",
            primary["delta"] >= gates["minimum_primary_ndcg_point_delta"],
            primary["delta"],
            gates["minimum_primary_ndcg_point_delta"],
        )
        check(
            "recall_non_regression",
            recall["delta"] >= gates["recall_non_regression_margin"],
            recall["delta"],
            gates["recall_non_regression_margin"],
        )
        if triplet_delta is not None:
            check(
                "hard_negative_non_regression",
                triplet_delta >= gates["hard_negative_non_regression_margin"],
                triplet_delta,
                gates["hard_negative_non_regression_margin"],
            )
        coverage_delta = comparison["cohort_covered_at_k"]["delta"]
        check(
            "cohort_coverage_non_regression",
            coverage_delta >= gates["cohort_coverage_non_regression_margin"],
            coverage_delta,
            gates["cohort_coverage_non_regression_margin"],
        )
        baseline_execution_rate = baseline["coverage"]["eligible_query_execution_rate"]
        check(
            "query_execution_non_regression",
            execution_rate >= baseline_execution_rate,
            execution_rate,
            baseline_execution_rate,
        )
    return {
        "status": "pass" if checks and all(item["passed"] for item in checks) else "fail",
        "checks": checks,
    }


def default_gates() -> dict[str, Any]:
    return {
        "require_zero_correctness_errors": True,
        "minimum_query_surfaces_per_fact": 3,
        "minimum_eligible_evaluation_rate": 1.0,
        "minimum_eligible_query_execution_rate": 1.0,
        "minimum_hard_negative_declaration_rate": 1.0,
        "minimum_macro_ndcg": 0.70,
        "minimum_macro_recall": 0.85,
        "minimum_cohort_coverage": 0.90,
        "minimum_each_cohort_coverage": 0.80,
        "minimum_hard_negative_triplet_accuracy": 0.90,
        "minimum_hard_negative_triplet_coverage": 1.0,
        "minimum_hard_negative_margin_ci_lower": 0.0,
        "primary_ndcg_non_inferiority_margin": -0.01,
        "minimum_primary_ndcg_point_delta": 0.0,
        "recall_non_regression_margin": -0.01,
        "hard_negative_non_regression_margin": -0.01,
        "cohort_coverage_non_regression_margin": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidate-universes", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=[DEFAULT_K], default=DEFAULT_K)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    queries = load_rows(args.queries)
    candidate_universes = load_rows(args.candidate_universes)
    inventory = load_inventory(args.inventory)
    qrels = load_rows(args.qrels)
    hard_negatives = load_rows(args.hard_negatives)
    candidate_rows = load_rows(args.candidate)
    require_unique(queries, ("query_id",), "queries")
    require_unique_documents(qrels, "qrels", ("document_id", "command_id"))
    require_unique_documents(
        hard_negatives,
        "hard negatives",
        ("document_id", "negative_command_id"),
    )
    validate_benchmark_rows(queries, qrels, hard_negatives)
    validate_candidate_universes(queries, candidate_universes, args.k)
    query_ids = {str(row["query_id"]) for row in queries}
    candidate_contract = validate_rankings(
        candidate_rows,
        "candidate rankings",
        known_query_ids=query_ids,
        strict_schema=True,
    )
    validate_ranking_cardinality(queries, candidate_rows, "candidate rankings", args.k)

    gates = default_gates()
    if args.gates:
        override = json.loads(args.gates.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise EvaluationError("gate configuration must be an object")
        unknown = set(override) - set(gates)
        if unknown:
            raise EvaluationError(f"unknown gate configuration keys: {sorted(unknown)}")
        gates.update(override)
    validate_inventory_queries(
        inventory,
        queries,
        int(gates["minimum_query_surfaces_per_fact"]),
    )

    candidate = evaluate_rankings(
        queries, qrels, hard_negatives, candidate_rows, args.k, inventory
    )
    attach_uncertainty(candidate, args.k, args.bootstrap_samples, args.seed + 100)
    baseline = None
    comparison = None
    input_paths = {
        "queries": args.queries,
        "candidate_universes": args.candidate_universes,
        "inventory": args.inventory,
        "qrels": args.qrels,
        "hard_negatives": args.hard_negatives,
        "candidate_rankings": args.candidate,
    }
    if args.baseline:
        baseline_rows = load_rows(args.baseline)
        baseline_contract = validate_rankings(
            baseline_rows,
            "baseline rankings",
            known_query_ids=query_ids,
            strict_schema=True,
        )
        validate_ranking_cardinality(queries, baseline_rows, "baseline rankings", args.k)
        baseline = evaluate_rankings(
            queries, qrels, hard_negatives, baseline_rows, args.k, inventory
        )
        attach_uncertainty(baseline, args.k, args.bootstrap_samples, args.seed + 200)
        comparison = compare_runs(
            candidate,
            baseline,
            args.k,
            args.bootstrap_samples,
            args.seed,
        )
        if (
            candidate_contract != baseline_contract
            or candidate_contract["score_kind"] != "cosine_distance"
        ):
            comparison["median_hard_negative_margin"] = {
                "candidate": candidate["macro"].get("median_hard_negative_margin"),
                "baseline": baseline["macro"].get("median_hard_negative_margin"),
                "delta": None,
            }
        input_paths["baseline_rankings"] = args.baseline

    if args.gates:
        input_paths["gates"] = args.gates

    report = {
        "schema_version": "livefire.rag.evidence-benchmark-comparison/1",
        "primary_metric": f"macro_ndcg_at_{args.k}",
        "scoring_contract": "declared_per_ranking_input",
        "candidate_score_contract": candidate_contract,
        "baseline_score_contract": baseline_contract if args.baseline else None,
        "candidate": candidate,
        "baseline": baseline,
        "comparison": comparison,
        "promotion": promotion_decision(candidate, baseline, comparison, gates, args.k),
        "gates": gates,
        "inputs": {
            role: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for role, path in input_paths.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
