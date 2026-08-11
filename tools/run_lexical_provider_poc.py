#!/usr/bin/env python3
"""Run a deterministic same-corpus BM25 baseline for the provider POC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
K1 = 1.2
B = 0.75
SCORER_ID = "single_field_bm25_semantic_text_ascii_camel_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [match.group(0).lower() for match in TOKEN_RE.finditer(camel_split)]


class Bm25Corpus:
    def __init__(self, documents: list[dict[str, Any]], k1: float = K1, b: float = B):
        if not documents:
            raise ValueError("BM25 corpus must not be empty")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.document_frequencies: Counter[str] = Counter()
        for document in documents:
            projection = str(document.get("semantic_text", ""))
            frequencies = Counter(tokenize(projection))
            self.term_frequencies.append(frequencies)
            length = sum(frequencies.values())
            self.lengths.append(length)
            self.document_frequencies.update(frequencies.keys())
        self.average_length = sum(self.lengths) / len(self.lengths)

    def score(self, query: str, document_index: int) -> float:
        query_frequencies = Counter(tokenize(query))
        frequencies = self.term_frequencies[document_index]
        document_length = self.lengths[document_index]
        score = 0.0
        population = len(self.documents)
        for term, query_frequency in query_frequencies.items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self.document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0 + (population - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalization = 1.0 - self.b + self.b * document_length / self.average_length
            saturation = (
                term_frequency * (self.k1 + 1.0)
                / (term_frequency + self.k1 * length_normalization)
            )
            score += query_frequency * inverse_document_frequency * saturation
        return score

    def search(self, query: str, top_n: int) -> list[tuple[float, dict[str, Any]]]:
        scored = [
            (self.score(query, index), document)
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]["command_id"]))
        return scored[:top_n]


def load_index(index_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    manifest_path = index_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    document_object = manifest["objects"]["documents"]
    document_path = index_path / document_object["path"]
    observed_sha256 = sha256_file(document_path)
    if observed_sha256 != document_object["sha256"]:
        raise ValueError("documents object digest does not match the sealed index manifest")
    documents = [json.loads(line) for line in document_path.read_text(encoding="utf-8").splitlines()]
    if len(documents) != manifest["documents_count"]:
        raise ValueError("documents object count does not match the sealed index manifest")
    command_ids = [document["command_id"] for document in documents]
    if len(command_ids) != len(set(command_ids)):
        raise ValueError("documents object contains duplicate command_id")
    return manifest, documents, document_path


def pointer(score: float, rank: int, document: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: document[key]
        for key in ("event_time", "host_id", "principal_key", "shell_family", "source_kind")
        if key in document
    }
    return {
        "rank": rank,
        "command_id": document["command_id"],
        "bm25_score_millionths": int(round(score * 1_000_000)),
        "preview": document.get("preview", ""),
        "source_ref": document["source_pointer"],
        "metadata": metadata,
    }


def lexical_results(
    suite: dict[str, Any], manifest: dict[str, Any], documents: list[dict[str, Any]]
) -> dict[str, Any]:
    top_n = int(suite["policy"]["top_n"])
    cases = [
        case
        for case in suite["cases"]
        if case.get("kind") == "acceptance" and case.get("tool") == "cli.search"
    ]
    corpus = Bm25Corpus(documents)
    calls = []
    for case in cases:
        ranked = corpus.search(case["query"], top_n)
        candidates = [pointer(score, rank, document) for rank, (score, document) in enumerate(ranked, 1)]
        output = {
            "schema_version": "livefire.rag.lexical-result/1",
            "kind": "pointer",
            "tool": "cli.search",
            "index": manifest["component"],
            "scorer": {
                "id": SCORER_ID,
                "k1": K1,
                "b": B,
                "projection": "semantic_text only; exactly the text embedded by the dense system",
                "tokenizer": "ASCII alphanumeric/underscore tokens after lower-camel boundary splitting; lowercase; no stemming or stop-word removal",
                "tie_break": "score_desc_command_id_asc",
            },
            "pointers": candidates,
            "coverage": {
                "status": "complete",
                "indexed_commands": len(documents),
                "eligible_commands": len(documents),
                "requested_top_n": top_n,
                "returned_count": len(candidates),
                "exhausted": len(documents) <= top_n,
            },
        }
        calls.append({"case_id": case["case_id"], "response": output})
    return {
        "schema_version": "livefire.rag.lexical-provider-poc-results/1",
        "run_id": "deterministic-lexical-baseline",
        "suite_id": suite["suite_id"],
        "index": manifest["component"],
        "documents": manifest["objects"]["documents"],
        "scorer_id": SCORER_ID,
        "calls": calls,
    }


def load_checker(repository_root: Path):
    path = repository_root / "tools/check_provider_poc.py"
    spec = importlib.util.spec_from_file_location("provider_poc_checker_for_lexical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load provider POC checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_report(
    suite: dict[str, Any], manifest: dict[str, Any], dense: dict[str, Any], lexical: dict[str, Any]
) -> dict[str, Any]:
    checker = load_checker(Path(__file__).resolve().parents[1])
    dense_check = checker.build_report(suite, dense, "acceptance")
    lexical_check = checker.build_report(suite, lexical, "acceptance")
    dense_by_id = {row["case_id"]: row for row in dense_check["cases"]}
    lexical_by_id = {row["case_id"]: row for row in lexical_check["cases"]}
    cases = []
    buckets: dict[str, list[str]] = {
        "both_full_positive_behavior": [],
        "both_full_lexical_lower_hard_negative_exposure": [],
        "dense_full_lexical_partial": [],
        "dense_only_positive_behavior": [],
        "lexical_full_dense_partial_or_zero": [],
        "neither_full_positive_behavior": [],
        "boundary_diagnostic": [],
    }
    dense_positive_passed = 0
    lexical_positive_passed = 0
    positive_total = 0
    dense_hard_negatives_observed = 0
    lexical_hard_negatives_observed = 0
    hard_negative_total = 0
    for case in [row for row in suite["cases"] if row.get("kind") == "acceptance"]:
        case_id = case["case_id"]
        dense_case = dense_by_id[case_id]
        lexical_case = lexical_by_id[case_id]
        dense_pass = dense_case["passed"]
        lexical_pass = lexical_case["passed"]
        dense_positive = sum(row["passed"] for row in dense_case["behaviors"])
        lexical_positive = sum(row["passed"] for row in lexical_case["behaviors"])
        case_positive_total = len(dense_case["behaviors"])
        dense_negative = sum(row["passed"] for row in dense_case["hard_negatives"])
        lexical_negative = sum(row["passed"] for row in lexical_case["hard_negatives"])
        case_negative_total = len(dense_case["hard_negatives"])
        dense_positive_passed += dense_positive
        lexical_positive_passed += lexical_positive
        positive_total += case_positive_total
        dense_hard_negatives_observed += dense_negative
        lexical_hard_negatives_observed += lexical_negative
        hard_negative_total += case_negative_total
        if case["expected_disposition"] == "expected_boundary_failure":
            bucket = "boundary_diagnostic"
        elif dense_positive == case_positive_total and lexical_positive == case_positive_total:
            bucket = (
                "both_full_lexical_lower_hard_negative_exposure"
                if lexical_negative < dense_negative
                else "both_full_positive_behavior"
            )
        elif dense_positive == case_positive_total and lexical_positive > 0:
            bucket = "dense_full_lexical_partial"
        elif dense_positive == case_positive_total:
            bucket = "dense_only_positive_behavior"
        elif lexical_positive == case_positive_total:
            bucket = "lexical_full_dense_partial_or_zero"
        else:
            bucket = "neither_full_positive_behavior"
        buckets[bucket].append(case_id)
        cases.append(
            {
                "case_id": case_id,
                "expected_disposition": case["expected_disposition"],
                "dense_pass": dense_pass,
                "lexical_pass": lexical_pass,
                "comparison": bucket,
                "dense_outcome": dense_by_id[case_id]["outcome"],
                "lexical_outcome": lexical_by_id[case_id]["outcome"],
                "positive_behavior_checks": {
                    "total": case_positive_total,
                    "dense_passed": dense_positive,
                    "lexical_passed": lexical_positive,
                },
                "hard_negative_exposure": {
                    "declared": case_negative_total,
                    "dense_exposed": dense_negative,
                    "lexical_exposed": lexical_negative,
                    "preferred_direction": "lower_is_better",
                    "interpretation": "These are adjudicated near-miss or wrong-polarity results; top-10 exposure is a weakness.",
                },
                "dense_behavior_matches": {
                    row["label"]: row["matched_ranks"] for row in dense_by_id[case_id]["behaviors"]
                },
                "lexical_behavior_matches": {
                    row["label"]: row["matched_ranks"] for row in lexical_by_id[case_id]["behaviors"]
                },
            }
        )
    return {
        "schema_version": "livefire.rag.provider-poc-effectiveness-comparison/1",
        "status": "qualitative_only",
        "suite_id": suite["suite_id"],
        "same_candidate_universe": {
            "index": manifest["component"],
            "documents_count": manifest["documents_count"],
            "documents_sha256": manifest["objects"]["documents"]["sha256"],
            "top_n": suite["policy"]["top_n"],
        },
        "systems": {
            "dense": {"checker": dense_check, "name": "Qwen3-Embedding-8B Q4 exact cosine"},
            "lexical": {"checker": lexical_check, "name": SCORER_ID},
        },
        "qualitative_outcomes": buckets,
        "positive_behavior_summary": {
            "declared_checks": positive_total,
            "dense_passed": dense_positive_passed,
            "lexical_passed": lexical_positive_passed,
        },
        "hard_negative_exposure_summary": {
            "declared": hard_negative_total,
            "dense_exposed": dense_hard_negatives_observed,
            "lexical_exposed": lexical_hard_negatives_observed,
            "preferred_direction": "lower_is_better",
            "interpretation": "The acceptance checker requires these rows only to reproduce known dense behavior; effectiveness treats their presence as a weakness.",
        },
        "cases": cases,
        "claim_boundary": [
            "Acceptance predicates are frozen qualitative smoke checks, not qrels.",
            "No nDCG, Recall, precision, statistical significance, or general model superiority is claimed.",
            "Q9 passing means the expected flat-query boundary failure was reproduced, not that either retriever answered the composite investigation.",
        ],
    }


def markdown_comparison(report: dict[str, Any]) -> str:
    lines = [
        "# Dense versus lexical provider POC",
        "",
        "Status: **qualitative only**",
        "",
        f"Both systems used the same {report['same_candidate_universe']['documents_count']}-document "
        f"object (`{report['same_candidate_universe']['documents_sha256']}`) and frozen top "
        f"{report['same_candidate_universe']['top_n']} Q1-Q9 queries.",
        "",
        "| Case | Expected disposition | Dense positive checks | Lexical positive checks | HN exposure dense/lexical | Strict checker | Qualitative result |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['expected_disposition']} | "
            f"{case['positive_behavior_checks']['dense_passed']}/{case['positive_behavior_checks']['total']} | "
            f"{case['positive_behavior_checks']['lexical_passed']}/{case['positive_behavior_checks']['total']} | "
            f"{case['hard_negative_exposure']['dense_exposed']}/{case['hard_negative_exposure']['lexical_exposed']} | "
            f"dense={'pass' if case['dense_pass'] else 'fail'}, lexical={'pass' if case['lexical_pass'] else 'fail'} | "
            f"{case['comparison']} |"
        )
    outcomes = report["qualitative_outcomes"]
    lines.extend(
        [
            "",
            "## Outcome counts",
            "",
            f"- Positive behavior checks: dense {report['positive_behavior_summary']['dense_passed']}/{report['positive_behavior_summary']['declared_checks']}; lexical {report['positive_behavior_summary']['lexical_passed']}/{report['positive_behavior_summary']['declared_checks']}.",
            f"- Both complete: {', '.join(outcomes['both_full_positive_behavior']) or 'none'}",
            f"- Both complete, lexical lower hard-negative exposure: {', '.join(outcomes['both_full_lexical_lower_hard_negative_exposure']) or 'none'}",
            f"- Dense complete, lexical partial: {', '.join(outcomes['dense_full_lexical_partial']) or 'none'}",
            f"- Dense only: {', '.join(outcomes['dense_only_positive_behavior']) or 'none'}",
            f"- Lexical complete, dense partial/zero: {', '.join(outcomes['lexical_full_dense_partial_or_zero']) or 'none'}",
            f"- Neither complete: {', '.join(outcomes['neither_full_positive_behavior']) or 'none'}",
            f"- Boundary diagnostic: {', '.join(outcomes['boundary_diagnostic']) or 'none'}",
            f"- Known hard-negative exposure (lower is better): dense {report['hard_negative_exposure_summary']['dense_exposed']}/{report['hard_negative_exposure_summary']['declared']}; lexical {report['hard_negative_exposure_summary']['lexical_exposed']}/{report['hard_negative_exposure_summary']['declared']}.",
            "",
            "## Claim boundary",
            "",
        ]
    )
    lines.extend(f"- {statement}" for statement in report["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dense-results", type=Path, required=True)
    parser.add_argument("--out-results", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    dense = json.loads(args.dense_results.read_text(encoding="utf-8"))
    manifest, documents, _ = load_index(args.index)
    lexical = lexical_results(suite, manifest, documents)
    report = comparison_report(suite, manifest, dense, lexical)
    write_json(args.out_results, lexical)
    write_json(args.out_report, report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_comparison(report), encoding="utf-8")
    print(json.dumps({"report": str(args.out_report), "results": str(args.out_results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
