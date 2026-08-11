from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/evaluate_fact_evidence.py"
FIXTURE = ROOT / "fixtures/fact-evidence-synthetic"


class FactEvidenceEvaluatorTest(unittest.TestCase):
    @staticmethod
    def load_module():
        spec = importlib.util.spec_from_file_location("fact_eval", TOOL)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_synthetic_candidate_improves_over_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--inventory",
                    str(FIXTURE / "inventory.json"),
                    "--queries",
                    str(FIXTURE / "queries.jsonl"),
                    "--candidate-universes",
                    str(FIXTURE / "candidate-universes.jsonl"),
                    "--qrels",
                    str(FIXTURE / "qrels.jsonl"),
                    "--hard-negatives",
                    str(FIXTURE / "hard-negatives.jsonl"),
                    "--candidate",
                    str(FIXTURE / "candidate-rankings.jsonl"),
                    "--baseline",
                    str(FIXTURE / "baseline-rankings.jsonl"),
                    "--gates",
                    str(FIXTURE / "gates.json"),
                    "--bootstrap-samples",
                    "500",
                    "--seed",
                    "7",
                    "--out",
                    str(report_path),
                ],
                check=True,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "livefire.rag.evidence-benchmark-comparison/1")
        self.assertEqual(report["promotion"]["status"], "pass")
        self.assertEqual(report["candidate"]["coverage"]["facts_total"], 5)
        self.assertEqual(report["candidate"]["coverage"]["eligible_facts"], 4)
        self.assertEqual(report["candidate"]["coverage"]["evaluated_facts"], 4)
        self.assertGreater(report["comparison"]["ndcg_at_20"]["delta"], 0)
        self.assertGreaterEqual(
            report["candidate"]["macro"]["hard_negative_triplet_accuracy"],
            report["baseline"]["macro"]["hard_negative_triplet_accuracy"],
        )

    def test_duplicate_ranked_document_is_rejected(self) -> None:
        module = self.load_module()
        rows = [
            {"query_id": "q", "document_id": "d", "rank": 1},
            {"query_id": "q", "document_id": "d", "rank": 2},
        ]
        with self.assertRaises(module.EvaluationError):
            module.validate_rankings(rows, "test")

    def test_macro_average_uses_fact_not_query_surface(self) -> None:
        module = self.load_module()
        queries = [
            {"query_id": "q1a", "fact_id": "f1", "eligibility": "eligible"},
            {"query_id": "q1b", "fact_id": "f1", "eligibility": "eligible"},
            {"query_id": "q2", "fact_id": "f2", "eligibility": "eligible"},
        ]
        qrels = [
            {"query_id": "q1a", "document_id": "p1", "relevance": 3},
            {"query_id": "q1b", "document_id": "p1", "relevance": 3},
            {"query_id": "q1b", "document_id": "n1", "relevance": 0},
            {"query_id": "q2", "document_id": "p2", "relevance": 3},
        ]
        rankings = [
            {"query_id": "q1a", "document_id": "p1", "rank": 1},
            {"query_id": "q1b", "document_id": "n1", "rank": 1},
            {"query_id": "q2", "document_id": "p2", "rank": 1},
        ]
        result = module.evaluate_rankings(queries, qrels, [], rankings, 1)
        self.assertAlmostEqual(result["macro"]["ndcg_at_1"], 0.75)
        self.assertEqual(result["coverage"]["facts_total"], 2)
        self.assertEqual(result["coverage"]["queries_total"], 3)

    def test_duplicate_evidence_groups_do_not_inflate_recall(self) -> None:
        module = self.load_module()
        queries = [{"query_id": "q", "fact_id": "f", "eligibility": "eligible"}]
        qrels = [
            {
                "query_id": "q",
                "document_id": "duplicate-1",
                "relevance": 3,
                "evidence_group_id": "same-evidence",
            },
            {
                "query_id": "q",
                "document_id": "duplicate-2",
                "relevance": 3,
                "evidence_group_id": "same-evidence",
            },
        ]
        rankings = [
            {"query_id": "q", "document_id": "duplicate-1", "rank": 1},
            {"query_id": "q", "document_id": "duplicate-2", "rank": 2},
        ]
        result = module.evaluate_rankings(queries, qrels, [], rankings, 20)
        self.assertEqual(result["per_query"][0]["relevant_documents"], 1)
        self.assertEqual(result["per_query"][0]["recall_at_20"], 1.0)
        self.assertEqual(result["per_query"][0]["ndcg_at_20"], 1.0)

    def test_normative_ranking_fields_are_supported(self) -> None:
        module = self.load_module()
        queries = [{"query_id": "q", "fact_id": "f", "eligibility": "eligible"}]
        qrels = [
            {"query_id": "q", "command_id": "positive", "relevance_grade": 3},
            {"query_id": "q", "command_id": "negative", "relevance_grade": 0},
        ]
        negatives = [{"query_id": "q", "negative_command_id": "negative"}]
        rankings = [
            {
                "query_id": "q",
                "command_id": "positive",
                "rank": 1,
                "distance_millionths": 100000,
            },
            {
                "query_id": "q",
                "command_id": "negative",
                "rank": 2,
                "distance_millionths": 300000,
            },
        ]
        module.validate_rankings(rankings, "normative")
        result = module.evaluate_rankings(queries, qrels, negatives, rankings, 20)
        self.assertEqual(result["macro"]["ndcg_at_20"], 1.0)
        self.assertAlmostEqual(result["macro"]["median_hard_negative_margin"], 0.2)

    def test_missing_query_surface_scores_zero_and_reduces_execution_coverage(self) -> None:
        module = self.load_module()
        queries = [
            {"query_id": "q1", "fact_id": "f", "eligibility": "eligible"},
            {"query_id": "q2", "fact_id": "f", "eligibility": "eligible"},
        ]
        qrels = [
            {"query_id": "q1", "document_id": "p", "relevance": 3},
            {"query_id": "q2", "document_id": "p", "relevance": 3},
        ]
        rankings = [{"query_id": "q1", "document_id": "p", "rank": 1}]
        result = module.evaluate_rankings(queries, qrels, [], rankings, 20)
        self.assertEqual(result["macro"]["ndcg_at_20"], 0.5)
        self.assertEqual(result["coverage"]["eligible_query_execution_rate"], 0.5)
        self.assertEqual(result["per_query"][1]["status"], "failed")

    def test_unjudged_top_result_fails_closed(self) -> None:
        module = self.load_module()
        queries = [{"query_id": "q", "fact_id": "f", "eligibility": "eligible"}]
        qrels = [{"query_id": "q", "document_id": "p", "relevance": 3}]
        rankings = [
            {"query_id": "q", "document_id": "p", "rank": 1},
            {"query_id": "q", "document_id": "unjudged", "rank": 2},
        ]
        with self.assertRaises(module.EvaluationError):
            module.evaluate_rankings(queries, qrels, [], rankings, 20)

    def test_missing_triplet_score_counts_as_failure_and_missing_coverage(self) -> None:
        module = self.load_module()
        queries = [{"query_id": "q", "fact_id": "f", "eligibility": "eligible"}]
        qrels = [
            {"query_id": "q", "document_id": "p", "relevance": 3},
            {"query_id": "q", "document_id": "n", "relevance": 0},
        ]
        negatives = [
            {"query_id": "q", "positive_command_id": "p", "negative_command_id": "n"}
        ]
        rankings = [{"query_id": "q", "document_id": "p", "rank": 1, "distance": 0.1}]
        result = module.evaluate_rankings(queries, qrels, negatives, rankings, 20)
        self.assertEqual(result["macro"]["hard_negative_triplet_accuracy"], 0.0)
        self.assertEqual(result["macro"]["hard_negative_triplet_coverage"], 0.0)

    def test_native_higher_is_better_scores_support_triplets(self) -> None:
        module = self.load_module()
        queries = [{"query_id": "q", "fact_id": "f", "eligibility": "eligible"}]
        qrels = [
            {"query_id": "q", "document_id": "p", "relevance": 3},
            {"query_id": "q", "document_id": "n", "relevance": 0},
        ]
        negatives = [
            {"query_id": "q", "positive_command_id": "p", "negative_command_id": "n"}
        ]
        rankings = [
            {
                "query_id": "q", "document_id": "p", "rank": 1,
                "score_kind": "bm25", "score_direction": "higher_is_better", "score": 4.0,
            },
            {
                "query_id": "q", "document_id": "n", "rank": 2,
                "score_kind": "bm25", "score_direction": "higher_is_better", "score": 2.5,
            },
        ]
        module.validate_rankings(rankings, "native")
        result = module.evaluate_rankings(queries, qrels, negatives, rankings, 20)
        self.assertEqual(result["macro"]["hard_negative_triplet_accuracy"], 1.0)
        self.assertAlmostEqual(result["macro"]["median_hard_negative_margin"], 1.5)

    def test_ranking_score_order_must_match_declared_direction(self) -> None:
        module = self.load_module()
        rows = [
            {
                "query_id": "q", "document_id": "first", "rank": 1,
                "score_kind": "bm25", "score_direction": "higher_is_better", "score": 1.0,
            },
            {
                "query_id": "q", "document_id": "second", "rank": 2,
                "score_kind": "bm25", "score_direction": "higher_is_better", "score": 2.0,
            },
        ]
        with self.assertRaises(module.EvaluationError):
            module.validate_rankings(rows, "wrong-order")

    def test_expected_top_k_cardinality_prevents_result_suppression(self) -> None:
        module = self.load_module()
        queries = [
            {
                "query_id": "q", "fact_id": "f", "eligibility": "eligible",
                "expected_top_k_cardinality": 2,
            }
        ]
        rankings = [{"query_id": "q", "document_id": "p", "rank": 1}]
        with self.assertRaises(module.EvaluationError):
            module.validate_ranking_cardinality(queries, rankings, "suppressed", 20)

    def test_every_active_query_requires_a_declared_hard_negative(self) -> None:
        module = self.load_module()
        queries = module.load_rows(FIXTURE / "queries.jsonl")
        qrels = module.load_rows(FIXTURE / "qrels.jsonl")
        negatives = [
            row
            for row in module.load_rows(FIXTURE / "hard-negatives.jsonl")
            if row["query_id"] != "cloud-access-key"
        ]
        with self.assertRaises(module.EvaluationError):
            module.validate_benchmark_rows(queries, qrels, negatives)

    def test_real_suite_requires_the_exact_three_query_surfaces(self) -> None:
        module = self.load_module()
        inventory = {
            "suite_contract": "livefire-23-cloud-53-bots-v1",
            "atoms": [
                {
                    "atom_id": "fact",
                    "eligibility": "eligible_native",
                    "cohort": "cloud",
                    "resampling_cluster_id": "cluster",
                    "incident_id": "incident",
                }
            ],
        }
        queries = [
            {
                "query_id": f"q{index}",
                "fact_id": "fact",
                "eligibility": "eligible",
                "status": "active",
                "surface": surface,
                "cohort": "cloud",
                "resampling_cluster_id": "cluster",
                "incident_id": "incident",
            }
            for index, surface in enumerate(
                ["analyst_question", "terse_soc", "canonical_fact"], 1
            )
        ]
        with self.assertRaises(module.EvaluationError):
            module.validate_inventory_queries(inventory, queries, 3)

    def test_strict_ranking_rejects_string_correctness_flags(self) -> None:
        module = self.load_module()
        row = {
            "schema_version": "livefire.rag.evidence-ranking-row/1",
            "system_id": "candidate",
            "query_id": "q",
            "command_id": "p",
            "rank": 1,
            "score_kind": "cosine_distance",
            "score_direction": "lower_is_better",
            "distance_millionths": 100000,
            "pointer_resolved": "false",
            "filter_compliant": True,
        }
        with self.assertRaises(module.EvaluationError):
            module.validate_rankings(
                [row], "strict", known_query_ids={"q"}, strict_schema=True
            )


if __name__ == "__main__":
    unittest.main()
