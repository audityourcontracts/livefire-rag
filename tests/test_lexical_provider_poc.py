from __future__ import annotations

import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_lexical_provider_poc.py"


class LexicalProviderPocTest(unittest.TestCase):
    @staticmethod
    def load_module():
        spec = importlib.util.spec_from_file_location("lexical_provider_poc", TOOL)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_tokenizer_splits_camel_case_and_preserves_security_identifiers(self) -> None:
        module = self.load_module()
        self.assertEqual(
            module.tokenize("CreateAccessKey web_admin script-block"),
            ["create", "access", "key", "web_admin", "script", "block"],
        )

    def test_bm25_prefers_more_specific_matching_document(self) -> None:
        module = self.load_module()
        documents = [
            {"command_id": "b", "semantic_text": "disable firewall", "preview": "netsh"},
            {"command_id": "a", "semantic_text": "disable firewall windows", "preview": "netsh"},
            {"command_id": "c", "semantic_text": "read a file", "preview": "type"},
        ]
        corpus = module.Bm25Corpus(documents)
        ranked = corpus.search("disable firewall windows", 3)
        self.assertEqual(ranked[0][1]["command_id"], "a")
        self.assertGreater(ranked[0][0], ranked[1][0])

    def test_equal_scores_use_command_id_ascending_tie_break(self) -> None:
        module = self.load_module()
        documents = [
            {"command_id": "cmd-z", "semantic_text": "same words", "preview": ""},
            {"command_id": "cmd-a", "semantic_text": "same words", "preview": ""},
            {"command_id": "cmd-m", "semantic_text": "unrelated", "preview": ""},
        ]
        ranked = module.Bm25Corpus(documents).search("same words", 3)
        self.assertEqual([row[1]["command_id"] for row in ranked[:2]], ["cmd-a", "cmd-z"])
        zero_ranked = module.Bm25Corpus(documents).search("absent", 3)
        self.assertEqual(
            [row[1]["command_id"] for row in zero_ranked],
            ["cmd-a", "cmd-m", "cmd-z"],
        )

    def test_comparison_treats_hard_negative_exposure_as_weakness(self) -> None:
        module = self.load_module()
        suite = json.loads(
            (ROOT / "fixtures/provider-poc/acceptance-suite.v1.json").read_text()
        )
        dense = json.loads(
            (ROOT / "fixtures/provider-poc/synthetic-provider-results.pass.json").read_text()
        )
        lexical = deepcopy(dense)
        for case_id, hard_negative_id in (
            ("Q4", "q4-hard-negative"),
            ("Q7", "q7-hard-negative"),
        ):
            call = next(row for row in lexical["calls"] if row["case_id"] == case_id)
            candidates = call["response"]["rankings"][0]["candidates"]
            candidates[:] = [row for row in candidates if row["command_id"] != hard_negative_id]
        manifest = {
            "component": {"id": "index", "version": "1", "sha256": "0" * 64},
            "documents_count": 3806,
            "objects": {"documents": {"sha256": "1" * 64}},
        }
        report = module.comparison_report(suite, manifest, dense, lexical)
        self.assertEqual(report["hard_negative_exposure_summary"]["preferred_direction"], "lower_is_better")
        self.assertEqual(report["hard_negative_exposure_summary"]["dense_exposed"], 2)
        self.assertEqual(report["hard_negative_exposure_summary"]["lexical_exposed"], 0)
        q7 = next(row for row in report["cases"] if row["case_id"] == "Q7")
        self.assertEqual(
            q7["comparison"], "both_full_lexical_lower_hard_negative_exposure"
        )
        q9 = next(row for row in report["cases"] if row["case_id"] == "Q9")
        self.assertEqual(q9["comparison"], "boundary_diagnostic")


if __name__ == "__main__":
    unittest.main()
