from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_provider_poc.py"
SUITE = ROOT / "fixtures/provider-poc/acceptance-suite.v1.json"
RESULTS = ROOT / "fixtures/provider-poc/synthetic-provider-results.pass.json"


class ProviderPocCheckerTest(unittest.TestCase):
    @staticmethod
    def load_module():
        spec = importlib.util.spec_from_file_location("provider_poc_check", TOOL)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def setUp(self) -> None:
        self.module = self.load_module()
        self.suite = json.loads(SUITE.read_text(encoding="utf-8"))
        self.results = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_complete_synthetic_run_passes_and_reports_q9_boundary(self) -> None:
        report = self.module.build_report(self.suite, self.results)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["acceptance_passed"], 9)
        self.assertEqual(report["summary"]["diagnostics_passed"], 2)
        self.assertTrue(report["non_cherry_pick"]["complete"])
        q9 = next(row for row in report["cases"] if row["case_id"] == "Q9")
        self.assertEqual(q9["outcome"], "expected_boundary_failure")
        self.assertTrue(q9["boundary"]["required_absent"]["passed"])

    def test_missing_case_fails_non_cherry_pick_gate(self) -> None:
        partial = deepcopy(self.results)
        partial["calls"] = [call for call in partial["calls"] if call["case_id"] != "Q8"]
        report = self.module.build_report(self.suite, partial)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["non_cherry_pick"]["missing_case_ids"], ["Q8"])

    def test_acceptance_scope_has_frozen_q1_q9_denominator(self) -> None:
        report = self.module.build_report(self.suite, self.results, "acceptance")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["scope"], "acceptance")
        self.assertEqual(report["non_cherry_pick"]["declared_cases"], 9)
        self.assertEqual(
            report["non_cherry_pick"]["ignored_out_of_scope_case_ids"], ["S1", "S2"]
        )
        self.assertEqual(report["summary"]["diagnostics_total"], 0)

    def test_actual_pointer_results_and_jsonl_call_envelopes_are_accepted(self) -> None:
        pointer_results = deepcopy(self.results)
        for call in pointer_results["calls"]:
            legacy = call["response"]
            output = {
                "schema_version": "livefire.rag.semantic-result/1",
                "kind": "pointer",
                "tool": legacy["tool"],
                "pointers": legacy["rankings"][0]["candidates"],
            }
            call["response"] = {
                "protocol": "livefire.tool/1",
                "id": f"call-{call['case_id']}",
                "result": {"response_kind": "call", "output": output},
            }
        report = self.module.build_report(self.suite, pointer_results)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["acceptance_passed"], 9)

    def test_q9_returning_both_facets_does_not_fake_expected_boundary(self) -> None:
        changed = deepcopy(self.results)
        q9 = next(call for call in changed["calls"] if call["case_id"] == "Q9")
        candidates = q9["response"]["rankings"][0]["candidates"]
        candidates.append(
            {
                "rank": 4,
                "command_id": "q9-upload",
                "preview": "python s3-upload.py --bucket frothlywebcode --file archive.tar.gz",
            }
        )
        report = self.module.build_report(self.suite, changed)
        q9_result = next(row for row in report["cases"] if row["case_id"] == "Q9")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(q9_result["outcome"], "boundary_not_reproduced")

    def test_missing_mandatory_hard_negative_fails(self) -> None:
        changed = deepcopy(self.results)
        q7 = next(call for call in changed["calls"] if call["case_id"] == "Q7")
        q7["response"]["rankings"][0]["candidates"][4]["preview"] = (
            "web_admin ListAccessKeys AccessDenied"
        )
        report = self.module.build_report(self.suite, changed)
        q7_result = next(row for row in report["cases"] if row["case_id"] == "Q7")
        self.assertEqual(report["status"], "fail")
        self.assertFalse(q7_result["hard_negatives"][0]["passed"])

    def test_markdown_lists_every_case(self) -> None:
        report = self.module.build_report(self.suite, self.results)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(self.module.markdown_report(report), encoding="utf-8")
            text = path.read_text(encoding="utf-8")
        for case_id in [f"Q{index}" for index in range(1, 10)] + ["S1", "S2"]:
            self.assertIn(f"| {case_id} |", text)


if __name__ == "__main__":
    unittest.main()
