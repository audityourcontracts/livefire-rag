from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from livefire_rag.evidence_builder import _build_evidence_pack_for_test
from livefire_rag.evidence_index import promote_evidence_pack
from livefire_rag.evidence_pilot import build_evidence_pilot_sample
from livefire_rag.evidence_pilot_eval import (
    EvidencePilotEvaluationError,
    run_evidence_pilot_evaluation,
)
from livefire_rag.evidence_projection import projection_policy_ref
from tests.test_evidence_index import PROFILE, SOURCE_ADMISSION, fake_embed
from tests.test_evidence_pilot import SNAPSHOT, _rows


ROOT = Path(__file__).resolve().parents[1]
SDK_SPECS = ROOT.parent / "livefire-sdk/specs"
QUERY_FIXTURE = ROOT / "fixtures/generic-evidence-pilot-queries.v1.json"


def query_embed(query: str, deadline_unix_ms: int):
    del deadline_unix_ms
    text = PROFILE["query_composition"].format(
        query_instruction=PROFILE["query_instruction"], query=query
    )
    return fake_embed([text])[0]


class EvidencePilotEvaluationTests(unittest.TestCase):
    def _index(self, root: Path) -> Path:
        pack = root / "projection"
        _build_evidence_pack_for_test(
            pack,
            row_sources={
                "ocsf_process_activity": _rows(
                    4, relation="ocsf_process_activity"
                )
            },
            index_id="test.evaluation.projection", version="1",
            source_snapshot=SNAPSHOT, projection_policy=projection_policy_ref(),
            batch_size=2,
        )
        pilot = root / "pilot"
        build_evidence_pilot_sample(
            pack, pilot, component_id="test.evaluation.pilot", version="1",
            sdk_specs=SDK_SPECS,
        )
        index = root / "index"
        promote_evidence_pack(
            pack, index, relation_sources=(), source_snapshot=SNAPSHOT,
            projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
            embedding_profile=PROFILE,
            embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
            embedding_profile_version="1", embedder=fake_embed,
            embedding_conformance_fixture=(
                ROOT / "fixtures/generic-evidence-embedding-conformance.v1.json"
            ),
            source_admission_receipt=SOURCE_ADMISSION,
            index_id="test.evaluation.pilot-index", version="1",
            pilot_sample=pilot, resume_dir=root / "resume", batch_size=2,
        )
        return index

    def test_runs_entire_frozen_matrix_and_seals_diagnostic_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._index(root)
            output = root / "evaluation"
            manifest = run_evidence_pilot_evaluation(
                index, QUERY_FIXTURE, output, sdk_specs=SDK_SPECS,
                embed_query=query_embed, component_id="test.pilot-evaluation",
                version="1", top_n=3,
            )
            self.assertEqual(manifest["scope_status"], "sample_only_not_corpus_coverage")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["query_count"], 15)
            self.assertEqual(report["ranking_run_count"], 45)
            self.assertEqual(report["comparison_count"], 45)
            self.assertTrue(report["returned_pointer_closure"])
            self.assertEqual(
                report["quality_claim_status"],
                "diagnostic_only_no_qrels_no_retrieval_quality_claim",
            )
            plan = json.loads((output / "execution-plan.json").read_text())
            self.assertEqual(
                [row["mode"] for row in plan["runs"][:3]],
                ["lexical", "dense", "fused"],
            )
            rankings = [
                json.loads(line)
                for line in (output / "rankings.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rankings), 45)
            for ranking in rankings:
                self.assertEqual(ranking["output"]["coverage"]["status"], "partial")
                self.assertIn(
                    "pilot_sample_not_corpus_coverage",
                    ranking["output"]["coverage"]["reason_codes"],
                )
                self.assertIn(
                    "relation_family_diagnostic_only_not_relevance", ranking
                )

    def test_rejects_fixture_drift_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self._index(root)
            fixture = json.loads(QUERY_FIXTURE.read_text())
            fixture["status"] = "edited_after_ranking"
            drifted = root / "drifted.json"
            drifted.write_text(json.dumps(fixture), encoding="utf-8")
            output = root / "evaluation"
            with self.assertRaisesRegex(
                EvidencePilotEvaluationError, "frozen digest"
            ):
                run_evidence_pilot_evaluation(
                    index, drifted, output, sdk_specs=SDK_SPECS,
                    embed_query=query_embed, component_id="test.pilot-evaluation",
                    version="1", top_n=3,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
