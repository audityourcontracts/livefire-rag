from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from livefire_rag.evidence_builder import _build_evidence_pack_for_test
from livefire_rag.canonical import (
    artifact_ref, canonical_json_bytes, canonical_sha256_omitting, write_canonical_json,
)
from livefire_rag.evidence_pilot import (
    EvidencePilotError,
    build_evidence_pilot_sample,
    pilot_index_binding,
    verify_evidence_pilot_sample,
)
from livefire_rag.evidence_projection import projection_policy_ref
from livefire_rag.evidence_index import EvidenceIndex, promote_evidence_pack, verify_promoted_evidence_index
from tests.test_evidence_index import PROFILE, SOURCE_ADMISSION, fake_embed


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"
SNAPSHOT = {"id": "test.snapshot", "version": "1", "sha256": "1" * 64}


def _rows(count: int, *, relation: str):
    for index in range(count):
        typed = {
            "ocsf": {"time": 1_700_000_000_000 + index, "activity_id": 1, "class_uid": 1007},
            "process": {"cmd_line": f"tool --structural-id {index}", "name": "tool"},
            "actor": {"user": {"name": "alice"}},
        }
        if relation == "ocsf_api_activity":
            typed = {
                "ocsf": {"time": 1_700_000_000_000 + index, "activity_id": 1, "class_uid": 6003},
                "api": {"operation": f"operation-{index}", "service": {"name": "example"}},
                "actor": {"user": {"name": "alice"}},
            }
        yield {
            "event_id": f"{relation}-{index}", "typed_event_json": typed,
            "support_ref": f"support:{relation}:{index}",
            "source_object_sha256": ("a" if relation == "ocsf_process_activity" else "b") * 64,
            "row_group": 0, "row_ordinal": index,
        }


class EvidencePilotTests(unittest.TestCase):
    def test_builds_deterministic_census_and_stratified_sample_with_occurrence_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "projection"
            _build_evidence_pack_for_test(
                pack,
                row_sources={
                    "ocsf_process_activity": _rows(2005, relation="ocsf_process_activity"),
                    "ocsf_api_activity": _rows(3, relation="ocsf_api_activity"),
                },
                index_id="test.projection", version="1", source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), batch_size=128,
            )
            first = build_evidence_pilot_sample(
                pack, root / "pilot-a", component_id="test.pilot", version="1",
                sdk_specs=SDK_SPECS,
            )
            second = build_evidence_pilot_sample(
                pack, root / "pilot-b", component_id="test.pilot", version="1",
                sdk_specs=SDK_SPECS,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["source_counts"], {"documents": 2008, "occurrences": 2008})
            self.assertEqual(first["selected_counts"], {"documents": 2003, "occurrences": 2003})
            self.assertEqual(first["scope_status"], "sample_only_not_corpus_coverage")
            self.assertFalse(first["closure"]["corpus_miss_definitive"])
            self.assertEqual(
                (root / "pilot-a" / "documents.jsonl").read_bytes(),
                (root / "pilot-b" / "documents.jsonl").read_bytes(),
            )
            selection = [json.loads(line) for line in (root / "pilot-a" / "selection.jsonl").read_text().splitlines()]
            reasons = {row["relation"]: row["selection_reason"] for row in selection}
            self.assertEqual(reasons["ocsf_api_activity"], "relation_census")
            self.assertEqual(reasons["ocsf_process_activity"], "relation_stratified_hash_min")
            self.assertEqual(verify_evidence_pilot_sample(root / "pilot-a", projection_pack=pack, sdk_specs=SDK_SPECS), first)
            binding = pilot_index_binding(first)
            self.assertEqual(binding["selected_document_count"], 2003)
            self.assertFalse(binding["corpus_miss_definitive"])

            with (root / "pilot-a" / "documents.jsonl").open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaisesRegex(EvidencePilotError, "artifact mismatch"):
                verify_evidence_pilot_sample(root / "pilot-a", sdk_specs=SDK_SPECS)

    def test_rejects_resealed_selection_metadata_that_does_not_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "projection"
            _build_evidence_pack_for_test(
                pack, row_sources={"ocsf_process_activity": _rows(4, relation="ocsf_process_activity")},
                index_id="test.projection", version="1", source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), batch_size=2,
            )
            pilot = root / "pilot"
            build_evidence_pilot_sample(pack, pilot, component_id="test.pilot", version="1", sdk_specs=SDK_SPECS)
            rows = [json.loads(line) for line in (pilot / "selection.jsonl").read_text().splitlines()]
            rows[0]["rank_sha256"] = "f" * 64
            (pilot / "selection.jsonl").write_bytes(b"".join(
                canonical_json_bytes(row, newline=True) for row in rows
            ))
            manifest = json.loads((pilot / "manifest.json").read_text())
            selection_ref = artifact_ref(pilot / "selection.jsonl", "selection.jsonl", "application/x-ndjson")
            manifest["objects"]["selection"] = selection_ref
            locked = [manifest["objects"][role] for role in manifest["objects"] if role != "object_lock"]
            locked.sort(key=lambda item: item["path"])
            write_canonical_json(pilot / "objects.lock.json", {"schema_version": "livefire.object-lock/1", "objects": locked})
            manifest["objects"]["object_lock"] = artifact_ref(pilot / "objects.lock.json", "objects.lock.json", "application/json")
            manifest["component"]["sha256"] = canonical_sha256_omitting(manifest, ("component", "sha256"))
            write_canonical_json(pilot / "manifest.json", manifest)
            with self.assertRaisesRegex(EvidencePilotError, "selection metadata does not replay"):
                verify_evidence_pilot_sample(pilot, projection_pack=pack, sdk_specs=SDK_SPECS)

    def test_promotes_and_queries_pilot_with_explicit_partial_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "projection"
            _build_evidence_pack_for_test(
                pack, row_sources={"ocsf_process_activity": _rows(4, relation="ocsf_process_activity")},
                index_id="test.projection", version="1", source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), batch_size=2,
            )
            pilot = root / "pilot"
            pilot_manifest = build_evidence_pilot_sample(
                pack, pilot, component_id="test.pilot", version="1", sdk_specs=SDK_SPECS,
            )
            manifest = promote_evidence_pack(
                pack, root / "index", relation_sources=(), source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
                embedding_profile=PROFILE,
                embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
                embedding_profile_version="1", embedder=fake_embed,
                embedding_conformance_fixture=REPOSITORY / "fixtures/generic-evidence-embedding-conformance.v1.json",
                source_admission_receipt=SOURCE_ADMISSION,
                index_id="test.pilot.index", version="1", pilot_sample=pilot,
                resume_dir=root / "resume", batch_size=2,
            )
            self.assertEqual(manifest["pilot_sample"], pilot_index_binding(pilot_manifest))
            self.assertEqual(
                verify_promoted_evidence_index(root / "index", pilot_sample=pilot, sdk_specs=SDK_SPECS),
                manifest,
            )
            index = EvidenceIndex.open(root / "index", sdk_specs=SDK_SPECS)
            try:
                document = index.connection.execute(
                    "SELECT semantic_projection.text FROM evidence_documents ORDER BY document_id LIMIT 1"
                ).fetchone()[0]
                result = index.search_dense({
                    "schema_version": "livefire.rag.evidence-search.input/1",
                    "query": "structural tool", "top_n": 1,
                    "retrieval": {"methods": ["dense"], "fusion": "none"},
                    "filters": {},
                }, fake_embed([document])[0], max_occurrences=2)
                self.assertEqual(result["coverage"]["status"], "partial")
                self.assertIn("pilot_sample_not_corpus_coverage", result["coverage"]["reason_codes"])
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
