from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from livefire_rag.canonical import canonical_json_bytes
from livefire_rag.evidence_builder import _build_evidence_pack_for_test
from livefire_rag.evidence_bundle import package_evidence_provider_bundle
from livefire_rag.evidence_index import promote_evidence_pack
from livefire_rag.evidence_loadout import (
    prepare_evidence_loadout,
    validate_evidence_wire,
)
from livefire_rag.evidence_pilot import build_evidence_pilot_sample
from livefire_rag.evidence_projection import projection_policy_ref
from tests.test_evidence_index import PROFILE, SOURCE_ADMISSION, fake_embed
from tests.test_evidence_pilot import SNAPSHOT, _rows
from tests.test_generic_evidence_provider import build_index


ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT.parent / "livefire-sdk"
SDK_SPECS = SDK_ROOT / "specs"


class EvidenceLoadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp())
        self.index = self.temp / "index"
        self.index.mkdir()
        build_index(self.index)
        self.bundle = self.temp / "bundle"
        package_evidence_provider_bundle(ROOT, self.bundle, SDK_SPECS)
        self.queries = [
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "disable firewall",
                "top_n": 2,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
            },
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "missing",
                "top_n": 2,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {"attribute_predicates": [{
                    "namespace": "ocsf", "path": "/missing",
                    "operator": "eq", "value": "absent",
                }]},
            },
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _prepare(self) -> Path:
        destination = self.temp / "loadout"
        prepare_evidence_loadout(
            self.index, self.bundle, destination,
            sdk_specs=SDK_SPECS, queries=self.queries,
            embedding_endpoint="http://127.0.0.1:65534",
        )
        return destination

    def test_prepare_refuses_test_only_fast_index_before_opening(self) -> None:
        (self.index / "index.json").write_text(
            json.dumps({
                "schema_version": "livefire.rag.fast-index/4",
                "test_only": True,
            }),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError, "test-only indexes cannot be prepared as provider loadouts"
        ):
            self._prepare()
        self.assertFalse((self.temp / "loadout").exists())

    def test_prepare_is_byte_deterministic_and_explicitly_local_test(self) -> None:
        destination = self._prepare()
        first = {
            name: (destination / name).read_bytes()
            for name in (
                "index-admission-receipt.json", "tool-binding-lock.json",
                "requests.jsonl", "loadout.json",
            )
        }
        receipt = json.loads(first["index-admission-receipt.json"])
        loadout = json.loads(first["loadout.json"])
        self.assertTrue(receipt["authority_signature"].startswith("local-test:"))
        self.assertEqual(loadout["admission_status"], "local_test_only_not_production_admitted")
        self.assertFalse(receipt["checks"]["deterministic_rebuild"])
        self.assertEqual(
            receipt["reason_codes"],
            ["local_test_only_not_production_admitted"],
        )
        shutil.rmtree(destination)
        self._prepare()
        self.assertEqual(first, {name: (destination / name).read_bytes() for name in first})

    def test_sdk_invoke_then_validate_and_export_hydration_requests(self) -> None:
        binary = SDK_ROOT / "target/debug/livefire-sdk"
        if not binary.is_file():
            self.skipTest("livefire-sdk debug binary is unavailable")
        loadout = self._prepare()
        wire = self.temp / "wire.jsonl"
        completed = subprocess.run(
            [
                str(binary), "--specs", str(SDK_SPECS), "invoke",
                "--program", str(self.bundle / "bin/livefire-rag-evidence-provider"),
                "--requests", str(loadout / "requests.jsonl"),
                "--timeout-ms", "30000",
            ],
            cwd=self.temp, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        wire.write_text(completed.stdout, encoding="utf-8")
        report_path = self.temp / "wire-report.json"
        hydration_path = self.temp / "hydration-requests.jsonl"
        report = validate_evidence_wire(
            wire, loadout, sdk_specs=SDK_SPECS,
            report_path=report_path, hydration_requests_path=hydration_path,
        )
        self.assertTrue(report["valid"])
        self.assertEqual([row["kind"] for row in report["calls"]], ["pointer", "miss"])
        hydration = [json.loads(line) for line in hydration_path.read_text().splitlines()]
        self.assertEqual(len(hydration), report["unique_hydration_pointer_count"])
        self.assertGreater(len(hydration), 0)
        self.assertEqual(
            hydration[0]["schema_version"],
            "livefire.rag.local-test-hydration-request/1",
        )
        self.assertIn("source_pointer", hydration[0])
        self.assertIn("discoveries", hydration[0])

    def test_sealed_pilot_invoke_preserves_non_corpus_scope_end_to_end(self) -> None:
        binary = SDK_ROOT / "target/debug/livefire-sdk"
        if not binary.is_file():
            self.skipTest("livefire-sdk debug binary is unavailable")
        pack = self.temp / "pilot-projection"
        _build_evidence_pack_for_test(
            pack,
            row_sources={
                "ocsf_process_activity": _rows(
                    4, relation="ocsf_process_activity"
                )
            },
            index_id="test.loadout.projection", version="1",
            source_snapshot=SNAPSHOT, projection_policy=projection_policy_ref(),
            batch_size=2,
        )
        pilot = self.temp / "pilot"
        build_evidence_pilot_sample(
            pack, pilot, component_id="test.loadout.pilot", version="1",
            sdk_specs=SDK_SPECS,
        )
        pilot_index = self.temp / "pilot-index"
        manifest = promote_evidence_pack(
            pack, pilot_index, relation_sources=(), source_snapshot=SNAPSHOT,
            projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
            embedding_profile=PROFILE,
            embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
            embedding_profile_version="1", embedder=fake_embed,
            embedding_conformance_fixture=(
                ROOT / "fixtures/generic-evidence-embedding-conformance.v1.json"
            ),
            source_admission_receipt=SOURCE_ADMISSION,
            index_id="test.loadout.pilot-index", version="1",
            pilot_sample=pilot, resume_dir=self.temp / "pilot-resume", batch_size=2,
        )
        queries = [
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "tool structural-id", "top_n": 2,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
            },
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "tool structural-id", "top_n": 2,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {"attribute_predicates": [{
                    "namespace": "ocsf", "path": "/missing",
                    "operator": "eq", "value": "absent",
                }]},
            },
        ]
        loadout = self.temp / "pilot-loadout"
        prepared = prepare_evidence_loadout(
            pilot_index, self.bundle, loadout, sdk_specs=SDK_SPECS,
            queries=queries, embedding_endpoint="http://127.0.0.1:65534",
        )
        self.assertEqual(prepared["pilot_sample"], manifest["pilot_sample"])
        receipt = json.loads((loadout / "index-admission-receipt.json").read_text())
        self.assertEqual(receipt["reason_codes"], [
            "local_test_only_not_production_admitted",
            "pilot_sample_not_corpus_coverage",
        ])
        completed = subprocess.run(
            [
                str(binary), "--specs", str(SDK_SPECS), "invoke",
                "--program", str(self.bundle / "bin/livefire-rag-evidence-provider"),
                "--requests", str(loadout / "requests.jsonl"),
                "--timeout-ms", "30000",
            ],
            cwd=self.temp, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        wire = self.temp / "pilot-wire.jsonl"
        wire.write_text(completed.stdout, encoding="utf-8")
        report = validate_evidence_wire(wire, loadout, sdk_specs=SDK_SPECS)
        self.assertEqual(report["pilot_sample"], manifest["pilot_sample"])
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        pointer = responses[2]["result"]["output"]
        miss = responses[3]["result"]["output"]
        self.assertEqual([pointer["kind"], miss["kind"]], ["pointer", "miss"])
        for output in (pointer, miss):
            self.assertEqual(output["coverage"]["status"], "partial")
            self.assertIn(
                "pilot_sample_not_corpus_coverage",
                output["coverage"]["reason_codes"],
            )
        self.assertIn("not a corpus-wide miss", miss["miss"]["message"])

    def test_wire_validator_rejects_call_identity_tampering(self) -> None:
        loadout = self._prepare()
        requests = [json.loads(line) for line in (loadout / "requests.jsonl").read_text().splitlines()]
        index_manifest = json.loads((self.index / "manifest.json").read_text())
        responses = [
            {"protocol": "livefire.tool/1", "id": "1", "result": {
                "response_kind": "handshake", "provider": requests[1]["params"]["provider"],
                "protocol": "livefire.tool/1", "tools": requests[1]["params"]["tools"],
                "accepted_index_formats": [json.loads((loadout / "tool-binding-lock.json").read_text())["index_format"]],
            }},
            {"protocol": "livefire.tool/1", "id": "2", "result": {
                "response_kind": "open", "session_id": "fixture",
                "binding_lock_sha256": requests[1]["params"]["binding_lock_sha256"],
            }},
        ]
        for request in requests[2:-2]:
            responses.append({"protocol": "livefire.tool/1", "id": request["id"], "result": {
                "response_kind": "call", "output": {
                    "schema_version": "livefire.rag.evidence-search.output/1",
                    "kind": "miss", "tool": "evidence.search",
                    "index": index_manifest["component"],
                    "source_snapshots": index_manifest["source_snapshots"],
                    "query_sha256": "f" * 64,
                    "coverage": {"status": "complete", "indexed_documents": 2, "eligible_documents": 0, "eligible_occurrences": 0, "definitive": False, "reason_codes": []},
                    "selection": {"requested_top_n": 2, "returned_count": 0, "eligible_count": 0, "exhausted": True, "deterministic": True, "tie_break": "ranking_score_desc_document_id_asc"},
                    "miss": {"reason": "no_eligible_occurrences", "message": "none"},
                },
            }})
        responses.extend([
            {"protocol": "livefire.tool/1", "id": requests[-2]["id"], "result": {"response_kind": "health", "status": "ready", "binding_lock_sha256": requests[1]["params"]["binding_lock_sha256"]}},
            {"protocol": "livefire.tool/1", "id": requests[-1]["id"], "result": {"response_kind": "close", "closed": True}},
        ])
        wire = self.temp / "tampered-wire.jsonl"
        wire.write_bytes(b"".join(canonical_json_bytes(row, newline=True) for row in responses))
        with self.assertRaisesRegex(ValueError, "query identity"):
            validate_evidence_wire(wire, loadout, sdk_specs=SDK_SPECS)


if __name__ == "__main__":
    unittest.main()
