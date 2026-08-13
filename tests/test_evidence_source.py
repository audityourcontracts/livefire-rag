from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from livefire_rag.canonical import sha256_file
from livefire_rag.evidence_builder import (
    EvidencePackError,
    build_evidence_pack,
    source_record_profile_material,
    source_record_profile_ref,
)
from livefire_rag.evidence_projection import (
    RELATION_DOCUMENT_KINDS,
    projection_policy_material,
    projection_policy_ref,
)
from livefire_rag.evidence_source import SnapshotAdmissionError, admit_typed_snapshot


class EvidenceSourceAdmissionTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        semantic = root / "semantic"
        semantic.mkdir(parents=True)
        objects = []
        connection = duckdb.connect()
        try:
            for relation in sorted(RELATION_DOCUMENT_KINDS):
                path = semantic / f"{relation}.parquet"
                payload = json.dumps(
                    {
                        "semantic_class": relation.removeprefix("ocsf_"),
                        "ocsf": {"time": 1_700_000_000_000, "activity_name": "Observe"},
                    }
                )
                connection.execute(
                    "CREATE OR REPLACE TABLE fixture AS SELECT ?::VARCHAR AS event_id, "
                    "?::VARCHAR AS typed_event_json, ?::VARCHAR AS support_ref",
                    [f"event-{relation}", payload, f"support-{relation}"],
                )
                connection.execute("COPY fixture TO ? (FORMAT parquet)", [str(path)])
                objects.append(
                    {
                        "relation": relation,
                        "path": f"semantic/{relation}.parquet",
                        "rows": 1,
                        "sha256": sha256_file(path),
                    }
                )
        finally:
            connection.close()
        typed_event_count = len(objects)
        objects.append(
            {
                "relation": "events",
                "path": "semantic/events.parquet",
                "rows": typed_event_count,
                "sha256": "c" * 64,
            }
        )
        receipt = root / "build-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "output_logical_sha256": "a" * 64,
                    "snapshot_manifest": {
                        "schema_version": 1,
                        "logical_sha256": "a" * 64,
                        "dataset_sha256": "b" * 64,
                        "mapping_pack_sha256": "d" * 64,
                        "relation_contract_sha256": "e" * 64,
                        "objects": objects,
                    },
                    "runnable_snapshot": {
                        "schema_version": 1,
                        "component": {
                            "id": "fixture.ocsf.snapshot",
                            "version": "1",
                            "sha256": "a" * 64,
                        },
                        "dataset_sha256": "b" * 64,
                        "mapping_pack": {"sha256": "d" * 64},
                        "relation_contract": {"sha256": "e" * 64},
                        "source_rows": typed_event_count,
                        "normalized_events": typed_event_count,
                    },
                    "closure": {
                        "input_rows": typed_event_count,
                        "mapped_source_records": typed_event_count,
                        "event_rows": typed_event_count,
                        "mapped_events": typed_event_count,
                        "unresolved_provenance_fields": 0,
                        "provenance_digest_mismatches": 0,
                        "rejected_malformed_records": 0,
                        "unsupported_records": 0,
                    },
                    "completeness_receipt": {
                        "normalized_snapshot_sha256": "a" * 64,
                        "dataset_sha256": "b" * 64,
                        "mapping_pack_sha256": "d" * 64,
                        "relation_contract_sha256": "e" * 64,
                        "metrics": {
                            "source_rows": typed_event_count,
                            "mapped_source_records": typed_event_count,
                            "normalized_events": typed_event_count,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return receipt

    def test_admits_complete_receipt_bound_typed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            admitted = admit_typed_snapshot(
                root,
                receipt,
                snapshot_id="fixture.ocsf.snapshot",
                snapshot_version="1",
            )
            self.assertEqual(len(admitted.relations), 18)
            self.assertEqual(sum(admitted.expected_rows.values()), 18)
            self.assertEqual(admitted.component["sha256"], "a" * 64)
            self.assertEqual(len(admitted.receipt_sha256), 64)

    def test_admission_preserves_optional_identity_bearing_component_uri(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["runnable_snapshot"]["component"]["uri"] = "urn:test:ocsf-snapshot:1"
            receipt.write_text(json.dumps(value), encoding="utf-8")
            admitted = admit_typed_snapshot(root, receipt)
            self.assertEqual(admitted.component["uri"], "urn:test:ocsf-snapshot:1")

            value["runnable_snapshot"]["component"]["uri"] = ""
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SnapshotAdmissionError, "closed runnable_snapshot.component"):
                admit_typed_snapshot(root, receipt)

    def test_rejects_missing_relation_and_changed_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["snapshot_manifest"]["objects"] = [
                item
                for item in value["snapshot_manifest"]["objects"]
                if item.get("relation") != "ocsf_user_inventory"
            ]
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SnapshotAdmissionError, "omits typed relations"):
                admit_typed_snapshot(
                    root,
                    receipt,
                    snapshot_id="fixture.ocsf.snapshot",
                    snapshot_version="1",
                )

    def test_rejects_snapshot_manifest_logical_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["snapshot_manifest"]["logical_sha256"] = "f" * 64
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                SnapshotAdmissionError, "manifest logical digest"
            ):
                admit_typed_snapshot(root, receipt)

    def test_builder_fences_admitted_object_digest_through_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            admitted = admit_typed_snapshot(
                root,
                receipt,
                snapshot_id="fixture.ocsf.snapshot",
                snapshot_version="1",
            )
            changed = admitted.relations[0].path
            changed.write_bytes(changed.read_bytes() + b"changed-after-admission")
            with self.assertRaisesRegex(EvidencePackError, "receipt-fenced object digest"):
                build_evidence_pack(
                    root / "pack",
                    admitted.relations,
                    index_id="fixture.evidence.pack",
                    version="1",
                    source_snapshot=admitted.component,
                    projection_policy=projection_policy_ref(),
                )
            self.assertFalse((root / "pack").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._snapshot(root)
            target = root / "semantic" / "ocsf_api_activity.parquet"
            target.write_bytes(target.read_bytes() + b"changed")
            with self.assertRaisesRegex(SnapshotAdmissionError, "digest mismatch"):
                admit_typed_snapshot(
                    root,
                    receipt,
                    snapshot_id="fixture.ocsf.snapshot",
                    snapshot_version="1",
                )

    def test_projection_policy_identity_is_stable_and_scenario_blind(self) -> None:
        reference = projection_policy_ref()
        self.assertEqual(reference, projection_policy_ref())
        self.assertEqual(reference["version"], "2")
        self.assertEqual(
            projection_policy_material(),
            json.loads((Path(__file__).parents[1] / "specs/evidence-projection-policy.v2.json").read_text()),
        )
        self.assertEqual(
            source_record_profile_material(),
            json.loads((Path(__file__).parents[1] / "specs/typed-parquet-record-profile.v1.json").read_text()),
        )
        self.assertEqual(source_record_profile_ref(), source_record_profile_ref())
        encoded = json.dumps(projection_policy_material(), sort_keys=True).lower()
        for prohibited in ("fro" + "thly", "bots" + "v3", "expected" + "-answer"):
            self.assertNotIn(prohibited, encoded)

    def test_generic_index_build_dependency_graph_has_no_scenario_literals(self) -> None:
        root = Path(__file__).parents[1]
        paths = [
            root / "src/livefire_rag/evidence_builder.py",
            root / "src/livefire_rag/evidence_projection.py",
            root / "src/livefire_rag/evidence_source.py",
            root / "specs/evidence-projection-policy.v2.json",
            root / "specs/typed-parquet-record-profile.v1.json",
        ]
        material = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
        prohibited = [
            "fro" + "thly",
            "coin" + "hive",
            "bots" + "v3",
            "h" + "door",
            "known" + "-answer",
            "expected" + "-answer",
        ]
        for literal in prohibited:
            self.assertNotIn(literal, material)


if __name__ == "__main__":
    unittest.main()
