from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from livefire_rag.canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from livefire_rag.evidence_builder import (
    EvidencePackCorrupt,
    RelationSource,
    _build_evidence_pack_for_test as build_evidence_pack,
    _verify_evidence_pack,
    build_evidence_pack as build_admitted_evidence_pack,
    verify_evidence_pack,
)
from livefire_rag.evidence_projection import project_event, projection_policy_ref
from livefire_rag.evidence_schema import (
    GENERIC_EVIDENCE_SCHEMA_NAMES,
    _offline_registry,
    generic_schema_root,
    validate_evidence_pack_schemas,
)


ZERO_SHA = "0" * 64
SNAPSHOT = {"id": "test.ocsf.snapshot", "version": "1", "sha256": ZERO_SHA}
POLICY = projection_policy_ref()
REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"


def _row(event_id: str, event_time: int, *, command: str = "echo hello") -> dict[str, object]:
    ordinal = {"event-a": 0, "event-b": 1, "event-c": 2}.get(event_id, 0)
    return {
        "event_id": event_id,
        "typed_event_json": {
            "class_uid": 1007,
            "activity_id": 1,
            "activity_name": "Launch",
            "time": event_time,
            "process": {"cmd_line": command, "name": "sh"},
        },
        "support_ref": f"support:{event_id}",
        "source_object_sha256": "2" * 64,
        "row_group": 0,
        "row_ordinal": ordinal,
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EvidenceBuilderTests(unittest.TestCase):
    def _build(self, root: Path, name: str, rows: list[dict[str, object]]) -> dict[str, object]:
        return build_evidence_pack(
            root / name,
            row_sources={"ocsf_process_activity": rows},
            index_id="test.evidence.pack",
            version="1",
            source_snapshot=SNAPSHOT,
            projection_policy=POLICY,
            projector=project_event,
            batch_size=2,
        )

    def test_public_builder_requires_fenced_sources_and_generic_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = RelationSource("ocsf_process_activity", root / "source.parquet")
            with self.assertRaisesRegex(ValueError, "receipt-fenced object digests"):
                build_admitted_evidence_pack(
                    root / "unfenced",
                    [source],
                    index_id="test.evidence.pack",
                    version="1",
                    source_snapshot=SNAPSHOT,
                    projection_policy=POLICY,
                )

            fenced = RelationSource(
                "ocsf_process_activity",
                root / "source.parquet",
                expected_sha256="2" * 64,
                expected_rows=0,
            )
            with self.assertRaisesRegex(ValueError, "built-in generic evidence policy"):
                build_admitted_evidence_pack(
                    root / "wrong-policy",
                    [fenced],
                    index_id="test.evidence.pack",
                    version="1",
                    source_snapshot=SNAPSHOT,
                    projection_policy={**POLICY, "sha256": "3" * 64},
                )

    def test_duplicate_semantic_groups_preserve_every_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_row("event-b", 1_700_000_001_000), _row("event-a", 1_700_000_000_000)]
            manifest = self._build(root, "pack", rows)

            documents = _jsonl(root / "pack" / "documents.jsonl")
            occurrences = _jsonl(root / "pack" / "occurrences.jsonl")
            coverage = json.loads(
                (root / "pack" / "coverage-report.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["closure"]["source_record_count"], 2)
            self.assertEqual(manifest["closure"]["terminal_disposition_count"], 2)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["occurrence_count"], 2)
            self.assertEqual(len(occurrences), 2)
            self.assertEqual({row["document_ids"][0] for row in occurrences}, {documents[0]["document_id"]})
            self.assertEqual(
                {row["source_pointer"]["record_id"] for row in occurrences},
                {"event-a", "event-b"},
            )
            self.assertTrue(coverage["closure"]["all_source_records_dispositioned"])

    def test_build_is_reproducible_independent_of_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _row("event-b", 1_700_000_001_000),
                _row("event-a", 1_700_000_000_000),
                _row("event-c", 1_700_000_002_000, command="uname -a"),
            ]
            first = self._build(root, "first", rows)
            second = self._build(root, "second", list(reversed(rows)))

            self.assertEqual(first, second)
            for artifact in (
                "documents.jsonl",
                "occurrences.jsonl",
                "coverage-report.json",
                "objects.lock.json",
                "manifest.json",
            ):
                self.assertEqual((root / "first" / artifact).read_bytes(), (root / "second" / artifact).read_bytes())

    def test_component_uris_are_accepted_preserved_and_identity_bearing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = {**SNAPSHOT, "uri": "urn:test:snapshot:1"}
            policy = {**POLICY, "uri": "urn:test:projection-policy:1"}
            manifest = build_evidence_pack(
                root / "pack",
                row_sources={"ocsf_process_activity": [_row("event-a", 1_700_000_000_000)]},
                index_id="test.evidence.uri-pack",
                version="1",
                index_uri="urn:test:evidence-pack:1",
                source_snapshot=snapshot,
                projection_policy=policy,
                projector=project_event,
            )
            self.assertEqual(manifest["component"]["uri"], "urn:test:evidence-pack:1")
            self.assertEqual(manifest["source_snapshots"], [snapshot])
            self.assertEqual(manifest["projection_policy"], policy)
            occurrence = _jsonl(root / "pack" / "occurrences.jsonl")[0]
            self.assertEqual(occurrence["source_pointer"]["snapshot"], snapshot)
            self.assertEqual(occurrence["projection_policy"], policy)
            material = json.loads(json.dumps(manifest))
            identity = material["component"].pop("sha256")
            self.assertEqual(identity, canonical_sha256_omitting(manifest, ("component", "sha256")))

            changed = json.loads(json.dumps(manifest))
            changed["component"]["uri"] = "urn:test:evidence-pack:changed"
            self.assertNotEqual(identity, canonical_sha256_omitting(changed, ("component", "sha256")))

    def test_empty_component_uris_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "source_snapshot.uri"):
                build_evidence_pack(
                    root / "bad-snapshot",
                    row_sources={"ocsf_process_activity": [_row("event-a", 1)]},
                    index_id="test.evidence.pack",
                    version="1",
                    source_snapshot={**SNAPSHOT, "uri": ""},
                    projection_policy=POLICY,
                    projector=project_event,
                )
            with self.assertRaisesRegex(ValueError, "index_uri"):
                build_evidence_pack(
                    root / "bad-index",
                    row_sources={"ocsf_process_activity": [_row("event-a", 1)]},
                    index_id="test.evidence.pack",
                    version="1",
                    index_uri="",
                    source_snapshot=SNAPSHOT,
                    projection_policy=POLICY,
                    projector=project_event,
                )

    def test_structured_only_rows_are_terminally_accounted_without_document(self) -> None:
        def structured_only_projector(
            relation_name: str,
            event_id: str,
            typed_event_json: str | dict[str, object],
            support_ref: str,
        ) -> dict[str, object]:
            result = project_event(relation_name, event_id, typed_event_json, support_ref)
            self.assertEqual(result["terminal_disposition"], "structured_only_occurrence")
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_evidence_pack(
                root / "pack",
                row_sources={"unknown_relation": [_row("event-a", 1_700_000_000_000)]},
                index_id="test.evidence.pack",
                version="1",
                source_snapshot=SNAPSHOT,
                projection_policy=POLICY,
                projector=structured_only_projector,
            )
            occurrence = _jsonl(root / "pack" / "occurrences.jsonl")[0]
            self.assertEqual(manifest["closure"]["document_count"], 0)
            self.assertEqual(manifest["closure"]["terminal_disposition_count"], 1)
            self.assertEqual(occurrence["document_ids"], [])
            self.assertEqual(occurrence["terminal_disposition"], "structured_only_occurrence")

    def test_occurrence_exact_attributes_are_unmodified_typed_scalars_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _row("event-a", 1_700_000_000_000)
            typed = row["typed_event_json"]
            assert isinstance(typed, dict)
            typed.update(
                {
                    "actor": "  operator@example.test  ",
                    "status": " Success\n",
                    "password": "do-not-publish",
                    "large_counter": 278_037_780_140_032_000,
                }
            )
            # Production Parquet stores typed_event_json as an exact JSON
            # string; retain the out-of-JCS-range integer lexeme for the
            # projector's explicit omission accounting.
            row["typed_event_json"] = json.dumps(typed, separators=(",", ":"))
            self._build(root, "pack", [row])
            occurrence = _jsonl(root / "pack" / "occurrences.jsonl")[0]
            exact = {item["path"]: item["value"] for item in occurrence["exact_attributes"]}
            self.assertEqual(exact["/actor"], "  operator@example.test  ")
            self.assertEqual(exact["/status"], " Success\n")
            self.assertEqual(exact["/class_uid"], 1007)
            self.assertNotIn("/process/cmd_line", exact)
            self.assertNotIn("/password", exact)
            self.assertNotIn("/large_counter", exact)
            self.assertFalse(any("redacted" in str(value) for value in exact.values()))
            accounting = occurrence["exact_attribute_projection"]
            self.assertEqual(accounting["selected_count"], len(exact))
            self.assertTrue(accounting["source_hydration_required"])
            self.assertIn(
                "exact_attribute_subset_requires_source_hydration",
                occurrence["reason_codes"],
            )

    def test_builder_rejects_projector_that_fabricates_an_exact_value(self) -> None:
        def dishonest_projector(
            relation_name: str,
            event_id: str,
            typed_event_json: str | dict[str, object],
            support_ref: str,
        ) -> dict[str, object]:
            projected = project_event(
                relation_name, event_id, typed_event_json, support_ref
            )
            for attribute in projected["exact_attributes"]:
                if attribute["path"] == "/activity_name":
                    attribute["value"] = "normalized launch"
                    break
            return projected

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "differs from the typed JSON scalar"):
                build_evidence_pack(
                    root / "pack",
                    row_sources={
                        "ocsf_process_activity": [_row("event-a", 1_700_000_000_000)]
                    },
                    index_id="test.evidence.pack",
                    version="1",
                    source_snapshot=SNAPSHOT,
                    projection_policy=POLICY,
                    projector=dishonest_projector,
                )
            self.assertFalse((root / "pack").exists())

    def test_builder_rejects_invalid_rfc6901_exact_attribute_escape(self) -> None:
        def invalid_pointer_projector(
            relation_name: str,
            event_id: str,
            typed_event_json: str | dict[str, object],
            support_ref: str,
        ) -> dict[str, object]:
            projected = project_event(
                relation_name, event_id, typed_event_json, support_ref
            )
            projected["exact_attributes"][0]["path"] = "/~2invalid"
            return projected

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid RFC 6901 escape"):
                build_evidence_pack(
                    root / "pack",
                    row_sources={
                        "ocsf_process_activity": [
                            _row("event-a", 1_700_000_000_000)
                        ]
                    },
                    index_id="test.evidence.pack",
                    version="1",
                    source_snapshot=SNAPSHOT,
                    projection_policy=POLICY,
                    projector=invalid_pointer_projector,
                )
            self.assertFalse((root / "pack").exists())

    def test_null_typed_event_is_accounted_as_structured_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _row("event-a", 1_700_000_000_000)
            row["typed_event_json"] = None
            manifest = self._build(root, "pack", [row])
            occurrence = _jsonl(root / "pack" / "occurrences.jsonl")[0]
            self.assertEqual(manifest["closure"]["terminal_disposition_count"], 1)
            self.assertEqual(occurrence["terminal_disposition"], "structured_only_occurrence")
            self.assertEqual(
                occurrence["reason_codes"],
                [
                    "typed_event_unavailable",
                    "exact_attribute_subset_requires_source_hydration",
                ],
            )

    def test_refuses_overwrite_and_duplicate_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _row("event-a", 1_700_000_000_000)
            self._build(root, "pack", [row])
            with self.assertRaises(FileExistsError):
                self._build(root, "pack", [row])
            with self.assertRaisesRegex(ValueError, "duplicate occurrence identity"):
                self._build(root, "duplicate", [row, row])
            self.assertFalse((root / "duplicate").exists())

    def test_verifier_rejects_artifact_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", [_row("event-a", 1_700_000_000_000)])
            occurrence_path = root / "pack" / "occurrences.jsonl"
            occurrence_path.write_bytes(occurrence_path.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(EvidencePackCorrupt, "artifact digest mismatch"):
                _verify_evidence_pack(
                    root / "pack",
                    source_snapshot=SNAPSHOT,
                    relation_sources=None,
                    projection_policy=POLICY,
                    projector=project_event,
                    trusted_builder=True,
                )

    def test_manifest_self_digest_and_canonical_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._build(root, "pack", [_row("event-a", 1_700_000_000_000)])
            material = json.loads(json.dumps(manifest))
            digest = material["component"].pop("sha256")
            self.assertEqual(digest, sha256_bytes(canonical_json_bytes(material)))
            for name in ("documents.jsonl", "occurrences.jsonl"):
                for raw in (root / "pack" / name).read_bytes().splitlines(keepends=True):
                    self.assertEqual(raw, canonical_json_bytes(json.loads(raw), newline=True))

    def test_streams_parquet_and_builds_exact_row_locators(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "ocsf_process_activity.parquet"
            connection = duckdb.connect()
            try:
                rows = [
                    ("event-a", json.dumps(_row("event-a", 1_700_000_000_000)["typed_event_json"]), "support:event-a"),
                    ("event-b", json.dumps(_row("event-b", 1_700_000_001_000)["typed_event_json"]), "support:event-b"),
                ]
                connection.execute(
                    "CREATE TABLE source(event_id VARCHAR, typed_event_json VARCHAR, support_ref VARCHAR)"
                )
                connection.executemany("INSERT INTO source VALUES (?, ?, ?)", rows)
                connection.execute("COPY source TO ? (FORMAT parquet)", [str(parquet_path)])
            finally:
                connection.close()
            manifest = build_evidence_pack(
                root / "pack",
                relations={"ocsf_process_activity": parquet_path},
                index_id="test.evidence.parquet-pack",
                version="1",
                source_snapshot=SNAPSHOT,
                projection_policy=POLICY,
                projector=project_event,
                batch_size=1,
            )
            occurrences = _jsonl(root / "pack" / "occurrences.jsonl")
            locators = sorted(
                (row["source_pointer"]["locator"] for row in occurrences),
                key=lambda locator: locator["row_ordinal"],
            )
            self.assertEqual([locator["row_group"] for locator in locators], [0, 0])
            self.assertEqual([locator["row_ordinal"] for locator in locators], [0, 1])
            self.assertEqual(len({locator["object_sha256"] for locator in locators}), 1)
            self.assertEqual(manifest["closure"]["source_record_count"], 2)

            tampered = occurrences[0]
            tampered["source_pointer"]["locator"]["row_ordinal"] = 999_999
            tampered["occurrence_id"] = "occ-" + sha256_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "livefire.rag.evidence-occurrence-identity/1",
                        "source_pointer": tampered["source_pointer"],
                    }
                )
            )
            occurrences.sort(key=lambda row: row["occurrence_id"])
            occurrence_path = root / "pack" / "occurrences.jsonl"
            occurrence_path.write_bytes(
                b"".join(canonical_json_bytes(row, newline=True) for row in occurrences)
            )
            occurrence_ref = artifact_ref(
                occurrence_path, "occurrences.jsonl", "application/x-ndjson"
            )
            lock_path = root / "pack" / "objects.lock.json"
            object_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            object_lock["objects"] = [
                occurrence_ref if item["path"] == "occurrences.jsonl" else item
                for item in object_lock["objects"]
            ]
            object_lock["objects"].sort(key=lambda item: item["path"])
            write_canonical_json(lock_path, object_lock)
            manifest_path = root / "pack" / "manifest.json"
            resealed = json.loads(manifest_path.read_text(encoding="utf-8"))
            resealed["objects"]["occurrences"] = occurrence_ref
            resealed["objects"]["object_lock"] = artifact_ref(
                lock_path, "objects.lock.json", "application/json"
            )
            resealed["component"]["sha256"] = ""
            resealed["component"]["sha256"] = canonical_sha256_omitting(
                resealed, ("component", "sha256")
            )
            write_canonical_json(manifest_path, resealed)
            with self.assertRaisesRegex(
                EvidencePackCorrupt, "source row has no exact occurrence pointer"
            ):
                verify_evidence_pack(
                    root / "pack",
                    source_snapshot=SNAPSHOT,
                    relation_sources=[
                        RelationSource(
                            "ocsf_process_activity",
                            parquet_path,
                            expected_sha256=sha256_file(parquet_path),
                            expected_rows=2,
                        )
                    ],
                    projection_policy=POLICY,
                    rag_specs=REPOSITORY / "specs",
                    sdk_specs=SDK_SPECS,
                )

    def test_source_aware_verifier_replays_document_projection(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "ocsf_process_activity.parquet"
            connection = duckdb.connect()
            try:
                event = _row("event-a", 1_700_000_000_000)
                connection.execute(
                    "CREATE TABLE source(event_id VARCHAR, typed_event_json VARCHAR, support_ref VARCHAR)"
                )
                connection.execute(
                    "INSERT INTO source VALUES (?, ?, ?)",
                    ["event-a", json.dumps(event["typed_event_json"]), "support:event-a"],
                )
                connection.execute("COPY source TO ? (FORMAT parquet)", [str(parquet_path)])
            finally:
                connection.close()

            build_evidence_pack(
                root / "pack",
                relations={"ocsf_process_activity": parquet_path},
                index_id="test.evidence.replay-pack",
                version="1",
                source_snapshot=SNAPSHOT,
                projection_policy=POLICY,
                projector=project_event,
            )

            document_path = root / "pack" / "documents.jsonl"
            document = _jsonl(document_path)[0]
            document["semantic_projection"]["text"] += " fabricated behavior"
            document["document_sha256"] = canonical_sha256_omitting(
                document, ("document_sha256",)
            )
            document_path.write_bytes(canonical_json_bytes(document, newline=True))
            document_ref = artifact_ref(
                document_path, "documents.jsonl", "application/x-ndjson"
            )

            lock_path = root / "pack" / "objects.lock.json"
            object_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            object_lock["objects"] = [
                document_ref if item["path"] == "documents.jsonl" else item
                for item in object_lock["objects"]
            ]
            object_lock["objects"].sort(key=lambda item: item["path"])
            write_canonical_json(lock_path, object_lock)

            manifest_path = root / "pack" / "manifest.json"
            resealed = json.loads(manifest_path.read_text(encoding="utf-8"))
            resealed["objects"]["documents"] = document_ref
            resealed["objects"]["object_lock"] = artifact_ref(
                lock_path, "objects.lock.json", "application/json"
            )
            resealed["component"]["sha256"] = ""
            resealed["component"]["sha256"] = canonical_sha256_omitting(
                resealed, ("component", "sha256")
            )
            write_canonical_json(manifest_path, resealed)

            with self.assertRaisesRegex(
                EvidencePackCorrupt, "document does not replay from source"
            ):
                verify_evidence_pack(
                    root / "pack",
                    source_snapshot=SNAPSHOT,
                    relation_sources=[
                        RelationSource(
                            "ocsf_process_activity",
                            parquet_path,
                            expected_sha256=sha256_file(parquet_path),
                            expected_rows=1,
                        )
                    ],
                    projection_policy=POLICY,
                    rag_specs=REPOSITORY / "specs",
                    sdk_specs=SDK_SPECS,
                )

    def test_empty_parquet_relation_is_closed_and_source_verified(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "ocsf_process_activity.parquet"
            connection = duckdb.connect()
            try:
                connection.execute(
                    "COPY (SELECT NULL::VARCHAR AS event_id, "
                    "NULL::VARCHAR AS typed_event_json, NULL::VARCHAR AS support_ref "
                    "WHERE false) TO ? (FORMAT parquet)",
                    [str(parquet_path)],
                )
            finally:
                connection.close()
            source = RelationSource(
                "ocsf_process_activity",
                parquet_path,
                expected_sha256=sha256_file(parquet_path),
                expected_rows=0,
            )
            snapshot = {**SNAPSHOT, "uri": "urn:test:empty-snapshot:1"}
            policy = {**POLICY, "uri": "urn:test:empty-policy:1"}
            manifest = build_evidence_pack(
                root / "pack",
                relations=[source],
                index_id="test.evidence.empty-pack",
                version="1",
                source_snapshot=snapshot,
                projection_policy=policy,
                projector=project_event,
            )
            verified = verify_evidence_pack(
                root / "pack",
                source_snapshot=snapshot,
                relation_sources=[source],
                projection_policy=policy,
                rag_specs=REPOSITORY / "specs",
                sdk_specs=SDK_SPECS,
            )
            self.assertEqual(manifest, verified)
            self.assertEqual(manifest["closure"]["source_record_count"], 0)
            coverage = json.loads(
                (root / "pack" / "coverage-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(coverage["relation_coverage"][0]["source_record_count"], 0)

    def test_parquet_pointer_ordinals_reset_at_row_group_boundaries(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "ocsf_process_activity.parquet"
            connection = duckdb.connect()
            try:
                payload = json.dumps(_row("event-a", 1_700_000_000_000)["typed_event_json"])
                connection.execute(
                    "CREATE TABLE source AS SELECT 'event-' || range AS event_id, "
                    "? AS typed_event_json, 'support:' || range AS support_ref FROM range(5000)",
                    [payload],
                )
                connection.execute(
                    "COPY source TO ? (FORMAT parquet, ROW_GROUP_SIZE 2048)",
                    [str(parquet_path)],
                )
            finally:
                connection.close()
            build_evidence_pack(
                root / "pack",
                relations={"ocsf_process_activity": parquet_path},
                index_id="test.evidence.multi-row-group-pack",
                version="1",
                source_snapshot=SNAPSHOT,
                projection_policy=POLICY,
                projector=project_event,
                batch_size=1024,
            )
            locators = {
                row["source_pointer"]["record_id"]: row["source_pointer"]["locator"]
                for row in _jsonl(root / "pack" / "occurrences.jsonl")
            }
            self.assertEqual((locators["event-0"]["row_group"], locators["event-0"]["row_ordinal"]), (0, 0))
            self.assertEqual(
                (locators["event-2047"]["row_group"], locators["event-2047"]["row_ordinal"]),
                (0, 2047),
            )
            self.assertEqual(
                (locators["event-2048"]["row_group"], locators["event-2048"]["row_ordinal"]),
                (1, 0),
            )
            self.assertEqual(
                (locators["event-4096"]["row_group"], locators["event-4096"]["row_ordinal"]),
                (2, 0),
            )

    def test_pack_rows_conform_to_offline_contracts(self) -> None:
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource

        repository = Path(__file__).resolve().parents[1]
        registry = Registry()
        schemas: dict[str, dict[str, object]] = {}
        for path in [
            *sorted((repository.parent / "livefire-sdk" / "specs").glob("*.json")),
            *sorted((repository / "specs").glob("*.json")),
        ]:
            schema = json.loads(path.read_text(encoding="utf-8"))
            if "$id" in schema:
                registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
                schemas[path.name] = schema
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", [_row("event-a", 1_700_000_000_000)])
            validations = (
                ("evidence-projection-pack.v1.schema.json", "manifest.json"),
                ("evidence-document.v1.schema.json", "documents.jsonl"),
                ("evidence-occurrence-row.v1.schema.json", "occurrences.jsonl"),
                ("evidence-coverage-report.v1.schema.json", "coverage-report.json"),
            )
            for schema_name, artifact_name in validations:
                value = json.loads((root / "pack" / artifact_name).read_text(encoding="utf-8").splitlines()[0])
                Draft202012Validator(
                    schemas[schema_name], registry=registry, format_checker=FormatChecker()
                ).validate(value)
            counts = validate_evidence_pack_schemas(
                root / "pack",
                rag_specs=repository / "specs",
                sdk_specs=repository.parent / "livefire-sdk" / "specs",
            )
            self.assertEqual(counts["documents"], 1)
            self.assertEqual(counts["occurrences"], 1)

    def test_generic_registry_excludes_scenario_schemas(self) -> None:
        _, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        self.assertTrue(set(GENERIC_EVIDENCE_SCHEMA_NAMES) <= set(schemas))
        self.assertEqual(
            {
                name
                for name in schemas
                if name in {
                    "component-ref.v1.schema.json",
                    "artifact-ref.v1.schema.json",
                    "source-record-pointer.v1.schema.json",
                }
            },
            {
                "component-ref.v1.schema.json",
                "artifact-ref.v1.schema.json",
                "source-record-pointer.v1.schema.json",
            },
        )
        self.assertNotIn("evidence-qrel-row.v1.schema.json", schemas)
        self.assertNotIn("evidence-benchmark-run.v1.schema.json", schemas)

    def test_fast_provider_input_and_pointer_miss_outputs_validate_offline(self) -> None:
        from jsonschema import Draft202012Validator, ValidationError

        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        input_validator = Draft202012Validator(
            schemas["fast-evidence-search.input.v1.schema.json"], registry=registry
        )
        output_validator = Draft202012Validator(
            schemas["fast-evidence-search.output.v1.schema.json"], registry=registry
        )
        input_validator.validate(
            {
                "schema_version": "livefire.rag.fast-search.input/1",
                "query": "encoded PowerShell command",
                "mode": "fused",
                "top_n": 20,
                "filters": {"relations": ["ocsf_process_activity"]},
            }
        )
        component = {"id": "test.index", "version": "1", "sha256": "a" * 64}
        common = {
            "schema_version": "livefire.rag.fast-search.output/1",
            "tool": "evidence.search",
            "index": component,
            "source_snapshots": [
                {"id": "test.snapshot", "version": "1", "sha256": "b" * 64}
            ],
            "query": "encoded PowerShell command",
            "coverage": {
                "status": "complete",
                "indexed_documents": 6,
                "definitive": False,
                "reason_codes": [
                    "candidate_occurrences_require_authoritative_hydration"
                ],
            },
        }
        pointer = {
            **common,
            "kind": "pointer",
            "selection": {
                "requested_top_n": 20,
                "returned_count": 1,
                "deterministic": True,
                "tie_break": "score_desc_document_id_asc",
            },
            "candidates": [
                {
                    "rank": 1,
                    "document_id": "sha256:document",
                    "scores": {"retrieval": 0.8, "dense": 0.8, "lexical": None},
                    "eligible_evidence_count": 1,
                    "evidence_exhausted": True,
                    "evidence": [
                        {
                            "schema_version": "livefire.ocsf-hydration-ref/1",
                            "snapshot": {"id": "test.snapshot", "version": "1", "sha256": "b" * 64},
                            "mapping": {"id": "test.mapping", "version": "1", "sha256": "c" * 64},
                            "relation": "ocsf_process_activity",
                            "event_id": "event-1",
                            "support_ref": "support:event-1",
                        }
                    ],
                }
            ],
        }
        output_validator.validate(pointer)
        output_validator.validate(
            {
                **common,
                "kind": "miss",
                "selection": {
                    "requested_top_n": 20,
                    "returned_count": 0,
                    "deterministic": True,
                    "tie_break": "score_desc_document_id_asc",
                },
                "miss": {
                    "reason": "no_ranked_candidates",
                    "message": "No indexed semantic document matched the query.",
                },
            }
        )
        broken = json.loads(json.dumps(pointer))
        del broken["candidates"][0]["eligible_evidence_count"]
        with self.assertRaises(ValidationError):
            output_validator.validate(broken)

    def test_wheel_declares_all_generic_contract_and_policy_resources(self) -> None:
        configuration = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
        project = configuration["project"]
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertIn("License :: OSI Approved :: Apache Software License", project["classifiers"])
        self.assertEqual(
            (REPOSITORY / "LICENSE").read_text(encoding="utf-8"),
            (REPOSITORY.parent / "livefire" / "LICENSE").read_text(encoding="utf-8"),
        )
        forced = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        expected = {
            *GENERIC_EVIDENCE_SCHEMA_NAMES,
            "evidence-projection-policy.v1.json",
            "evidence-projection-policy.v2.json",
            "evidence-derivation-policy.v1.json",
            "evidence-pilot-sampling-policy.v1.json",
            "evidence-pilot-geometry-policy.v1.json",
            "typed-parquet-record-profile.v1.json",
            "fast-vector-binary-profile.v1.json",
            "fast-lexical-profile.v1.json",
            "fast-lexical-profile.v2.json",
            "fast-occurrence-lookup-profile.v1.json",
        }
        self.assertEqual({Path(destination).name for destination in forced.values()}, expected)
        self.assertIn("/LICENSE", configuration["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
        self.assertEqual(generic_schema_root(), REPOSITORY / "specs")


if __name__ == "__main__":
    unittest.main()
