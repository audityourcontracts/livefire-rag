from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from livefire_rag.canonical import canonical_json_bytes, sha256_bytes
from livefire_rag.evidence_derivation import (
    EvidenceDerivationCorrupt,
    _build_derivation_pack_for_test as build_derivation_pack,
    build_evidence_derivation_pack,
    derivation_policy_material,
    derivation_policy_ref,
    MAX_DERIVED_NODES,
    _safe_token,
    verify_evidence_derivation_pack,
)
from livefire_rag.evidence_projection import RELATION_DOCUMENT_KINDS
from livefire_rag.evidence_projection import semantic_safe_value


REPOSITORY = Path(__file__).resolve().parents[1]
SNAPSHOT = {"id": "test.snapshot", "version": "1", "sha256": "1" * 64}
BASE_PACK = {"id": "test.projection-pack", "version": "1", "sha256": "2" * 64}


def _occurrence(event_id: str, relation: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "occurrence_id": "occ-" + sha256_bytes(event_id.encode()),
        "relation_name": relation,
    }


def _typed(event_id: str, value: dict[str, object]) -> dict[str, object]:
    return {"event_id": event_id, "typed_event_json": value, "support_ref": f"support:{event_id}"}


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture() -> dict[str, object]:
    typed_rows: dict[str, list[dict[str, object]]] = {
        relation: [] for relation in RELATION_DOCUMENT_KINDS
    }
    occurrences: list[dict[str, str]] = []
    participants: list[dict[str, str]] = []
    entities = [
        {
            "entity_id": "ent-device",
            "kind": "device",
            "display_name": "private-host.internal",
            "canonical_value": "private-host.internal",
            "support_ref": "support:entity-device",
        },
        {
            "entity_id": "ent-service",
            "kind": "service",
            "display_name": "collector-account@example.invalid",
            "canonical_value": "collector-account@example.invalid",
            "support_ref": "support:entity-service",
        },
        {
            "entity_id": "ent-orphan",
            "kind": "resource",
            "display_name": "orphan-private-value",
            "canonical_value": "orphan-private-value",
            "support_ref": "support:orphan",
        },
    ]

    def add(
        relation: str,
        event_id: str,
        payload: dict[str, object],
        roles: tuple[tuple[str, str], ...] = (("actor", "ent-service"), ("resource", "ent-device")),
    ) -> None:
        typed_rows[relation].append(_typed(event_id, payload))
        occurrences.append(_occurrence(event_id, relation))
        for role, entity_id in roles:
            participants.append(
                {
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "role": role,
                    "support_ref": f"support:{event_id}",
                }
            )

    add(
        "ocsf_ext_livefire_system_metric",
        "metric-a",
        {"header": {"time": 299_999, "metadata": {"type": "metric-source"}}, "metric": "cpu.load", "unit": "percent", "value_milli": 1000},
    )
    add(
        "ocsf_ext_livefire_system_metric",
        "metric-b",
        {"header": {"time": 299_000, "metadata": {"type": "metric-source"}}, "metric": "cpu.load", "unit": "percent", "value_milli": 3000},
    )
    add(
        "ocsf_ext_livefire_system_metric",
        "metric-boundary",
        {"header": {"time": 300_000, "metadata": {"type": "metric-source"}}, "metric": "cpu.load", "value_milli": 9000},
    )
    add(
        "ocsf_ext_livefire_system_metric",
        "metric-no-scope",
        {"header": {"time": 1}, "metric": "cpu.load", "value_milli": 5},
        (),
    )

    add(
        "ocsf_network_activity",
        "network-a",
        {"ocsf": {"time": 10, "metadata": {"type": "flow"}}, "action": "connect", "protocol_stack": "tcp:tls", "bytes": 7, "status": "allowed"},
    )
    add(
        "ocsf_network_activity",
        "network-b",
        {"ocsf": {"time": 20, "metadata": {"type": "flow"}}, "action": "connect", "protocol_stack": "tcp:tls", "bytes": 11, "status": "allowed"},
    )
    add(
        "ocsf_dns_activity",
        "dns-a",
        {"ocsf": {"time": 25}, "query_type": "AAAA", "transport": "udp", "rcode": "NOERROR"},
    )
    add(
        "ocsf_http_activity",
        "http-a",
        {"ocsf": {"time": 30}, "method": "POST", "protocol": "https", "response": {"status_code": 201}},
    )

    for event_id, time_ms, state in (
        ("state-a", 100, "disabled"),
        ("state-same", 200, "disabled"),
        ("state-b", 300, "enabled"),
    ):
        add(
            "ocsf_ext_livefire_configuration_snapshot",
            event_id,
            {
                "header": {"time": time_ms, "metadata": {"type": "configuration"}},
                "snapshot_kind": "service configuration",
                "subject_kind": "setting",
                "subject_instance_id": "stable-settings-object",
                "subject": "remote access",
                "state": state,
            },
        )

    # Exercise every typed relation in entity coverage without inventing a
    # derivation for relations to which a family does not structurally apply.
    for index, relation in enumerate(sorted(RELATION_DOCUMENT_KINDS)):
        if typed_rows[relation]:
            continue
        add(
            relation,
            f"generic-{index}",
            {"header": {"time": 1_000 + index, "metadata": {"type": "generic"}}, "activity_name": "observe"},
        )

    relationships = [
        {
            "relationship_id": "rel-1",
            "kind": "observed_on",
            "source_id": "ent-service",
            "target_id": "ent-device",
            "event_id": "network-a",
            "support_ref": "support:network-a",
        }
    ]
    return {
        "typed_rows": typed_rows,
        "occurrences": occurrences,
        "participants": participants,
        "entities": entities,
        "relationships": relationships,
    }


class EvidenceDerivationTests(unittest.TestCase):
    def test_nested_network_and_graph_taxonomy_values_cannot_leak(self) -> None:
        rendered = _safe_token(
            {
                "operation": "connect",
                "computer_name": "private-host.internal",
                "nested": [{
                    "x_api_key": "opaque-private-credential",
                    "account_id": 123456789012345678,
                    "bytes": 123456,
                }],
            },
            path="network.operation",
        )
        self.assertIn("connect", rendered)
        self.assertNotIn("private-host.internal", rendered)
        self.assertNotIn("opaque-private-credential", rendered)
        self.assertNotIn("123456789012345678", rendered)
        self.assertNotIn("123456", rendered)
        self.assertIn("1e5", rendered)
        self.assertIn("<redacted:", rendered)

        deep: object = "unreachable-secret"
        for _ in range(20):
            deep = {"nested": deep}
        self.assertIn("<omitted:depth-bound>", _safe_token(deep))
        self.assertEqual(
            _safe_token({str(index): index for index in range(100)}),
            "<omitted:container-bound>",
        )
        wide_nested = [list(range(32)) for _ in range(32)]
        with patch(
            "livefire_rag.evidence_derivation.semantic_safe_value",
            wraps=semantic_safe_value,
        ) as safe_value:
            bounded = _safe_token(wide_nested, maximum=10_000)
        self.assertIn("<omitted:scalar-bound>", bounded)
        self.assertEqual(safe_value.call_count, 128)

        class CountingDict(dict[str, object]):
            visits = 0

            def items(self):  # type: ignore[override]
                type(self).visits += 1
                return super().items()

        null_tree: object = None
        for _ in range(8):
            null_tree = CountingDict(
                {str(index): null_tree for index in range(32)}
            )
        null_bounded = _safe_token(null_tree, maximum=100_000)
        self.assertIn("<omitted:node-bound>", null_bounded)
        self.assertLessEqual(CountingDict.visits, MAX_DERIVED_NODES)

        fixture = _fixture()
        fixture["participants"][0]["role"] = "private-host.internal"
        fixture["relationships"][0]["kind"] = "opaque-private-relationship"
        fixture["entities"][0]["kind"] = "private-neighbor-kind"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "derivation"
            build_derivation_pack(
                root,
                typed_rows=fixture["typed_rows"],
                occurrences=fixture["occurrences"],
                participants=fixture["participants"],
                entities=fixture["entities"],
                relationships=fixture["relationships"],
                component_id="test.derivation",
                version="1",
                source_snapshot=SNAPSHOT,
                base_projection_pack=BASE_PACK,
            )
            text = "\n".join(
                row["semantic_projection"]["text"]
                for row in _jsonl(root / "documents.jsonl")
                if row["document_kind"] == "entity"
            )
        for secret in (
            "private-host.internal",
            "opaque-private-relationship",
            "private-neighbor-kind",
        ):
            self.assertNotIn(secret, text)
        self.assertIn("<redacted:", text)

    def _build(self, root: Path, name: str, fixture: dict[str, object]) -> dict[str, object]:
        return build_derivation_pack(
            root / name,
            typed_rows=fixture["typed_rows"],
            occurrences=fixture["occurrences"],
            participants=fixture["participants"],
            entities=fixture["entities"],
            relationships=fixture["relationships"],
            component_id="test.derivation-pack",
            version="1",
            source_snapshot=SNAPSHOT,
            base_projection_pack=BASE_PACK,
        )

    def test_all_four_families_and_complete_relation_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture()
            manifest = self._build(root, "pack", fixture)
            documents = _jsonl(root / "pack" / "documents.jsonl")
            memberships = _jsonl(root / "pack" / "memberships.jsonl")
            coverage = json.loads((root / "pack" / "coverage-report.json").read_text())

            self.assertEqual(
                {document["document_kind"] for document in documents},
                {"metric_window", "network_window", "state_transition", "entity"},
            )
            self.assertEqual(set(coverage["base_relation_counts"]), set(RELATION_DOCUMENT_KINDS))
            self.assertEqual(
                coverage["closure"]["base_source_record_count"],
                sum(coverage["base_relation_counts"].values()),
            )
            self.assertEqual(manifest["closure"]["derived_document_count"], len(documents))
            self.assertEqual(manifest["closure"]["derivation_membership_count"], len(memberships))
            self.assertEqual(coverage["families"]["metric_window"]["reason_counts"], {"missing_canonical_scope": 1})
            self.assertEqual(coverage["families"]["entity"]["orphan_entity_count"], 1)
            verify_evidence_derivation_pack(root / "pack")

    def test_metric_windows_are_epoch_aligned_exact_and_do_not_infer_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            documents = [
                row for row in _jsonl(root / "pack" / "documents.jsonl")
                if row["document_kind"] == "metric_window"
            ]
            self.assertEqual(len(documents), 2)
            first = next(row for row in documents if row["derivation"]["aggregate_material"]["sample_count"] == 2)
            aggregate = first["derivation"]["aggregate_material"]
            self.assertEqual(aggregate["minimum_value_milli"], 1000)
            self.assertEqual(aggregate["maximum_value_milli"], 3000)
            self.assertEqual(aggregate["sum_value_milli"], "4000")
            self.assertEqual(aggregate["mean_value_milli"], {"numerator": "4000", "denominator": 2})
            self.assertEqual(first["time_range"]["start"], "1970-01-01T00:00:00.000Z")
            boundary = next(row for row in documents if row is not first)
            self.assertEqual(boundary["time_range"]["start"], "1970-01-01T00:05:00.000Z")
            self.assertIn("unit: absent", boundary["semantic_projection"]["text"])

    def test_network_missing_measures_are_counted_not_zero_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            documents = [
                row for row in _jsonl(root / "pack" / "documents.jsonl")
                if row["document_kind"] == "network_window"
            ]
            network = next(row for row in documents if row["relation_identities"][0]["relation"] == "ocsf_network_activity")
            measures = network["derivation"]["aggregate_material"]["measures"]
            self.assertEqual(measures["bytes"], {"observed_count": 2, "missing_count": 0, "minimum": 7, "maximum": 11, "sum": "18"})
            self.assertEqual(measures["packets"], {"observed_count": 0, "missing_count": 2})

    def test_transition_uses_adjacent_observed_state_and_exact_two_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            transitions = [
                row for row in _jsonl(root / "pack" / "documents.jsonl")
                if row["document_kind"] == "state_transition"
            ]
            self.assertEqual(len(transitions), 1)
            members = [
                row for row in _jsonl(root / "pack" / "memberships.jsonl")
                if row["derived_document_id"] == transitions[0]["document_id"]
            ]
            self.assertEqual({row["input_role"] for row in members}, {"before", "after"})
            self.assertEqual(
                {row["occurrence_id"] for row in members},
                {_occurrence("state-same", "x")["occurrence_id"], _occurrence("state-b", "x")["occurrence_id"]},
            )
            coverage = json.loads((root / "pack" / "coverage-report.json").read_text())
            self.assertEqual(coverage["families"]["state_transition"]["outcome_counts"]["unchanged"], 1)
            self.assertEqual(coverage["families"]["state_transition"]["outcome_counts"]["transition"], 1)

    def test_same_time_conflicting_states_are_ambiguous_not_ordered_by_event_id(self) -> None:
        fixture = _fixture()
        for event_id, state in (("ambiguous-a", "one"), ("ambiguous-b", "two")):
            relation = "ocsf_ext_livefire_configuration_snapshot"
            fixture["typed_rows"][relation].append(
                _typed(
                    event_id,
                    {
                        "header": {"time": 500},
                        "snapshot_kind": "policy",
                        "subject_instance_id": "stable-policy-object",
                        "subject": "mode",
                        "state": state,
                    },
                )
            )
            fixture["occurrences"].append(_occurrence(event_id, relation))
            for role, entity_id in (("actor", "ent-service"), ("resource", "ent-device")):
                fixture["participants"].append(
                    {"event_id": event_id, "entity_id": entity_id, "role": role, "support_ref": f"support:{event_id}"}
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", fixture)
            coverage = json.loads((root / "pack" / "coverage-report.json").read_text())
            self.assertEqual(coverage["families"]["state_transition"]["outcome_counts"]["ambiguous_same_time"], 2)

    def test_configuration_fields_without_explicit_subject_instance_never_transition(self) -> None:
        fixture = _fixture()
        relation = "ocsf_ext_livefire_configuration_snapshot"
        for event_id, time_ms, state in (
            ("field-record-a", 800, "alpha"),
            ("field-record-b", 900, "beta"),
        ):
            fixture["typed_rows"][relation].append(
                _typed(
                    event_id,
                    {
                        "header": {"time": time_ms},
                        "snapshot_kind": "record observations",
                        "subject": "same_field_name",
                        "state": state,
                    },
                )
            )
            fixture["occurrences"].append(_occurrence(event_id, relation))
            fixture["participants"].append(
                {
                    "event_id": event_id,
                    "entity_id": "ent-device",
                    "role": "resource",
                    "support_ref": f"support:{event_id}",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", fixture)
            transition_members = {
                row["occurrence_id"]
                for row in _jsonl(root / "pack" / "memberships.jsonl")
                if row["input_role"] in {"before", "after"}
            }
            self.assertNotIn(_occurrence("field-record-a", relation)["occurrence_id"], transition_members)
            self.assertNotIn(_occurrence("field-record-b", relation)["occurrence_id"], transition_members)
            coverage = json.loads((root / "pack" / "coverage-report.json").read_text())
            self.assertGreaterEqual(
                coverage["families"]["state_transition"]["reason_counts"][
                    "missing_stable_subject_instance"
                ],
                2,
            )

    def test_incomplete_rows_are_explicitly_ineligible_and_never_disappear(self) -> None:
        fixture = _fixture()
        for relation, event_id, payload in (
            ("ocsf_ext_livefire_system_metric", "metric-malformed", "{"),
            ("ocsf_network_activity", "network-missing-time", {"action": "observe"}),
        ):
            fixture["typed_rows"][relation].append(
                {"event_id": event_id, "typed_event_json": payload, "support_ref": f"support:{event_id}"}
            )
            fixture["occurrences"].append(_occurrence(event_id, relation))
            fixture["participants"].append(
                {"event_id": event_id, "entity_id": "ent-device", "role": "resource", "support_ref": f"support:{event_id}"}
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", fixture)
            coverage = json.loads((root / "pack" / "coverage-report.json").read_text())
            metric = coverage["families"]["metric_window"]
            network = coverage["families"]["network_window"]
            self.assertEqual(metric["reason_counts"]["typed_event_unavailable"], 1)
            self.assertEqual(network["reason_counts"]["missing_time"], 1)
            self.assertEqual(
                metric["eligible_source_record_count"] + metric["ineligible_source_record_count"],
                metric["applicable_source_record_count"],
            )
            self.assertEqual(
                network["eligible_source_record_count"] + network["ineligible_source_record_count"],
                network["applicable_source_record_count"],
            )

    def test_state_series_never_mix_distinct_canonical_entity_scopes(self) -> None:
        fixture = _fixture()
        fixture["entities"].append(
            {
                "entity_id": "ent-other-device",
                "kind": "device",
                "display_name": "another-private-host",
                "canonical_value": "another-private-host",
                "support_ref": "support:other-device",
            }
        )
        relation = "ocsf_ext_livefire_configuration_snapshot"
        for event_id, entity_id, state in (
            ("isolated-a", "ent-device", "alpha"),
            ("isolated-b", "ent-other-device", "beta"),
        ):
            fixture["typed_rows"][relation].append(
                _typed(
                    event_id,
                    {
                        "header": {"time": 700},
                        "snapshot_kind": "isolated fixture",
                        "subject": "same field",
                        "state": state,
                    },
                )
            )
            fixture["occurrences"].append(_occurrence(event_id, relation))
            fixture["participants"].append(
                {"event_id": event_id, "entity_id": entity_id, "role": "resource", "support_ref": f"support:{event_id}"}
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", fixture)
            transitions = [
                row
                for row in _jsonl(root / "pack" / "documents.jsonl")
                if row["document_kind"] == "state_transition"
            ]
            isolated_ids = {
                _occurrence("isolated-a", relation)["occurrence_id"],
                _occurrence("isolated-b", relation)["occurrence_id"],
            }
            memberships = _jsonl(root / "pack" / "memberships.jsonl")
            transition_ids = {row["document_id"] for row in transitions}
            self.assertFalse(
                any(
                    row["derived_document_id"] in transition_ids
                    and row["occurrence_id"] in isolated_ids
                    for row in memberships
                )
            )

    def test_entity_semantics_exclude_exact_identifiers_but_membership_is_resolvable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            documents = _jsonl(root / "pack" / "documents.jsonl")
            semantic = "\n".join(str(row["semantic_projection"]) for row in documents)
            for secret in (
                "ent-device",
                "ent-service",
                "private-host.internal",
                "collector-account@example.invalid",
                "orphan-private-value",
            ):
                self.assertNotIn(secret, semantic)
            entity_documents = [row for row in documents if row["document_kind"] == "entity"]
            self.assertEqual(len(entity_documents), 2)
            self.assertTrue(all(row["occurrence_count"] > 0 for row in entity_documents))
            entity_document_ids = {row["document_id"] for row in entity_documents}
            entity_memberships = [
                row
                for row in _jsonl(root / "pack" / "memberships.jsonl")
                if row["derived_document_id"] in entity_document_ids
            ]
            self.assertTrue(entity_memberships)
            self.assertEqual(
                {row["entity_id"] for row in entity_memberships},
                {"ent-device", "ent-service"},
            )

    def test_build_is_byte_deterministic_under_all_input_orderings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _fixture()
            second = _fixture()
            second["occurrences"] = list(reversed(second["occurrences"]))
            second["participants"] = list(reversed(second["participants"]))
            second["entities"] = list(reversed(second["entities"]))
            second["relationships"] = list(reversed(second["relationships"]))
            second["typed_rows"] = {
                relation: list(reversed(rows)) for relation, rows in reversed(list(second["typed_rows"].items()))
            }
            self._build(root, "first", first)
            self._build(root, "second", second)
            for name in ("documents.jsonl", "memberships.jsonl", "coverage-report.json", "objects.lock.json", "manifest.json"):
                self.assertEqual((root / "first" / name).read_bytes(), (root / "second" / name).read_bytes(), name)

    def test_dangling_graph_references_fail_closed(self) -> None:
        fixture = _fixture()
        fixture["participants"].append(
            {"event_id": "metric-a", "entity_id": "ent-absent", "role": "resource", "support_ref": "support:bad"}
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Exception, "absent entity"):
                self._build(Path(directory), "pack", fixture)

    def test_verifier_rejects_membership_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            path = root / "pack" / "memberships.jsonl"
            rows = _jsonl(path)
            rows[0]["input_role"] = "tampered"
            path.write_bytes(b"".join(canonical_json_bytes(row, newline=True) for row in rows))
            with self.assertRaisesRegex(EvidenceDerivationCorrupt, "artifact mismatch"):
                verify_evidence_derivation_pack(root / "pack")

    def test_policy_file_is_exact_and_schemas_are_valid_draft_2020_12(self) -> None:
        policy = json.loads((REPOSITORY / "specs" / "evidence-derivation-policy.v1.json").read_text())
        self.assertEqual(policy, derivation_policy_material())
        self.assertEqual(derivation_policy_ref()["sha256"], sha256_bytes(canonical_json_bytes(policy)))
        for name in (
            "evidence-derived-document.v1.schema.json",
            "evidence-derivation-membership-row.v1.schema.json",
            "evidence-derivation-coverage.v1.schema.json",
            "evidence-derivation-pack.v1.schema.json",
        ):
            Draft202012Validator.check_schema(json.loads((REPOSITORY / "specs" / name).read_text()))

    def test_every_emitted_artifact_conforms_to_derivation_schemas(self) -> None:
        registry = Registry()
        schemas: dict[str, dict[str, object]] = {}
        for root in (REPOSITORY / "specs", REPOSITORY.parent / "livefire-sdk" / "specs"):
            for path in root.glob("*.schema.json"):
                value = json.loads(path.read_text())
                if "$id" not in value:
                    continue
                registry = registry.with_resource(value["$id"], Resource.from_contents(value))
                schemas[path.name] = value
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, "pack", _fixture())
            for filename, schema_name in (
                ("manifest.json", "evidence-derivation-pack.v1.schema.json"),
                ("coverage-report.json", "evidence-derivation-coverage.v1.schema.json"),
            ):
                Draft202012Validator(schemas[schema_name], registry=registry).validate(
                    json.loads((root / "pack" / filename).read_text())
                )
            for filename, schema_name in (
                ("documents.jsonl", "evidence-derived-document.v1.schema.json"),
                ("memberships.jsonl", "evidence-derivation-membership-row.v1.schema.json"),
            ):
                validator = Draft202012Validator(schemas[schema_name], registry=registry)
                for row in _jsonl(root / "pack" / filename):
                    validator.validate(row)

    def test_public_api_has_no_selectors_and_source_has_no_scenario_literals(self) -> None:
        parameters = set(inspect.signature(build_evidence_derivation_pack).parameters)
        self.assertEqual(
            parameters,
            {"output_dir", "snapshot_root", "receipt_path", "base_projection_pack", "component_id", "version", "component_uri"},
        )
        source = (REPOSITORY / "src" / "livefire_rag" / "evidence_derivation.py").read_text().lower()
        for forbidden in ("botsv", "froth", "attack technique", "evidentiary fact", "expected event id"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
