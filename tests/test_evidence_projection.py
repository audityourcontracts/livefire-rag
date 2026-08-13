from __future__ import annotations

import json
import unittest
from pathlib import Path

from livefire_rag.canonical import canonical_json_bytes, sha256_bytes
from livefire_rag.evidence_projection import (
    MAX_FACET_TEXT_CHARS,
    MAX_LEAVES,
    MAX_SEMANTIC_TEXT_CHARS,
    RELATION_DOCUMENT_KINDS,
    project_event,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceProjectionTest(unittest.TestCase):
    def test_closed_relation_set_has_total_generic_projection(self) -> None:
        expected = {
            "ocsf_api_activity",
            "ocsf_application_lifecycle",
            "ocsf_authentication",
            "ocsf_cloud_resources_inventory_info",
            "ocsf_datastore_activity",
            "ocsf_detection_finding",
            "ocsf_dns_activity",
            "ocsf_email_activity",
            "ocsf_entity_management",
            "ocsf_event_log_activity",
            "ocsf_ext_livefire_configuration_snapshot",
            "ocsf_ext_livefire_system_metric",
            "ocsf_file_activity",
            "ocsf_http_activity",
            "ocsf_inventory_info",
            "ocsf_network_activity",
            "ocsf_process_activity",
            "ocsf_user_inventory",
        }
        self.assertEqual(set(RELATION_DOCUMENT_KINDS), expected)
        for relation in sorted(expected):
            with self.subTest(relation=relation):
                result = project_event(
                    relation,
                    f"event-{relation}",
                    {
                        "semantic_class": relation.removeprefix("ocsf_"),
                        "ocsf": {
                            "time": 1_700_000_000_000,
                            "class_uid": 1001,
                            "activity_name": "Observe",
                            "status": "Success",
                        },
                    },
                    f"support-{relation}",
                )
                expected_disposition = (
                    "structured_only_occurrence"
                    if relation == "ocsf_ext_livefire_system_metric"
                    else "direct_semantic_document"
                )
                self.assertEqual(result["terminal_disposition"], expected_disposition)
                self.assertNotEqual(result["semantic_text"], "")
                self.assertEqual(result["event_metadata"]["event_time_availability"], "available")
                self.assertEqual(len(result["projection_sha256"]), 64)
                self.assertEqual(len(result["semantic_group_sha256"]), 64)

    def test_api_projection_is_role_separated_and_identifier_safe(self) -> None:
        event = {
            "semantic_class": "api",
            "ocsf": {
                "time": "2025-04-02T01:02:03Z",
                "class_uid": 6003,
                "activity_name": "Update",
                "status": "Success",
            },
            "service": "compute",
            "operation": "ChangeResourcePolicy",
            "actor": "analyst@example.test",
            "source_address": "192.0.2.40",
            "resource": "arn:example:compute:region:000000000000:resource/private-name",
            "resources": [{"name": "private-child", "type": "virtual machine"}],
            "credential_id": "AKIAABCDEFGHIJKLMNOP",
        }
        projected = project_event("ocsf_api_activity", "evt-api", event, "support-api")
        all_text = " ".join(
            projected[name]
            for name in ("semantic_text", "action_text", "target_text", "context_text", "outcome_text")
        )
        self.assertIn("operation=ChangeResourcePolicy", projected["action_text"])
        self.assertIn("service=compute", projected["target_text"])
        self.assertIn("status=Success", projected["outcome_text"])
        for secret in (
            "analyst@example.test",
            "192.0.2.40",
            "private-name",
            "private-child",
            "AKIAABCDEFGHIJKLMNOP",
        ):
            self.assertNotIn(secret, all_text)
        self.assertEqual(projected["structured_fields"]["actor"], "analyst@example.test")
        self.assertEqual(projected["structured_fields"]["source_address"], "192.0.2.40")
        self.assertIn("private-name", projected["structured_fields"]["resource"])
        self.assertEqual(
            projected["structured_fields"]["resources[0].name"], "private-child"
        )
        # A credential-looking value is redacted even when a producer labels it as an ID.
        self.assertEqual(
            projected["structured_fields"]["credential_id"],
            "<redacted:cloud-credential>",
        )
        exact = {item["path"]: item["value"] for item in projected["exact_attributes"]}
        self.assertEqual(exact["/actor"], "analyst@example.test")
        self.assertEqual(exact["/source_address"], "192.0.2.40")
        self.assertIn("private-name", exact["/resource"])
        self.assertEqual(exact["/resources/0/name"], "private-child")
        self.assertNotIn("/credential_id", exact)
        self.assertTrue(projected["exact_attribute_metadata"]["source_hydration_required"])
        self.assertIn(
            {"reason": "unsafe_credential_value", "count": 1},
            projected["exact_attribute_metadata"]["omission_counts"],
        )

    def test_process_free_text_redacts_in_band_values_but_retains_behavior(self) -> None:
        command = (
            "client --mode scan --password swordfish --host 198.51.100.8 "
            "--token=opaque-value user@example.test"
        )
        result = project_event(
            "ocsf_process_activity",
            "evt-process",
            {
                "semantic_class": "process",
                "ocsf": {"time": 1_700_000_001, "activity_name": "Launch"},
                "process": {"command_line": command, "name": "client"},
                "password": "second-secret",
                "secret_key": 123456,
                "status": "running",
            },
            "support-process",
        )
        self.assertIn("client", result["action_text"])
        self.assertIn("--mode scan", result["action_text"])
        for value in (
            "swordfish",
            "opaque-value",
            "second-secret",
            "198.51.100.8",
            "user@example.test",
        ):
            self.assertNotIn(value, json.dumps(result))
        self.assertIn("<redacted:secret>", result["action_text"])
        self.assertEqual(
            result["structured_fields"]["password"]["classification"], "secret"
        )
        self.assertEqual(
            result["structured_fields"]["secret_key"]["classification"], "secret"
        )
        self.assertNotIn("123456", result["semantic_text"])
        exact_paths = {item["path"] for item in result["exact_attributes"]}
        self.assertNotIn("/process/command_line", exact_paths)
        self.assertNotIn("/password", exact_paths)
        self.assertNotIn("/secret_key", exact_paths)
        self.assertTrue(result["exact_attribute_metadata"]["source_hydration_required"])

    def test_exact_attributes_copy_safe_scalars_without_normalization(self) -> None:
        event = {
            "actor": "  operator@example.test  ",
            "status": " Success\n",
            "enabled": True,
            "retry_count": 3,
            "ratio": 0.125,
        }
        projected = project_event(
            "ocsf_api_activity", "exact-scalars", event, "support-exact"
        )
        exact = {item["path"]: item["value"] for item in projected["exact_attributes"]}
        self.assertEqual(exact, {f"/{key}": value for key, value in event.items()})
        metadata = projected["exact_attribute_metadata"]
        self.assertEqual(metadata["selected_count"], 5)
        self.assertEqual(metadata["known_omitted_scalar_count"], 0)
        self.assertFalse(metadata["scan_truncated"])
        self.assertFalse(metadata["source_hydration_required"])

    def test_configuration_metric_and_network_rows_remain_semantic(self) -> None:
        fixtures = [
            (
                "ocsf_ext_livefire_configuration_snapshot",
                {
                    "semantic_class": "extension",
                    "class": "configuration_snapshot",
                    "snapshot_kind": "service configuration",
                    "subject": "resource-9087",
                    "subject_kind": "datastore",
                    "state": "noncompliant",
                    "observer_service": "configuration monitor",
                },
                "state",
                "noncompliant",
            ),
            (
                "ocsf_ext_livefire_system_metric",
                {
                    "semantic_class": "extension",
                    "class": "system_metric",
                    "metric": "cpu utilization",
                    "value_milli": 98250,
                    "unit": "percent_milli",
                    "subject": "process-775",
                    "device": "device-441",
                },
                "structured_only",
                "<quantity:1e4>",
            ),
            (
                "ocsf_network_activity",
                {
                    "semantic_class": "network",
                    "action": "connect",
                    "protocol_stack": "ip:tcp:https",
                    "src_ip": "203.0.113.4",
                    "dst_ip": "198.51.100.3",
                    "bytes_out": 2048,
                    "duration_millis": 3100,
                    "status": "allowed",
                },
                "activity",
                "connect",
            ),
        ]
        for relation, event, kind, expected_text in fixtures:
            with self.subTest(relation=relation):
                result = project_event(relation, "event", event, "support")
                expected_disposition = (
                    "structured_only_occurrence"
                    if relation == "ocsf_ext_livefire_system_metric"
                    else "direct_semantic_document"
                )
                self.assertEqual(result["terminal_disposition"], expected_disposition)
                self.assertEqual(result["document_kind"], kind)
                self.assertIn(expected_text, result["semantic_text"])
        metric = project_event(
            fixtures[1][0], "metric", fixtures[1][1], "support-metric"
        )
        self.assertEqual(metric["structured_fields"]["value_milli"], 98250)
        self.assertEqual(
            metric["disposition_reason"], "awaits_deterministic_window_derivation"
        )

    def test_projection_and_group_digests_have_deliberate_identity_boundaries(self) -> None:
        event_a = {
            "operation": "ListResources",
            "status": "Success",
            "actor": "operator-a@example.test",
            "support_ref": "typed-support-a",
            "process": {"pid": 101},
            "hostIdentifier": "host-a",
            "ocsf": {
                "time": "2025-01-01T00:00:00Z",
                "class_uid": 6003,
                "unmapped": {"$token/0": "volatile-a"},
            },
        }
        # Different map insertion order is byte-identical after projection.
        event_b = {
            "ocsf": {
                "unmapped": {"$token/0": "volatile-a"},
                "class_uid": 6003,
                "time": "2025-01-01T00:00:00Z",
            },
            "hostIdentifier": "host-a",
            "process": {"pid": 101},
            "support_ref": "typed-support-a",
            "status": "Success",
            "actor": "operator-a@example.test",
            "operation": "ListResources",
        }
        first = project_event("ocsf_api_activity", "event-a", event_a, "support-a")
        reordered = project_event("ocsf_api_activity", "event-a", event_b, "support-a")
        occurrence = project_event(
            "ocsf_api_activity",
            "event-b",
            {
                **event_a,
                "actor": "operator-b@example.test",
                "support_ref": "typed-support-b",
                "process": {"pid": 202},
                "hostIdentifier": "host-b",
                "ocsf": {
                    **event_a["ocsf"],
                    "time": "2025-01-02T00:00:00Z",
                    "unmapped": {"$token/0": "volatile-b"},
                },
            },
            "support-b",
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first["projection_sha256"], occurrence["projection_sha256"])
        self.assertEqual(first["semantic_group_sha256"], occurrence["semantic_group_sha256"])
        projection_material = {key: value for key, value in first.items() if key != "projection_sha256"}
        self.assertEqual(
            first["projection_sha256"], sha256_bytes(canonical_json_bytes(projection_material))
        )

    def test_event_time_variants_and_missing_time_are_explicit(self) -> None:
        milliseconds = project_event(
            "ocsf_event_log_activity", "a", {"ocsf": {"time": 1_700_000_000_000}}, "s-a"
        )
        iso = project_event(
            "ocsf_event_log_activity", "b", {"event_time": "2023-11-14T22:13:20Z"}, "s-b"
        )
        missing = project_event("ocsf_event_log_activity", "c", {"action": "observe"}, "s-c")
        unparsed = project_event(
            "ocsf_event_log_activity", "d", {"timestamp": "producer-local-clock"}, "s-d"
        )
        self.assertEqual(milliseconds["event_metadata"]["event_time"], "2023-11-14T22:13:20.000Z")
        self.assertEqual(iso["event_metadata"]["event_time_availability"], "available")
        self.assertEqual(missing["event_metadata"]["event_time_availability"], "missing")
        self.assertIsNone(missing["event_metadata"]["event_time"])
        self.assertEqual(unparsed["event_metadata"]["event_time_availability"], "present_unparsed")

    def test_non_jcs_safe_integer_is_semantic_only_and_requires_hydration(self) -> None:
        large = 278_037_780_140_032_000
        projected = project_event(
            "ocsf_ext_livefire_configuration_snapshot",
            "large-integer",
            {"state": "observed", "exact_counter": large},
            "support-large-integer",
        )
        self.assertEqual(projected["structured_fields"]["exact_counter"], str(large))
        self.assertIn(str(large), projected["semantic_text"])
        self.assertNotIn(
            "/exact_counter", {item["path"] for item in projected["exact_attributes"]}
        )
        self.assertIn(
            {"reason": "non_jcs_safe_integer", "count": 1},
            projected["exact_attribute_metadata"]["omission_counts"],
        )
        self.assertTrue(projected["exact_attribute_metadata"]["source_hydration_required"])
        self.assertEqual(len(projected["projection_sha256"]), 64)

    def test_semantic_group_drops_transport_identity_noise_and_buckets_quantities(self) -> None:
        first_event = {
            "ocsf": {
                "time": 1_700_000_000_000,
                "activity_id": 6,
                "class_uid": 4001,
                "metadata": {
                    "version": "1.8.0",
                    "product": {"name": "normalizer-a", "vendor_name": "vendor-a"},
                },
                "unmapped": {
                    "endtime": "2025-01-01T00:00:01Z",
                    "src_mac": "00:11:22:33:44:55",
                    "dest_mac": "66:77:88:99:aa:bb",
                    "src_content": "opaque-packet-a",
                    "bytes": 145,
                },
            },
            "support_ref": "support-a",
            "protocol_stack": "ip:tcp:https",
            "status": "allowed",
        }
        second_event = json.loads(json.dumps(first_event))
        second_event["ocsf"]["time"] = 1_700_000_010_000
        second_event["ocsf"]["metadata"] = {
            "version": "9.9.9",
            "product": {"name": "normalizer-b", "vendor_name": "vendor-b"},
        }
        second_event["ocsf"]["unmapped"].update(
            {
                "endtime": "2025-01-01T00:10:01Z",
                "src_mac": "aa:aa:aa:aa:aa:aa",
                "dest_mac": "bb:bb:bb:bb:bb:bb",
                "src_content": "opaque-packet-b",
                "bytes": 199,
            }
        )
        second_event["support_ref"] = "support-b"

        first = project_event("ocsf_network_activity", "event-a", first_event, "support-a")
        second = project_event("ocsf_network_activity", "event-b", second_event, "support-b")
        self.assertEqual(first["semantic_group_sha256"], second["semantic_group_sha256"])
        self.assertIn("<quantity:1e2>", first["semantic_text"])
        for forbidden in (
            "00:11:22:33:44:55",
            "opaque-packet-a",
            "normalizer-a",
            "vendor-a",
            "2025-01-01T00:00:01Z",
            "support-a",
        ):
            self.assertNotIn(forbidden, first["semantic_text"])
        self.assertNotEqual(first["projection_sha256"], second["projection_sha256"])

        changed = json.loads(json.dumps(second_event))
        changed["status"] = "blocked"
        blocked = project_event("ocsf_network_activity", "event-c", changed, "support-c")
        self.assertNotEqual(first["semantic_group_sha256"], blocked["semantic_group_sha256"])

    def test_unknown_relation_and_invalid_json_are_terminal_not_silent(self) -> None:
        unknown = project_event(
            "ocsf_future_activity",
            "future-event",
            {"action": "generic action", "status": "unknown"},
            "future-support",
        )
        invalid = project_event(
            "ocsf_api_activity", "broken-event", "{not-json", "broken-support"
        )
        self.assertEqual(unknown["terminal_disposition"], "structured_only_occurrence")
        self.assertEqual(unknown["disposition_reason"], "unknown_typed_relation")
        self.assertIn("generic action", unknown["semantic_text"])
        self.assertEqual(invalid["terminal_disposition"], "structured_only_occurrence")
        self.assertEqual(invalid["disposition_reason"], "typed_event_unavailable")
        self.assertEqual(invalid["event_metadata"]["event_time_availability"], "missing")
        self.assertIn("input_error", invalid["event_metadata"])
        self.assertTrue(invalid["exact_attribute_metadata"]["source_hydration_required"])

        for unavailable in (None, [], "{not-json"):
            projected = project_event(
                "ocsf_api_activity",
                "evt-unavailable",
                unavailable,
                "support-unavailable",
            )
            self.assertEqual(projected["exact_attributes"], [])
            self.assertTrue(
                projected["exact_attribute_metadata"]["source_hydration_required"]
            )
    def test_recursive_projection_is_hard_bounded(self) -> None:
        event = {
            f"field_{index:03d}": "x" * 400 for index in range(MAX_LEAVES * 2)
        }
        result = project_event("ocsf_inventory_info", "large", event, "support-large")
        self.assertLessEqual(len(result["structured_fields"]), MAX_LEAVES)
        self.assertTrue(result["event_metadata"]["projection_truncated"])
        self.assertEqual(result["event_metadata"]["projection_leaf_limit"], MAX_LEAVES)
        self.assertLessEqual(len(result["semantic_text"]), MAX_SEMANTIC_TEXT_CHARS)
        for facet in ("action_text", "target_text", "context_text", "outcome_text"):
            self.assertLessEqual(len(result[facet]), MAX_FACET_TEXT_CHARS)
        exact_metadata = result["exact_attribute_metadata"]
        self.assertEqual(exact_metadata["selected_count"], 256)
        self.assertIn(
            {"reason": "attribute_limit", "count": MAX_LEAVES * 2 - 256},
            exact_metadata["omission_counts"],
        )
        self.assertTrue(exact_metadata["source_hydration_required"])

    def test_typed_behavior_has_priority_over_large_unmapped_bags(self) -> None:
        result = project_event(
            "ocsf_process_activity",
            "priority-event",
            {
                "ocsf": {
                    "time": 1_700_000_000_000,
                    "unmapped": {f"field_{index:03d}": "noise" for index in range(300)},
                },
                "process": {"command_line": "portable-tool --mode inspect"},
                "status": "denied",
            },
            "priority-support",
        )
        self.assertTrue(result["event_metadata"]["projection_truncated"])
        self.assertIn("portable-tool --mode inspect", result["action_text"])
        self.assertIn("status=denied", result["outcome_text"])
        self.assertIn("portable-tool --mode inspect", result["semantic_text"])
        self.assertIn("status=denied", result["semantic_text"])

    def test_each_semantic_role_has_a_reserved_embedding_text_budget(self) -> None:
        long_value = "generic-value-" * 20
        result = project_event(
            "ocsf_api_activity",
            "role-budget-event",
            {
                "request": {f"action_{index}": long_value for index in range(8)},
                "resource": {f"target_{index}": long_value for index in range(8)},
                **{f"context_{index}": long_value for index in range(8)},
                "state": "disabled",
                "status": "denied",
                "severity_name": "Critical",
            },
            "role-budget-support",
        )
        self.assertLessEqual(len(result["semantic_text"]), MAX_SEMANTIC_TEXT_CHARS)
        self.assertIn("state=disabled", result["semantic_text"])
        self.assertIn("status=denied", result["semantic_text"])
        self.assertIn("severity_name=Critical", result["semantic_text"])

    def test_vendor_identifier_aliases_and_opaque_secret_fields_are_safe(self) -> None:
        result = project_event(
            "ocsf_authentication",
            "alias-event",
            {
                "activity_name": "Login",
                "computer_name": "SECRET-WORKSTATION",
                "source_hostname": "private.internal",
                "workstation_name": "FINANCE-LAPTOP",
                "user_name": "operator@example.invalid",
                "x_api_key": "opaque-value-without-standard-prefix",
                "aws_secret_access_key": "another-opaque-value",
                "passphrase": "correct horse battery staple",
                "status": "success",
            },
            "alias-support",
        )
        semantic = " ".join(
            result[field]
            for field in (
                "semantic_text",
                "action_text",
                "target_text",
                "context_text",
                "outcome_text",
            )
        )
        for forbidden in (
            "SECRET-WORKSTATION",
            "private.internal",
            "FINANCE-LAPTOP",
            "operator@example.invalid",
            "opaque-value-without-standard-prefix",
            "another-opaque-value",
            "correct horse battery staple",
        ):
            self.assertNotIn(forbidden, semantic)
        self.assertEqual(result["structured_fields"]["x_api_key"]["classification"], "secret")
        exact_paths = {row["path"] for row in result["exact_attributes"]}
        self.assertNotIn("/x_api_key", exact_paths)
        self.assertNotIn("/aws_secret_access_key", exact_paths)
        self.assertNotIn("/passphrase", exact_paths)

    def test_projector_source_contains_no_hunt_or_benchmark_literals(self) -> None:
        source = (ROOT / "src/livefire_rag/evidence_projection.py").read_text(encoding="utf-8").lower()
        prohibited = [
            "fro" + "thly",
            "coin" + "hive",
            "bots" + "v3",
            "h" + "door",
            "known" + "-answer",
            "expected" + "-answer",
        ]
        for literal in prohibited:
            self.assertNotIn(literal, source)


if __name__ == "__main__":
    unittest.main()
