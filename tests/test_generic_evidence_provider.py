from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import duckdb
import numpy as np

from livefire_rag.canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    write_canonical_json,
)
from livefire_rag.evidence_bundle import (
    COVERAGE_SCHEMA_REF,
    DERIVATION_MEMBERSHIP_SCHEMA_REF,
    DERIVED_DOCUMENT_SCHEMA_REF,
    DOCUMENT_SCHEMA_REF,
    EMBEDDING_SCHEMA_REF,
    INDEX_FORMAT_DESCRIPTOR,
    INDEX_FORMAT_REF,
    INPUT_SCHEMA_REF,
    OCCURRENCE_SCHEMA_REF,
    OUTPUT_SCHEMA_REF,
    PHYSICAL_PROFILE_REF,
    PROTOCOL,
    PROVIDER_EXECUTABLE_ARTIFACT,
    PROVIDER_REF,
    RETRIEVAL_POLICY_REF,
    TOOL_REF,
    package_evidence_provider_bundle,
)
from livefire_rag.evidence_provider import EvidenceProvider, ProviderError
from livefire_rag.evidence_service import EvidenceError, EvidenceIndex, EvidenceService
from livefire_rag.evidence_service import validate_evidence_value


ROOT = Path(__file__).resolve().parents[1]
SDK_SPECS = ROOT.parent / "livefire-sdk/specs"
ZERO = "0" * 64


def _ref(name: str, material: object | None = None) -> dict[str, str]:
    return component_ref(f"test.{name}", "1", material if material is not None else {"name": name})


def _write_parquet(path: Path, rows: list[dict]) -> None:
    jsonl = path.with_suffix(".jsonl")
    with jsonl.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row, newline=True))
    connection = duckdb.connect(":memory:")
    input_sql = str(jsonl).replace("'", "''")
    output_sql = str(path).replace("'", "''")
    connection.execute(
        f"COPY (SELECT * FROM read_json_auto('{input_sql}', format='newline_delimited')) "
        f"TO '{output_sql}' (FORMAT parquet)",
    )
    connection.close()
    jsonl.unlink()


def _document(document_id: str, text: str, kind: str, count: int, policy: dict) -> dict:
    value = {
        "schema_version": "livefire.rag.evidence-document/1",
        "document_id": document_id,
        "document_sha256": "",
        "document_kind": kind,
        "representation": "direct",
        "searchable": True,
        "projection_policy": policy,
        "relation_identities": [
            {
                "namespace": "ocsf",
                "relation": "process_activity",
                "ocsf_category_uid": 1,
                "ocsf_class_uid": 1007,
                "ocsf_activity_id": 1,
            }
        ],
        "semantic_projection": {
            "text": text,
            "facets": [{"name": "action", "values": [text.split()[0]]}],
        },
        "occurrence_count": count,
        "exact_attributes": [],
    }
    value["document_sha256"] = canonical_sha256_omitting(value, ("document_sha256",))
    return value


def _occurrence(
    occurrence_id: str,
    document_id: str,
    host: str,
    snapshot: dict,
    snapshot_profile: dict,
    projection_policy: dict,
) -> dict:
    return {
        "schema_version": "livefire.rag.evidence-occurrence-row/1",
        "occurrence_id": occurrence_id,
        "event_time": "2026-01-01T00:00:00Z",
        "relation_identity": {
            "namespace": "ocsf",
            "relation": "process_activity",
            "ocsf_category_uid": 1,
            "ocsf_class_uid": 1007,
            "ocsf_activity_id": 1,
        },
        "source_pointer": {
            "schema_version": "livefire.source-record-pointer/1",
            "snapshot": snapshot,
            "snapshot_profile": snapshot_profile,
            "record_id": occurrence_id,
            "record_sha256": component_ref("record", occurrence_id, {"id": occurrence_id})["sha256"],
            "locator": {"kind": "record_id_only"},
        },
        "projection_policy": projection_policy,
        "terminal_disposition": "direct_semantic_document",
        "document_ids": [document_id],
        "reason_codes": [],
        "exact_attributes": [{"namespace": "ocsf", "path": "/device/name", "value": host}],
        "exact_attribute_projection": {
            "contract": "bounded_value_exact_typed_json_scalar_subset",
            "selected_count": 1,
            "scalars_scanned": 1,
            "known_omitted_scalar_count": 0,
            "omitted_subtree_count": 0,
            "omission_counts": [],
            "scan_truncated": False,
            "source_hydration_required": False,
            "limits": {
                "max_attributes": 256,
                "max_scalars_scanned": 512,
                "max_list_items": 64,
                "max_string_utf8_bytes": 1024,
                "max_path_chars": 1024,
            },
        },
    }


def build_index(root: Path, *, include_derivation: bool = False) -> tuple[dict, dict]:
    snapshot = _ref("snapshot")
    snapshot_profile = _ref("snapshot-profile")
    projection_policy = _ref("projection-policy")
    embedding_policy = json.loads(
        (ROOT / "profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json")
        .read_text(encoding="utf-8")
    )
    embedding_profile = component_ref("test.embedding-profile", "1", embedding_policy)
    write_canonical_json(root / "embedding-profile.json", embedding_policy)
    documents = [
        _document("doc-firewall", "disable firewall powershell", "activity", 2, projection_policy),
        _document("doc-storage", "upload archive object storage", "activity", 1, projection_policy),
    ]
    occurrences = [
        _occurrence("occ-a", "doc-firewall", "host-a", snapshot, snapshot_profile, projection_policy),
        _occurrence("occ-b", "doc-firewall", "host-b", snapshot, snapshot_profile, projection_policy),
        _occurrence("occ-c", "doc-storage", "host-c", snapshot, snapshot_profile, projection_policy),
    ]
    derivation_policy = _ref("derivation-policy")
    derived_documents: list[dict] = []
    memberships: list[dict] = []
    if include_derivation:
        document_id = "ddoc-" + "a" * 64
        derived = {
            "schema_version": "livefire.rag.evidence-derived-document/1",
            "document_id": document_id,
            "document_sha256": "",
            "document_kind": "metric_window",
            "representation": "derived",
            "searchable": True,
            "source_snapshot": snapshot,
            "base_projection_pack": _ref("projection-pack"),
            "derivation_policy": derivation_policy,
            "relation_identities": [{"namespace": "ocsf", "relation": "process_activity"}],
            "semantic_projection": {"text": "repeated firewall changes over a fixed window", "facets": []},
            "derivation": {
                "group_sha256": "2" * 64,
                "input_count": 2,
                "input_set_sha256": "3" * 64,
                "closure_state": "complete",
                "completeness_state": "complete",
                "aggregate_material": {"count": 2},
            },
            "occurrence_count": 2,
        }
        derived["document_sha256"] = canonical_sha256_omitting(derived, ("document_sha256",))
        derived_documents.append(derived)
        for number, occurrence_id in enumerate(("occ-a", "occ-b"), 1):
            membership = {
                "schema_version": "livefire.rag.evidence-derivation-membership-row/1",
                "membership_id": "dmem-" + str(number) * 64,
                "membership_sha256": "",
                "derived_document_id": document_id,
                "occurrence_id": occurrence_id,
                "input_role": "window_member",
                "derivation_policy": derivation_policy,
            }
            membership["membership_sha256"] = canonical_sha256_omitting(
                membership, ("membership_sha256",)
            )
            memberships.append(membership)
    all_documents = [*documents, *derived_documents]
    basis = []
    for position in range(len(all_documents)):
        vector = [0.0] * embedding_policy["dimensions"]
        vector[position] = 1.0
        basis.append(vector)
    if include_derivation:
        basis[-1][0] = 2 ** -0.5
        basis[-1][1] = 2 ** -0.5
        basis[-1][2] = 0.0
    embeddings = [
        {
            "schema_version": "livefire.rag.evidence-embedding-row/1",
            "document_id": document["document_id"],
            "document_sha256": document["document_sha256"],
            "purpose": "semantic_search",
            "embedding_profile": embedding_profile,
            "dimensions": embedding_policy["dimensions"],
            "normalization": "l2",
            "vector": vector,
        }
        for document, vector in zip(all_documents, basis, strict=True)
    ]
    _write_parquet(root / "documents.parquet", documents)
    _write_parquet(root / "occurrences.parquet", occurrences)
    _write_parquet(root / "embeddings.parquet", embeddings)
    if include_derivation:
        _write_parquet(root / "derivation-documents.parquet", derived_documents)
        _write_parquet(root / "derivation-memberships.parquet", memberships)
    coverage = {
        "schema_version": "livefire.rag.evidence-coverage-report/1",
        "source_snapshots": [snapshot],
        "projection_policy": projection_policy,
        "derivation_policies": [derivation_policy] if include_derivation else [],
        "closure": {
            "source_record_count": 3,
            "terminal_disposition_count": 3,
            "unaccounted_record_count": 0,
            "multiply_dispositioned_record_count": 0,
            "all_source_records_dispositioned": True,
            "by_terminal_disposition": {
                "direct_semantic_document": 3,
                "semantic_group_occurrence": 0,
                "derived_document_input": 0,
                "structured_only_occurrence": 0,
                "rejected": 0,
            },
        },
        "documents": {
            "total": 2,
            "searchable": 2,
            "by_kind": {
                "activity": 2,
                "state": 0,
                "state_transition": 0,
                "metric_window": 0,
                "network_window": 0,
                "entity": 0,
                "detection": 0,
                "structured_only": 0,
            },
        },
        "relation_coverage": [],
        "pointer_resolution": {
            "pointer_count": 3,
            "resolved_count": 3,
            "unresolved_count": 0,
            "all_pointers_resolved": True,
        },
        "reason_counts": [],
    }
    write_canonical_json(root / "coverage-report.json", coverage)
    fixture_artifact = {
        "path": "occurrences.parquet", "sha256": "4" * 64, "bytes": 1,
        "media_type": "application/vnd.apache.parquet",
    }
    base_manifest = {
        "schema_version": "livefire.index/1",
        "index_id": "test.base-index", "index_version": "1",
        "index_kind": "generic_evidence", "format": INDEX_FORMAT_REF,
        "builder": _ref("builder"),
        "source_bindings": [{
            "source_snapshot": snapshot,
            "source_snapshot_profile": snapshot_profile,
            "source_admission_receipt": _ref("source-admission"),
            "record_identity_policy": _ref("record-identity"),
        }],
        "policies": {"projection": projection_policy, "embedding": embedding_profile},
        "objects": [fixture_artifact], "source_pointer_table": fixture_artifact,
        "coverage": {"source_records": 3, "indexed_documents": len(all_documents), "excluded_records": 0},
        "query_time_contract": {
            "mode": "local_component", "network": ["loopback:fixture"],
            "secret_handles": [], "vendor_services": [],
            "required_local_components": [embedding_profile],
        },
        "governance": {"inherits_source_confidentiality": True, "inherits_source_retention": True},
    }
    write_canonical_json(root / "base-index-manifest.json", base_manifest)
    write_canonical_json(root / "index-format-descriptor.json", INDEX_FORMAT_DESCRIPTOR)
    build_report = {
        "schema_version": "local-test.evidence-index-build-report/1",
        "admission_status": "local_test_not_production_admitted",
    }
    write_canonical_json(root / "build-report.json", build_report)
    objects = {
        "documents": artifact_ref(root / "documents.parquet", "documents.parquet", "application/vnd.apache.parquet"),
        "occurrences": artifact_ref(root / "occurrences.parquet", "occurrences.parquet", "application/vnd.apache.parquet"),
        "embeddings": artifact_ref(root / "embeddings.parquet", "embeddings.parquet", "application/vnd.apache.parquet"),
        "coverage_report": artifact_ref(root / "coverage-report.json", "coverage-report.json", "application/json"),
        "embedding_profile": artifact_ref(root / "embedding-profile.json", "embedding-profile.json", "application/json"),
        "base_manifest": artifact_ref(root / "base-index-manifest.json", "base-index-manifest.json", "application/json"),
        "format_descriptor": artifact_ref(root / "index-format-descriptor.json", "index-format-descriptor.json", "application/json"),
        "build_report": artifact_ref(root / "build-report.json", "build-report.json", "application/json"),
    }
    if include_derivation:
        objects.update(
            {
                "derivation_documents": artifact_ref(root / "derivation-documents.parquet", "derivation-documents.parquet", "application/vnd.apache.parquet"),
                "derivation_memberships": artifact_ref(root / "derivation-memberships.parquet", "derivation-memberships.parquet", "application/vnd.apache.parquet"),
            }
        )
    lock = {
        "schema_version": "livefire.object-lock/1",
        "objects": sorted(objects.values(), key=lambda value: (value["path"], value["sha256"])),
    }
    write_canonical_json(root / "objects.lock.json", lock)
    objects["object_lock"] = artifact_ref(root / "objects.lock.json", "objects.lock.json", "application/vnd.livefire.object-lock+json")
    manifest = {
        "schema_version": "livefire.rag.evidence-index/1",
        "component": {"id": "test.evidence-index", "version": "1", "sha256": ""},
        "projection_pack": _ref("projection-pack"),
        "base_index_manifest": component_ref("test.evidence-index.base", "1", base_manifest),
        "index_format_descriptor": INDEX_FORMAT_REF,
        "physical_profile": PHYSICAL_PROFILE_REF,
        "source_snapshots": [snapshot],
        "document_kinds": [
            "activity", "state", "state_transition", "metric_window", "network_window", "entity", "detection", "structured_only"
        ],
        "row_schemas": {
            "evidence_document": DOCUMENT_SCHEMA_REF,
            "evidence_occurrence": OCCURRENCE_SCHEMA_REF,
            "evidence_embedding": EMBEDDING_SCHEMA_REF,
            "coverage_report": COVERAGE_SCHEMA_REF,
        },
        "projection_policy": projection_policy,
        "derivation_policies": [derivation_policy] if include_derivation else [],
        "embedding_profiles": [embedding_profile],
        "objects": objects,
        "coverage": {
            "source_record_count": 3,
            "terminal_disposition_count": 3,
            "document_count": len(all_documents),
            "searchable_document_count": len(all_documents),
            "unaccounted_record_count": 0,
            "unresolved_pointer_count": 0,
        },
        "query_contract": {
            "canonical_format": "parquet",
            "source_filters_apply_to_occurrences": True,
            "semantic_groups_preserve_occurrences": True,
            "derived_caches_authoritative": False,
            "candidate_results_are_evidence": False,
            "tie_break": "ranking_score_desc_document_id_asc",
        },
    }
    if include_derivation:
        manifest["derivation_packs"] = [_ref("derivation-pack")]
        manifest["row_schemas"].update(
            {
                "derivation_document": DERIVED_DOCUMENT_SCHEMA_REF,
                "derivation_membership": DERIVATION_MEMBERSHIP_SCHEMA_REF,
            }
        )
        manifest["coverage"].update(
            {"derived_document_count": len(derived_documents), "derivation_membership_count": len(memberships)}
        )
    manifest["component"]["sha256"] = canonical_sha256_omitting(manifest, ("component", "sha256"))
    write_canonical_json(root / "manifest.json", manifest)
    return manifest, embedding_policy


def build_loadout(root: Path, manifest: dict, profile: dict) -> tuple[dict, dict]:
    profile_path = root / "embedding-policy.json"
    write_canonical_json(profile_path, profile)
    receipt = {
        "schema_version": "livefire.index-admission/1",
        "receipt_id": "local-test-receipt",
        "receipt_version": "1",
        "build_request_sha256": ZERO,
        "build_report_sha256": "1" * 64,
        "index_manifest_sha256": manifest["component"]["sha256"],
        "verifier": _ref("local-test-verifier"),
        "checks": {
            "object_digests": True,
            "source_binding": True,
            "safe_paths": True,
            "schema_profiles": True,
            "coverage_closure": True,
            "pointer_closure": True,
            "offline_query_conformance": True,
            "conformance": True,
            "deterministic_rebuild": True,
        },
        "disposition": "admitted",
        "authority_signature": "local-test:not-a-production-signature",
    }
    receipt_path = root / "index-admission-receipt.json"
    write_canonical_json(receipt_path, receipt)
    receipt_ref = component_ref("test.index-admission-receipt", "1", receipt)
    contract = {
        "mode": "local_component",
        "network": ["loopback:http://127.0.0.1:65534"],
        "secret_handles": [],
        "vendor_services": [],
    }
    limits = {
        "request_bytes": 65536,
        "result_bytes": 1048576,
        "wall_time_ms": 30000,
        "memory_bytes": 268435456,
        "max_candidates": 1000,
    }
    lock = {
        "schema_version": "livefire.tool-binding-lock/1",
        "descriptor": TOOL_REF,
        "provider": PROVIDER_REF,
        "executable": PROVIDER_EXECUTABLE_ARTIFACT,
        "input_schema": INPUT_SCHEMA_REF,
        "output_schema": OUTPUT_SCHEMA_REF,
        "index": manifest["component"],
        "index_format": INDEX_FORMAT_REF,
        "index_admission_receipt": receipt_ref,
        "source_snapshots": manifest["source_snapshots"],
        "retrieval_policy": RETRIEVAL_POLICY_REF,
        "query_time_contract": contract,
        "protocol": PROTOCOL,
        "limits": limits,
    }
    lock_path = root / "tool-binding-lock.json"
    write_canonical_json(lock_path, lock)
    binding_sha = component_ref("test.binding", "1", lock)["sha256"]
    mounts = [
        {"logical_name": "evidence-index", "role": "index", "component": manifest["component"], "access": "read_only", "process_path": str(root)},
        {"logical_name": "tool-binding-lock", "role": "policy", "component": {"id": "test.binding", "version": "1", "sha256": binding_sha}, "access": "read_only", "process_path": str(lock_path)},
        {"logical_name": "index-admission-receipt", "role": "policy", "component": receipt_ref, "access": "read_only", "process_path": str(receipt_path)},
        {"logical_name": "embedding-profile", "role": "model", "component": manifest["embedding_profiles"][0], "access": "read_only", "process_path": str(profile_path)},
    ]
    params = {
        "provider": PROVIDER_REF,
        "tools": [TOOL_REF],
        "indexes": [manifest["component"]],
        "source_snapshots": manifest["source_snapshots"],
        "binding_lock_sha256": binding_sha,
        "query_time_contract": contract,
        "limits": limits,
        "mounts": mounts,
    }
    return params, lock


class GenericEvidenceProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp())
        self.index_root = self.temp / "index"
        self.index_root.mkdir()
        self.manifest, self.profile = build_index(self.index_root)
        self.index = EvidenceIndex.open(self.index_root, expected_format=INDEX_FORMAT_REF, sdk_specs=SDK_SPECS)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _request(self, request_id: str, method: str, params: dict) -> dict:
        return {
            "protocol": PROTOCOL,
            "id": request_id,
            "method": method,
            "params": params,
            "context": {"trace_id": f"trace-{request_id}", "deadline_unix_ms": int(time.time() * 1000) + 5000},
        }

    def test_occurrence_filters_run_before_document_eligibility(self) -> None:
        service = EvidenceService(self.index, sdk_specs=SDK_SPECS)
        result = service.search(
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "firewall",
                "top_n": 5,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {"attribute_predicates": [{"namespace": "ocsf", "path": "/device/name", "operator": "eq", "value": "host-a"}]},
            },
            int(time.time() * 1000) + 5000,
        )
        self.assertEqual(result["kind"], "pointer")
        self.assertEqual(result["candidates"][0]["matching_occurrence_count"], 1)
        self.assertEqual([row["occurrence_id"] for row in result["candidates"][0]["source_occurrences"]], ["occ-a"])
        self.assertEqual(result["coverage"]["status"], "complete")
        self.assertNotIn(
            "pilot_sample_not_corpus_coverage", result["coverage"]["reason_codes"]
        )

    def test_pilot_results_and_misses_are_explicitly_sample_scoped(self) -> None:
        self.index.manifest["pilot_sample"] = {
            "scope_status": "sample_only_not_corpus_coverage",
            "admission_status": "local_evaluation_only_not_sdk_admitted",
            "corpus_miss_definitive": False,
        }
        service = EvidenceService(self.index, sdk_specs=SDK_SPECS)
        pointer = service.search(
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "firewall", "top_n": 1,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
            },
            int(time.time() * 1000) + 5000,
        )
        miss = service.search(
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "firewall", "top_n": 1,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {"attribute_predicates": [{
                    "namespace": "ocsf", "path": "/missing",
                    "operator": "eq", "value": "absent",
                }]},
            },
            int(time.time() * 1000) + 5000,
        )
        for output in (pointer, miss):
            self.assertEqual(output["coverage"]["status"], "partial")
            self.assertIn(
                "pilot_sample_not_corpus_coverage", output["coverage"]["reason_codes"]
            )
        self.assertIn("not a corpus-wide miss", miss["miss"]["message"])

    def test_explicit_miss_and_missing_negative_attribute_does_not_match(self) -> None:
        service = EvidenceService(self.index, sdk_specs=SDK_SPECS)
        result = service.search(
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "anything",
                "top_n": 5,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {"attribute_predicates": [{"namespace": "ocsf", "path": "/missing", "operator": "not_eq", "value": "x"}]},
            },
            int(time.time() * 1000) + 5000,
        )
        self.assertEqual(result["kind"], "miss")
        self.assertNotIn("candidates", result)
        self.assertEqual(result["miss"]["reason"], "no_eligible_occurrences")

    def test_dense_and_fused_ranking_are_deterministic(self) -> None:
        service = EvidenceService(
            self.index,
            embed_query=lambda query, deadline: np.asarray(
                [1.0, *([0.0] * (self.profile["dimensions"] - 1))]
            ),
            sdk_specs=SDK_SPECS,
        )
        arguments = {
            "schema_version": "livefire.rag.evidence-search.input/1",
            "query": "firewall upload",
            "top_n": 2,
            "retrieval": {"methods": ["dense", "lexical"], "fusion": "reciprocal_rank"},
        }
        first = service.search(arguments, int(time.time() * 1000) + 5000)
        second = service.search(arguments, int(time.time() * 1000) + 5000)
        self.assertEqual(first, second)
        self.assertEqual(first["candidates"][0]["document_id"], "doc-firewall")
        self.assertIsNotNone(first["candidates"][0]["scores"]["fused_score_millionths"])

    def test_entity_filter_fails_closed(self) -> None:
        service = EvidenceService(self.index, sdk_specs=SDK_SPECS)
        with self.assertRaisesRegex(EvidenceError, "entity-membership"):
            service.search(
                {
                    "schema_version": "livefire.rag.evidence-search.input/1",
                    "query": "firewall",
                    "top_n": 1,
                    "retrieval": {"methods": ["lexical"], "fusion": "none"},
                    "filters": {"entity_ids": ["user:alice"]},
                },
                int(time.time() * 1000) + 5000,
            )

    def test_derived_membership_uses_only_matching_base_occurrences(self) -> None:
        derived_root = self.temp / "derived-index"
        derived_root.mkdir()
        build_index(derived_root, include_derivation=True)
        index = EvidenceIndex.open(
            derived_root, expected_format=INDEX_FORMAT_REF, sdk_specs=SDK_SPECS
        )
        service = EvidenceService(index, sdk_specs=SDK_SPECS)
        result = service.search(
            {
                "schema_version": "livefire.rag.evidence-search.input/1",
                "query": "repeated fixed window",
                "top_n": 5,
                "retrieval": {"methods": ["lexical"], "fusion": "none"},
                "filters": {
                    "attribute_predicates": [
                        {
                            "namespace": "ocsf",
                            "path": "/device/name",
                            "operator": "eq",
                            "value": "host-a",
                        }
                    ]
                },
            },
            int(time.time() * 1000) + 5000,
        )
        candidate = next(
            row for row in result["candidates"] if row["document_kind"] == "metric_window"
        )
        self.assertEqual(candidate["matching_occurrence_count"], 1)
        self.assertEqual(
            [row["occurrence_id"] for row in candidate["source_occurrences"]], ["occ-a"]
        )

    def test_provider_exact_binding_lifecycle_and_miss(self) -> None:
        params, _ = build_loadout(self.index_root, self.manifest, self.profile)
        provider = EvidenceProvider(sdk_specs=SDK_SPECS)
        handshake = provider.handle(self._request("1", "handshake", {}))
        self.assertEqual(handshake["tools"], [TOOL_REF])
        opened = provider.handle(self._request("2", "open", params))
        session_id = opened["session_id"]
        called = provider.handle(
            self._request(
                "3",
                "call",
                {
                    "session_id": session_id,
                    "tool": TOOL_REF,
                    "arguments": {
                        "schema_version": "livefire.rag.evidence-search.input/1",
                        "query": "firewall",
                        "top_n": 2,
                        "retrieval": {"methods": ["lexical"], "fusion": "none"},
                    },
                },
            )
        )
        self.assertEqual(called["output"]["kind"], "pointer")
        self.assertEqual(provider.handle(self._request("4", "health", {"session_id": session_id}))["status"], "ready")
        self.assertTrue(provider.handle(self._request("5", "close", {"session_id": session_id}))["closed"])

    def test_provider_rejects_digest_only_binding_claim(self) -> None:
        params, _ = build_loadout(self.index_root, self.manifest, self.profile)
        params["binding_lock_sha256"] = "f" * 64
        provider = EvidenceProvider(sdk_specs=SDK_SPECS)
        provider.handle(self._request("1", "handshake", {}))
        with self.assertRaisesRegex(ProviderError, "mounted lock"):
            provider.handle(self._request("2", "open", params))

    def test_development_profile_cannot_use_a_production_claiming_receipt(self) -> None:
        params, lock = build_loadout(self.index_root, self.manifest, self.profile)
        receipt_path = self.index_root / "index-admission-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["authority_signature"] = "production-authority:fixture"
        write_canonical_json(receipt_path, receipt)
        receipt_ref = component_ref("test.index-admission-receipt", "1", receipt)
        lock["index_admission_receipt"] = receipt_ref
        lock_path = self.index_root / "tool-binding-lock.json"
        write_canonical_json(lock_path, lock)
        binding_sha = component_ref("test.binding", "1", lock)["sha256"]
        params["binding_lock_sha256"] = binding_sha
        by_name = {row["logical_name"]: row for row in params["mounts"]}
        by_name["tool-binding-lock"]["component"]["sha256"] = binding_sha
        by_name["index-admission-receipt"]["component"] = receipt_ref
        provider = EvidenceProvider(sdk_specs=SDK_SPECS)
        provider.handle(self._request("1", "handshake", {}))
        with self.assertRaisesRegex(ProviderError, "development-only"):
            provider.handle(self._request("2", "open", params))

    def test_development_bundle_is_sdk_valid_and_contains_no_index(self) -> None:
        bundle = self.temp / "bundle"
        package_evidence_provider_bundle(ROOT, bundle, SDK_SPECS)
        self.assertFalse((bundle / "documents.parquet").exists())
        self.assertFalse((bundle / "lib/livefire_rag/evidence_geometry.py").exists())
        self.assertFalse((bundle / "lib/livefire_rag/evidence_pilot_eval.py").exists())
        self.assertTrue((bundle / "lib/rfc8785/_impl.py").is_file())
        binary = ROOT.parent / "livefire-sdk/target/debug/livefire-sdk"
        if not binary.is_file():
            self.skipTest("livefire-sdk debug binary is unavailable")
        completed = subprocess.run(
            [str(binary), "--specs", str(SDK_SPECS), "validate-bundle", "--manifest", str(bundle / "plugin.json"), "--root", str(bundle)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        isolated = subprocess.run(
            [
                sys.executable, "-S", "-c",
                "import rfc8785; assert rfc8785.dumps({'b': 2, 'a': 1}) == b'{\"a\":1,\"b\":2}'",
            ],
            cwd=self.temp,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(bundle / "lib"),
                "PYTHONNOUSERSITE": "1",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(isolated.returncode, 0, isolated.stderr)

    def test_rust_sdk_invoke_runs_packaged_pointer_miss_health_close(self) -> None:
        binary = ROOT.parent / "livefire-sdk/target/debug/livefire-sdk"
        if not binary.is_file():
            self.skipTest("livefire-sdk debug binary is unavailable")
        bundle = self.temp / "bundle-invoke"
        package_evidence_provider_bundle(ROOT, bundle, SDK_SPECS)
        params, _ = build_loadout(self.index_root, self.manifest, self.profile)
        deadline = int(time.time() * 1000) + 30_000

        def request(request_id: str, method: str, params_value: dict) -> dict:
            return {
                "protocol": PROTOCOL, "id": request_id, "method": method,
                "params": params_value,
                "context": {"trace_id": f"rust-{request_id}", "deadline_unix_ms": deadline},
            }

        pointer_arguments = {
            "schema_version": "livefire.rag.evidence-search.input/1",
            "query": "firewall", "top_n": 2,
            "retrieval": {"methods": ["lexical"], "fusion": "none"},
        }
        miss_arguments = {
            **pointer_arguments,
            "filters": {"attribute_predicates": [{
                "namespace": "ocsf", "path": "/missing",
                "operator": "eq", "value": "absent",
            }]},
        }
        requests = [
            request("1", "handshake", {}),
            request("2", "open", params),
            request("3", "call", {"session_id": "${session_id}", "tool": TOOL_REF, "arguments": pointer_arguments}),
            request("4", "call", {"session_id": "${session_id}", "tool": TOOL_REF, "arguments": miss_arguments}),
            request("5", "health", {"session_id": "${session_id}"}),
            request("6", "close", {"session_id": "${session_id}"}),
        ]
        transcript = self.temp / "invoke-requests.jsonl"
        with transcript.open("wb") as handle:
            for row in requests:
                handle.write(canonical_json_bytes(row, newline=True))
        completed = subprocess.run(
            [
                str(binary), "--specs", str(SDK_SPECS), "invoke",
                "--program", str(bundle / "bin/livefire-rag-evidence-provider"),
                "--requests", str(transcript), "--timeout-ms", "30000",
            ],
            cwd=self.temp, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        responses = [json.loads(line) for line in completed.stdout.splitlines() if line]
        self.assertEqual([row["id"] for row in responses], [str(n) for n in range(1, 7)])
        pointer = responses[2]["result"]["output"]
        miss = responses[3]["result"]["output"]
        validate_evidence_value("evidence-search.output.v1.schema.json", pointer, sdk_specs=SDK_SPECS)
        validate_evidence_value("evidence-search.output.v1.schema.json", miss, sdk_specs=SDK_SPECS)
        self.assertEqual(pointer["kind"], "pointer")
        self.assertGreater(len(pointer["candidates"]), 0)
        self.assertEqual(miss["kind"], "miss")
        self.assertEqual(responses[4]["result"]["status"], "ready")
        self.assertTrue(responses[5]["result"]["closed"])


if __name__ == "__main__":
    unittest.main()
