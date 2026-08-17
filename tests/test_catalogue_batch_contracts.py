from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from livefire_rag.canonical import canonical_sha256_omitting
from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"
SHA = "a" * 64


def request_row() -> dict[str, object]:
    return {
        "query_id": "q-1",
        "query": "encoded process",
        "mode": "fused",
        "top_n": 20,
        "relations": ["ocsf_process_activity"],
    }


def occurrence() -> dict[str, object]:
    return {
        "event_time_ms": None,
        "relation": "ocsf_process_activity",
        "snapshot_sha256": SHA,
        "mapping_sha256": SHA,
        "event_id": "event-1",
        "support_ref": "support-1",
    }


def dataset() -> dict[str, object]:
    component = {"id": "component", "version": "1", "sha256": SHA}
    return {
        "id": "dataset-1",
        "version": "1",
        "source_snapshot": component,
        "mapping": component,
        "included_relations": ["ocsf_process_activity"],
        "excluded_relations": [],
        "structured_only_relations": [],
    }


def result_row() -> dict[str, object]:
    return {
        "schema_version": "livefire.rag.catalogue-batch-search-result/1",
        "query_id": "q-1",
        "catalogue_sha256": SHA,
        "query": "encoded process",
        "mode": "fused",
        "top_n": 20,
        "relations": ["ocsf_process_activity"],
        "rank_merge": "reciprocal_rank_fusion_v1",
        "hits": [
            {
                "rank": 1,
                "reciprocal_rank_score": 1 / 61,
                "dataset": dataset(),
                "dataset_sha256": SHA,
                "index_sha256": SHA,
                "index_rank": 1,
                "hit": {
                    "rank": 1,
                    "document_id": "document-1",
                    "semantic_text": "encoded process",
                    "score": 1.0,
                    "dense_score": 1.0,
                    "lexical_score": 0.5,
                    "eligible_occurrence_count": 1,
                    "occurrences_exhausted": True,
                    "occurrences": [occurrence()],
                },
            }
        ],
    }


def run_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "livefire.rag.catalogue-batch-search-run/1",
        "component_sha256": "0" * 64,
        "status": "complete",
        "catalogue_sha256": SHA,
        "embedding_profile": {"id": "profile", "version": "1", "sha256": SHA},
        "requests": {"path": "requests.jsonl", "sha256": SHA, "bytes": 100, "rows": 1},
        "results": {"path": "results.jsonl", "sha256": SHA, "bytes": 200, "rows": 1},
        "request_count": 1,
        "result_count": 1,
        "modes": ["fused"],
        "top_n_values": [20],
        "relation_filters": [["ocsf_process_activity"]],
        "request_shapes": [
            {
                "mode": "fused",
                "top_n": 20,
                "relations": ["ocsf_process_activity"],
                "rows": 1,
            }
        ],
        "model": {
            "status": "used",
            "configured_model": "local-model",
            "returned_model": "local-model",
            "calls": 1,
        },
        "query_vectors": [
            {"composed_query_sha256": SHA, "vector_sha256": SHA, "dimensions": 4096}
        ],
        "rank_merge": {"policy": "reciprocal_rank_fusion_v1", "k": 60},
    }
    manifest["component_sha256"] = canonical_sha256_omitting(
        manifest, ("component_sha256",)
    )
    return manifest


def review_row() -> dict[str, object]:
    return {
        "schema_version": "livefire.rag.catalogue-review-pool-row/1",
        "candidate_id": "candidate-" + SHA,
        "query_id": "q-1",
        "query": "encoded process",
        "dataset_id": "dataset-1",
        "document_id": "document-1",
        "semantic_text": "encoded process",
        "eligible_occurrence_count": 1,
        "occurrences": [occurrence()],
    }


class CatalogueBatchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        cls.validators = {
            name: Draft202012Validator(schemas[name], registry=registry)
            for name in (
                "catalogue-batch-search-request.v1.schema.json",
                "catalogue-batch-search-result.v1.schema.json",
                "catalogue-batch-search-run.v1.schema.json",
                "catalogue-review-pool-row.v1.schema.json",
                "catalogue-review-pool-manifest.v1.schema.json",
                "query-vector-set.v1.schema.json",
            )
        }

    def test_request_and_raw_result_rows_are_strict(self) -> None:
        self.validators["catalogue-batch-search-request.v1.schema.json"].validate(
            request_row()
        )
        self.validators["catalogue-batch-search-result.v1.schema.json"].validate(
            result_row()
        )
        for row, schema in [
            (request_row(), "catalogue-batch-search-request.v1.schema.json"),
            (result_row(), "catalogue-batch-search-result.v1.schema.json"),
        ]:
            row["unknown"] = True
            with self.assertRaises(ValidationError):
                self.validators[schema].validate(row)
        blank = request_row()
        blank["query"] = "   "
        with self.assertRaises(ValidationError):
            self.validators["catalogue-batch-search-request.v1.schema.json"].validate(blank)

    def test_run_manifest_schema_and_self_digest(self) -> None:
        manifest = run_manifest()
        self.validators["catalogue-batch-search-run.v1.schema.json"].validate(manifest)
        self.assertEqual(
            manifest["component_sha256"],
            canonical_sha256_omitting(manifest, ("component_sha256",)),
        )
        lexical = copy.deepcopy(manifest)
        lexical["modes"] = ["lexical"]
        lexical["request_shapes"][0]["mode"] = "lexical"
        lexical["model"] = {
            "status": "not_used_all_lexical",
            "configured_model": "local-model",
            "returned_model": None,
            "calls": 0,
        }
        lexical["query_vectors"] = []
        lexical["component_sha256"] = canonical_sha256_omitting(
            lexical, ("component_sha256",)
        )
        self.validators["catalogue-batch-search-run.v1.schema.json"].validate(lexical)

        sealed = copy.deepcopy(manifest)
        sealed["model"] = {
            "status": "sealed_query_vector_set",
            "configured_model": "local-model",
            "returned_model": "local-model",
            "calls": 0,
            "query_vector_set_sha256": SHA,
        }
        sealed["component_sha256"] = canonical_sha256_omitting(
            sealed, ("component_sha256",)
        )
        self.validators["catalogue-batch-search-run.v1.schema.json"].validate(sealed)

    def test_query_vector_set_schema_is_strict(self) -> None:
        component = {"id": "component", "version": "1", "sha256": SHA}
        execution = {
            "executor_image": component,
            "executor_image_build": component,
            "runtime": component,
            "worker_binary": component,
            "model_artifact": component,
            "embedding_profile": component,
            "accelerator": {
                "provider": "runpod",
                "model": "A100",
                "architecture": "ampere",
                "compute_capability": "8.0",
                "count": 1,
            },
            "returned_model": "model",
        }
        manifest = {
            "schema_version": "livefire.rag.query-vector-set/1",
            "component_sha256": SHA,
            "status": "complete",
            "query_plan": {"path": "queries.jsonl", "bytes": 100, "sha256": SHA},
            "request_rows": 2,
            "semantic_request_rows": 2,
            "execution": {
                "embedding_profile": component,
                "embedding_policy": component,
                "execution_identity_sha256": SHA,
                "execution": execution,
                "executor_image_build_receipt": component,
            },
            "vectors": {
                "path": "vectors.f32le",
                "bytes": 32768,
                "sha256": SHA,
                "rows": 2,
                "dimensions": 4096,
                "dtype": "f32le",
                "normalization": "l2",
                "order_sha256": SHA,
            },
            "queries": [
                {
                    "ordinal": ordinal,
                    "query_id": f"q-{ordinal}",
                    "raw_query_sha256": SHA,
                    "composed_query_sha256": SHA,
                    "vector_sha256": SHA,
                }
                for ordinal in range(2)
            ],
        }
        validator = self.validators["query-vector-set.v1.schema.json"]
        validator.validate(manifest)
        manifest["raw_vector"] = [1.0]
        with self.assertRaises(ValidationError):
            validator.validate(manifest)

    def test_review_contract_excludes_system_labels(self) -> None:
        row = review_row()
        self.validators["catalogue-review-pool-row.v1.schema.json"].validate(row)
        for forbidden in (
            "mode",
            "rank",
            "score",
            "index_sha256",
            "catalogue_sha256",
            "expected_relation",
            "relevance",
        ):
            leaked = copy.deepcopy(row)
            leaked[forbidden] = "leak"
            with self.assertRaises(ValidationError):
                self.validators["catalogue-review-pool-row.v1.schema.json"].validate(
                    leaked
                )

    def test_review_manifest_is_strict_and_sealed(self) -> None:
        manifest: dict[str, object] = {
            "schema_version": "livefire.rag.catalogue-review-pool-manifest/1",
            "component_sha256": "0" * 64,
            "status": "people_have_not_yet_marked_relevance",
            "system_labels_hidden": True,
            "query_fixture_sha256": SHA,
            "raw_batch_run_sha256": SHA,
            "review_pool": {
                "path": "review-pool.jsonl",
                "sha256": SHA,
                "bytes": 100,
                "rows": 1,
            },
            "unique_query_count": 1,
            "unique_candidate_count": 1,
        }
        manifest["component_sha256"] = canonical_sha256_omitting(
            manifest, ("component_sha256",)
        )
        self.validators["catalogue-review-pool-manifest.v1.schema.json"].validate(
            manifest
        )
        self.assertEqual(
            manifest["component_sha256"],
            canonical_sha256_omitting(manifest, ("component_sha256",)),
        )


if __name__ == "__main__":
    unittest.main()
