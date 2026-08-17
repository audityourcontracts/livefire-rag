from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from livefire_rag.canonical import canonical_json_bytes, canonical_sha256_omitting, sha256_bytes
from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"


def _digest(fill: str) -> str:
    return fill * 64


def _component(component_id: str, fill: str, version: str = "1") -> dict[str, object]:
    return {"id": component_id, "version": version, "sha256": _digest(fill)}


def _object(key: str, fill: str) -> dict[str, object]:
    return {"key": key, "bytes": 10, "sha256": _digest(fill)}


def _execution() -> dict[str, object]:
    return {
        "executor_image": _component("test.executor-image", "1"),
        "executor_image_build": _component("test.executor-image-build", "c"),
        "runtime": _component("test.runtime", "2"),
        "worker_binary": _component("test.worker", "3"),
        "model_artifact": _component("test.model", "4"),
        "embedding_profile": _component("test.embedding-profile", "5"),
        "accelerator": {
            "provider": "runpod",
            "model": "NVIDIA H100 PCIe",
            "architecture": "hopper-sm90",
            "compute_capability": "9.0",
            "count": 1,
        },
        "returned_model": "Qwen/Qwen3-Embedding-8B",
    }


def _query_vector_set() -> dict[str, object]:
    return {
        "component_sha256": _digest("d"),
        "manifest": _object("query-vectors/manifest.json", "e"),
        "query_plan": _object("query-vectors/queries.jsonl", "9"),
        "vectors": _object("query-vectors/vectors.f32le", "f"),
    }


def _seal(value: dict[str, object]) -> dict[str, object]:
    value["component_sha256"] = canonical_sha256_omitting(value, ("component_sha256",))
    return value


def _assignment() -> dict[str, object]:
    return _seal(
        {
            "component_sha256": _digest("0"),
            "worker_id": "worker-0000",
            "task_start": 0,
            "task_end": 1,
            "ordinal_start": 0,
            "ordinal_end": 2,
            "token_count": 32,
            "tasks": [
                {
                    "task_id": "task-0000",
                    "task_ordinal": 0,
                    "ordinal_start": 0,
                    "ordinal_end": 2,
                    "token_count": 32,
                    "result_key": "results/task-0000.parquet",
                    "receipt_key": "receipts/task-0000.json",
                    "report_key": "reports/task-0000.json",
                }
            ],
        }
    )


def _bundle() -> dict[str, object]:
    artifact = lambda key, component_fill, object_fill: {
        "component_sha256": _digest(component_fill),
        "object": _object(key, object_fill),
    }
    return _seal(
        {
            "schema_version": "livefire.rag.runpod-embedding-bundle/1",
            "component_sha256": _digest("0"),
            "prepared_corpus_sha256": _digest("6"),
            "plan_sha256": _digest("7"),
            "embedding_profile_sha256": _digest("8"),
            "tokenizer_sha256": _digest("9"),
            "model_sha256": _digest("a"),
            "document_count": 2,
            "task_count": 1,
            "total_tokens": 32,
            "artifacts": {
                "prepared_manifest": artifact("input/prepared.json", "6", "b"),
                "embedding_plan": artifact("input/plan.json", "7", "c"),
                "document_token_counts": _object("input/document-token-counts.parquet", "0"),
                "embedding_profile": artifact("input/profile.json", "8", "d"),
                "executor_image_build": artifact("input/executor-image-build.json", "c", "b"),
                "executable_tokenizer": artifact("input/tokenizer.json", "9", "e"),
                "conformance_fixture": _object("input/tei-conformance.json", "a"),
                "query_plan": _object("input/query/queries.jsonl", "9"),
                "worker_binary": artifact("input/rag-worker", "3", "f"),
                "model_manifest": artifact("input/model.json", "a", "1"),
                "model_objects": [_object("model/model.safetensors", "2")],
                "prepared_documents": [
                    {
                        "prepared_path": "documents/part-000000.parquet",
                        "object": _object("prepared/documents/part-000000.parquet", "3"),
                    }
                ],
            },
            "execution": _execution(),
            "query_vector_output": {
                "worker_id": "worker-0000",
                "manifest_key": "query-vectors/manifest.json",
                "query_plan_key": "query-vectors/queries.jsonl",
                "vectors_key": "query-vectors/vectors.f32le",
            },
            "assignments": [_assignment()],
        }
    )


def _attempt(bundle: dict[str, object]) -> dict[str, object]:
    assignment = bundle["assignments"][0]
    return _seal(
        {
            "schema_version": "livefire.rag.runpod-worker-attempt/1",
            "component_sha256": _digest("0"),
            "bundle_sha256": bundle["component_sha256"],
            "assignment_sha256": assignment["component_sha256"],
            "worker_id": "worker-0000",
            "attempt_id": "attempt-0001",
            "attempt_number": 1,
            "outcome": "completed",
            "machine": {
                "pod_id": "pod-1",
                "machine_id": "machine-1",
            },
            "execution": _execution(),
            "started_at_ms": 1,
            "completed_at_ms": 2,
            "requests": 1,
            "retries": 0,
            "outputs": [
                {
                    "task_id": "task-0000",
                    "result": _object("results/task-0000.parquet", "4"),
                    "receipt": _object("receipts/task-0000.json", "5"),
                    "report": _object("reports/task-0000.json", "6"),
                }
            ],
            "query_vector_set": _query_vector_set(),
        }
    )


def _run_report(bundle: dict[str, object], attempt: dict[str, object]) -> dict[str, object]:
    return _seal(
        {
            "schema_version": "livefire.rag.runpod-run-report/1",
            "component_sha256": _digest("0"),
            "bundle_sha256": bundle["component_sha256"],
            "execution": _execution(),
            "worker_count": 1,
            "task_count": 1,
            "document_count": 2,
            "total_tokens": 32,
            "vector_objects": 1,
            "receipt_objects": 1,
            "report_objects": 1,
            "query_vector_set": _query_vector_set(),
            "selected_attempts": [
                {
                    "worker_id": "worker-0000",
                    "attempt_id": "attempt-0001",
                    "marker_component_sha256": attempt["component_sha256"],
                    "marker": _object("attempts/worker-0000/attempt-0001/marker.json", "7"),
                }
            ],
        }
    )


def _embedding_policy_v3() -> dict[str, object]:
    artifact_set = _model_artifact_set()
    revision = artifact_set["revision"]
    objects = artifact_set["objects"]
    artifact_set_sha256 = sha256_bytes(canonical_json_bytes(artifact_set))
    tokenizer_object = next(value for value in objects if value["path"] == "tokenizer.json")
    tokenizer = {
        "id": "test.tokenizer",
        "version": revision,
        "sha256": tokenizer_object["sha256"],
    }
    accelerator = _tei_accelerator()
    return {
        "schema_version": "livefire.rag.embedding-policy/3",
        "admission_status": "development_only",
        "purpose": "semantic_search",
        "model_repository": "Qwen/Qwen3-Embedding-8B",
        "model_revision": revision,
        "model_snapshot_completeness": "complete_hugging_face_snapshot",
        "model_artifact_set": {
            "id": "test.model-artifact-set",
            "version": revision,
            "sha256": artifact_set_sha256,
        },
        "model_objects": objects,
        "tokenizer": tokenizer,
        "executable_tokenizer": {
            "repository": "Qwen/Qwen3-Embedding-8B",
            "revision": revision,
            "format": "hugging_face_tokenizer_json",
            "object": tokenizer_object,
            "add_special_tokens": True,
        },
        "tei_image": {
            "component": _component("test.tei-image", "4"),
            "repository": "ghcr.io/huggingface/text-embeddings-inference",
            "digest": f"sha256:{_digest('4')}",
        },
        "executor_image": {
            "component": _component("test.executor-image", "d"),
            "repository": "ghcr.io/example/livefire-rag-worker",
            "digest": f"sha256:{_digest('d')}",
        },
        "executor_image_build": _component("test.executor-image-build", "c"),
        "runtime": _component("test.tei-runtime", "5"),
        "inference_engine": _component("test.tei-engine", "6"),
        "load_policy": {
            "component": _component("test.load-policy", "7"),
            "model_source": "mounted_complete_snapshot",
            "revision_policy": "exact",
            "local_files_only": True,
            "trust_remote_code": False,
            "safetensors_only": True,
        },
        "runtime_mode": "tei_loopback_worker",
        "api_contract": "openai_compatible_v1_embeddings",
        "api_model_key": "Qwen/Qwen3-Embedding-8B",
        "dimensions": 4096,
        "checkpoint_compute_dtype": "float16",
        "api_vector_dtype": "float32",
        "stored_vector_dtype": "f32le",
        "pooling": "last_token",
        "normalization": "l2",
        "maximum_tokens": 8192,
        "document_format": "{semantic_text}",
        "query_instruction": "Retrieve relevant evidence.",
        "query_composition": "Instruct: {query_instruction}\nQuery: {query}",
        "batching": {
            "maximum_batch_items": 8,
            "maximum_batch_tokens": 65536,
            "maximum_concurrent_requests": 4,
            "order": "preserve_input_order",
            "overlength": "reject",
        },
        "response_limits": {"request_timeout_ms": 120000, "maximum_response_bytes": 1048576},
        "output_processing": {
            "client_normalization": "none",
            "required_l2_norm_tolerance_millionths": 100,
        },
        "accelerator": accelerator,
        "conformance": {
            "mode": "exact_digest",
            "measured": True,
            "fixture": {
                "path": "conformance/tei.json",
                "media_type": "application/json",
                "bytes": 10,
                "sha256": _digest("8"),
            },
            "input_count": 2,
            "returned_model": "Qwen/Qwen3-Embedding-8B",
            "normalized_output_sha256": _digest("9"),
            "accelerator": accelerator,
            "candidate_sha256": _digest("a"),
            "initial_result_sha256": _digest("b"),
            "fresh_pod_replay_result_sha256": _digest("c"),
        },
    }


def _model_artifact_set() -> dict[str, object]:
    return json.loads(
        (REPOSITORY / "profiles/qwen3-embedding-8b-upstream-model-artifacts.v1.json").read_text(
            encoding="utf-8"
        )
    )


def _artifact(path: str, media_type: str, fill: str) -> dict[str, object]:
    return {"path": path, "media_type": media_type, "bytes": 10, "sha256": _digest(fill)}


def _tei_accelerator() -> dict[str, object]:
    return {
        "provider": "runpod",
        "gpu_model_id": "NVIDIA H100 PCIe",
        "compute_capability": "9.0",
        "architecture_image_class": "sm90-cuda12",
        "gpu_count": 1,
    }


def _tei_execution(artifact_set_sha256: str, tokenizer_sha256: str) -> dict[str, object]:
    revision = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    return {
        "model_artifact_set": {
            "id": "test.model-artifact-set",
            "version": revision,
            "sha256": artifact_set_sha256,
        },
        "tokenizer": {"id": "test.tokenizer", "version": revision, "sha256": tokenizer_sha256},
        "tei_image": _component("test.tei-image", "4", "1.9.3"),
        "executor_image": _component("test.executor-image", "d", "1"),
        "executor_image_build": _component("test.executor-image-build", "c", "1"),
        "runtime": _component("test.runtime", "5"),
        "inference_engine": _component("test.tei-engine", "6", "1.9.3"),
        "load_policy": _component("test.load-policy", "7"),
        "worker_binary": _component("test.worker", "8"),
        "served_model": "Qwen/Qwen3-Embedding-8B",
        "dimensions": 4096,
        "pooling": "last_token",
        "forced_runtime_dtype": "float16",
        "api_vector_dtype": "float32",
        "normalization": "l2",
        "maximum_tokens": 8192,
        "maximum_client_batch_size": 8,
        "maximum_batch_tokens": 65536,
        "maximum_concurrent_requests": 4,
        "request_timeout_ms": 120000,
        "maximum_response_bytes": 1048576,
    }


def _conformance_candidate() -> dict[str, object]:
    artifact_set = _model_artifact_set()
    artifact_set_sha256 = sha256_bytes(canonical_json_bytes(artifact_set))
    revision = artifact_set["revision"]
    objects = artifact_set["objects"]
    tokenizer_object = next(value for value in objects if value["path"] == "tokenizer.json")
    fixture_path = REPOSITORY / "fixtures/qwen3-embedding-8b-tei-conformance.v1.json"
    fixture_bytes = fixture_path.read_bytes()
    execution = _tei_execution(artifact_set_sha256, tokenizer_object["sha256"])
    return _seal(
        {
            "schema_version": "livefire.rag.runpod-tei-conformance-candidate/1",
            "component_sha256": _digest("0"),
            "model_repository": artifact_set["repository"],
            "model_revision": revision,
            "model_snapshot_completeness": "complete_upstream_snapshot",
            "model_manifest": {
                "component_sha256": artifact_set_sha256,
                "object": _artifact("model/manifest.json", "application/json", "d"),
            },
            "model_objects": objects,
            "tokenizer": {
                "component": execution["tokenizer"],
                "repository": artifact_set["repository"],
                "revision": revision,
                "format": "hugging_face_tokenizer_json",
                "object": tokenizer_object,
                "add_special_tokens": True,
            },
            "tei_image": {
                "component": execution["tei_image"],
                "repository": "ghcr.io/huggingface/text-embeddings-inference",
                "digest": f"sha256:{execution['tei_image']['sha256']}",
            },
            "executor_image": {
                "component": execution["executor_image"],
                "repository": "ghcr.io/example/livefire-rag-worker",
                "digest": f"sha256:{execution['executor_image']['sha256']}",
            },
            "executor_image_build": {
                "component": execution["executor_image_build"],
                "object": _artifact(
                    "build/executor-image-build-receipt.json", "application/json", "b"
                ),
            },
            "runtime": execution["runtime"],
            "inference_engine": execution["inference_engine"],
            "load_policy": {
                "component": execution["load_policy"],
                "model_source": "mounted_complete_snapshot",
                "revision_policy": "exact",
                "local_files_only": True,
                "trust_remote_code": False,
                "safetensors_only": True,
            },
            "worker_binary": {
                "component_sha256": execution["worker_binary"]["sha256"],
                "object": _artifact("bin/rag-runpod-worker", "application/octet-stream", "8"),
            },
            "execution": execution,
            "accelerator": _tei_accelerator(),
            "fixture": {
                "object": {
                    "path": "fixtures/qwen3-embedding-8b-tei-conformance.v1.json",
                    "media_type": "application/json",
                    "bytes": len(fixture_bytes),
                    "sha256": sha256_bytes(fixture_bytes),
                },
                "input_count": 1,
            },
            "expected_output_key": "results/normalized-vectors.json",
            "output_format": "rfc8785_f32_vectors_lf_v1",
        }
    )


def _conformance_result(candidate: dict[str, object]) -> dict[str, object]:
    output = _artifact("results/normalized-vectors.json", "application/json", "f")
    return _seal(
        {
            "schema_version": "livefire.rag.runpod-tei-conformance-result/1",
            "component_sha256": _digest("0"),
            "candidate_sha256": candidate["component_sha256"],
            "run_id": "run-1",
            "machine": {
                "pod_id": "pod-1",
                "machine_id": "machine-1",
                "gpu_device_id": "gpu-0",
                "accelerator": candidate["accelerator"],
            },
            "execution": candidate["execution"],
            "outcome": "completed",
            "returned_model": candidate["execution"]["served_model"],
            "normalized_output": {
                "object": output,
                "format": candidate["output_format"],
                "vector_count": candidate["fixture"]["input_count"],
                "dimensions": candidate["execution"]["dimensions"],
                "dtype": "float32",
                "normalized_output_sha256": output["sha256"],
            },
            "model_load_ms": 1000,
            "request_ms": 50,
            "failure_code": None,
        }
    )


class CloudEmbeddingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        cls.validators = {
            name: Draft202012Validator(schemas[name], registry=registry)
            for name in [
                "embedding-policy.v3.schema.json",
                "tei-model-artifact-set.v1.schema.json",
                "runpod-embedding-bundle.v1.schema.json",
                "runpod-worker-attempt.v1.schema.json",
                "runpod-run-report.v1.schema.json",
                "runpod-tei-conformance-candidate.v1.schema.json",
                "runpod-tei-conformance-result.v1.schema.json",
            ]
        }

    def test_pinned_model_artifact_set_validates_with_exact_canonical_digest(self) -> None:
        artifact_set = _model_artifact_set()
        self.validators["tei-model-artifact-set.v1.schema.json"].validate(artifact_set)
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(artifact_set)),
            "99beb578f3ca8c20eb204484178bf08fea6f0d7f016ab49ca33a8590e1af2dcb",
        )
        self.assertEqual(len(artifact_set["objects"]), 17)

    def test_v3_profile_validates_and_binds_the_complete_model_object_set(self) -> None:
        profile = _embedding_policy_v3()
        self.validators["embedding-policy.v3.schema.json"].validate(profile)
        material = {
            "schema_version": "livefire.rag.tei-model-artifact-set/1",
            "repository": profile["model_repository"],
            "revision": profile["model_revision"],
            "objects": profile["model_objects"],
        }
        self.assertEqual(
            profile["model_artifact_set"]["sha256"],
            sha256_bytes(canonical_json_bytes(material)),
        )
        self.assertEqual(
            profile["tokenizer"]["sha256"],
            profile["executable_tokenizer"]["object"]["sha256"],
        )

    def test_v3_profile_is_strict_and_requires_a_complete_checkpoint_shape(self) -> None:
        extra = _embedding_policy_v3()
        extra["provider_url"] = "https://example.invalid"
        with self.assertRaises(ValidationError):
            self.validators["embedding-policy.v3.schema.json"].validate(extra)

        incomplete = _embedding_policy_v3()
        incomplete["model_objects"] = [
            value
            for value in incomplete["model_objects"]
            if value["path"] != "model-00004-of-00004.safetensors"
        ]
        with self.assertRaises(ValidationError):
            self.validators["embedding-policy.v3.schema.json"].validate(incomplete)

        unsupported_dtype = _embedding_policy_v3()
        unsupported_dtype["checkpoint_compute_dtype"] = "bfloat16"
        with self.assertRaises(ValidationError):
            self.validators["embedding-policy.v3.schema.json"].validate(unsupported_dtype)

    def test_conformance_candidate_and_result_validate_with_exact_self_digests(self) -> None:
        candidate = _conformance_candidate()
        self.validators["runpod-tei-conformance-candidate.v1.schema.json"].validate(candidate)
        self.assertEqual(
            candidate["component_sha256"],
            canonical_sha256_omitting(candidate, ("component_sha256",)),
        )

        result = _conformance_result(candidate)
        self.validators["runpod-tei-conformance-result.v1.schema.json"].validate(result)
        self.assertEqual(result["candidate_sha256"], candidate["component_sha256"])
        self.assertEqual(
            result["component_sha256"],
            canonical_sha256_omitting(result, ("component_sha256",)),
        )

    def test_cloud_bundle_and_nested_assignment_validate_with_exact_self_digests(self) -> None:
        bundle = _bundle()
        self.validators["runpod-embedding-bundle.v1.schema.json"].validate(bundle)
        assignment = bundle["assignments"][0]
        self.assertEqual(
            assignment["component_sha256"],
            canonical_sha256_omitting(assignment, ("component_sha256",)),
        )
        self.assertEqual(
            bundle["component_sha256"],
            canonical_sha256_omitting(bundle, ("component_sha256",)),
        )

        extra = copy.deepcopy(bundle)
        extra["endpoint"] = "https://example.invalid"
        with self.assertRaises(ValidationError):
            self.validators["runpod-embedding-bundle.v1.schema.json"].validate(extra)

        changed_output = copy.deepcopy(bundle)
        changed_output["query_vector_output"]["manifest_key"] = "other/manifest.json"
        with self.assertRaises(ValidationError):
            self.validators["runpod-embedding-bundle.v1.schema.json"].validate(changed_output)

    def test_worker_attempt_enforces_outcome_shape_and_exact_self_digest(self) -> None:
        bundle = _bundle()
        attempt = _attempt(bundle)
        self.validators["runpod-worker-attempt.v1.schema.json"].validate(attempt)
        self.assertEqual(
            attempt["component_sha256"],
            canonical_sha256_omitting(attempt, ("component_sha256",)),
        )

        impossible = copy.deepcopy(attempt)
        impossible["failure_code"] = "remote_failure"
        with self.assertRaises(ValidationError):
            self.validators["runpod-worker-attempt.v1.schema.json"].validate(impossible)

        missing_query_vectors = copy.deepcopy(attempt)
        del missing_query_vectors["query_vector_set"]
        with self.assertRaises(ValidationError):
            self.validators["runpod-worker-attempt.v1.schema.json"].validate(
                missing_query_vectors
            )

    def test_run_report_validates_with_exact_self_digest_and_strict_selection(self) -> None:
        bundle = _bundle()
        attempt = _attempt(bundle)
        report = _run_report(bundle, attempt)
        self.validators["runpod-run-report.v1.schema.json"].validate(report)
        self.assertEqual(
            report["component_sha256"],
            canonical_sha256_omitting(report, ("component_sha256",)),
        )

        missing = copy.deepcopy(report)
        missing["selected_attempts"] = []
        with self.assertRaises(ValidationError):
            self.validators["runpod-run-report.v1.schema.json"].validate(missing)


if __name__ == "__main__":
    unittest.main()
