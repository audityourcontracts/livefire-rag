from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"
SHA = "a" * 64


def _git() -> dict[str, object]:
    return {"status": "unavailable", "commit": None, "working_tree_dirty": None}


def _machine() -> dict[str, object]:
    return {
        "status": "partial",
        "operating_system": "macos",
        "operating_system_version": None,
        "architecture": "aarch64",
        "cpu_model": None,
        "logical_cpu_count": 18,
        "ram_bytes": None,
    }


def _lm_studio() -> dict[str, object]:
    return {
        "status": "partial",
        "version": None,
        "configured_model": "qwen-local",
        "returned_model": "qwen-local",
        "endpoint_kind": "local_openai_compatible",
        "batch_size": 16,
        "requests_in_flight": 2,
        "cold_load_micros": None,
    }


def _resources() -> dict[str, object]:
    return {
        "status": "not_measured",
        "rust_peak_rss_bytes": None,
        "lm_studio_peak_rss_bytes": None,
    }


def _transport() -> dict[str, object]:
    return {
        "status": "partial",
        "request_body_bytes": None,
        "response_body_bytes": None,
        "submitted_text_bytes": 40,
        "decoded_vector_bytes": 32,
    }


def _latency(warmups: int = 1) -> dict[str, int]:
    return {
        "warmups": warmups,
        "samples": 2,
        "min_micros": 10,
        "p50_micros": 10,
        "p95_micros": 20,
        "max_micros": 20,
    }


def _task_report() -> dict[str, object]:
    attempt = {
        "attempt": 1,
        "input_rows": 2,
        "input_text_bytes": 20,
        "vector_bytes": 32,
        "elapsed_micros": 900,
        "backoff_micros": 0,
        "outcome": "success",
    }
    execution = {
        "rows": 2,
        "batches": 1,
        "attempts": 1,
        "retries": 0,
        "unique_input_text_bytes": 20,
        "sent_input_text_bytes": 20,
        "vector_bytes": 32,
        "shard_bytes": 96,
        "elapsed_micros": 1_000,
        "request_elapsed_micros": 900,
        "retry_backoff_micros": 0,
        "peak_in_flight": 1,
        "batch_reports": [
            {
                "batch_ordinal": 0,
                "row_start": 0,
                "row_end": 2,
                "input_text_bytes": 20,
                "vector_bytes": 32,
                "elapsed_micros": 900,
                "backoff_micros": 0,
                "attempts": [attempt],
            }
        ],
    }
    transport = _transport()
    transport["submitted_text_bytes"] = 20
    return {
        "schema_version": "livefire.rag.embedding-task-run-report/1",
        "plan_sha256": SHA,
        "source_snapshot_sha256": SHA,
        "prepared_corpus_sha256": SHA,
        "embedding_profile_sha256": SHA,
        "tokenizer_sha256": SHA,
        "task_id": "task-1",
        "task_index": 0,
        "ordinal_start": 0,
        "ordinal_end": 2,
        "document_count": 2,
        "token_count": 14,
        "receipt_sha256": SHA,
        "outcome": "executed",
        "started_unix_ms": 1_000,
        "finished_unix_ms": 1_001,
        "git": _git(),
        "machine": _machine(),
        "lm_studio": _lm_studio(),
        "transport_bytes": transport,
        "resource_usage": _resources(),
        "artifact_sizes": {
            "status": "partial",
            "vector_shard_bytes": 96,
            "receipt_bytes": 400,
            "task_report_bytes": None,
        },
        "execution": execution,
    }


def _run_summary() -> dict[str, object]:
    return {
        "schema_version": "livefire.rag.embedding-run-summary/1",
        "status": "finalized",
        "source_snapshot_sha256": SHA,
        "prepared_corpus_sha256": SHA,
        "plan_sha256": SHA,
        "embedding_profile_sha256": SHA,
        "tokenizer_sha256": SHA,
        "git": _git(),
        "machine": _machine(),
        "lm_studio": _lm_studio(),
        "resource_usage": _resources(),
        "artifact_sizes": {
            "status": "observed",
            "prepared_corpus_bytes": 1_000,
            "embedding_plan_bytes": 500,
            "embedding_profile_bytes": 200,
            "vector_shards_bytes": 96,
            "receipts_bytes": 400,
            "task_reports_bytes": 1_000,
        },
        "tasks": 1,
        "documents": 2,
        "tokens": 14,
        "unique_input_text_bytes": 20,
        "sent_input_text_bytes": 20,
        "vector_payload_bytes": 32,
        "vector_shard_bytes": 96,
        "transport_bytes": _transport(),
        "requests": 1,
        "retries": 0,
        "execution_reports_complete": True,
        "calendar_span_micros": 1_000,
        "wall_time_micros": 1_000,
        "task_elapsed_micros_sum": 1_000,
        "request_elapsed_micros": 900,
        "retry_backoff_micros": 0,
        "peak_in_flight": 1,
        "documents_per_second": 2_000.0,
        "tokens_per_second": 14_000.0,
        "request_latency_micros": {"p50": 900, "p95": 900, "samples": 1},
        "length_bucket_throughput": [
            {
                "basis": "exact_model_input_tokens",
                "minimum_tokens": 1,
                "maximum_tokens": 128,
                "documents": 2,
                "tokens": 14,
                "shared_wall_time_micros": 1_000,
                "documents_per_second": 2_000.0,
                "tokens_per_second": 14_000.0,
            }
        ],
    }


def _query_report() -> dict[str, object]:
    stage = {"latency": _latency(), "hits": 2, "result_sha256": SHA}
    lm_studio = _lm_studio()
    lm_studio["batch_size"] = None
    lm_studio["requests_in_flight"] = None
    return {
        "schema_version": "livefire.rag.query-benchmark/1",
        "query_id": "query-1",
        "query_sha256": SHA,
        "source_snapshot_sha256": SHA,
        "index_component_sha256": SHA,
        "index_schema_version": "livefire.rag.fast-index/2",
        "embedding_profile_id": "profile",
        "embedding_profile_version": "1",
        "embedding_profile_sha256": SHA,
        "configured_model": "qwen-local",
        "returned_model": "qwen-local",
        "git": _git(),
        "machine": _machine(),
        "lm_studio": lm_studio,
        "resource_usage": _resources(),
        "artifact_sizes": {"status": "observed", "index_bytes": 10_000},
        "transport_bytes": _transport(),
        "top_n": 20,
        "relation_filters": [],
        "warmups": 1,
        "repeats": 2,
        "embedding_warmups": 1,
        "embedding_repeats": 2,
        "query_embedding": {
            "calls": 3,
            "latency": _latency(1),
            "returned_model": "qwen-local",
            "vector_dimensions": 4,
            "vector_sha256": SHA,
        },
        "dense_index_only": copy.deepcopy(stage),
        "fused_index_only": copy.deepcopy(stage),
        "lexical_index_only": copy.deepcopy(stage),
        "end_to_end_fused": None,
        "total_model_calls": 3,
    }


def _overlap_report() -> dict[str, object]:
    overlap = {
        "full_hits": 2,
        "reduced_hits": 2,
        "shared_hits": 1,
        "overlap_fraction_of_full": 0.5,
        "jaccard": 1.0 / 3.0,
        "full_document_ids": [f"sha256:{'1' * 64}", f"sha256:{'2' * 64}"],
        "reduced_document_ids": [f"sha256:{'1' * 64}", f"sha256:{'3' * 64}"],
    }
    return {
        "schema_version": "livefire.rag.index-overlap/1",
        "query_id": "query-1",
        "full_index_sha256": SHA,
        "reduced_index_sha256": "b" * 64,
        "full_profile_sha256": "c" * 64,
        "reduced_profile_sha256": "d" * 64,
        "reduced_dimensions": 2048,
        "top_n": 2,
        "dense": copy.deepcopy(overlap),
        "fused": copy.deepcopy(overlap),
    }


class PerformanceReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        cls.validators = {
            name: Draft202012Validator(schemas[name], registry=registry)
            for name in [
                "embedding-task-run-report.v1.schema.json",
                "embedding-run-summary.v1.schema.json",
                "query-benchmark.v1.schema.json",
                "index-overlap.v1.schema.json",
            ]
        }

    def test_reports_validate_with_explicit_unavailable_measurements(self) -> None:
        self.validators["embedding-task-run-report.v1.schema.json"].validate(
            _task_report()
        )
        self.validators["embedding-run-summary.v1.schema.json"].validate(
            _run_summary()
        )
        self.validators["query-benchmark.v1.schema.json"].validate(_query_report())
        self.validators["index-overlap.v1.schema.json"].validate(_overlap_report())

    def test_reports_reject_content_and_secret_bearing_extra_fields(self) -> None:
        task = _task_report()
        task["endpoint_url"] = "http://localhost:1234/v1/embeddings"
        with self.assertRaises(ValidationError):
            self.validators["embedding-task-run-report.v1.schema.json"].validate(task)

        query = _query_report()
        query["query"] = "raw query text"
        with self.assertRaises(ValidationError):
            self.validators["query-benchmark.v1.schema.json"].validate(query)

        overlap = _overlap_report()
        overlap["query"] = "raw query text"
        with self.assertRaises(ValidationError):
            self.validators["index-overlap.v1.schema.json"].validate(overlap)

    def test_completed_summary_cannot_be_a_progress_message(self) -> None:
        summary = _run_summary()
        summary["status"] = "running"
        with self.assertRaises(ValidationError):
            self.validators["embedding-run-summary.v1.schema.json"].validate(summary)


if __name__ == "__main__":
    unittest.main()
