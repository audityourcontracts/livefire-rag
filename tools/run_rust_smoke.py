#!/usr/bin/env python3
"""Run the tiny OCSF -> LM Studio -> Rust RAG -> Python analysis smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from livefire_rag.evidence_schema import _offline_registry
from livefire_rag_analysis import evaluate_retrieval_run, write_pca_report


def _run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_lm_studio(endpoint: str, expected_model: str) -> None:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/v1/models", timeout=5) as response:
        models = json.load(response)
    keys = {
        item.get("id")
        for item in models.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if expected_model not in keys:
        raise RuntimeError(
            f"LM Studio does not report required model {expected_model!r}; available={sorted(keys)}"
        )


def _provider_smoke(
    executable: Path,
    repository: Path,
    index: Path,
    embedding_endpoint: str,
    query: str,
    transcript_path: Path,
) -> dict[str, Any]:
    """Exercise the standalone SDK JSONL lifecycle over the generated index."""

    manifest = json.loads((index / "index.json").read_text(encoding="utf-8"))
    source_sha256 = manifest["source"]["snapshot_sha256"]
    lock_sha256 = "d" * 64
    index_ref = {"id": "smoke.fast-index", "version": "1", "sha256": "c" * 64}
    source_ref = {"id": "smoke.ocsf-snapshot", "version": "1", "sha256": source_sha256}

    process = subprocess.Popen(
        [str(executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("provider process did not expose its JSONL streams")
    transcript: list[dict[str, Any]] = []

    def exchange(identifier: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request = {
            "protocol": "livefire.tool/1",
            "id": identifier,
            "method": method,
            "params": params,
            "context": {
                "trace_id": "rust-smoke",
                "deadline_unix_ms": 9_999_999_999_999,
            },
        }
        process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"provider exited before responding to {method}")
        response = json.loads(line)
        transcript.append({"request": request, "response": response})
        if response.get("id") != identifier or response.get("protocol") != "livefire.tool/1":
            raise RuntimeError(f"provider returned an invalid envelope for {method}")
        if "error" in response:
            raise RuntimeError(f"provider {method} failed: {response['error']}")
        return response["result"]

    try:
        handshake = exchange("1", "handshake", {})
        provider_ref = handshake["provider"]
        tool_ref = handshake["tools"][0]
        opened = exchange(
            "2",
            "open",
            {
                "provider": provider_ref,
                "tools": [tool_ref],
                "indexes": [index_ref],
                "source_snapshots": [source_ref],
                "binding_lock_sha256": lock_sha256,
                "query_time_contract": {"embedding_endpoint": embedding_endpoint},
                "limits": {
                    "request_bytes": 1_048_576,
                    "result_bytes": 1_048_576,
                    "wall_time_ms": 300_000,
                    "memory_bytes": 1_073_741_824,
                    "max_candidates": 20,
                },
                "mounts": [
                    {
                        "logical_name": "evidence-index",
                        "role": "index",
                        "component": index_ref,
                        "access": "read_only",
                        "process_path": str(index),
                    }
                ],
            },
        )
        session_id = opened["session_id"]
        called = exchange(
            "3",
            "call",
            {
                "session_id": session_id,
                "tool": tool_ref,
                "arguments": {
                    "schema_version": "livefire.rag.fast-search.input/1",
                    "query": query,
                    "mode": "lexical",
                    "top_n": 3,
                },
            },
        )
        health = exchange("4", "health", {"session_id": session_id})
        closed = exchange("5", "close", {"session_id": session_id})
    finally:
        process.stdin.close()
        stderr = process.stderr.read()
        status = process.wait(timeout=10)
        transcript_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in transcript),
            encoding="utf-8",
        )
        if status != 0:
            raise RuntimeError(f"provider exited with status {status}: {stderr}")

    output = called["output"]
    if output.get("kind") != "pointer" or not output.get("candidates"):
        raise RuntimeError("provider lexical smoke did not return a candidate pointer")
    registry, schemas = _offline_registry(
        repository / "specs", repository.parent / "livefire-sdk" / "specs"
    )
    Draft202012Validator(
        schemas["fast-evidence-search.output.v1.schema.json"], registry=registry
    ).validate(output)
    if health.get("status") != "ready" or closed.get("closed") is not True:
        raise RuntimeError("provider health/close lifecycle did not complete")
    return {
        "scope": "direct_jsonl_lifecycle_not_sdk_admission",
        "requests": len(transcript),
        "tool": tool_ref,
        "result_kind": output["kind"],
        "returned_candidates": len(output["candidates"]),
        "output_schema_validated": True,
        "transcript": transcript_path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument(
        "--embedding-profile",
        type=Path,
        default=Path("profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json"),
    )
    parser.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    parser.add_argument("--mode", choices=("dense", "lexical", "fused"), default="fused")
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    work = arguments.work.resolve()
    if work.exists():
        raise FileExistsError(f"refusing to overwrite smoke directory: {work}")
    work.mkdir(parents=True)
    profile_path = (repository / arguments.embedding_profile).resolve()
    profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
    expected_model = profile_json.get("api_model_key", profile_json.get("model"))
    if not isinstance(expected_model, str):
        raise RuntimeError("embedding profile does not identify an API model")
    _check_lm_studio(arguments.embedding_endpoint, expected_model)

    _run(
        [
            "cargo", "build", "-q",
            "-p", "rag-builder",
            "-p", "rag-provider",
            "-p", "rag-testkit",
        ],
        cwd=repository,
    )
    snapshot = work / "snapshot"
    _run(
        [
            str(repository / "target/debug/make-smoke-snapshot"),
            "--out", str(snapshot),
        ],
        cwd=repository,
    )
    index = work / "index"
    build_output = json.loads(
        _run(
            [
                str(repository / "target/debug/rag"), "build",
                "--snapshot", str(snapshot),
                "--out", str(index),
                "--embedding-profile", str(profile_path),
                "--embedding-endpoint", arguments.embedding_endpoint,
                "--resume", str(work / "embedding-cache.sqlite3"),
                "--embedding-batch-size", "16",
            ],
            cwd=repository,
        )
    )
    cached_index = work / "index-cached"
    cached_build = json.loads(
        _run(
            [
                str(repository / "target/debug/rag"), "build",
                "--snapshot", str(snapshot),
                "--out", str(cached_index),
                "--embedding-profile", str(profile_path),
                "--embedding-endpoint", "http://127.0.0.1:9",
                "--resume", str(work / "embedding-cache.sqlite3"),
                "--embedding-batch-size", "16",
            ],
            cwd=repository,
        )
    )
    comparable = [
        "index.json",
        "documents.parquet",
        "occurrences.parquet",
        "vectors.f32",
        "lexical/index.json",
    ]
    if any(_sha256(index / name) != _sha256(cached_index / name) for name in comparable):
        raise RuntimeError("cached rebuild changed a stable index artifact")
    expected_documents = build_output["index"]["documents"]["rows"]
    if cached_build["embedded"] != 0 or cached_build["cache_hits"] != expected_documents:
        raise RuntimeError("cached rebuild attempted model inference or missed cached vectors")

    result_root = work / "search-results"
    result_root.mkdir()
    run_rows: list[dict[str, Any]] = []
    for query in _json_lines(snapshot / "smoke-queries.jsonl"):
        output_text = _run(
            [
                str(repository / "target/debug/rag"), "query",
                "--index", str(index),
                "--query", query["query"],
                "--mode", arguments.mode,
                "--top-n", "6",
                "--embedding-endpoint", arguments.embedding_endpoint,
            ],
            cwd=repository,
        )
        output = json.loads(output_text)
        (result_root / f"{query['query_id']}.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run_rows.extend(
            {
                "query_id": query["query_id"],
                "document_id": hit["document_id"],
                "rank": hit["rank"],
                "score": hit["score"],
            }
            for hit in output["hits"]
        )
    run_path = work / f"run-{arguments.mode}.jsonl"
    run_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in run_rows),
        encoding="utf-8",
    )
    evaluation = evaluate_retrieval_run(
        run_path,
        qrels=snapshot / "smoke-qrels.jsonl",
        cutoffs=(1, 3, 6),
    )
    evaluation_path = work / f"evaluation-{arguments.mode}.json"
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    geometry = write_pca_report(index, work / "geometry", seed=0, mark_count=2)
    provider = _provider_smoke(
        repository / "target/debug/rag-provider",
        repository,
        index,
        arguments.embedding_endpoint,
        "encoded PowerShell command",
        work / "provider-wire.jsonl",
    )
    report = {
        "schema_version": "livefire.rag.rust-smoke-report/1",
        "scope": "synthetic_interface_smoke_not_retrieval_quality_benchmark",
        "mode": arguments.mode,
        "embedding_endpoint": arguments.embedding_endpoint,
        "embedding_model": expected_model,
        "build": build_output,
        "cached_rebuild": {
            "cache_hits": cached_build["cache_hits"],
            "embedded": cached_build["embedded"],
            "stable_artifacts_byte_identical": True,
            "embedding_endpoint_was_unreachable": True,
        },
        "evaluation": evaluation,
        "provider": provider,
        "geometry": {
            "documents": geometry["population"]["documents"],
            "dimensions": geometry["population"]["dimensions"],
            "image": "geometry/pca.png",
        },
    }
    (work / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
