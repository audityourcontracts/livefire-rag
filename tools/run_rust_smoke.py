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

import duckdb
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
    embedding_profile: Path,
    embedding_endpoint: str,
    query: str,
    transcript_path: Path,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Package, admit locally, and invoke the provider through the SDK harness."""

    sdk = repository.parent / "livefire-sdk"
    registry, schemas = _offline_registry(repository / "specs", sdk / "specs")
    Draft202012Validator(
        schemas["fast-index-manifest.v2.schema.json"], registry=registry
    ).validate(json.loads((index / "index.json").read_text(encoding="utf-8")))
    Draft202012Validator(
        schemas["fast-build-report.v1.schema.json"], registry=registry
    ).validate(json.loads((index / "build-report.json").read_text(encoding="utf-8")))
    bundle = transcript_path.parent / "provider-bundle"
    loadout = transcript_path.parent / "provider-loadout"
    _run([str(repository / "target/debug/rag-package-tool"), "--provider", str(executable), "--sdk-specs", str(sdk / "specs"), "--out", str(bundle)], cwd=repository)
    _run([str(repository / "target/debug/rag-prepare-local-tool"), "--index", str(index), "--bundle", str(bundle), "--embedding-profile", str(embedding_profile), "--source-receipt", str(snapshot / "build-receipt.json") if snapshot else "", "--embedding-endpoint", embedding_endpoint, "--query", query, "--out", str(loadout)], cwd=repository)
    _run(["cargo", "run", "-q", "--manifest-path", str(sdk / "Cargo.toml"), "-p", "livefire-sdk", "--", "--specs", str(sdk / "specs"), "validate-bundle", "--manifest", str(bundle / "plugin.json"), "--root", str(bundle)], cwd=repository)
    for schema, document in [("index-manifest.v1", index / "sdk-index-manifest.json"), ("index-admission-receipt.v1", loadout / "index-admission-receipt.json"), ("tool-binding-lock.v1", loadout / "tool-binding-lock.json")]:
        _run(["cargo", "run", "-q", "--manifest-path", str(sdk / "Cargo.toml"), "-p", "livefire-sdk", "--", "--specs", str(sdk / "specs"), "validate", "--schema", schema, str(document)], cwd=repository)
    wire = _run(["cargo", "run", "-q", "--manifest-path", str(sdk / "Cargo.toml"), "-p", "livefire-sdk", "--", "--specs", str(sdk / "specs"), "invoke", "--program", str(bundle / "bin/rag-provider"), "--requests", str(loadout / "requests.jsonl")], cwd=repository)
    transcript_path.write_text(wire, encoding="utf-8")
    responses = [json.loads(line) for line in wire.splitlines()]
    if any("error" in response for response in responses):
        raise RuntimeError("SDK provider lifecycle returned an error")
    output = responses[2]["result"]["output"]
    if output.get("kind") != "pointer" or not output.get("candidates"):
        raise RuntimeError("provider lexical smoke did not return a candidate pointer")
    Draft202012Validator(
        schemas["fast-evidence-search.output.v1.schema.json"], registry=registry
    ).validate(output)
    if responses[3]["result"].get("status") != "ready" or responses[4]["result"].get("closed") is not True:
        raise RuntimeError("provider health/close lifecycle did not complete")
    hydration_refs = [ref for candidate in output["candidates"] for ref in candidate["evidence"]]
    if any(ref.get("schema_version") != "livefire.ocsf-hydration-ref/1" for ref in hydration_refs):
        raise RuntimeError("provider returned a non-hydration candidate reference")
    if snapshot is None:
        raise RuntimeError("synthetic source snapshot is required for hydration closure validation")
    receipt = json.loads((snapshot / "build-receipt.json").read_text(encoding="utf-8"))
    source_component = receipt["runnable_snapshot"]["component"]
    mapping_component = receipt["runnable_snapshot"]["mapping_pack"]
    by_relation: dict[str, set[tuple[str, str]]] = {}
    for ref in hydration_refs:
        relation = ref["relation"]
        if relation not in by_relation:
            rows = duckdb.sql(
                "SELECT event_id, support_ref FROM read_parquet(?)",
                params=[str(snapshot / "semantic" / f"{relation}.parquet")],
            ).fetchall()
            by_relation[relation] = set(rows)
        if ref["snapshot"] != source_component or ref["mapping"] != mapping_component or (ref["event_id"], ref["support_ref"]) not in by_relation[relation]:
            raise RuntimeError("provider hydration reference is outside the authoritative synthetic source closure")
    hydration_requests = [{"operation":"hydrate_event_envelopes","contract_version":1,"arguments":{"event_ids":[ref["event_id"] for ref in hydration_refs[position:position+20]]}} for position in range(0,len(hydration_refs),20)]
    (transcript_path.parent / "ocsf-hydration-requests.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in hydration_requests),encoding="utf-8")
    return {
        "scope": "sdk_validated_local_test_admission_not_production",
        "requests": len(responses),
        "tool": responses[0]["result"]["tools"][0],
        "result_kind": output["kind"],
        "returned_candidates": len(output["candidates"]),
        "output_schema_validated": True,
        "hydration_refs": len(hydration_refs),
        "hydration_reference_closure_validated": True,
        "hydration_requests": "ocsf-hydration-requests.jsonl",
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
        "occurrence-index.sqlite3",
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
        profile_path,
        arguments.embedding_endpoint,
        "encoded PowerShell command",
        work / "provider-wire.jsonl",
        snapshot,
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
