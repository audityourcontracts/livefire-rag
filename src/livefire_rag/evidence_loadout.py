"""Deterministic local-test loadouts and wire validation for ``evidence.search``.

These helpers never mint production admission or hydrate source records.  They
only make the development provider lifecycle reproducible and export immutable
source pointers for a separate authoritative adapter.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    canonical_json_bytes,
    component_ref,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from .evidence_bundle import (
    INDEX_FORMAT_REF,
    INPUT_SCHEMA_REF,
    OUTPUT_SCHEMA_REF,
    PROTOCOL,
    PROVIDER_EXECUTABLE_ARTIFACT,
    PROVIDER_REF,
    RETRIEVAL_POLICY_REF,
    TOOL_REF,
)
from .evidence_index import EvidenceIndex
from .evidence_service import validate_evidence_value, validate_sdk_value


DEFAULT_DEADLINE_UNIX_MS = 4_102_444_800_000  # 2100-01-01, deterministic POC only.
DEFAULT_LIMITS = {
    "request_bytes": 65_536,
    "result_bytes": 1_048_576,
    "wall_time_ms": 30_000,
    "memory_bytes": 268_435_456,
    "max_candidates": 1_000,
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _bundle_identity(bundle: Path) -> None:
    manifest = _load_object(bundle / "plugin.json", "bundle manifest")
    entrypoint = manifest.get("entrypoints", {}).get("provider", {})
    if entrypoint.get("component") != PROVIDER_REF:
        raise ValueError("bundle provider component does not match evidence.search")
    executable = bundle / PROVIDER_EXECUTABLE_ARTIFACT["path"]
    if (
        not executable.is_file()
        or executable.stat().st_size != PROVIDER_EXECUTABLE_ARTIFACT["bytes"]
        or sha256_file(executable) != PROVIDER_EXECUTABLE_ARTIFACT["sha256"]
        or entrypoint.get("executable") != PROVIDER_EXECUTABLE_ARTIFACT
    ):
        raise ValueError("bundle provider executable does not match evidence.search")


def _request(request_id: str, method: str, params: Mapping[str, Any], deadline: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "id": request_id,
        "method": method,
        "params": dict(params),
        "context": {
            "trace_id": f"local-test-evidence-{request_id}",
            "deadline_unix_ms": deadline,
        },
    }


def prepare_evidence_loadout(
    index_root: Path,
    bundle_root: Path,
    out_dir: Path,
    *,
    sdk_specs: Path,
    queries: Sequence[Mapping[str, Any]],
    embedding_endpoint: str = "http://127.0.0.1:1234",
    deadline_unix_ms: int = DEFAULT_DEADLINE_UNIX_MS,
) -> dict[str, Any]:
    """Create a deterministic, explicitly local-test provider loadout."""

    index_root = Path(index_root).resolve()
    bundle_root = Path(bundle_root).resolve()
    out_dir = Path(out_dir).resolve()
    sdk_specs = Path(sdk_specs).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite loadout path: {out_dir}")
    if not queries:
        raise ValueError("at least one evidence.search query is required")
    if not embedding_endpoint.startswith("http://127.0.0.1:") and not embedding_endpoint.startswith(
        "http://localhost:"
    ):
        raise ValueError("embedding endpoint must be loopback HTTP")
    if isinstance(deadline_unix_ms, bool) or not isinstance(deadline_unix_ms, int) or deadline_unix_ms < 1:
        raise ValueError("deadline_unix_ms must be positive")
    normalized_queries = [dict(query) for query in queries]
    for query in normalized_queries:
        validate_evidence_value(
            "evidence-search.input.v1.schema.json", query, sdk_specs=sdk_specs
        )
    _bundle_identity(bundle_root)
    fast_manifest_path = index_root / "index.json"
    if fast_manifest_path.is_file():
        fast_manifest = _load_object(fast_manifest_path, "fast index manifest")
        if fast_manifest.get("test_only") is True:
            raise ValueError("test-only indexes cannot be prepared as provider loadouts")
    with EvidenceIndex.open(index_root, sdk_specs=sdk_specs) as index:
        manifest = index.manifest
    pilot_sample = manifest.get("pilot_sample")
    if pilot_sample is not None and (
        not isinstance(pilot_sample, dict)
        or pilot_sample.get("scope_status") != "sample_only_not_corpus_coverage"
        or pilot_sample.get("admission_status")
        != "local_evaluation_only_not_sdk_admitted"
        or pilot_sample.get("corpus_miss_definitive") is not False
    ):
        raise ValueError("pilot index scope is not explicitly local and non-definitive")

    build_report_path = index_root / "build-report.json"
    if not build_report_path.is_file():
        raise ValueError("promoted index build report is absent")
    request_material = {
        "schema_version": "livefire.rag.local-test-loadout-request/1",
        "index": manifest["component"],
        "provider": PROVIDER_REF,
        "tool": TOOL_REF,
        "queries": normalized_queries,
        "embedding_endpoint": embedding_endpoint,
        "limits": DEFAULT_LIMITS,
    }
    verifier_material = {
        "schema_version": "livefire.rag.local-test-index-verifier/1",
        "checks": [
            "object_digests", "source_binding", "safe_paths", "schema_profiles",
            "coverage_closure", "pointer_closure", "offline_query_conformance",
            "conformance",
        ],
        "authority": "local-test-only",
        "coverage_scope": (
            "sealed_selected_document_occurrences_only"
            if pilot_sample is not None
            else "sealed_index"
        ),
    }
    verifier_ref = component_ref(
        "com.ayc.livefire-rag.local-test-index-verifier", "1", verifier_material
    )
    receipt_unsigned = {
        "schema_version": "livefire.index-admission/1",
        "receipt_id": "com.ayc.livefire-rag.local-test-index-admission",
        "receipt_version": "1",
        "build_request_sha256": sha256_bytes(canonical_json_bytes(request_material)),
        "build_report_sha256": sha256_file(build_report_path),
        "index_manifest_sha256": manifest["component"]["sha256"],
        "verifier": verifier_ref,
        "checks": {
            "object_digests": True,
            "source_binding": True,
            "safe_paths": True,
            "schema_profiles": True,
            "coverage_closure": True,
            "pointer_closure": True,
            "offline_query_conformance": True,
            "conformance": True,
            "deterministic_rebuild": False,
        },
        "disposition": "admitted",
        "reason_codes": [
            "local_test_only_not_production_admitted",
            *(
                ["pilot_sample_not_corpus_coverage"]
                if pilot_sample is not None
                else []
            ),
        ],
    }
    receipt = {
        **receipt_unsigned,
        "authority_signature": "local-test:" + sha256_bytes(canonical_json_bytes(receipt_unsigned)),
    }
    validate_sdk_value(
        "index-admission-receipt.v1.schema.json", receipt, sdk_specs=sdk_specs
    )
    receipt_ref = component_ref(
        "com.ayc.livefire-rag.local-test-index-admission", "1", receipt
    )
    contract = {
        "mode": "local_component",
        "network": [f"loopback:{embedding_endpoint}"],
        "secret_handles": [],
        "vendor_services": [],
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
        "limits": dict(DEFAULT_LIMITS),
    }
    validate_sdk_value("tool-binding-lock.v1.schema.json", lock, sdk_specs=sdk_specs)
    lock_sha256 = sha256_bytes(canonical_json_bytes(lock))
    lock_ref = {
        "id": "com.ayc.livefire-rag.local-test-tool-binding",
        "version": "1",
        "sha256": lock_sha256,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        receipt_path = staging / "index-admission-receipt.json"
        lock_path = staging / "tool-binding-lock.json"
        write_canonical_json(receipt_path, receipt)
        write_canonical_json(lock_path, lock)
        mounts = [
            {"logical_name": "evidence-index", "role": "index", "component": manifest["component"], "access": "read_only", "process_path": str(index_root)},
            {"logical_name": "tool-binding-lock", "role": "policy", "component": lock_ref, "access": "read_only", "process_path": str(out_dir / "tool-binding-lock.json")},
            {"logical_name": "index-admission-receipt", "role": "policy", "component": receipt_ref, "access": "read_only", "process_path": str(out_dir / "index-admission-receipt.json")},
            {"logical_name": "embedding-profile", "role": "model", "component": manifest["embedding_profiles"][0], "access": "read_only", "process_path": str(index_root / "embedding-profile.json")},
        ]
        open_params = {
            "provider": PROVIDER_REF,
            "tools": [TOOL_REF],
            "indexes": [manifest["component"]],
            "source_snapshots": manifest["source_snapshots"],
            "binding_lock_sha256": lock_sha256,
            "query_time_contract": contract,
            "limits": dict(DEFAULT_LIMITS),
            "mounts": mounts,
        }
        requests = [
            _request("1", "handshake", {}, deadline_unix_ms),
            _request("2", "open", open_params, deadline_unix_ms),
        ]
        for position, query in enumerate(normalized_queries, 3):
            requests.append(_request(str(position), "call", {
                "session_id": "${session_id}", "tool": TOOL_REF, "arguments": query,
            }, deadline_unix_ms))
        requests.extend([
            _request(str(len(requests) + 1), "health", {"session_id": "${session_id}"}, deadline_unix_ms),
            _request(str(len(requests) + 2), "close", {"session_id": "${session_id}"}, deadline_unix_ms),
        ])
        for request in requests:
            validate_sdk_value(
                "tool-provider-protocol.v1.schema.json", request, sdk_specs=sdk_specs
            )
        transcript_path = staging / "requests.jsonl"
        with transcript_path.open("wb") as handle:
            for request in requests:
                handle.write(canonical_json_bytes(request, newline=True))
        loadout = {
            "schema_version": "livefire.rag.local-test-evidence-loadout/1",
            "admission_status": "local_test_only_not_production_admitted",
            "index": manifest["component"],
            "source_snapshots": manifest["source_snapshots"],
            "provider": PROVIDER_REF,
            "tool": TOOL_REF,
            "receipt": receipt_ref,
            "binding_lock": lock_ref,
            "bundle": str(bundle_root),
            "provider_program": str(bundle_root / PROVIDER_EXECUTABLE_ARTIFACT["path"]),
            "requests_sha256": sha256_file(transcript_path),
            "call_ids": [str(position) for position in range(3, 3 + len(normalized_queries))],
        }
        if pilot_sample is not None:
            loadout["pilot_sample"] = pilot_sample
        write_canonical_json(staging / "loadout.json", loadout)
        os.replace(staging, out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return loadout


def validate_evidence_wire(
    wire_path: Path,
    loadout_dir: Path,
    *,
    sdk_specs: Path,
    report_path: Path | None = None,
    hydration_requests_path: Path | None = None,
) -> dict[str, Any]:
    """Validate successful SDK wire output and export pointers for hydration."""

    wire_path = Path(wire_path).resolve()
    loadout_dir = Path(loadout_dir).resolve()
    sdk_specs = Path(sdk_specs).resolve()
    loadout = _load_object(loadout_dir / "loadout.json", "local-test loadout")
    if loadout.get("admission_status") != "local_test_only_not_production_admitted":
        raise ValueError("loadout is not explicitly local-test only")
    lock = _load_object(loadout_dir / "tool-binding-lock.json", "tool binding lock")
    validate_sdk_value("tool-binding-lock.v1.schema.json", lock, sdk_specs=sdk_specs)
    lock_sha256 = sha256_bytes(canonical_json_bytes(lock))
    if loadout.get("binding_lock", {}).get("sha256") != lock_sha256:
        raise ValueError("loadout binding-lock identity does not match its bytes")
    requests_path = loadout_dir / "requests.jsonl"
    if sha256_file(requests_path) != loadout.get("requests_sha256"):
        raise ValueError("loadout transcript identity does not match its bytes")
    requests = _load_transcript(requests_path)
    for request in requests:
        validate_sdk_value(
            "tool-provider-protocol.v1.schema.json", request, sdk_specs=sdk_specs
        )
    receipt = _load_object(
        loadout_dir / "index-admission-receipt.json", "index admission receipt"
    )
    validate_sdk_value(
        "index-admission-receipt.v1.schema.json", receipt, sdk_specs=sdk_specs
    )
    if (
        not receipt.get("authority_signature", "").startswith("local-test:")
        or component_ref(
            "com.ayc.livefire-rag.local-test-index-admission", "1", receipt
        ) != lock["index_admission_receipt"]
    ):
        raise ValueError("local-test receipt identity does not match the binding lock")
    index_manifest = _load_object(
        Path(next(
            mount["process_path"]
            for mount in requests[1]["params"]["mounts"]
            if mount["logical_name"] == "evidence-index"
        )) / "manifest.json",
        "evidence index manifest",
    )
    if lock["index"] != index_manifest.get("component") or lock["index"] != loadout.get("index"):
        raise ValueError("wire loadout index identity is inconsistent")
    if loadout.get("pilot_sample") != index_manifest.get("pilot_sample"):
        raise ValueError("loadout pilot scope differs from the sealed index manifest")

    responses = _load_transcript(wire_path)
    by_id: dict[str, dict[str, Any]] = {}
    for response in responses:
        validate_sdk_value(
            "tool-provider-protocol.v1.schema.json", response, sdk_specs=sdk_specs
        )
        response_id = response["id"]
        if response_id in by_id:
            raise ValueError(f"duplicate wire response id: {response_id}")
        if "error" in response:
            raise ValueError(f"provider returned an error for request {response_id}")
        by_id[response_id] = response
    expected_ids = [str(position) for position in range(1, len(loadout["call_ids"]) + 5)]
    if list(by_id) != expected_ids:
        raise ValueError("wire response IDs/order do not match the loadout transcript")
    handshake = by_id["1"]["result"]
    if (
        handshake.get("provider") != PROVIDER_REF
        or handshake.get("tools") != [TOOL_REF]
        or handshake.get("accepted_index_formats") != [INDEX_FORMAT_REF]
    ):
        raise ValueError("wire handshake provider identity does not match")
    if by_id["2"]["result"].get("binding_lock_sha256") != lock_sha256:
        raise ValueError("wire open binding-lock identity does not match")

    hydration: dict[bytes, dict[str, Any]] = {}
    call_summaries = []
    for call_id in loadout["call_ids"]:
        result = by_id[call_id]["result"]
        if result.get("response_kind") != "call":
            raise ValueError(f"wire request {call_id} is not a call result")
        output = result.get("output")
        validate_evidence_value(
            "evidence-search.output.v1.schema.json", output, sdk_specs=sdk_specs
        )
        request_arguments = next(
            request["params"]["arguments"] for request in requests if request["id"] == call_id
        )
        if output["query_sha256"] != sha256_bytes(canonical_json_bytes(request_arguments)):
            raise ValueError(f"call {call_id} query identity differs from its request")
        if output["index"] != lock["index"] or output["source_snapshots"] != lock["source_snapshots"]:
            raise ValueError(f"call {call_id} output identity differs from the binding lock")
        candidates = output.get("candidates", [])
        call_summaries.append({
            "call_id": call_id,
            "kind": output["kind"],
            "query_sha256": output["query_sha256"],
            "candidate_count": len(candidates),
        })
        for candidate in candidates:
            for occurrence in candidate["source_occurrences"]:
                pointer = occurrence["source_pointer"]
                key = canonical_json_bytes(pointer)
                item = hydration.setdefault(key, {
                    "schema_version": "livefire.rag.local-test-hydration-request/1",
                    "source_pointer": pointer,
                    "discoveries": [],
                })
                item["discoveries"].append({
                    "call_id": call_id,
                    "query_sha256": output["query_sha256"],
                    "rank": candidate["rank"],
                    "document_id": candidate["document_id"],
                    "occurrence_id": occurrence["occurrence_id"],
                    "relation_identity": occurrence["relation_identity"],
                })
    hydration_rows = []
    for key in sorted(hydration):
        item = hydration[key]
        item["discoveries"].sort(
            key=lambda row: (row["call_id"], row["rank"], row["document_id"], row["occurrence_id"])
        )
        hydration_rows.append(item)
    health = by_id[expected_ids[-2]]["result"]
    if (
        health.get("response_kind") != "health"
        or health.get("binding_lock_sha256") != lock_sha256
    ):
        raise ValueError("wire health response is absent or out of order")
    if by_id[expected_ids[-1]]["result"] != {"response_kind": "close", "closed": True}:
        raise ValueError("wire close response is absent or invalid")
    hydration_bytes = b"".join(
        canonical_json_bytes(row, newline=True) for row in hydration_rows
    )
    report = {
        "schema_version": "livefire.rag.local-test-evidence-wire-validation/1",
        "admission_status": "local_test_only_not_production_admitted",
        "valid": True,
        "binding_lock_sha256": lock_sha256,
        "index": lock["index"],
        "response_count": len(responses),
        "wire_sha256": sha256_file(wire_path),
        "calls": call_summaries,
        "unique_hydration_pointer_count": len(hydration_rows),
        "hydration_requests_sha256": sha256_bytes(hydration_bytes),
        "hydration_status": (
            "requests_exported_not_hydrated"
            if hydration_requests_path is not None
            else "requests_not_exported_not_hydrated"
        ),
    }
    if "pilot_sample" in loadout:
        pilot_sample = loadout["pilot_sample"]
        for call in call_summaries:
            output = by_id[call["call_id"]]["result"]["output"]
            if (
                output["coverage"]["status"] != "partial"
                or "pilot_sample_not_corpus_coverage"
                not in output["coverage"]["reason_codes"]
                or (
                    output["kind"] == "miss"
                    and "not a corpus-wide miss" not in output["miss"]["message"]
                )
            ):
                raise ValueError(
                    f"call {call['call_id']} does not preserve pilot-sample scope"
                )
        report["pilot_sample"] = pilot_sample
    report_path = Path(report_path) if report_path is not None else None
    hydration_requests_path = (
        Path(hydration_requests_path) if hydration_requests_path is not None else None
    )
    for path, label in (
        (report_path, "report"),
        (hydration_requests_path, "hydration requests"),
    ):
        if path is not None and path.exists():
            raise FileExistsError(f"refusing to overwrite {label}: {path}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(report_path, report)
    if hydration_requests_path is not None:
        hydration_requests_path.parent.mkdir(parents=True, exist_ok=True)
        hydration_requests_path.write_bytes(hydration_bytes)
    return report


def _load_transcript(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for position, line in enumerate(Path(path).read_bytes().splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"transcript line {position} must be an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"transcript is unreadable: {path}") from error
    if not rows:
        raise ValueError(f"transcript is empty: {path}")
    return rows


__all__ = ["prepare_evidence_loadout", "validate_evidence_wire"]
