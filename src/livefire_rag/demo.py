"""Run the frozen Q1-Q9/S1-S2 acceptance cases through the actual provider core."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .canonical import sha256_file, write_canonical_json
from .contracts import (
    PROTOCOL,
    PROVIDER_REF,
    SEARCH_TOOL_ID,
    SIMILAR_TOOL_ID,
    TOOL_REFS,
    development_binding,
    development_binding_ref,
)
from .index import SemanticIndex
from .provider import Provider


def _wire_request(request_id: str, method: str, params: dict[str, Any], deadline: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "id": request_id,
        "method": method,
        "params": params,
        "context": {"trace_id": f"provider-poc-{request_id}", "deadline_unix_ms": deadline},
    }


def _select_seed(index: SemanticIndex, case_id: str) -> str:
    for document in index.documents:
        text = document["semantic_text"].lower()
        if case_id == "S1" and document.get("source_kind") == "source_powershell_script_block" and "cachedgrouppolicysettings" in text:
            return document["command_id"]
        if case_id == "S2" and "createaccesskey" in text and "accessdenied" in text:
            return document["command_id"]
    raise ValueError(f"could not select the frozen {case_id} diagnostic seed")


def run_demo(
    suite_path: Path,
    index_path: Path,
    out_path: Path,
    requests_out: Path | None = None,
    *,
    embedding_endpoint: str,
    per_call_deadline_ms: int = 30_000,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("acceptance suite has no cases")
    index = SemanticIndex.open(index_path)
    provider = Provider(embedding_endpoint)
    transcript: list[dict[str, Any]] = []
    replay_requests: list[dict[str, Any]] = []

    sequence = 1
    deadline = int(time.time() * 1000) + per_call_deadline_ms
    request = _wire_request(str(sequence), "handshake", {}, deadline)
    response = provider.handle(request)
    transcript.append({"request": request, "response": {"protocol": PROTOCOL, "id": str(sequence), "result": response}})
    replay_requests.append(request)
    sequence += 1
    binding = development_binding(index.manifest)
    binding_ref = development_binding_ref(binding)
    open_params = {
        "provider": PROVIDER_REF,
        "tools": [TOOL_REFS[SEARCH_TOOL_ID], TOOL_REFS[SIMILAR_TOOL_ID]],
        "indexes": [index.manifest["component"]],
        "source_snapshots": index.manifest["source_snapshots"],
        "binding_lock_sha256": binding_ref["sha256"],
        "query_time_contract": {"embedding_endpoint": embedding_endpoint},
        "limits": {"result_bytes": 1048576, "wall_time_ms": per_call_deadline_ms},
        "mounts": [
            {
                "logical_name": "semantic-command-index",
                "role": "index",
                "component": index.manifest["component"],
                "access": "read_only",
                "process_path": str(index.root),
            }
        ],
    }
    request = _wire_request(str(sequence), "open", open_params, int(time.time() * 1000) + per_call_deadline_ms)
    response = provider.handle(request)
    transcript.append({"request": request, "response": {"protocol": PROTOCOL, "id": str(sequence), "result": response}})
    replay_requests.append(request)
    session_id = response["session_id"]
    sequence += 1

    calls = []
    seeds: dict[str, str] = {}
    top_n = int(suite.get("policy", {}).get("top_n", 10))
    for case in cases:
        case_id = case["case_id"]
        if case["tool"] == "cli.search":
            tool = TOOL_REFS[SEARCH_TOOL_ID]
            arguments = {
                "schema_version": "livefire.rag.cli-search.input/1",
                "query": case["query"],
                "time_range": {"start": "1970-01-01T00:00:00Z", "end_exclusive": "2100-01-01T00:00:00Z"},
                "top_n": top_n,
            }
        elif case["tool"] == "cli.similar":
            tool = TOOL_REFS[SIMILAR_TOOL_ID]
            seed_id = _select_seed(index, case_id)
            seeds[case_id] = seed_id
            arguments = {
                "schema_version": "livefire.rag.cli-similar.input/1",
                "command_id": seed_id,
                "top_n": top_n,
                "exclude_seed": True,
            }
        else:
            raise ValueError(f"unsupported suite tool: {case['tool']}")
        request = _wire_request(
            str(sequence),
            "call",
            {"session_id": session_id, "tool": tool, "arguments": arguments},
            int(time.time() * 1000) + per_call_deadline_ms,
        )
        provider_result = provider.handle(request)
        wire_response = {"protocol": PROTOCOL, "id": str(sequence), "result": provider_result}
        transcript.append({"request": request, "response": wire_response})
        replay_request = json.loads(json.dumps(request))
        replay_request["params"]["session_id"] = "${session_id}"
        replay_requests.append(replay_request)
        calls.append({"case_id": case_id, "response": wire_response})
        sequence += 1

    request = _wire_request(str(sequence), "health", {"session_id": session_id}, int(time.time() * 1000) + per_call_deadline_ms)
    response = provider.handle(request)
    transcript.append({"request": request, "response": {"protocol": PROTOCOL, "id": str(sequence), "result": response}})
    replay_request = json.loads(json.dumps(request))
    replay_request["params"]["session_id"] = "${session_id}"
    replay_requests.append(replay_request)
    sequence += 1
    request = _wire_request(str(sequence), "close", {"session_id": session_id}, int(time.time() * 1000) + per_call_deadline_ms)
    response = provider.handle(request)
    transcript.append({"request": request, "response": {"protocol": PROTOCOL, "id": str(sequence), "result": response}})
    replay_request = json.loads(json.dumps(request))
    replay_request["params"]["session_id"] = "${session_id}"
    replay_requests.append(replay_request)

    result = {
        "schema_version": "livefire.rag.provider-poc-results/1",
        "run_id": f"provider-poc-{int(time.time())}",
        "suite": {"id": suite["suite_id"], "sha256": sha256_file(suite_path)},
        "index": index.manifest["component"],
        "embedding_profile": index.manifest["embedding_profile"],
        "provider": PROVIDER_REF,
        "development_binding": {
            "component": binding_ref,
            "value": binding,
            "not_sdk_tool_binding_lock": True,
        },
        "selected_seeds": seeds,
        "calls": calls,
        "transcript": transcript,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out_path, result)
    requests_out = requests_out or out_path.with_name("provider-requests.jsonl")
    requests_out.parent.mkdir(parents=True, exist_ok=True)
    with requests_out.open("wb") as handle:
        for replay_request in replay_requests:
            replay_request["context"]["deadline_unix_ms"] = 4_102_444_800_000
            handle.write(
                json.dumps(replay_request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
    return {
        "results": str(out_path),
        "requests": str(requests_out),
        "cases": len(calls),
        "index": index.manifest["component"],
        "selected_seeds": seeds,
    }
