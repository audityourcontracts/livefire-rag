"""Historical Python provider retained only for tests and comparisons."""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .contracts import (
    INDEX_FORMAT_REF,
    PROTOCOL,
    PROVIDER_REF,
    SEARCH_TOOL_ID,
    SIMILAR_TOOL_ID,
    TOOL_REFS,
    ContractError,
    require_exact_keys,
    require_object,
)
from .embedding import EmbeddingError
from .index import IndexErrorBase, SemanticIndex
from .service import DeadlineExceeded, SemanticService
from .schema_validation import validate_semantic_result


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail


@dataclass
class Session:
    binding_lock_sha256: str
    service: SemanticService
    tool_ids: set[str]
    result_bytes: int


class Provider:
    def __init__(self, default_embedding_endpoint: str = "http://127.0.0.1:1234") -> None:
        self.default_embedding_endpoint = default_embedding_endpoint
        self.handshaken = False
        self.sessions: dict[str, Session] = {}

    def _deadline(self, request: dict[str, Any]) -> int:
        context = require_object(request.get("context"), "context")
        require_exact_keys(context, {"trace_id", "deadline_unix_ms"}, {"trace_id", "deadline_unix_ms"}, "context")
        if not isinstance(context["trace_id"], str) or not context["trace_id"]:
            raise ProviderError("invalid_request", "context.trace_id must be non-empty")
        deadline = context["deadline_unix_ms"]
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 1:
            raise ProviderError("invalid_request", "context.deadline_unix_ms must be positive")
        if int(time.time() * 1000) > deadline:
            raise ProviderError("deadline_exceeded", "request deadline has expired")
        return deadline

    def handle(self, request: Any) -> dict[str, Any]:
        value = require_object(request, "request")
        require_exact_keys(value, {"protocol", "id", "method", "params", "context"}, {"protocol", "id", "method", "params", "context"}, "request")
        if value["protocol"] != PROTOCOL or not isinstance(value["id"], str) or not value["id"]:
            raise ProviderError("protocol_error", "invalid protocol or request id")
        deadline = self._deadline(value)
        params = require_object(value["params"], "params")
        method = value["method"]
        if method == "handshake":
            require_exact_keys(params, set(), set(), "params")
            self.handshaken = True
            return {
                "response_kind": "handshake",
                "provider": PROVIDER_REF,
                "protocol": PROTOCOL,
                "tools": [TOOL_REFS[SEARCH_TOOL_ID], TOOL_REFS[SIMILAR_TOOL_ID]],
                "accepted_index_formats": [INDEX_FORMAT_REF],
            }
        if not self.handshaken:
            raise ProviderError("protocol_error", "handshake is required before session methods")
        if method == "open":
            return self._open(params)
        if method in {"call", "health", "close"}:
            session_id = params.get("session_id")
            if not isinstance(session_id, str) or session_id not in self.sessions:
                raise ProviderError("not_found", "session was not found")
            if method == "call":
                return self._call(self.sessions[session_id], params, deadline)
            if method == "health":
                require_exact_keys(params, {"session_id"}, {"session_id"}, "params")
                return {"response_kind": "health", "status": "ready", "binding_lock_sha256": self.sessions[session_id].binding_lock_sha256}
            require_exact_keys(params, {"session_id"}, {"session_id"}, "params")
            del self.sessions[session_id]
            return {"response_kind": "close", "closed": True}
        raise ProviderError("invalid_request", "unsupported method")

    def _open(self, params: dict[str, Any]) -> dict[str, Any]:
        allowed = {"provider", "tools", "indexes", "source_snapshots", "binding_lock_sha256", "query_time_contract", "limits", "mounts"}
        require_exact_keys(params, allowed, allowed, "params")
        if params["provider"] != PROVIDER_REF:
            raise ProviderError("invalid_binding", "provider component does not match this executable")
        tools = params["tools"]
        if not isinstance(tools, list) or not tools:
            raise ProviderError("invalid_binding", "at least one tool binding is required")
        tool_ids = set()
        for tool in tools:
            if tool not in TOOL_REFS.values():
                raise ProviderError("invalid_binding", "tool component is not implemented by this provider")
            tool_ids.add(tool["id"])
        mounts = params["mounts"]
        if not isinstance(mounts, list) or len(mounts) != 1:
            raise ProviderError("invalid_binding", "exactly one index mount is required")
        mount = require_object(mounts[0], "mount")
        require_exact_keys(
            mount,
            {"logical_name", "role", "component", "access", "process_path"},
            {"logical_name", "role", "component", "access", "process_path"},
            "mount",
        )
        if mount.get("role") != "index" or mount.get("access") != "read_only" or not isinstance(mount.get("process_path"), str):
            raise ProviderError("policy_denied", "provider requires one read-only index mount")
        index = SemanticIndex.open(Path(mount["process_path"]))
        if params["indexes"] != [index.manifest["component"]] or mount.get("component") != index.manifest["component"]:
            raise ProviderError("invalid_binding", "mounted index identity does not match open binding")
        if params["source_snapshots"] != index.manifest["source_snapshots"]:
            raise ProviderError("invalid_binding", "source snapshot binding does not match index manifest")
        binding = params["binding_lock_sha256"]
        if not isinstance(binding, str) or len(binding) != 64 or any(c not in "0123456789abcdef" for c in binding):
            raise ProviderError("invalid_binding", "binding_lock_sha256 is invalid")
        contract = require_object(params["query_time_contract"], "query_time_contract")
        endpoint = contract.get("embedding_endpoint", self.default_embedding_endpoint)
        if not isinstance(endpoint, str):
            raise ProviderError("invalid_binding", "embedding_endpoint must be a string")
        limits = require_object(params["limits"], "limits")
        result_bytes = limits.get("result_bytes", 1048576)
        if isinstance(result_bytes, bool) or not isinstance(result_bytes, int) or result_bytes < 1024:
            raise ProviderError("invalid_binding", "limits.result_bytes must be an integer >= 1024")
        session_id = "rag_" + uuid.uuid4().hex
        self.sessions[session_id] = Session(binding, SemanticService(index, endpoint), tool_ids, result_bytes)
        return {"response_kind": "open", "session_id": session_id, "binding_lock_sha256": binding}

    def _call(self, session: Session, params: dict[str, Any], deadline: int) -> dict[str, Any]:
        require_exact_keys(params, {"session_id", "tool", "arguments"}, {"session_id", "tool", "arguments"}, "params")
        tool = params["tool"]
        if not isinstance(tool, dict) or tool.get("id") not in session.tool_ids or tool != TOOL_REFS.get(tool.get("id")):
            raise ProviderError("policy_denied", "tool is not granted to this session")
        if tool["id"] == SEARCH_TOOL_ID:
            output = session.service.search(params["arguments"], deadline)
        elif tool["id"] == SIMILAR_TOOL_ID:
            output = session.service.similar(params["arguments"], deadline)
        else:
            raise ProviderError("policy_denied", "tool is not implemented")
        validate_semantic_result(output)
        encoded = json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > session.result_bytes:
            raise ProviderError("resource_exhausted", "tool result exceeds the session byte limit")
        return {"response_kind": "call", "output": output}


def error_response(request_id: str, error: Exception) -> dict[str, Any]:
    code = getattr(error, "code", "invalid_request" if isinstance(error, (ValueError, ContractError)) else "internal_error")
    retryable = getattr(error, "retryable", isinstance(error, EmbeddingError) and code == "unavailable")
    body: dict[str, Any] = {"code": code, "message": str(error) or "provider error", "retryable": retryable}
    detail = getattr(error, "detail", None)
    if detail is not None:
        body["detail"] = detail
    return {"protocol": PROTOCOL, "id": request_id or "unknown", "error": body}


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, *, embedding_endpoint: str = "http://127.0.0.1:1234") -> int:
    provider = Provider(embedding_endpoint)
    for line in stdin:
        request_id = "unknown"
        try:
            if not line.endswith("\n"):
                raise ProviderError("protocol_error", "JSONL request must end with LF")
            request = json.loads(line)
            if isinstance(request, dict) and isinstance(request.get("id"), str):
                request_id = request["id"]
            result = provider.handle(request)
            response = {"protocol": PROTOCOL, "id": request_id, "result": result}
        except Exception as error:
            response = error_response(request_id, error)
        stdout.write(json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stdout.flush()
    return 0
