"""Livefire SDK JSONL provider for the generic ``evidence.search`` tool."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from .canonical import canonical_json_bytes, sha256_bytes
from .embedding import embed_query
from .evidence_bundle import (
    INDEX_FORMAT_REF,
    INPUT_SCHEMA_REF,
    OUTPUT_SCHEMA_REF,
    PROTOCOL,
    PROVIDER_EXECUTABLE_ARTIFACT,
    PROVIDER_REF,
    RETRIEVAL_POLICY_REF,
    TOOL_DESCRIPTOR,
    TOOL_REF,
)
from .evidence_service import (
    EvidenceBindingError,
    EvidenceError,
    EvidenceIndex,
    EvidenceService,
    EvidenceUnavailable,
    validate_sdk_value,
)


class ProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail


@dataclass
class Session:
    binding_lock_sha256: str
    service: EvidenceService
    result_bytes: int
    max_candidates: int


def _exact_keys(
    value: Any,
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderError("invalid_request", f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown or missing:
        detail = {"unknown": unknown, "missing": missing}
        raise ProviderError("invalid_request", f"{label} has unknown or missing fields", detail=detail)
    return value


def _read_json_mount(path_text: str, filename: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if path.is_dir():
        path = path / filename
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderError("corrupt_artifact", f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ProviderError("corrupt_artifact", f"{label} must be a JSON object")
    return value


def _mounts_by_name(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ProviderError("invalid_binding", "mounts must be an array")
    mounts: dict[str, dict[str, Any]] = {}
    required = {"logical_name", "role", "component", "access", "process_path"}
    for position, item in enumerate(value):
        mount = _exact_keys(item, allowed=required, required=required, label=f"mounts[{position}]")
        name = mount["logical_name"]
        if not isinstance(name, str) or not name or name in mounts:
            raise ProviderError("invalid_binding", "mount logical names must be unique and non-empty")
        if mount["access"] != "read_only":
            raise ProviderError("policy_denied", "evidence provider accepts read-only mounts only")
        if not isinstance(mount["process_path"], str) or not mount["process_path"]:
            raise ProviderError("invalid_binding", "mount process_path must be non-empty")
        mounts[name] = mount
    expected = {
        "evidence-index",
        "tool-binding-lock",
        "index-admission-receipt",
        "embedding-profile",
    }
    if set(mounts) != expected:
        raise ProviderError(
            "invalid_binding",
            "open requires exactly the evidence index, binding lock, admission receipt, and embedding profile mounts",
        )
    expected_roles = {
        "evidence-index": "index",
        "tool-binding-lock": "policy",
        "index-admission-receipt": "policy",
        "embedding-profile": "model",
    }
    for name, role in expected_roles.items():
        if mounts[name]["role"] != role:
            raise ProviderError("invalid_binding", f"{name} mount has the wrong role")
    return mounts


def _loopback_endpoint(contract: Mapping[str, Any]) -> str:
    network = contract.get("network")
    if not isinstance(network, list):
        raise ProviderError("invalid_binding", "query_time_contract.network must be an array")
    endpoints = [item.removeprefix("loopback:") for item in network if isinstance(item, str) and item.startswith("loopback:http")]
    if len(endpoints) != 1:
        raise ProviderError(
            "invalid_binding",
            "local_component mode requires exactly one loopback:http embedding endpoint",
        )
    return endpoints[0]


class EvidenceProvider:
    def __init__(self, *, sdk_specs: Path | None = None) -> None:
        self.sdk_specs = sdk_specs
        self.handshaken = False
        self.sessions: dict[str, Session] = {}

    def handle(self, request: Any) -> dict[str, Any]:
        value = _exact_keys(
            request,
            allowed={"protocol", "id", "method", "params", "context"},
            required={"protocol", "id", "method", "params", "context"},
            label="request",
        )
        if value["protocol"] != PROTOCOL or not isinstance(value["id"], str) or not value["id"]:
            raise ProviderError("protocol_error", "protocol and request id are invalid")
        context = _exact_keys(
            value["context"],
            allowed={"trace_id", "deadline_unix_ms"},
            required={"trace_id", "deadline_unix_ms"},
            label="context",
        )
        if not isinstance(context["trace_id"], str) or not context["trace_id"]:
            raise ProviderError("invalid_request", "context.trace_id must be non-empty")
        deadline = context["deadline_unix_ms"]
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 1:
            raise ProviderError("invalid_request", "context.deadline_unix_ms must be positive")
        if int(time.time() * 1000) >= deadline:
            raise ProviderError("deadline_exceeded", "request deadline has expired")
        params = value["params"]
        method = value["method"]
        if method == "handshake":
            _exact_keys(params, allowed=set(), required=set(), label="params")
            self.handshaken = True
            return {
                "response_kind": "handshake",
                "provider": PROVIDER_REF,
                "protocol": PROTOCOL,
                "tools": [TOOL_REF],
                "accepted_index_formats": [INDEX_FORMAT_REF],
            }
        if not self.handshaken:
            raise ProviderError("protocol_error", "handshake is required before session methods")
        if method == "open":
            return self._open(params)
        if method not in {"call", "health", "close"}:
            raise ProviderError("invalid_request", "unsupported method")
        if not isinstance(params, dict):
            raise ProviderError("invalid_request", "params must be an object")
        session_id = params.get("session_id")
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise ProviderError("not_found", "session was not found")
        if method == "call":
            return self._call(self.sessions[session_id], params, deadline)
        _exact_keys(params, allowed={"session_id"}, required={"session_id"}, label="params")
        if method == "health":
            return {
                "response_kind": "health",
                "status": "ready",
                "binding_lock_sha256": self.sessions[session_id].binding_lock_sha256,
            }
        session = self.sessions.pop(session_id)
        session.service.index.close()
        return {"response_kind": "close", "closed": True}

    def _open(self, params: Any) -> dict[str, Any]:
        fields = {
            "provider",
            "tools",
            "indexes",
            "source_snapshots",
            "binding_lock_sha256",
            "query_time_contract",
            "limits",
            "mounts",
        }
        params = _exact_keys(params, allowed=fields, required=fields, label="params")
        mounts = _mounts_by_name(params["mounts"])
        lock = _read_json_mount(
            mounts["tool-binding-lock"]["process_path"], "tool-binding-lock.json", "tool binding lock"
        )
        sdk_specs = self._sdk_specs()
        validate_sdk_value("tool-binding-lock.v1.schema.json", lock, sdk_specs=sdk_specs)
        binding_digest = sha256_bytes(canonical_json_bytes(lock))
        if params["binding_lock_sha256"] != binding_digest:
            raise ProviderError("invalid_binding", "binding_lock_sha256 does not match the mounted lock")
        binding_component = mounts["tool-binding-lock"]["component"]
        if not isinstance(binding_component, dict) or binding_component.get("sha256") != binding_digest:
            raise ProviderError("invalid_binding", "binding-lock mount component does not match its bytes")
        required_lock = {
            "schema_version",
            "descriptor",
            "provider",
            "executable",
            "input_schema",
            "output_schema",
            "index",
            "index_format",
            "index_admission_receipt",
            "source_snapshots",
            "retrieval_policy",
            "query_time_contract",
            "protocol",
            "limits",
        }
        allowed_lock = required_lock | {"provider_permissions_sha256"}
        _exact_keys(lock, allowed=allowed_lock, required=required_lock, label="tool binding lock")
        expected_values = {
            "schema_version": "livefire.tool-binding-lock/1",
            "descriptor": TOOL_REF,
            "provider": PROVIDER_REF,
            "executable": PROVIDER_EXECUTABLE_ARTIFACT,
            "input_schema": INPUT_SCHEMA_REF,
            "output_schema": OUTPUT_SCHEMA_REF,
            "index_format": INDEX_FORMAT_REF,
            "retrieval_policy": RETRIEVAL_POLICY_REF,
            "protocol": PROTOCOL,
        }
        for field, expected in expected_values.items():
            if lock[field] != expected:
                raise ProviderError("invalid_binding", f"binding lock {field} is incompatible")
        if params["provider"] != PROVIDER_REF or params["provider"] != lock["provider"]:
            raise ProviderError("invalid_binding", "provider identity does not match the binding")
        if params["tools"] != [TOOL_REF] or params["tools"] != [lock["descriptor"]]:
            raise ProviderError("invalid_binding", "evidence.search requires its exact singleton tool binding")
        if params["query_time_contract"] != lock["query_time_contract"]:
            raise ProviderError("invalid_binding", "query-time contract differs from the binding lock")
        query_contract = _exact_keys(
            lock["query_time_contract"],
            allowed={"mode", "network", "secret_handles", "vendor_services"},
            required={"mode", "network", "secret_handles", "vendor_services"},
            label="query_time_contract",
        )
        if (
            query_contract["mode"] != "local_component"
            or query_contract["secret_handles"] != []
            or query_contract["vendor_services"] != []
        ):
            raise ProviderError(
                "invalid_binding",
                "evidence.search requires a secret-free local_component query contract",
            )
        if params["limits"] != lock["limits"]:
            raise ProviderError("invalid_binding", "runtime limits differ from the binding lock")
        self._validate_limits(lock["limits"])

        index = EvidenceIndex.open(
            Path(mounts["evidence-index"]["process_path"]),
            sdk_specs=sdk_specs,
        )
        if index.manifest["index_format_descriptor"] != INDEX_FORMAT_REF:
            index.close()
            raise ProviderError("invalid_binding", "evidence index format is incompatible")
        if params["indexes"] != [index.component] or lock["index"] != index.component:
            raise ProviderError("invalid_binding", "index identity does not match the binding lock")
        if mounts["evidence-index"]["component"] != index.component:
            raise ProviderError("invalid_binding", "index mount component does not match the index")
        if params["source_snapshots"] != index.source_snapshots or lock["source_snapshots"] != index.source_snapshots:
            raise ProviderError("invalid_binding", "source snapshots do not match the admitted index")

        receipt = _read_json_mount(
            mounts["index-admission-receipt"]["process_path"],
            "index-admission-receipt.json",
            "index admission receipt",
        )
        validate_sdk_value("index-admission-receipt.v1.schema.json", receipt, sdk_specs=sdk_specs)
        receipt_digest = sha256_bytes(canonical_json_bytes(receipt))
        receipt_component = mounts["index-admission-receipt"]["component"]
        if receipt_component != lock["index_admission_receipt"] or receipt_component.get("sha256") != receipt_digest:
            raise ProviderError("invalid_binding", "index admission receipt identity does not match")
        required_checks = {
            "object_digests",
            "source_binding",
            "safe_paths",
            "schema_profiles",
            "coverage_closure",
            "pointer_closure",
            "offline_query_conformance",
            "conformance",
        }
        checks = receipt.get("checks")
        if (
            receipt.get("schema_version") != "livefire.index-admission/1"
            or receipt.get("disposition") != "admitted"
            or not isinstance(receipt.get("authority_signature"), str)
            or not receipt["authority_signature"]
            or receipt.get("index_manifest_sha256") != index.component["sha256"]
            or not isinstance(checks, dict)
            or any(checks.get(name) is not True for name in required_checks)
        ):
            raise ProviderError("invalid_binding", "index admission receipt is not an admitted receipt for this index")
        if index.manifest.get("pilot_sample") is not None and not receipt[
            "authority_signature"
        ].startswith("local-test:"):
            raise ProviderError(
                "invalid_binding",
                "pilot sample indexes require an explicit local-test receipt",
            )

        profile = _read_json_mount(
            mounts["embedding-profile"]["process_path"], "embedding-policy.json", "embedding profile"
        )
        profile_ref = mounts["embedding-profile"]["component"]
        if profile_ref != index.embedding_profile or sha256_bytes(canonical_json_bytes(profile)) != profile_ref.get("sha256"):
            raise ProviderError("invalid_binding", "embedding profile does not match the index")
        if profile.get("purpose") != "semantic_search" or profile.get("normalization") != "l2":
            raise ProviderError("invalid_binding", "embedding profile is incompatible with evidence.search")
        if profile.get("admission_status") == "development_only" and not receipt[
            "authority_signature"
        ].startswith("local-test:"):
            raise ProviderError(
                "invalid_binding",
                "development-only embedding profiles require an explicit local-test receipt",
            )
        if index.profile.get("dimensions") != profile.get("dimensions"):
            raise ProviderError("invalid_binding", "embedding profile dimension does not match the index")
        endpoint = _loopback_endpoint(lock["query_time_contract"])

        def query_embedding(query: str, deadline_unix_ms: int):
            try:
                text = profile["query_composition"].format(
                    query_instruction=profile["query_instruction"], query=query
                )
                return embed_query(
                    endpoint,
                    profile["api_model_key"],
                    text,
                    dimensions=profile["dimensions"],
                    deadline_unix_ms=deadline_unix_ms,
                )
            except (KeyError, ValueError, TypeError) as error:
                raise EvidenceBindingError("embedding profile query contract is invalid") from error

        session_id = "evidence_" + uuid.uuid4().hex
        self.sessions[session_id] = Session(
            binding_digest,
            EvidenceService(index, embed_query=query_embedding, sdk_specs=self.sdk_specs),
            lock["limits"]["result_bytes"],
            lock["limits"]["max_candidates"],
        )
        return {
            "response_kind": "open",
            "session_id": session_id,
            "binding_lock_sha256": binding_digest,
        }

    def _sdk_specs(self) -> Path:
        if self.sdk_specs is not None:
            return Path(self.sdk_specs)
        module_root = Path(__file__).resolve().parent
        candidates = (
            module_root / "evidence_specs/sdk",
            module_root.parents[1] / "../livefire-sdk/specs",
        )
        for candidate in candidates:
            if (candidate / "tool-binding-lock.v1.schema.json").is_file() and (
                candidate / "index-admission-receipt.v1.schema.json"
            ).is_file():
                return candidate.resolve()
        raise ProviderError("invalid_binding", "offline Livefire SDK binding schemas are unavailable")

    @staticmethod
    def _validate_limits(limits: Any) -> None:
        expected = {"request_bytes", "result_bytes", "wall_time_ms", "memory_bytes", "max_candidates"}
        limits = _exact_keys(limits, allowed=expected, required=expected, label="limits")
        descriptor_limits = TOOL_DESCRIPTOR["limits"]
        for name in expected:
            value = limits[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProviderError("invalid_binding", f"limits.{name} must be positive")
        for name in ("request_bytes", "result_bytes", "wall_time_ms", "max_candidates"):
            if limits[name] > descriptor_limits[name]:
                raise ProviderError("invalid_binding", f"limits.{name} exceeds the tool descriptor ceiling")

    def _call(self, session: Session, params: Any, deadline: int) -> dict[str, Any]:
        params = _exact_keys(
            params,
            allowed={"session_id", "tool", "arguments"},
            required={"session_id", "tool", "arguments"},
            label="params",
        )
        if params["tool"] != TOOL_REF:
            raise ProviderError("policy_denied", "tool is not granted to this session")
        arguments = params["arguments"]
        if isinstance(arguments, dict):
            top_n = arguments.get("top_n")
            if isinstance(top_n, int) and not isinstance(top_n, bool) and top_n > session.max_candidates:
                raise ProviderError("resource_exhausted", "top_n exceeds the bound max_candidates")
        output = session.service.search(arguments, deadline)
        encoded = canonical_json_bytes(output)
        if len(encoded) > session.result_bytes:
            raise ProviderError("resource_exhausted", "tool result exceeds the session byte limit")
        return {"response_kind": "call", "output": output}


def error_response(request_id: str, error: Exception) -> dict[str, Any]:
    code = getattr(error, "code", "internal_error")
    if code not in {
        "invalid_request",
        "invalid_binding",
        "policy_denied",
        "not_found",
        "deadline_exceeded",
        "resource_exhausted",
        "corrupt_artifact",
        "protocol_error",
        "internal_error",
        "unavailable",
    }:
        code = "internal_error"
    retryable = bool(getattr(error, "retryable", isinstance(error, EvidenceUnavailable)))
    body: dict[str, Any] = {
        "code": code,
        "message": str(error) or "evidence provider error",
        "retryable": retryable,
    }
    detail = getattr(error, "detail", None)
    if detail is not None:
        body["detail"] = detail
    return {"protocol": PROTOCOL, "id": request_id or "unknown", "error": body}


def serve(
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    *,
    sdk_specs: Path | None = None,
) -> int:
    provider = EvidenceProvider(sdk_specs=sdk_specs)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Livefire generic evidence provider")
    parser.add_argument("--sdk-specs", type=Path)
    args = parser.parse_args(argv)
    return serve(sdk_specs=args.sdk_specs)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EvidenceProvider", "ProviderError", "error_response", "main", "serve"]
