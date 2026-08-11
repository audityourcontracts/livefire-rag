"""Normative component identities and closed request validation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
)


PROTOCOL = "livefire.tool/1"
PROVIDER_ID = "com.ayc.livefire-rag.provider"
PROVIDER_VERSION = "0.1.0"
INDEX_FORMAT_ID = "com.ayc.livefire-rag.semantic-index-format"
INDEX_FORMAT_VERSION = "1.0.0"
SEARCH_TOOL_ID = "com.ayc.livefire-rag.cli.search"
SIMILAR_TOOL_ID = "com.ayc.livefire-rag.cli.similar"
TOOL_VERSION = "1.0.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHELL_FAMILIES = {
    "powershell", "cmd", "posix_shell", "python", "cloud_cli", "direct_exec", "unknown"
}

SEARCH_DESCRIPTION = (
    "Return immutable source pointers matching a natural-language command investigation query."
)
SIMILAR_DESCRIPTION = (
    "Return immutable source pointers semantically similar to one indexed command."
)

PROVIDER_WRAPPER_TEXT = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.dont_write_bytecode = True\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))\n"
    "from livefire_rag.cli import main\n"
    "raise SystemExit(main(['provider', *sys.argv[1:]]))\n"
)


def _distribution_root() -> Path:
    # Editable checkout: <root>/src/livefire_rag. Bundle: <root>/lib/livefire_rag.
    return Path(__file__).resolve().parents[2]


def schema_root() -> Path:
    root = _distribution_root()
    for candidate in (root / "specs", root / "schemas"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Livefire RAG schemas are unavailable beside the provider implementation")


def schema_component_ref(name: str) -> dict[str, str]:
    value = json.loads((schema_root() / name).read_text(encoding="utf-8"))
    return component_ref(value["$id"], "1", value)


CLI_COMMON_SCHEMA_REF = schema_component_ref("cli-common.v1.schema.json")
SEARCH_INPUT_SCHEMA_REF = schema_component_ref("cli-search.input.v1.schema.json")
SIMILAR_INPUT_SCHEMA_REF = schema_component_ref("cli-similar.input.v1.schema.json")
SEMANTIC_RESULT_SCHEMA_REF = schema_component_ref("semantic-result.v1.schema.json")


INDEX_FORMAT_MATERIAL = {
    "schema_version": "livefire.rag.semantic-index-format/1",
    "documents": "rfc8785-jsonl",
    "vectors": "little-endian-float32-row-major",
    "object_lock": "livefire.object-lock/1",
    "distance": "cosine",
    "accumulation": "float64",
    "tie_break": "distance_asc_command_id_asc",
}
INDEX_FORMAT_REF = component_ref(INDEX_FORMAT_ID, INDEX_FORMAT_VERSION, INDEX_FORMAT_MATERIAL)


def _tool_descriptor(
    tool_id: str,
    name: str,
    description: str,
    input_schema: dict[str, str],
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": "livefire.tool-descriptor/1",
        "tool": {"id": tool_id, "version": TOOL_VERSION, "sha256": ""},
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "output_schema": SEMANTIC_RESULT_SCHEMA_REF,
        "result_semantics": "candidate_pointer",
        "evidence_policy": "pointer_only",
        "required_indexes": [
            {"format_id": INDEX_FORMAT_ID, "accepted_versions": [INDEX_FORMAT_VERSION]}
        ],
        "limits": {
            "request_bytes": 65536,
            "result_bytes": 1048576,
            "wall_time_ms": 30000,
            "max_candidates": 1000,
        },
        "determinism": "ranked_deterministic",
    }
    descriptor["tool"]["sha256"] = canonical_sha256_omitting(
        descriptor, ("tool", "sha256")
    )
    return descriptor


SEARCH_TOOL_DESCRIPTOR = _tool_descriptor(
    SEARCH_TOOL_ID, "cli.search", SEARCH_DESCRIPTION, SEARCH_INPUT_SCHEMA_REF
)
SIMILAR_TOOL_DESCRIPTOR = _tool_descriptor(
    SIMILAR_TOOL_ID, "cli.similar", SIMILAR_DESCRIPTION, SIMILAR_INPUT_SCHEMA_REF
)
SEARCH_TOOL_REF = SEARCH_TOOL_DESCRIPTOR["tool"]
SIMILAR_TOOL_REF = SIMILAR_TOOL_DESCRIPTOR["tool"]
TOOL_REFS = {SEARCH_TOOL_ID: SEARCH_TOOL_REF, SIMILAR_TOOL_ID: SIMILAR_TOOL_REF}
TOOL_DESCRIPTORS = {
    SEARCH_TOOL_ID: SEARCH_TOOL_DESCRIPTOR,
    SIMILAR_TOOL_ID: SIMILAR_TOOL_DESCRIPTOR,
}


def provider_object_lock() -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    objects = [
        {
            "path": "bin/livefire-rag-provider",
            "media_type": "text/x-python",
            "sha256": sha256_bytes(PROVIDER_WRAPPER_TEXT.encode("utf-8")),
            "bytes": len(PROVIDER_WRAPPER_TEXT.encode("utf-8")),
        }
    ]
    for source in sorted(source_dir.glob("*.py"), key=lambda path: path.name):
        data = source.read_bytes()
        objects.append(
            {
                "path": f"lib/livefire_rag/{source.name}",
                "media_type": "text/x-python",
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
    objects.sort(key=lambda item: (item["path"], item["sha256"]))
    return {"schema_version": "livefire.object-lock/1", "objects": objects}


PROVIDER_OBJECT_LOCK = provider_object_lock()
PROVIDER_REF = component_ref(PROVIDER_ID, PROVIDER_VERSION, PROVIDER_OBJECT_LOCK)


def development_binding(index_manifest: dict[str, Any]) -> dict[str, Any]:
    profile = index_manifest["embedding_profile"]
    return {
        "schema_version": "livefire.rag.development-binding-lock/1",
        "status": "development_only_not_admitted",
        "warning": (
            "This digest is a POC binding receipt, not a Livefire SDK ToolBindingLock or "
            "index-admission receipt."
        ),
        "provider": PROVIDER_REF,
        "tools": [SEARCH_TOOL_REF, SIMILAR_TOOL_REF],
        "input_schemas": [SEARCH_INPUT_SCHEMA_REF, SIMILAR_INPUT_SCHEMA_REF],
        "output_schema": SEMANTIC_RESULT_SCHEMA_REF,
        "index": index_manifest["component"],
        "index_format": INDEX_FORMAT_REF,
        "source_snapshots": index_manifest["source_snapshots"],
        "embedding_profile_sha256": component_ref(
            "livefire.rag.embedding-profile.inline", "1", profile
        )["sha256"],
        "query_time_contract": {
            "network": "loopback_only",
            "api_contract": profile["api_contract"]
            if "api_contract" in profile
            else "fixture_embedding",
            "api_model_key": profile["api_model_key"],
        },
        "limits": {"result_bytes": 1048576, "wall_time_ms": 30000},
        "result_semantics": "candidate_pointer_or_miss_never_evidence",
    }


def development_binding_object_lock(binding: dict[str, Any]) -> dict[str, Any]:
    data = canonical_json_bytes(binding, newline=True)
    return {
        "schema_version": "livefire.object-lock/1",
        "objects": [
            {
                "path": "bindings/development-binding-lock.json",
                "media_type": "application/vnd.livefire.rag.development-binding-lock+json",
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        ],
    }


def development_binding_ref(binding: dict[str, Any]) -> dict[str, str]:
    return component_ref(
        "com.ayc.livefire-rag.development-binding-lock",
        "1",
        development_binding_object_lock(binding),
    )


class ContractError(ValueError):
    pass


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def require_exact_keys(value: dict[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ContractError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ContractError(f"{label} is missing fields: {', '.join(missing)}")


def parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a date-time string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{label} must be an RFC3339 date-time") from error
    if parsed.tzinfo is None:
        raise ContractError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_filters(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    filters = require_object(value, "filters")
    allowed = {"principals", "host_ids", "shell_families", "source_snapshot_ids", "exclude_command_ids"}
    require_exact_keys(filters, allowed, set(), "filters")
    for field in ("host_ids", "source_snapshot_ids", "exclude_command_ids"):
        if field in filters and (
            not isinstance(filters[field], list)
            or any(not isinstance(item, str) or not item for item in filters[field])
        ):
            raise ContractError(f"filters.{field} must be an array of non-empty strings")
    if "shell_families" in filters and (
        not isinstance(filters["shell_families"], list)
        or any(item not in SHELL_FAMILIES for item in filters["shell_families"])
    ):
        raise ContractError("filters.shell_families contains an unsupported value")
    if "principals" in filters:
        if not isinstance(filters["principals"], list):
            raise ContractError("filters.principals must be an array")
        for index, principal in enumerate(filters["principals"]):
            item = require_object(principal, f"filters.principals[{index}]")
            require_exact_keys(item, {"namespace", "id"}, {"namespace", "id"}, f"filters.principals[{index}]")
            if any(not isinstance(item[key], str) or not item[key] for key in ("namespace", "id")):
                raise ContractError(f"filters.principals[{index}] fields must be non-empty strings")
    return filters


def validate_time_range(value: Any, *, required: bool) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ContractError("time_range is required")
        return None
    result = require_object(value, "time_range")
    require_exact_keys(result, {"start", "end_exclusive"}, {"start", "end_exclusive"}, "time_range")
    if parse_timestamp(result["start"], "time_range.start") >= parse_timestamp(
        result["end_exclusive"], "time_range.end_exclusive"
    ):
        raise ContractError("time_range.start must be before end_exclusive")
    return result


def validate_top_n(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
        raise ContractError("top_n must be an integer in [1,1000]")
    return value


def validate_search(arguments: Any) -> dict[str, Any]:
    request = require_object(arguments, "arguments")
    require_exact_keys(
        request,
        {"schema_version", "query", "time_range", "top_n", "filters"},
        {"schema_version", "query", "time_range", "top_n"},
        "arguments",
    )
    if request["schema_version"] != "livefire.rag.cli-search.input/1":
        raise ContractError("unsupported search schema_version")
    if not isinstance(request["query"], str) or not 1 <= len(request["query"]) <= 8192:
        raise ContractError("query must contain 1 to 8192 characters")
    validate_top_n(request["top_n"])
    validate_time_range(request["time_range"], required=True)
    validate_filters(request.get("filters"))
    return request


def validate_similar(arguments: Any) -> dict[str, Any]:
    request = require_object(arguments, "arguments")
    require_exact_keys(
        request,
        {"schema_version", "command_id", "top_n", "exclude_seed", "time_range", "filters"},
        {"schema_version", "command_id", "top_n"},
        "arguments",
    )
    if request["schema_version"] != "livefire.rag.cli-similar.input/1":
        raise ContractError("unsupported similar schema_version")
    if not isinstance(request["command_id"], str) or not request["command_id"]:
        raise ContractError("command_id must be a non-empty string")
    validate_top_n(request["top_n"])
    if "exclude_seed" in request and not isinstance(request["exclude_seed"], bool):
        raise ContractError("exclude_seed must be boolean")
    validate_time_range(request.get("time_range"), required=False)
    validate_filters(request.get("filters"))
    return request
