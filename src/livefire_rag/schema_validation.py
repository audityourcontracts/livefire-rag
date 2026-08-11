"""Offline SDK plus RAG JSON Schema registry for provider result validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from .contracts import schema_root


class ResultSchemaError(ValueError):
    pass


def _sdk_schema_root() -> Path:
    configured = os.environ.get("LIVEFIRE_SDK_SPECS")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    distribution = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            distribution / "schemas/sdk",
            distribution.parent / "livefire-sdk/specs",
        ]
    )
    for candidate in candidates:
        if (candidate / "component-ref.v1.schema.json").is_file() and (
            candidate / "source-record-pointer.v1.schema.json"
        ).is_file():
            return candidate
    raise ResultSchemaError(
        "offline SDK schemas are unavailable; set LIVEFIRE_SDK_SPECS or use the SDK bundle"
    )


def semantic_result_validator() -> Draft202012Validator:
    rag_root = schema_root()
    sdk_root = _sdk_schema_root()
    registry = Registry()
    schemas: dict[str, dict[str, Any]] = {}
    for path in [*sorted(sdk_root.glob("*.json")), *sorted(rag_root.glob("*.json"))]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("$id"):
            Draft202012Validator.check_schema(value)
            registry = registry.with_resource(value["$id"], Resource.from_contents(value))
            schemas[value["$id"]] = value
    schema_id = "https://livefire.dev/rag/semantic-result.v1.schema.json"
    if schema_id not in schemas:
        raise ResultSchemaError("semantic-result.v1 schema is absent from the offline registry")
    return Draft202012Validator(
        schemas[schema_id], registry=registry, format_checker=FormatChecker()
    )


_VALIDATOR: Draft202012Validator | None = None


def validate_semantic_result(value: Any) -> None:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = semantic_result_validator()
    errors = sorted(_VALIDATOR.iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ResultSchemaError(f"semantic result schema violation at {path}: {first.message}")
