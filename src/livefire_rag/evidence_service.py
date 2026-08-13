"""Validated provider-facing service over the disk-backed sealed evidence index."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from .evidence_index import EvidenceIndex, EvidenceIndexCorrupt, EvidenceIndexError
from .evidence_schema import _offline_registry, generic_schema_root


class EvidenceError(RuntimeError):
    code = "invalid_request"


class EvidenceIndexNotFound(EvidenceError):
    code = "not_found"


class EvidenceBindingError(EvidenceError):
    code = "invalid_binding"


class EvidenceDeadlineExceeded(EvidenceError):
    code = "deadline_exceeded"


class EvidenceUnavailable(EvidenceError):
    code = "unavailable"


def evidence_validator(name: str, *, sdk_specs: Path | None = None) -> Draft202012Validator:
    if sdk_specs is None:
        module_root = Path(__file__).resolve().parent
        candidates = (
            module_root / "evidence_specs" / "sdk",
            module_root.parents[1] / "../livefire-sdk/specs",
        )
        sdk_specs = next((path.resolve() for path in candidates if path.is_dir()), None)
    if sdk_specs is None:
        raise EvidenceBindingError("offline Livefire SDK schemas are unavailable")
    registry, schemas = _offline_registry(generic_schema_root(), sdk_specs)
    if name not in schemas:
        raise EvidenceBindingError(f"offline evidence schema is unavailable: {name}")
    return Draft202012Validator(
        schemas[name], registry=registry, format_checker=FormatChecker()
    )


def validate_evidence_value(name: str, value: Any, *, sdk_specs: Path | None = None) -> None:
    errors = sorted(
        evidence_validator(name, sdk_specs=sdk_specs).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        path = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise EvidenceError(f"{name} violation at {path}: {first.message}")


def validate_sdk_value(name: str, value: Any, *, sdk_specs: Path) -> None:
    registry = Registry()
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(sdk_specs).glob("*.schema.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceBindingError(f"offline SDK schema is unreadable: {path.name}") from error
        if isinstance(schema, dict) and isinstance(schema.get("$id"), str):
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
            schemas[path.name] = schema
    if name not in schemas:
        raise EvidenceBindingError(f"offline SDK schema is unavailable: {name}")
    validator = Draft202012Validator(
        schemas[name], registry=registry, format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        path = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise EvidenceBindingError(f"{name} violation at {path}: {first.message}")


def _parse_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise EvidenceError(f"{label} must be an RFC3339 date-time") from error
    if parsed.tzinfo is None:
        raise EvidenceError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


class EvidenceService:
    """Validate the public contract and delegate retrieval to the sealed index."""

    def __init__(
        self,
        index: EvidenceIndex,
        *,
        embed_query: Callable[[str, int], np.ndarray] | None = None,
        sdk_specs: Path | None = None,
    ) -> None:
        self.index = index
        self.embed_query = embed_query
        self.sdk_specs = sdk_specs

    def search(self, arguments: Any, deadline_unix_ms: int) -> dict[str, Any]:
        validate_evidence_value(
            "evidence-search.input.v1.schema.json", arguments, sdk_specs=self.sdk_specs
        )
        if int(time.time() * 1000) >= deadline_unix_ms:
            raise EvidenceDeadlineExceeded("call deadline exceeded")
        if arguments.get("time_range"):
            time_range = arguments["time_range"]
            if _parse_time(time_range["start"], "time_range.start") >= _parse_time(
                time_range["end_exclusive"], "time_range.end_exclusive"
            ):
                raise EvidenceError("time_range.start must be before time_range.end_exclusive")
        vector = None
        if "dense" in arguments["retrieval"]["methods"]:
            if self.embed_query is None:
                raise EvidenceUnavailable(
                    "dense retrieval requires the bound local embedding component"
                )
            vector = self.embed_query(arguments["query"], deadline_unix_ms)
        try:
            output = self.index.search(arguments, vector, max_occurrences=100)
        except EvidenceIndexCorrupt:
            raise
        except (EvidenceIndexError, ValueError, TypeError) as error:
            raise EvidenceError(str(error)) from error
        if int(time.time() * 1000) >= deadline_unix_ms:
            raise EvidenceDeadlineExceeded("call deadline exceeded")
        validate_evidence_value(
            "evidence-search.output.v1.schema.json", output, sdk_specs=self.sdk_specs
        )
        return output


__all__ = [
    "EvidenceBindingError", "EvidenceDeadlineExceeded", "EvidenceError",
    "EvidenceIndex", "EvidenceIndexCorrupt", "EvidenceIndexNotFound",
    "EvidenceService", "EvidenceUnavailable", "validate_evidence_value",
    "validate_sdk_value",
]
