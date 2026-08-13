"""Offline Draft 2020-12 validation for generic evidence-pack artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from .canonical import canonical_json_bytes, sha256_bytes


class EvidenceSchemaError(RuntimeError):
    """An evidence-pack artifact does not conform to its declared schema."""


GENERIC_EVIDENCE_SCHEMA_NAMES = (
    "embedding-policy.v1.schema.json",
    "evidence-common.v1.schema.json",
    "evidence-coverage-report.v1.schema.json",
    "evidence-derivation-coverage.v1.schema.json",
    "evidence-derivation-membership-row.v1.schema.json",
    "evidence-derivation-pack.v1.schema.json",
    "evidence-derived-document.v1.schema.json",
    "evidence-document.v1.schema.json",
    "evidence-embedding-row.v1.schema.json",
    "evidence-index-manifest.v1.schema.json",
    "evidence-occurrence-row.v1.schema.json",
    "evidence-pilot-coverage.v1.schema.json",
    "evidence-pilot-sample.v1.schema.json",
    "evidence-pilot-selection-row.v1.schema.json",
    "evidence-projection-pack.v1.schema.json",
    "evidence-search.input.v1.schema.json",
    "evidence-search.output.v1.schema.json",
)


def generic_schema_root(explicit: Path | None = None) -> Path:
    """Locate generic RAG schemas in a checkout or an installed wheel."""

    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    module_root = Path(__file__).resolve().parent
    candidates.extend((module_root / "evidence_specs", module_root.parents[1] / "specs"))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in GENERIC_EVIDENCE_SCHEMA_NAMES):
            return candidate
    if explicit is not None:
        raise EvidenceSchemaError(
            f"generic evidence schemas are incomplete or unavailable: {Path(explicit)}"
        )
    raise EvidenceSchemaError(
        "generic evidence schemas are unavailable in the installed package or source checkout"
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceSchemaError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceSchemaError(f"expected a JSON object: {path}")
    return value


def _schema_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                references.append(child)
            else:
                references.extend(_schema_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(_schema_references(child))
    return references


def _offline_registry(
    rag_specs: Path | None, sdk_specs: Path
) -> tuple[Registry, dict[str, dict[str, Any]]]:
    """Load only generic evidence schemas and their transitive SDK references."""

    registry = Registry()
    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    rag_root = generic_schema_root(rag_specs)
    sdk_root = Path(sdk_specs)
    if not sdk_root.is_dir():
        raise EvidenceSchemaError(
            f"SDK schemas are unavailable; pass an explicit SDK schema directory: {sdk_root}"
        )

    pending: list[Path] = [rag_root / name for name in GENERIC_EVIDENCE_SCHEMA_NAMES]
    loaded_paths: set[Path] = set()
    while pending:
        path = pending.pop(0).resolve()
        if path in loaded_paths:
            continue
        if not path.is_file():
            raise EvidenceSchemaError(f"required offline schema is missing: {path}")
        schema = _load_object(path)
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            raise EvidenceSchemaError(f"invalid schema: {path}") from error
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise EvidenceSchemaError(f"offline schema lacks $id: {path}")
        if schema_id in by_id and by_id[schema_id] != schema:
            raise EvidenceSchemaError(f"conflicting schema id: {schema_id}")
        registry = registry.with_resource(schema_id, Resource.from_contents(schema))
        by_id[schema_id] = schema
        if path.name in by_name and by_name[path.name] != schema:
            raise EvidenceSchemaError(f"conflicting schema filename: {path.name}")
        by_name[path.name] = schema
        loaded_paths.add(path)

        for reference in _schema_references(schema):
            reference_uri = urldefrag(urljoin(schema_id, reference)).url
            if not reference_uri or reference_uri == schema_id or reference_uri in by_id:
                continue
            parsed = urlparse(reference_uri)
            if parsed.netloc == "livefire.dev" and parsed.path.startswith("/rag/"):
                referenced_name = Path(parsed.path).name
                if referenced_name not in GENERIC_EVIDENCE_SCHEMA_NAMES:
                    raise EvidenceSchemaError(
                        f"generic evidence schema references an unscoped RAG schema: {reference_uri}"
                    )
                pending.append(rag_root / referenced_name)
            elif parsed.netloc == "livefire.dev" and parsed.path.startswith("/sdk/"):
                pending.append(sdk_root / Path(parsed.path).name)
            else:
                raise EvidenceSchemaError(f"unsupported offline schema reference: {reference_uri}")
    return registry, by_name


def validate_evidence_pack_schemas(
    root: Path,
    *,
    sdk_specs: Path,
    rag_specs: Path | None = None,
) -> dict[str, int]:
    """Validate every logical row in a projection pack against offline schemas."""

    pack = Path(root)
    registry, schemas = _offline_registry(
        Path(rag_specs) if rag_specs is not None else None, Path(sdk_specs)
    )
    required = {
        "manifest": "evidence-projection-pack.v1.schema.json",
        "coverage": "evidence-coverage-report.v1.schema.json",
        "documents": "evidence-document.v1.schema.json",
        "occurrences": "evidence-occurrence-row.v1.schema.json",
    }
    missing = set(required.values()) - set(schemas)
    if missing:
        raise EvidenceSchemaError(f"offline registry lacks schemas: {sorted(missing)}")

    def validator(name: str) -> Draft202012Validator:
        return Draft202012Validator(
            schemas[required[name]], registry=registry, format_checker=FormatChecker()
        )

    manifest = _load_object(pack / "manifest.json")
    row_bindings = {
        "evidence_document": "evidence-document.v1.schema.json",
        "evidence_occurrence": "evidence-occurrence-row.v1.schema.json",
        "coverage_report": "evidence-coverage-report.v1.schema.json",
    }
    for logical_name, schema_name in row_bindings.items():
        schema = schemas[schema_name]
        expected_ref = {
            "id": schema["$id"],
            "version": "1",
            "sha256": sha256_bytes(canonical_json_bytes(schema)),
        }
        if manifest.get("row_schemas", {}).get(logical_name) != expected_ref:
            raise EvidenceSchemaError(f"manifest does not bind the supplied {schema_name}")
    pointer_schema = schemas.get("source-record-pointer.v1.schema.json")
    if pointer_schema is None:
        raise EvidenceSchemaError("offline registry lacks source-record-pointer schema")
    expected_pointer_ref = {
        "id": pointer_schema["$id"],
        "version": "1",
        "sha256": sha256_bytes(canonical_json_bytes(pointer_schema)),
    }
    if manifest.get("pointer_contract", {}).get("pointer_schema") != expected_pointer_ref:
        raise EvidenceSchemaError("manifest does not bind the supplied pointer schema")

    try:
        validator("manifest").validate(manifest)
        validator("coverage").validate(_load_object(pack / "coverage-report.json"))
    except Exception as error:
        if isinstance(error, EvidenceSchemaError):
            raise
        raise EvidenceSchemaError(f"manifest or coverage schema failure: {error}") from error

    counts: dict[str, int] = {"manifest": 1, "coverage": 1, "documents": 0, "occurrences": 0}
    for logical_name, filename in (
        ("documents", "documents.jsonl"),
        ("occurrences", "occurrences.jsonl"),
    ):
        row_validator = validator(logical_name)
        path = pack / filename
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as error:
            raise EvidenceSchemaError(f"cannot open logical rows: {path}") from error
        with handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                    row_validator.validate(value)
                except Exception as error:
                    raise EvidenceSchemaError(
                        f"{filename}:{line_number}: schema validation failed: {error}"
                    ) from error
                counts[logical_name] += 1
    return counts


__all__ = [
    "EvidenceSchemaError",
    "GENERIC_EVIDENCE_SCHEMA_NAMES",
    "generic_schema_root",
    "validate_evidence_pack_schemas",
]
