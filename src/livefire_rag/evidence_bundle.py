"""SDK descriptors and development bundle packaging for ``evidence.search``."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
    write_canonical_json,
)
from .evidence_schema import (
    GENERIC_EVIDENCE_SCHEMA_NAMES,
    generic_schema_path,
    generic_schema_root,
)


PROTOCOL = "livefire.tool/1"
PROVIDER_ID = "com.ayc.livefire-rag.evidence-provider"
PROVIDER_VERSION = "0.1.0"
TOOL_ID = "com.ayc.livefire-rag.evidence.search"
TOOL_VERSION = "1.0.0"
INDEX_FORMAT_ID = "com.ayc.livefire-rag.evidence-index-format"
INDEX_FORMAT_VERSION = "1.0.0"
# The POC wrapper uses ambient compiled dependencies (DuckDB and NumPy).  Keep
# the target honest until a separately content-bound runtime is mounted.
TARGET = "python3.12-darwin-arm64-ambient-development"

PROVIDER_WRAPPER_TEXT = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.dont_write_bytecode = True\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))\n"
    "from livefire_rag.evidence_provider import main\n"
    "raise SystemExit(main())\n"
)
PROVIDER_EXECUTABLE_ARTIFACT = {
    "path": "bin/livefire-rag-evidence-provider",
    "media_type": "text/x-python",
    "sha256": sha256_bytes(PROVIDER_WRAPPER_TEXT.encode("utf-8")),
    "bytes": len(PROVIDER_WRAPPER_TEXT.encode("utf-8")),
}

RUNTIME_MODULES = (
    "__init__.py",
    "canonical.py",
    "embedding.py",
    "evidence_builder.py",
    "evidence_bundle.py",
    "evidence_derivation.py",
    "evidence_index.py",
    "evidence_projection.py",
    "evidence_provider.py",
    "evidence_schema.py",
    "evidence_service.py",
)
RFC8785_VERSION = "0.1.4"
VENDORED_RUNTIME_FILES = (
    "rfc8785/__init__.py",
    "rfc8785/_impl.py",
    "rfc8785/py.typed",
    f"rfc8785-{RFC8785_VERSION}.dist-info/LICENSE",
    f"rfc8785-{RFC8785_VERSION}.dist-info/METADATA",
)


def _runtime_vendor_root(source_dir: Path) -> Path:
    """Resolve vendored pure-Python runtime bytes in source or staged layouts."""

    staged_root = source_dir.parent
    if (staged_root / "rfc8785/__init__.py").is_file():
        return staged_root
    import rfc8785

    root = Path(rfc8785.__file__).resolve().parent.parent
    if not all((root / relative).is_file() for relative in VENDORED_RUNTIME_FILES):
        raise ValueError("rfc8785 runtime closure is unavailable for packaging")
    return root


def _schema_ref(name: str) -> dict[str, str]:
    schema = json.loads((generic_schema_root() / name).read_text(encoding="utf-8"))
    return component_ref(schema["$id"], "1", schema)


INPUT_SCHEMA_REF = _schema_ref("evidence-search.input.v1.schema.json")
OUTPUT_SCHEMA_REF = _schema_ref("evidence-search.output.v1.schema.json")
DOCUMENT_SCHEMA_REF = _schema_ref("evidence-document.v1.schema.json")
OCCURRENCE_SCHEMA_REF = _schema_ref("evidence-occurrence-row.v1.schema.json")
EMBEDDING_SCHEMA_REF = _schema_ref("evidence-embedding-row.v1.schema.json")
DERIVED_DOCUMENT_SCHEMA_REF = _schema_ref("evidence-derived-document.v1.schema.json")
DERIVATION_MEMBERSHIP_SCHEMA_REF = _schema_ref(
    "evidence-derivation-membership-row.v1.schema.json"
)
COVERAGE_SCHEMA_REF = _schema_ref("evidence-coverage-report.v1.schema.json")

PHYSICAL_PROFILE = {
    "schema_version": "livefire.rag.evidence-physical-profile/1",
    "canonical_format": "parquet",
    "document_order": "document_id_asc",
    "occurrence_order": "occurrence_id_asc",
    "embedding_order": "document_id_asc",
    "derived_caches_authoritative": False,
}
PHYSICAL_PROFILE_REF = component_ref(
    "com.ayc.livefire-rag.evidence-physical-profile", "1.0.0", PHYSICAL_PROFILE
)

VALIDATOR_PROFILE = {
    "schema_version": "livefire.rag.evidence-index-validator/1",
    "object_digests": True,
    "row_schemas": True,
    "occurrence_membership": True,
    "embedding_closure": True,
    "source_pointer_output": True,
}
VALIDATOR_REF = component_ref(
    "com.ayc.livefire-rag.evidence-index-validator", "1.0.0", VALIDATOR_PROFILE
)

RETRIEVAL_POLICY = {
    "schema_version": "livefire.rag.evidence-retrieval-policy/1",
    "occurrence_filters_first": True,
    "dense_distance": "exact_cosine_float64_accumulation",
    "lexical": {"algorithm": "bm25", "k1": 1.2, "b": 0.75},
    "fusion": {"algorithm": "reciprocal_rank", "rank_constant": 60},
    "tie_break": "ranking_score_desc_document_id_asc",
    "max_returned_occurrences_per_candidate": 100,
    "entity_filter": "requires_admitted_entity_membership_projection",
}
RETRIEVAL_POLICY_REF = component_ref(
    "com.ayc.livefire-rag.evidence-retrieval-policy", "1.0.0", RETRIEVAL_POLICY
)


def _index_format_descriptor() -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": "livefire.index-format-descriptor/1",
        "format": {"id": INDEX_FORMAT_ID, "version": INDEX_FORMAT_VERSION, "sha256": ""},
        "compatibility": {
            "rule": "exact_format_id_and_listed_version",
            "accepted_versions": [INDEX_FORMAT_VERSION],
        },
        "objects": [
            {"role": "documents", "required": True, "media_type": "application/vnd.apache.parquet", "row_schema": DOCUMENT_SCHEMA_REF},
            {"role": "occurrences", "required": True, "media_type": "application/vnd.apache.parquet", "row_schema": OCCURRENCE_SCHEMA_REF},
            {"role": "embeddings", "required": True, "media_type": "application/vnd.apache.parquet", "row_schema": EMBEDDING_SCHEMA_REF},
            {"role": "derivation_documents", "required": False, "media_type": "application/vnd.apache.parquet", "row_schema": DERIVED_DOCUMENT_SCHEMA_REF},
            {"role": "derivation_memberships", "required": False, "media_type": "application/vnd.apache.parquet", "row_schema": DERIVATION_MEMBERSHIP_SCHEMA_REF},
            {"role": "coverage_report", "required": True, "media_type": "application/json", "row_schema": COVERAGE_SCHEMA_REF},
        ],
        "pointer_table": {"required": True, "schema": OCCURRENCE_SCHEMA_REF},
        "physical_profile": PHYSICAL_PROFILE_REF,
        "validator": VALIDATOR_REF,
    }
    descriptor["format"]["sha256"] = canonical_sha256_omitting(
        descriptor, ("format", "sha256")
    )
    return descriptor


INDEX_FORMAT_DESCRIPTOR = _index_format_descriptor()
INDEX_FORMAT_REF = INDEX_FORMAT_DESCRIPTOR["format"]


def _tool_descriptor() -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": "livefire.tool-descriptor/1",
        "tool": {"id": TOOL_ID, "version": TOOL_VERSION, "sha256": ""},
        "name": "evidence.search",
        "description": (
            "Return ranked immutable source-record candidates from an admitted generic "
            "evidence index; candidates require authoritative hydration and verification."
        ),
        "input_schema": INPUT_SCHEMA_REF,
        "output_schema": OUTPUT_SCHEMA_REF,
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


TOOL_DESCRIPTOR = _tool_descriptor()
TOOL_REF = TOOL_DESCRIPTOR["tool"]


def provider_object_lock(source_dir: Path | None = None) -> dict[str, Any]:
    source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    vendor_root = _runtime_vendor_root(source_dir)
    wrapper = PROVIDER_WRAPPER_TEXT.encode("utf-8")
    objects = [
        {
            "path": "bin/livefire-rag-evidence-provider",
            "media_type": "text/x-python",
            "sha256": sha256_bytes(wrapper),
            "bytes": len(wrapper),
        }
    ]
    for name in RUNTIME_MODULES:
        data = (source_dir / name).read_bytes()
        objects.append(
            {
                "path": f"lib/livefire_rag/{name}",
                "media_type": "text/x-python",
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
    for relative in VENDORED_RUNTIME_FILES:
        data = (vendor_root / relative).read_bytes()
        objects.append(
            {
                "path": f"lib/{relative}",
                "media_type": (
                    "text/plain"
                    if relative.endswith(("LICENSE", "METADATA", "py.typed"))
                    else "text/x-python"
                ),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
    objects.sort(key=lambda item: (item["path"], item["sha256"]))
    return {"schema_version": "livefire.object-lock/1", "objects": objects}


PROVIDER_OBJECT_LOCK = provider_object_lock()
PROVIDER_REF = component_ref(PROVIDER_ID, PROVIDER_VERSION, PROVIDER_OBJECT_LOCK)


def _inventory(
    component: Mapping[str, Any],
    kind: str,
    root: Path,
    relative: str,
    media_type: str,
) -> dict[str, Any]:
    return {
        "component": dict(component),
        "kind": kind,
        "target": TARGET,
        "artifact": artifact_ref(root / relative, relative, media_type),
    }


def package_evidence_provider_bundle(
    repository_root: Path,
    out_dir: Path,
    sdk_specs: Path,
) -> dict[str, Any]:
    """Package provider code/contracts only; indexes and bindings stay external."""

    repository_root = Path(repository_root).resolve()
    out_dir = Path(out_dir).resolve()
    sdk_specs = Path(sdk_specs).resolve()
    schema_lock = json.loads((sdk_specs / "schema-set.lock.json").read_text(encoding="utf-8"))
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite bundle path: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        for relative in (
            "bin", "lib/livefire_rag", "lib/livefire_rag/evidence_specs/sdk",
            "lib/rfc8785", f"lib/rfc8785-{RFC8785_VERSION}.dist-info",
            "descriptors", "profiles",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)
        wrapper = staging / "bin/livefire-rag-evidence-provider"
        wrapper.write_text(PROVIDER_WRAPPER_TEXT, encoding="utf-8")
        wrapper.chmod(0o755)
        wrapper_artifact = artifact_ref(
            wrapper, "bin/livefire-rag-evidence-provider", "text/x-python"
        )
        launcher_ref = {
            "id": "com.ayc.livefire-rag.evidence-provider-launcher",
            "version": PROVIDER_VERSION,
            "sha256": sha256_bytes(wrapper.read_bytes()),
        }
        inventory = [
            {
                "component": launcher_ref,
                "kind": "tool_provider",
                "target": TARGET,
                "artifact": wrapper_artifact,
            }
        ]

        runtime_source = repository_root / "src/livefire_rag"
        for name in RUNTIME_MODULES:
            relative = f"lib/livefire_rag/{name}"
            shutil.copyfile(runtime_source / name, staging / relative)
            data = (staging / relative).read_bytes()
            inventory.append(
                _inventory(
                    {
                        "id": f"{PROVIDER_ID}.source.{Path(name).stem}",
                        "version": PROVIDER_VERSION,
                        "sha256": sha256_bytes(data),
                    },
                    "tool_provider",
                    staging,
                    relative,
                    "text/x-python",
                )
            )
        vendor_root = _runtime_vendor_root(runtime_source)
        for relative in VENDORED_RUNTIME_FILES:
            bundled_relative = f"lib/{relative}"
            shutil.copyfile(vendor_root / relative, staging / bundled_relative)
            data = (staging / bundled_relative).read_bytes()
            inventory.append(
                _inventory(
                    {
                        "id": f"org.pypi.rfc8785.runtime.{relative.replace('/', '.')}",
                        "version": RFC8785_VERSION,
                        "sha256": sha256_bytes(data),
                    },
                    "tool_provider" if relative.endswith(".py") else "sbom",
                    staging,
                    bundled_relative,
                    (
                        "text/x-python"
                        if relative.endswith(".py")
                        else "text/plain"
                    ),
                )
            )
        staged_provider_lock = provider_object_lock(staging / "lib/livefire_rag")
        if staged_provider_lock != PROVIDER_OBJECT_LOCK:
            raise ValueError("provider source changed while packaging")
        write_canonical_json(staging / "provider.objects.lock.json", staged_provider_lock)
        inventory.append(
            _inventory(
                PROVIDER_REF,
                "tool_provider",
                staging,
                "provider.objects.lock.json",
                "application/vnd.livefire.object-lock+json",
            )
        )

        schema_artifacts: dict[str, dict[str, Any]] = {}
        rag_specs = generic_schema_root(repository_root / "specs")
        bundle_schema_names = tuple(
            dict.fromkeys(
                (
                    *GENERIC_EVIDENCE_SCHEMA_NAMES,
                    "evidence-derived-document.v1.schema.json",
                    "evidence-derivation-membership-row.v1.schema.json",
                )
            )
        )
        for name in bundle_schema_names:
            relative = f"lib/livefire_rag/evidence_specs/{name}"
            shutil.copyfile(generic_schema_path(rag_specs, name), staging / relative)
            schema = json.loads((staging / relative).read_text(encoding="utf-8"))
            reference = component_ref(schema["$id"], "1", schema)
            inventory.append(_inventory(reference, "schema", staging, relative, "application/schema+json"))
            schema_artifacts[name] = inventory[-1]["artifact"]
        # The provider validates the mounted SDK binding lock and local/test
        # admission receipt against a fully offline registry. Inventory the
        # complete selected SDK schema set so transitive references never use
        # network retrieval or an adjacent repository checkout.
        sdk_schema_names = tuple(path.name for path in sorted(sdk_specs.glob("*.schema.json")))
        for name in sdk_schema_names:
            relative = f"lib/livefire_rag/evidence_specs/sdk/{name}"
            shutil.copyfile(sdk_specs / name, staging / relative)
            schema = json.loads((staging / relative).read_text(encoding="utf-8"))
            inventory.append(
                _inventory(component_ref(schema["$id"], "1", schema), "schema", staging, relative, "application/schema+json")
            )

        descriptor_relative = "descriptors/evidence-search.json"
        write_canonical_json(staging / descriptor_relative, TOOL_DESCRIPTOR)
        descriptor_artifact = artifact_ref(
            staging / descriptor_relative, descriptor_relative, "application/json"
        )
        inventory.append(
            {
                "component": TOOL_REF,
                "kind": "tool_descriptor",
                "target": TARGET,
                "artifact": descriptor_artifact,
            }
        )
        format_relative = "descriptors/evidence-index-format.json"
        write_canonical_json(staging / format_relative, INDEX_FORMAT_DESCRIPTOR)
        inventory.append(
            _inventory(
                INDEX_FORMAT_REF,
                "index_format",
                staging,
                format_relative,
                "application/json",
            )
        )
        for name, material, reference in (
            ("retrieval-policy.json", RETRIEVAL_POLICY, RETRIEVAL_POLICY_REF),
            ("physical-profile.json", PHYSICAL_PROFILE, PHYSICAL_PROFILE_REF),
            ("validator-profile.json", VALIDATOR_PROFILE, VALIDATOR_REF),
        ):
            relative = f"profiles/{name}"
            (staging / relative).write_bytes(canonical_json_bytes(material))
            inventory.append(_inventory(reference, "sbom", staging, relative, "application/json"))

        manifest: dict[str, Any] = {
            "schema_version": "livefire.plugin/1",
            "plugin": {"id": "com.ayc.livefire-rag.evidence", "version": "0.1.0", "sha256": ""},
            "sdk_compatibility": {
                "tool_protocol": PROTOCOL,
                "schema_set_sha256": schema_lock["schema_set_sha256"],
            },
            "artifacts": inventory,
            "entrypoints": {"provider": {"component": PROVIDER_REF, "executable": wrapper_artifact}},
            "tools": [
                {
                    "descriptor": TOOL_REF,
                    "descriptor_artifact": descriptor_artifact,
                    "name": TOOL_DESCRIPTOR["name"],
                    "description": TOOL_DESCRIPTOR["description"],
                    "input_schema": INPUT_SCHEMA_REF,
                    "output_schema": OUTPUT_SCHEMA_REF,
                    "effects": ["index.read", "embedding.loopback"],
                    "required_indexes": [INDEX_FORMAT_ID],
                }
            ],
            "permissions": {
                "tool_provider": {
                    "network": ["loopback:lmstudio"],
                    "secret_handles": [],
                    "source_mount": "none",
                    "index_mount": "read_only",
                    "staging_mount": "none",
                    "scratch_bytes": 268435456,
                }
            },
        }
        manifest["plugin"]["sha256"] = canonical_sha256_omitting(
            manifest, ("plugin", "sha256")
        )
        write_canonical_json(staging / "plugin.json", manifest)
        os.replace(staging, out_dir)
        return {
            "bundle": str(out_dir),
            "manifest": str(out_dir / "plugin.json"),
            "provider": str(out_dir / "bin/livefire-rag-evidence-provider"),
            "plugin": manifest["plugin"],
            "tool": TOOL_REF,
            "index_format": INDEX_FORMAT_REF,
            "artifacts": len(inventory),
            "admission_status": "local_test_bundle_not_production_admitted",
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "INDEX_FORMAT_DESCRIPTOR",
    "INDEX_FORMAT_REF",
    "DERIVED_DOCUMENT_SCHEMA_REF",
    "DERIVATION_MEMBERSHIP_SCHEMA_REF",
    "INPUT_SCHEMA_REF",
    "OUTPUT_SCHEMA_REF",
    "PROTOCOL",
    "PROVIDER_EXECUTABLE_ARTIFACT",
    "PROVIDER_REF",
    "RETRIEVAL_POLICY_REF",
    "TOOL_DESCRIPTOR",
    "TOOL_REF",
    "package_evidence_provider_bundle",
]
