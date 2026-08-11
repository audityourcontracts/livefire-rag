"""Build a content-closed POC plugin bundle for the standalone SDK."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
    write_canonical_json,
)
from .contracts import (
    PROVIDER_OBJECT_LOCK,
    PROVIDER_REF,
    PROVIDER_WRAPPER_TEXT,
    SEARCH_DESCRIPTION,
    SEARCH_INPUT_SCHEMA_REF,
    SEARCH_TOOL_DESCRIPTOR,
    SEARCH_TOOL_REF,
    SEMANTIC_RESULT_SCHEMA_REF,
    SIMILAR_DESCRIPTION,
    SIMILAR_INPUT_SCHEMA_REF,
    SIMILAR_TOOL_DESCRIPTOR,
    SIMILAR_TOOL_REF,
    development_binding,
    development_binding_object_lock,
    development_binding_ref,
)
from .index import SemanticIndex


TARGET = "python3-any"


def _schema_component(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return component_ref(value["$id"], "1", value)


def _inventory(
    component: dict[str, str],
    kind: str,
    root: Path,
    relative: str,
    media_type: str,
) -> dict[str, Any]:
    return {
        "component": component,
        "kind": kind,
        "target": TARGET,
        "artifact": artifact_ref(root / relative, relative, media_type),
    }


def _request(request_id: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "livefire.tool/1",
        "id": request_id,
        "method": method,
        "params": params,
        "context": {"trace_id": f"poc-{request_id}", "deadline_unix_ms": 4_102_444_800_000},
    }


def _write_transcript(path: Path, requests: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for request in requests:
            handle.write(canonical_json_bytes(request, newline=True))


def package_bundle(
    repository_root: Path,
    out_dir: Path,
    index_dir: Path,
    sdk_specs: Path,
) -> dict[str, Any]:
    index = SemanticIndex.open(index_dir)
    schema_lock = json.loads((sdk_specs / "schema-set.lock.json").read_text(encoding="utf-8"))
    schema_set_sha256 = schema_lock["schema_set_sha256"]
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite bundle path: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        for relative in (
            "bin",
            "lib/livefire_rag",
            "schemas",
            "schemas/sdk",
            "descriptors",
            "bindings",
            "requests",
        ):
            (staging / relative).mkdir(parents=True, exist_ok=True)

        wrapper = staging / "bin/livefire-rag-provider"
        wrapper.write_text(PROVIDER_WRAPPER_TEXT, encoding="utf-8")
        wrapper.chmod(0o755)
        launcher_ref = {
            "id": "com.ayc.livefire-rag.provider-launcher",
            "version": "0.1.0",
            "sha256": sha256_bytes(wrapper.read_bytes()),
        }

        inventory = [
            _inventory(
                launcher_ref,
                "tool_provider",
                staging,
                "bin/livefire-rag-provider",
                "text/x-python",
            )
        ]
        copied_runtime_objects = {"bin/livefire-rag-provider": wrapper.read_bytes()}
        for source in sorted((repository_root / "src/livefire_rag").glob("*.py")):
            relative = f"lib/livefire_rag/{source.name}"
            shutil.copyfile(source, staging / relative)
            data = (staging / relative).read_bytes()
            copied_runtime_objects[relative] = data
            source_component = {
                "id": f"com.ayc.livefire-rag.provider-source.{source.stem}",
                "version": "0.1.0",
                "sha256": sha256_bytes(data),
            }
            inventory.append(
                _inventory(source_component, "tool_provider", staging, relative, "text/x-python")
            )
        for entry in PROVIDER_OBJECT_LOCK["objects"]:
            data = copied_runtime_objects.get(entry["path"])
            if data is None or len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
                raise ValueError(f"provider object-lock mismatch for {entry['path']}")
        provider_lock_relative = "provider.objects.lock.json"
        write_canonical_json(staging / provider_lock_relative, PROVIDER_OBJECT_LOCK)
        inventory.append(
            _inventory(
                PROVIDER_REF,
                "tool_provider",
                staging,
                provider_lock_relative,
                "application/vnd.livefire.object-lock+json",
            )
        )

        rag_schema_names = [
            "cli-common.v1.schema.json",
            "cli-search.input.v1.schema.json",
            "cli-similar.input.v1.schema.json",
            "semantic-result.v1.schema.json",
            "development-binding-lock.v1.schema.json",
        ]
        schemas: dict[str, dict[str, str]] = {}
        for name in rag_schema_names:
            source = repository_root / "specs" / name
            relative = f"schemas/{name}"
            shutil.copyfile(source, staging / relative)
            schemas[name] = _schema_component(staging / relative)
            inventory.append(
                _inventory(schemas[name], "schema", staging, relative, "application/schema+json")
            )
        if schemas["cli-search.input.v1.schema.json"] != SEARCH_INPUT_SCHEMA_REF:
            raise ValueError("search input schema identity drifted from the provider")
        if schemas["cli-similar.input.v1.schema.json"] != SIMILAR_INPUT_SCHEMA_REF:
            raise ValueError("similar input schema identity drifted from the provider")
        if schemas["semantic-result.v1.schema.json"] != SEMANTIC_RESULT_SCHEMA_REF:
            raise ValueError("result schema identity drifted from the provider")

        for name in ("component-ref.v1.schema.json", "source-record-pointer.v1.schema.json"):
            relative = f"schemas/sdk/{name}"
            shutil.copyfile(sdk_specs / name, staging / relative)
            component = _schema_component(staging / relative)
            inventory.append(
                _inventory(component, "schema", staging, relative, "application/schema+json")
            )

        descriptors = [
            (
                SEARCH_TOOL_REF,
                SEARCH_TOOL_DESCRIPTOR,
                "descriptors/cli-search.json",
                ["index.read", "embedding.loopback"],
                SEARCH_DESCRIPTION,
            ),
            (
                SIMILAR_TOOL_REF,
                SIMILAR_TOOL_DESCRIPTOR,
                "descriptors/cli-similar.json",
                ["index.read"],
                SIMILAR_DESCRIPTION,
            ),
        ]
        descriptor_artifacts: dict[str, dict[str, Any]] = {}
        for tool_ref, descriptor, relative, _, _ in descriptors:
            write_canonical_json(staging / relative, descriptor)
            if canonical_sha256_omitting(descriptor, ("tool", "sha256")) != tool_ref["sha256"]:
                raise ValueError(f"tool descriptor identity mismatch: {descriptor['name']}")
            inventory.append(
                _inventory(tool_ref, "tool_descriptor", staging, relative, "application/json")
            )
            descriptor_artifacts[tool_ref["id"]] = inventory[-1]["artifact"]

        binding = development_binding(index.manifest)
        binding_relative = "bindings/development-binding-lock.json"
        write_canonical_json(staging / binding_relative, binding)
        binding_artifact = artifact_ref(
            staging / binding_relative,
            binding_relative,
            "application/vnd.livefire.rag.development-binding-lock+json",
        )
        binding_object_lock = development_binding_object_lock(binding)
        if binding_object_lock["objects"] != [binding_artifact]:
            raise ValueError("development binding object-lock drifted from its artifact")
        binding_lock_relative = "bindings/development-binding.objects.lock.json"
        write_canonical_json(staging / binding_lock_relative, binding_object_lock)
        binding_ref = development_binding_ref(binding)
        # The SDK POC manifest has no generic policy kind. This is a development
        # component bill of materials and is carried as `sbom`, never as admission.
        inventory.append(
            _inventory(
                binding_ref,
                "sbom",
                staging,
                binding_lock_relative,
                "application/vnd.livefire.object-lock+json",
            )
        )

        provider_lock_artifact = next(
            item["artifact"] for item in inventory if item["component"] == PROVIDER_REF
        )
        wrapper_artifact = inventory[0]["artifact"]
        manifest: dict[str, Any] = {
            "schema_version": "livefire.plugin/1",
            "plugin": {
                "id": "com.ayc.livefire-rag",
                "version": "0.1.0",
                "sha256": "",
            },
            "sdk_compatibility": {
                "tool_protocol": "livefire.tool/1",
                "schema_set_sha256": schema_set_sha256,
            },
            "artifacts": inventory,
            "entrypoints": {
                "provider": {"component": PROVIDER_REF, "executable": wrapper_artifact}
            },
            "tools": [],
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
        for tool_ref, descriptor, _, effects, _ in descriptors:
            manifest["tools"].append(
                {
                    "descriptor": tool_ref,
                    "descriptor_artifact": descriptor_artifacts[tool_ref["id"]],
                    "name": descriptor["name"],
                    "description": descriptor["description"],
                    "input_schema": descriptor["input_schema"],
                    "output_schema": descriptor["output_schema"],
                    "effects": effects,
                    "required_indexes": ["semantic-command-v1"],
                }
            )
        manifest["plugin"]["sha256"] = canonical_sha256_omitting(
            manifest, ("plugin", "sha256")
        )
        write_canonical_json(staging / "plugin.json", manifest)

        open_params = {
            "provider": PROVIDER_REF,
            "tools": [SEARCH_TOOL_REF, SIMILAR_TOOL_REF],
            "indexes": [index.manifest["component"]],
            "source_snapshots": index.manifest["source_snapshots"],
            "binding_lock_sha256": binding_ref["sha256"],
            "query_time_contract": {},
            "limits": binding["limits"],
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
        requests = [
            _request("1", "handshake", {}),
            _request("2", "open", open_params),
            _request(
                "3",
                "call",
                {
                    "session_id": "${session_id}",
                    "tool": SIMILAR_TOOL_REF,
                    "arguments": {
                        "schema_version": "livefire.rag.cli-similar.input/1",
                        "command_id": index.documents[0]["command_id"],
                        "top_n": 2,
                    },
                },
            ),
            _request("4", "health", {"session_id": "${session_id}"}),
            _request("5", "close", {"session_id": "${session_id}"}),
        ]
        _write_transcript(staging / "requests/provider-similar.requests.jsonl", requests)
        os.replace(staging, out_dir)
        return {
            "bundle": str(out_dir.resolve()),
            "manifest": str((out_dir / "plugin.json").resolve()),
            "provider": str((out_dir / "bin/livefire-rag-provider").resolve()),
            "provider_object_lock": provider_lock_artifact,
            "development_binding": {
                "component": binding_ref,
                "artifact": next(
                    item["artifact"] for item in inventory if item["component"] == binding_ref
                ),
                "binding_artifact": binding_artifact,
                "status": binding["status"],
            },
            "requests": str((out_dir / "requests/provider-similar.requests.jsonl").resolve()),
            "plugin": manifest["plugin"],
            "artifacts": len(inventory),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
