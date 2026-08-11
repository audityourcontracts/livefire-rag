"""Canonical JSON and digest helpers used by immutable index artifacts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import rfc8785


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    encoded = rfc8785.dumps(value)
    return encoded + (b"\n" if newline else b"")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path, relative_path: str, media_type: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "media_type": media_type,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def component_ref(component_id: str, version: str, material: Any) -> dict[str, str]:
    return {
        "id": component_id,
        "version": version,
        "sha256": sha256_bytes(canonical_json_bytes(material)),
    }


def canonical_sha256_omitting(value: Any, path: tuple[str, ...]) -> str:
    if not path:
        raise ValueError("cannot omit the document root")
    material = deepcopy(value)
    parent = material
    for part in path[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise ValueError(f"identity omission path does not exist: {'/'.join(path)}")
        parent = parent[part]
    if not isinstance(parent, dict) or path[-1] not in parent:
        raise ValueError(f"identity omission path does not exist: {'/'.join(path)}")
    del parent[path[-1]]
    return sha256_bytes(canonical_json_bytes(material))


def write_canonical_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value, newline=True))
