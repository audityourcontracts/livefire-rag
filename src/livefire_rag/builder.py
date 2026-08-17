"""Historical fixture and prototype promotion builders for tests only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, sha256_bytes
from .index import build_index


def build_fixture(fixture_path: Path, out_dir: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "livefire.rag.semantic-index-fixture/1":
        raise ValueError("unsupported fixture schema")
    documents = []
    vectors = []
    for row in fixture["rows"]:
        documents.append(row["document"])
        vectors.append(row["vector"])
    return build_index(
        out_dir,
        documents,
        np.asarray(vectors, dtype=np.float32),
        index_id=fixture["index"]["id"],
        version=fixture["index"]["version"],
        embedding_profile=fixture["embedding_profile"],
        source_snapshots=fixture["source_snapshots"],
        limitations=fixture.get("limitations", ["deterministic fixture index; not production evidence"]),
    )


def _load_prototype_module(repository_root: Path):
    script = repository_root / "tools/prototype_query_demo.py"
    spec = importlib.util.spec_from_file_location("livefire_rag_prototype_query_demo", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prototype query module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prototype_shell(document: Any) -> tuple[str, str]:
    if document.kind == "source_powershell_script_block":
        return "powershell_script_block", "powershell"
    if document.kind == "source_bash_history":
        return "process_command_line", "posix_shell"
    if document.kind == "ocsf_api_activity":
        return "cloud_api_action", "cloud_cli"
    lower = document.projection.lower()
    if "powershell" in lower:
        return "process_command_line", "powershell"
    if "cmd.exe" in lower or "executable: cmd" in lower:
        return "process_command_line", "cmd"
    return "process_command_line", "direct_exec"


def promote_prototype(repository_root: Path, prototype_dir: Path, out_dir: Path) -> dict[str, Any]:
    module = _load_prototype_module(repository_root)
    con = module.duckdb.connect()
    corpus, _, _ = module.build_corpus(con)
    con.close()
    cache = json.loads((prototype_dir / "corpus-cache.json").read_text(encoding="utf-8"))
    if module.corpus_digest(corpus) != cache.get("corpus_digest"):
        raise ValueError("rebuilt prototype corpus digest does not match the vector cache")
    profile = json.loads(module.PROFILE_PATH.read_text(encoding="utf-8"))
    profile_digest = module.sha256_json(profile)
    if profile_digest != cache.get("profile_digest"):
        raise ValueError("embedding profile digest does not match the vector cache")
    vectors = np.load(prototype_dir / "corpus-vectors.npy", allow_pickle=False)
    if list(vectors.shape) != cache.get("shape") or vectors.shape != (len(corpus), profile["dimensions"]):
        raise ValueError("prototype vector shape does not match rebuilt corpus")

    m21_snapshot = {
        "id": "livefire.ocsf.botsv3-m21-v1",
        "version": "m21-v1",
        "sha256": "1fda84fcd24790f67ca19c574628a9ab416fa5a6e55d4cab7fb9a1b62dbcbdd0",
    }
    openbots_snapshot = {
        "id": "openbots.v3.bronze",
        "version": "v3",
        "sha256": "61d85e27d31555263b1603fbe8f2a6bf9ee60df6cc5e65667aa489552d1c74d7",
    }
    profile_ref = {
        "id": "livefire.rag.prototype-source-profile",
        "version": "1",
        "sha256": sha256_bytes(canonical_json_bytes({"kind": "prototype-source-profile", "version": 1})),
    }

    # M21 process rows in the exploratory corpus omitted event time from metadata.
    process_connection = module.duckdb.connect()
    try:
        process_times = dict(process_connection.execute(
            "SELECT event_id, json_extract(typed_event_json, '$.ocsf.time')::BIGINT FROM read_parquet(?) ORDER BY event_id",
            [str(module.PROCESS_PARQUET)],
        ).fetchall())
    finally:
        process_connection.close()
    documents = []
    for document in corpus:
        snapshot = m21_snapshot if document.kind.startswith("ocsf_") else openbots_snapshot
        raw_time = document.metadata.get("event_time") or document.metadata.get("event_time_ms")
        if raw_time is None and document.kind == "ocsf_process_command":
            raw_time = process_times.get(document.metadata.get("event_id"))
        if raw_time is None:
            raise ValueError(f"prototype document lacks event time: {document.document_id}")
        if isinstance(raw_time, (int, float)) or str(raw_time).isdigit():
            from datetime import datetime, timezone
            event_time = datetime.fromtimestamp(int(raw_time) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            event_time = str(raw_time).replace(" ", "T")
            if event_time.endswith("+00"):
                event_time += ":00"
        observation_kind, shell = _prototype_shell(document)
        record_sha = hashlib.sha256(document.projection.encode("utf-8")).hexdigest()
        pointer = {
            "schema_version": "livefire.source-record-pointer/1",
            "snapshot": snapshot,
            "snapshot_profile": profile_ref,
            "record_id": document.document_id,
            "record_sha256": record_sha,
            "locator": {"kind": "record_id_only"},
            "support_refs": [document.locator, *document.aliases],
        }
        projected = {
            "schema_version": "livefire.rag.semantic-document/1",
            "command_id": document.document_id,
            "event_time": event_time,
            "observation_kind": observation_kind,
            "shell_family": shell,
            "semantic_text": document.projection,
            "preview": document.preview,
            "source_pointer": pointer,
            "source_kind": document.kind,
            "occurrences": document.occurrences,
            "limitations": ["prototype semantic deduplication retains representative metadata only"],
        }
        host = document.metadata.get("host")
        if host:
            projected["host_id"] = str(host)
        principal = document.metadata.get("principal") or document.metadata.get("actor")
        if principal:
            projected["principal_key"] = {"namespace": "prototype", "id": str(principal)}
        documents.append(projected)
    return build_index(
        out_dir,
        documents,
        vectors,
        index_id="com.ayc.livefire-rag.prototype-m21-openbots",
        version="development-1",
        embedding_profile=profile,
        source_snapshots=[m21_snapshot, openbots_snapshot],
        limitations=[
            "development_only and not admitted",
            "mixes M21 normalized rows with direct OpenBOTS authority reads",
            "semantic deduplication loses occurrence-complete filter semantics",
            "record_id_only pointers are prototype provenance hints, not admitted hydration pointers",
        ],
    )
