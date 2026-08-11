#!/usr/bin/env python3
"""Run an ad-hoc semantic retrieval demo over real BOTS v3 / M21 data.

This is deliberately not the production adapter, immutable index builder, or tool
provider. It exists to expose useful query behaviour and failure modes before
those components are implemented.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPENBOTS_GLOB = "/Users/michael/work/ayc/open-bots/v3/output/parquet/events/**/*.parquet"
M21_ROOT = Path("/Users/michael/work/ayc/livefire/outputs/ocsf/botsv3-m21-v1")
PROCESS_PARQUET = M21_ROOT / "semantic/ocsf_process_activity.parquet"
API_PARQUET = M21_ROOT / "semantic/ocsf_api_activity.parquet"
PROFILE_PATH = ROOT / "profiles/qwen3-embedding-8b-lmstudio-q4.dev.json"

PS_ENCODED_RE = re.compile(
    r"(?i)(?:-|/)(?:e|en|enc|enco|encod|encode|encodedcommand)\s+([A-Za-z0-9+/=]{20,})"
)
ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PASSWORD_BEFORE_ADD_RE = re.compile(
    r"(?i)(\buser\s+\S+\s+)(\S+)(\s+/add\b)"
)
PASSWORD_AFTER_ADD_RE = re.compile(
    r"(?i)(\buser\s+/add\s+\S+\s+)(\S+)"
)


@dataclass
class Document:
    kind: str
    projection: str
    preview: str
    locator: str
    metadata: dict[str, Any] = field(default_factory=dict)
    occurrences: int = 1
    aliases: list[str] = field(default_factory=list)
    document_id: str = ""

    def finish(self) -> "Document":
        material = f"{self.kind}\n{self.projection}".encode()
        self.document_id = "proto_" + hashlib.sha256(material).hexdigest()[:20]
        self.projection = self.projection[:24000]
        self.preview = one_line(redact(self.preview))[:700]
        return self


def one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def redact(value: str) -> str:
    value = ACCESS_KEY_RE.sub("<AWS_ACCESS_KEY_REDACTED>", value)
    value = PASSWORD_BEFORE_ADD_RE.sub(r"\1<PASSWORD_REDACTED>\3", value)
    return PASSWORD_AFTER_ADD_RE.sub(r"\1<PASSWORD_REDACTED>", value)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_identity(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_at_start": dirty}


def json_path(obj: Any, *parts: str, default: Any = None) -> Any:
    cur = obj
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_sysmon(raw: str) -> dict[str, str]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}
    values: dict[str, str] = {}
    for elem in root.iter():
        name = elem.attrib.get("Name")
        if name and elem.text:
            values[name] = elem.text
    return values


def printable_ratio(value: str) -> float:
    if not value:
        return 0.0
    return sum(ch.isprintable() or ch in "\r\n\t" for ch in value) / len(value)


def decode_powershell(command: str) -> str | None:
    match = PS_ENCODED_RE.search(command)
    if not match:
        return None
    token = match.group(1)
    token += "=" * (-len(token) % 4)
    try:
        payload = base64.b64decode(token, validate=False)
    except Exception:
        return None
    candidates = []
    for encoding in ("utf-16le", "utf-8", "latin-1"):
        try:
            decoded = payload.decode(encoding)
            candidates.append((printable_ratio(decoded), decoded))
        except UnicodeDecodeError:
            pass
    if not candidates:
        return None
    ratio, decoded = max(candidates, key=lambda item: item[0])
    return decoded[:16000] if ratio >= 0.70 else None


def process_projection(
    command: str,
    image: str | None,
    parent: str | None,
    user: str | None,
    host: str | None,
) -> tuple[str, str]:
    decoded = decode_powershell(command)
    fields = [
        "kind: process command line",
        f"executable: {image or ''}",
        f"parent executable: {parent or ''}",
        f"command: {command}",
    ]
    if decoded:
        fields.extend(
            [
                "static analysis: PowerShell encoded command decoded successfully",
                f"decoded PowerShell intent and script: {decoded}",
            ]
        )
    # Principal/host remain filter and evidence fields, not embedding inputs.
    context = f" host={host or 'unknown'} user={user or 'unknown'}"
    return redact("\n".join(fields)), redact(command + context)


def extract_ps_script(raw: str) -> str | None:
    marker = "Message=Creating Scriptblock text"
    pos = raw.find(marker)
    if pos < 0:
        return None
    start = raw.find("\n", pos)
    if start < 0:
        return None
    tail = raw[start + 1 :]
    for delimiter in ("\n\nScriptBlock ID:", "\r\n\r\nScriptBlock ID:"):
        end = tail.find(delimiter)
        if end >= 0:
            tail = tail[:end]
            break
    return tail.strip() or None


def grant_summary(grants: list[Any]) -> str:
    result = []
    for grant in grants:
        if not isinstance(grant, dict):
            continue
        grantee = str(grant.get("grantee") or "")
        if "AllUsers" in grantee:
            grantee = "global AllUsers (public)"
        elif "LogDelivery" in grantee:
            grantee = "S3 LogDelivery service"
        elif grantee:
            grantee = "canonical owner"
        result.append(f"{grantee}: {grant.get('permission')}")
    return ", ".join(result)


def allowlisted_request(unmapped: dict[str, Any]) -> dict[str, Any]:
    request = unmapped.get("requestParameters")
    if not isinstance(request, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("bucketName", "instanceType", "maxCount", "minCount", "userName"):
        if key in request:
            out[key] = request[key]
    instances = json_path(request, "instancesSet", "items", default=[])
    if isinstance(instances, list) and instances:
        item = instances[0]
        if isinstance(item, dict):
            out["instances"] = {
                key: item.get(key) for key in ("imageId", "minCount", "maxCount") if key in item
            }
    return out


def api_projection(event: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    ocsf = event.get("ocsf") or {}
    unmapped = ocsf.get("unmapped") or {}
    resources = event.get("resources") or []
    resource_names = []
    for item in resources:
        if isinstance(item, dict):
            resource_names.append(str(item.get("name") or item.get("uid") or item.get("data") or ""))
        else:
            resource_names.append(str(item))
    resource = event.get("resource") or json_path(event, "databucket", "name")
    if resource:
        resource_names.append(str(resource))
    request = allowlisted_request(unmapped)
    grants = grant_summary(event.get("authorization_grants") or [])
    service = str(event.get("service") or "")
    operation = str(event.get("operation") or "")
    status = str(event.get("status") or "success")
    region = str(unmapped.get("awsRegion") or "")
    projection = "\n".join(
        [
            "kind: cloud API activity",
            f"service: {service}",
            f"operation and action: {operation}",
            f"resources and targets: {', '.join(filter(None, resource_names))}",
            f"request fields: {json.dumps(request, sort_keys=True)}",
            f"authorization and access grants: {grants or 'none materialized'}",
            f"result status: {status}",
            f"region: {region}",
        ]
    )
    actor = event.get("actor_name") or event.get("actor") or "unknown"
    preview = (
        f"{service} {operation} target={','.join(filter(None, resource_names)) or '-'} "
        f"status={status} region={region or '-'} actor={actor} grants={grants or '-'}"
    )
    metadata = {
        "event_time_ms": json_path(ocsf, "time"),
        "actor": actor,
        "service": service,
        "operation": operation,
        "status": status,
        "region": region,
        "resources": list(filter(None, resource_names)),
    }
    return redact(projection), redact(preview), metadata


def add_document(store: dict[str, Document], doc: Document) -> None:
    doc.finish()
    key = hashlib.sha256(f"{doc.kind}\n{doc.projection}".encode()).hexdigest()
    if key not in store:
        store[key] = doc
        return
    prior = store[key]
    prior.occurrences += doc.occurrences
    if doc.locator != prior.locator and len(prior.aliases) < 8:
        prior.aliases.append(doc.locator)


def build_corpus(con: duckdb.DuckDBPyConnection) -> tuple[list[Document], dict[str, Any], list[dict[str, Any]]]:
    docs: dict[str, Document] = {}
    source_counts: Counter[str] = Counter()

    # M21's osquery process projection retains a useful cmdline in mapped columns.
    process_rows = con.execute(
        """
        SELECT event_id, support_ref, typed_event_json,
               json_extract_string(typed_event_json, '$.ocsf.unmapped.columns.cmdline') AS cmdline
        FROM read_parquet(?)
        WHERE cmdline IS NOT NULL AND cmdline <> ''
        ORDER BY event_id
        """,
        [str(PROCESS_PARQUET)],
    ).fetchall()
    for event_id, support_ref, raw_json, command in process_rows:
        event = json.loads(raw_json)
        image = event.get("image")
        parent = event.get("parent_image")
        user = json_path(event, "ocsf", "actor", "user", "account")
        host = json_path(event, "device", "hostname") or json_path(event, "ocsf", "device", "hostname")
        projection, preview = process_projection(command, image, parent, user, host)
        add_document(
            docs,
            Document(
                kind="ocsf_process_command",
                projection=projection,
                preview=preview,
                locator=f"m21:event:{event_id}",
                metadata={
                    "event_id": event_id,
                    "support_ref": support_ref,
                    "host": host,
                    "principal": user,
                    "image": image,
                    "parent_image": parent,
                },
            ),
        )
        source_counts["ocsf_process_rows"] += 1

    # Hydrate exact Sysmon command lines from the admitted source snapshot for this demo.
    sysmon_rows = con.execute(
        """
        SELECT event_address, row_id, event_time, host, source, raw
        FROM read_parquet(?)
        WHERE sourcetype = 'XmlWinEventLog:Microsoft-Windows-Sysmon/Operational'
          AND raw LIKE '%<EventID>1</EventID>%'
          AND raw LIKE '%CommandLine%'
        ORDER BY row_id
        """,
        [OPENBOTS_GLOB],
    ).fetchall()
    for address, row_id, event_time, host, source, raw in sysmon_rows:
        values = parse_sysmon(raw)
        command = values.get("CommandLine")
        if not command:
            continue
        projection, preview = process_projection(
            command,
            values.get("Image"),
            values.get("ParentImage"),
            values.get("User"),
            host,
        )
        add_document(
            docs,
            Document(
                kind="source_sysmon_process_command",
                projection=projection,
                preview=preview,
                locator=f"openbots:event_address:{address}",
                metadata={
                    "source_row_id": address,
                    "row_id": row_id,
                    "event_time": str(event_time),
                    "host": host,
                    "principal": values.get("User"),
                    "image": values.get("Image"),
                    "parent_image": values.get("ParentImage"),
                    "source": source,
                    "decoded_powershell": decode_powershell(command) is not None,
                },
            ),
        )
        source_counts["source_sysmon_rows"] += 1

    # PowerShell 4104 contains script content that M21 does not yet promote.
    ps_rows = con.execute(
        """
        SELECT event_address, row_id, event_time, host, source, raw
        FROM read_parquet(?)
        WHERE sourcetype = 'WinEventLog:Microsoft-Windows-PowerShell/Operational'
          AND raw LIKE '%EventCode=4104%'
        ORDER BY row_id
        """,
        [OPENBOTS_GLOB],
    ).fetchall()
    for address, row_id, event_time, host, source, raw in ps_rows:
        script = extract_ps_script(raw)
        if not script:
            continue
        projection = redact(
            "kind: PowerShell script block\n"
            "static analysis: exact PowerShell script-block content\n"
            f"script and intent: {script}"
        )
        add_document(
            docs,
            Document(
                kind="source_powershell_script_block",
                projection=projection,
                preview=f"PowerShell 4104 host={host} script={script}",
                locator=f"openbots:event_address:{address}",
                metadata={
                    "source_row_id": address,
                    "row_id": row_id,
                    "event_time": str(event_time),
                    "host": host,
                    "source": source,
                },
            ),
        )
        source_counts["source_powershell_4104_rows"] += 1

    bash_rows = con.execute(
        """
        SELECT event_address, row_id, event_time, host, source, raw
        FROM read_parquet(?)
        WHERE sourcetype = 'bash_history' AND raw IS NOT NULL AND raw <> ''
        ORDER BY row_id
        """,
        [OPENBOTS_GLOB],
    ).fetchall()
    for address, row_id, event_time, host, source, command in bash_rows:
        projection = redact(f"kind: Linux shell command\nshell: bash\ncommand and intent: {command}")
        add_document(
            docs,
            Document(
                kind="source_bash_history",
                projection=projection,
                preview=f"bash host={host} source={source} command={command}",
                locator=f"openbots:event_address:{address}",
                metadata={
                    "source_row_id": address,
                    "row_id": row_id,
                    "event_time": str(event_time),
                    "host": host,
                    "source": source,
                },
            ),
        )
        source_counts["source_bash_rows"] += 1

    api_events: list[dict[str, Any]] = []
    api_rows = con.execute(
        "SELECT event_id, support_ref, typed_event_json FROM read_parquet(?) ORDER BY event_id",
        [str(API_PARQUET)],
    ).fetchall()
    for event_id, support_ref, raw_json in api_rows:
        event = json.loads(raw_json)
        event["event_id"] = event_id
        event["support_ref"] = support_ref
        api_events.append(event)
        projection, preview, metadata = api_projection(event)
        metadata.update({"event_id": event_id, "support_ref": support_ref})
        add_document(
            docs,
            Document(
                kind="ocsf_api_activity",
                projection=projection,
                preview=preview,
                locator=f"m21:event:{event_id}",
                metadata=metadata,
            ),
        )
        source_counts["ocsf_api_rows"] += 1

    corpus = sorted(docs.values(), key=lambda doc: doc.document_id)
    stats = {
        "input_rows": dict(source_counts),
        "documents_after_semantic_deduplication": len(corpus),
        "documents_by_kind": dict(Counter(doc.kind for doc in corpus)),
        "occurrences_by_kind": dict(
            Counter({kind: sum(d.occurrences for d in corpus if d.kind == kind) for kind in {d.kind for d in corpus}})
        ),
        "decoded_powershell_documents": sum(
            1 for d in corpus if d.metadata.get("decoded_powershell")
        ),
    }
    return corpus, stats, api_events


def embed_request(endpoint: str, model: str, inputs: list[str], timeout: int) -> np.ndarray:
    body = json.dumps({"model": model, "input": inputs}).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    ordered = sorted(payload["data"], key=lambda item: item["index"])
    return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)


def batches(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for offset in range(0, len(values), size):
        yield offset, values[offset : offset + size]


def corpus_digest(corpus: list[Document]) -> str:
    return sha256_json(
        [
            {
                "id": d.document_id,
                "kind": d.kind,
                "projection_sha256": hashlib.sha256(d.projection.encode()).hexdigest(),
                "occurrences": d.occurrences,
            }
            for d in corpus
        ]
    )


def load_or_embed(
    corpus: list[Document],
    profile: dict[str, Any],
    out_dir: Path,
    endpoint: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    digest = corpus_digest(corpus)
    profile_digest = sha256_json(profile)
    vectors_path = out_dir / "corpus-vectors.npy"
    cache_path = out_dir / "corpus-cache.json"
    if vectors_path.exists() and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("corpus_digest") == digest and cache.get("profile_digest") == profile_digest:
            vectors = np.load(vectors_path)
            if vectors.shape == (len(corpus), profile["dimensions"]):
                return vectors, {**cache, "cache_hit": True}

    inputs = [doc.projection for doc in corpus]
    chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for offset, batch in batches(inputs, batch_size):
        batch_started = time.perf_counter()
        chunk = embed_request(endpoint, profile["api_model_key"], batch, timeout=900)
        chunks.append(chunk)
        elapsed = time.perf_counter() - batch_started
        print(
            f"embedded {min(offset + len(batch), len(inputs))}/{len(inputs)} "
            f"documents ({elapsed:.2f}s batch)",
            flush=True,
        )
    vectors = np.concatenate(chunks, axis=0)
    total = time.perf_counter() - started
    np.save(vectors_path, vectors)
    cache = {
        "corpus_digest": digest,
        "profile_digest": profile_digest,
        "shape": list(vectors.shape),
        "embedding_seconds": total,
        "documents_per_second": len(corpus) / total,
        "cache_hit": False,
    }
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return vectors, cache


QUERIES = [
    (
        "Q1",
        "Find encoded or obfuscated PowerShell that disables script-block logging or executes a payload stored in the registry.",
    ),
    (
        "Q2",
        "What persistence created a SYSTEM scheduled task that runs hidden PowerShell from registry-stored content?",
    ),
    (
        "Q3",
        "Find a PowerShell-spawned command that creates a local service or VNC user account.",
    ),
    (
        "Q4",
        "Find PowerShell-spawned network discovery or scanning and commands that disable the Windows firewall.",
    ),
    (
        "Q5",
        "Find Linux shell commands that uploaded the Frothly web archive to an S3 bucket.",
    ),
    (
        "Q6",
        "Who attempted to launch large EC2 fleets across many AWS regions, and why did the attempts fail?",
    ),
    (
        "Q7",
        "Which identity attempted to create an IAM user or access key and was denied?",
    ),
    (
        "Q8",
        "Find actions that made the Frothly S3 bucket publicly readable or writable and later tightened access.",
    ),
    (
        "Q9",
        "Show activity that staged the Frothly HTML archive in S3 and then changed who could access the bucket.",
    ),
]


def query_text(profile: dict[str, Any], query: str) -> str:
    return profile["query_composition"].format(
        query_instruction=profile["query_instruction"], query=query
    )


def exact_search(
    corpus: list[Document], vectors: np.ndarray, query_vector: np.ndarray, top_n: int
) -> list[dict[str, Any]]:
    # The admitted design calls for float64 accumulation and round-half-even millionths.
    dots = vectors.astype(np.float64) @ query_vector.astype(np.float64)
    distances = 1.0 - dots
    ids = np.asarray([doc.document_id for doc in corpus])
    order = np.lexsort((ids, distances))[:top_n]
    results = []
    for rank, idx in enumerate(order, 1):
        doc = corpus[int(idx)]
        distance = float(distances[int(idx)])
        results.append(
            {
                "rank": rank,
                "document_id": doc.document_id,
                "kind": doc.kind,
                "cosine_distance": distance,
                "cosine_distance_millionths": int(np.rint(distance * 1_000_000)),
                "preview": doc.preview,
                "locator": doc.locator,
                "occurrences": doc.occurrences,
                "metadata": doc.metadata,
            }
        )
    return results


def api_history_examples(api_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_events = sorted(
        api_events,
        key=lambda e: (json_path(e, "ocsf", "time", default=0), e["event_id"]),
    )
    candidates = [
        e
        for e in sorted_events
        if e.get("operation") in {"CreateAccessKey", "PutBucketAcl", "RunInstances"}
    ]
    examples = []
    for event in candidates:
        operation = event.get("operation")
        # Retain one representative event per useful condition.
        if operation == "RunInstances" and event.get("status") == "success":
            continue
        event_time = json_path(event, "ocsf", "time", default=0)
        actor = event.get("actor_name") or event.get("actor") or "unknown"
        prior = [e for e in sorted_events if json_path(e, "ocsf", "time", default=0) < event_time]
        prior_actor_operation = sum(
            1
            for e in prior
            if (e.get("actor_name") or e.get("actor") or "unknown") == actor
            and e.get("operation") == operation
        )
        prior_population_operation = sum(1 for e in prior if e.get("operation") == operation)
        actor_prior_ops = Counter(
            e.get("operation")
            for e in prior
            if (e.get("actor_name") or e.get("actor") or "unknown") == actor
        )
        examples.append(
            {
                "event_id": event["event_id"],
                "event_time_ms": event_time,
                "event_time_utc": datetime.fromtimestamp(event_time / 1000, tz=timezone.utc).isoformat(),
                "actor": actor,
                "operation": operation,
                "status": event.get("status"),
                "prior_same_actor_and_operation": prior_actor_operation,
                "prior_population_same_operation": prior_population_operation,
                "prior_actor_top_operations": actor_prior_ops.most_common(5),
                "interpretation": (
                    "history-only illustration; not an anomaly score and not semantic evidence"
                ),
            }
        )
        if len(examples) >= 8:
            break
    return examples


def select_similar_seeds(corpus: list[Document]) -> list[tuple[str, int]]:
    seeds = []
    for idx, doc in enumerate(corpus):
        lower = doc.projection.lower()
        if not any(name == "S1" for name, _ in seeds) and (
            "frombase64string" in lower or "encoded command decoded" in lower
        ):
            seeds.append(("S1", idx))
        if not any(name == "S2" for name, _ in seeds) and "createaccesskey" in lower:
            seeds.append(("S2", idx))
        if len(seeds) == 2:
            break
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports/prototype-rag-demo"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    profile = json.loads(PROFILE_PATH.read_text())
    con = duckdb.connect()
    print("building deterministic prototype corpus", flush=True)
    corpus, corpus_stats, api_events = build_corpus(con)
    digest = corpus_digest(corpus)
    print(json.dumps({"corpus_digest": digest, **corpus_stats}, indent=2), flush=True)

    vectors, embedding_stats = load_or_embed(
        corpus, profile, args.out, args.endpoint, args.batch_size
    )
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)

    query_inputs = [query_text(profile, query) for _, query in QUERIES]
    query_started = time.perf_counter()
    query_vectors = embed_request(
        args.endpoint, profile["api_model_key"], query_inputs, timeout=900
    )
    total_query_seconds = time.perf_counter() - query_started
    query_runs = []
    for (query_id, query), query_vector in zip(QUERIES, query_vectors):
        started = time.perf_counter()
        results = exact_search(corpus, vectors, query_vector, args.top_n)
        scan_ms = (time.perf_counter() - started) * 1000
        query_runs.append(
            {
                "query_id": query_id,
                "query": query,
                "top_n": args.top_n,
                "exact_scan_ms": scan_ms,
                "results": results,
            }
        )

    similar_runs = []
    for seed_id, idx in select_similar_seeds(corpus):
        results = exact_search(corpus, vectors, vectors[idx], args.top_n + 1)
        results = [r for r in results if r["document_id"] != corpus[idx].document_id][
            : args.top_n
        ]
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        similar_runs.append(
            {
                "query_id": seed_id,
                "seed": {
                    "document_id": corpus[idx].document_id,
                    "kind": corpus[idx].kind,
                    "preview": corpus[idx].preview,
                    "locator": corpus[idx].locator,
                },
                "results": results,
            }
        )

    report = {
        "schema_version": "livefire.rag.prototype-query-report/1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_kind": "ad_hoc_prototype",
        "conformance": "incomplete",
        "quality": "informational",
        "operations": "informational",
        "repository": git_identity(ROOT),
        "model": {
            "profile_path": str(PROFILE_PATH),
            "profile_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
            "model": profile["model_repository"],
            "model_revision": profile["model_revision"],
            "model_artifact_sha256": profile["model_artifact_set"]["sha256"],
            "api_model_key": profile["api_model_key"],
            "dimensions": profile["dimensions"],
            "dtype": profile["dtype"],
            "quantization": profile["quantization"],
            "normalization": profile["normalization"],
            "query_instruction": profile["query_instruction"],
        },
        "data": {
            "openbots_authority_glob": OPENBOTS_GLOB,
            "openbots_bronze_sha256": "61d85e27d31555263b1603fbe8f2a6bf9ee60df6cc5e65667aa489552d1c74d7",
            "m21_root": str(M21_ROOT),
            "m21_normalized_snapshot_logical_sha256": "1fda84fcd24790f67ca19c574628a9ab416fa5a6e55d4cab7fb9a1b62dbcbdd0",
            "flow": (
                "M21 typed semantics plus authority-linked exact source text -> "
                "prototype semantic projections -> LM Studio embeddings -> exact cosine scan"
            ),
        },
        "corpus": {"digest": digest, **corpus_stats},
        "embedding": {
            **embedding_stats,
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
            "query_batch_seconds": total_query_seconds,
            "distance_contract": (
                "stored float32 L2 vectors; dot product accumulated float64; "
                "cosine_distance=1-dot; millionths round-half-even"
            ),
        },
        "semantic_queries": query_runs,
        "similarity_queries": similar_runs,
        "history_examples": api_history_examples(api_events),
        "limitations": [
            "No production source adapter or admitted canonical command snapshot exists yet.",
            "No immutable index pack, index manifest admission, or standalone tool provider was exercised.",
            "This prototype hydrates exact command and script text directly from OpenBOTS; the production builder must receive it through a sealed command snapshot.",
            "No PowerShell AST parser was run; only bounded static Base64 decoding and exact 4104 content were used.",
            "No qrels or blinded judgments exist, so recall, precision, nDCG, and model quality cannot be claimed.",
            "Top-N always returns neighbors; cosine proximity is not proof of relevance, maliciousness, anomaly, or causality.",
            "Principal/population history counts are illustrations only; the four-component anomaly scorer is not implemented.",
            "Local locators are prototype provenance hints, not admitted LiveFire SDK source pointers.",
        ],
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== semantic retrieval results ===")
    for run in query_runs:
        print(f"\n{run['query_id']}: {run['query']}")
        for result in run["results"]:
            print(
                f"  {result['rank']}. d={result['cosine_distance']:.6f} "
                f"[{result['kind']}] {result['preview']} ({result['locator']})"
            )
    print("\n=== similar-command results ===")
    for run in similar_runs:
        print(f"\n{run['query_id']} seed: {run['seed']['preview']}")
        for result in run["results"]:
            print(
                f"  {result['rank']}. d={result['cosine_distance']:.6f} "
                f"[{result['kind']}] {result['preview']} ({result['locator']})"
            )
    print(f"\nfull machine-readable report: {report_path}")


if __name__ == "__main__":
    main()
