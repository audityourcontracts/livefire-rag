#!/usr/bin/env python3
"""Private stdin/stdout worker for the Rust projection-parity command.

Input rows contain source material and must be supplied only through stdin.
Output contains hashes and categorical values only.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from livefire_rag.canonical import canonical_json_bytes, sha256_bytes
from livefire_rag.evidence_projection import project_event


def _digest_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _digest_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _project(request: dict[str, Any]) -> dict[str, Any]:
    projected = project_event(
        request["relation"],
        request["event_id"],
        request["typed_event_json"],
        request["support_ref"],
    )
    searchable = projected["terminal_disposition"] == "direct_semantic_document"
    facets = {
        "action": projected["action_text"],
        "target": projected["target_text"],
        "context": projected["context_text"],
        "outcome": projected["outcome_text"],
    }
    event_time = {
        "event_time": projected["event_metadata"]["event_time"],
        "event_time_availability": projected["event_metadata"][
            "event_time_availability"
        ],
    }
    document_id = projected["semantic_group_id"] if searchable else ""
    semantic_text = projected["semantic_text"] if searchable else ""
    return {
        "sample_id": request["sample_id"],
        "searchable": searchable,
        "document_kind": projected["document_kind"],
        "document_id_sha256": _digest_text(document_id),
        "semantic_group_id_sha256": _digest_text(projected["semantic_group_id"]),
        "semantic_group_sha256_sha256": _digest_text(
            projected["semantic_group_sha256"]
        ),
        "semantic_text_sha256": _digest_text(semantic_text),
        "facets_sha256": _digest_json(facets),
        "event_time_summary_sha256": _digest_json(event_time),
    }


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        response = _project(request)
        sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
