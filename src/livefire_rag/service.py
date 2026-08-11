"""Typed standalone search and similar operations."""

from __future__ import annotations

import time
from typing import Any

from .contracts import ContractError, validate_filters, validate_search, validate_similar
from .embedding import embed_query
from .index import IndexNotFound, SemanticIndex


class DeadlineExceeded(RuntimeError):
    code = "deadline_exceeded"


def check_deadline(deadline_unix_ms: int) -> None:
    if int(time.time() * 1000) > deadline_unix_ms:
        raise DeadlineExceeded("call deadline exceeded")


class SemanticService:
    def __init__(self, index: SemanticIndex, embedding_endpoint: str = "http://127.0.0.1:1234") -> None:
        self.index = index
        self.embedding_endpoint = embedding_endpoint

    def search(self, arguments: Any, deadline_unix_ms: int) -> dict[str, Any]:
        request = validate_search(arguments)
        check_deadline(deadline_unix_ms)
        profile = self.index.manifest["embedding_profile"]
        query_text = profile["query_composition"].format(
            query_instruction=profile["query_instruction"], query=request["query"]
        )
        vector = embed_query(
            self.embedding_endpoint,
            profile["api_model_key"],
            query_text,
            dimensions=self.index.manifest["dimensions"],
            deadline_unix_ms=deadline_unix_ms,
        )
        eligible = self.index.eligible_indices(request["time_range"], validate_filters(request.get("filters")))
        check_deadline(deadline_unix_ms)
        ranked = self.index.exact_search(vector, eligible, request["top_n"])
        check_deadline(deadline_unix_ms)
        return self.index.pointer_output("cli.search", ranked, len(eligible), request["top_n"])

    def similar(self, arguments: Any, deadline_unix_ms: int) -> dict[str, Any]:
        request = validate_similar(arguments)
        check_deadline(deadline_unix_ms)
        seed_index = self.index.by_id.get(request["command_id"])
        if seed_index is None:
            raise IndexNotFound("seed command_id was not found in the bound index")
        filters = dict(validate_filters(request.get("filters")))
        excluded = list(filters.get("exclude_command_ids", []))
        if request.get("exclude_seed", True):
            excluded.append(request["command_id"])
        filters["exclude_command_ids"] = excluded
        eligible = self.index.eligible_indices(request.get("time_range"), filters)
        ranked = self.index.exact_search(self.index.vectors[seed_index], eligible, request["top_n"])
        check_deadline(deadline_unix_ms)
        return self.index.pointer_output("cli.similar", ranked, len(eligible), request["top_n"])
