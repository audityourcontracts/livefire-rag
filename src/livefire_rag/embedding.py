"""Bounded OpenAI-compatible loopback embedding client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np


class EmbeddingError(RuntimeError):
    code = "unavailable"


def embed_query(
    endpoint: str,
    model: str,
    text: str,
    *,
    dimensions: int,
    deadline_unix_ms: int,
) -> np.ndarray:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise EmbeddingError("embedding endpoint must be loopback HTTP")
    remaining = (deadline_unix_ms - int(time.time() * 1000)) / 1000.0
    if remaining <= 0:
        error = EmbeddingError("deadline exceeded before embedding request")
        error.code = "deadline_exceeded"
        raise error
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/embeddings",
        data=json.dumps({"model": model, "input": [text]}, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.001, remaining)) as response:
            payload: Any = json.load(response)
    except TimeoutError as error:
        failure = EmbeddingError("embedding request exceeded deadline")
        failure.code = "deadline_exceeded"
        raise failure from error
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise EmbeddingError("embedding provider unavailable or returned invalid JSON") from error
    if int(time.time() * 1000) > deadline_unix_ms:
        failure = EmbeddingError("embedding request exceeded deadline")
        failure.code = "deadline_exceeded"
        raise failure
    try:
        data = payload["data"]
        if not isinstance(data, list) or len(data) != 1 or data[0].get("index") != 0:
            raise ValueError
        vector = np.asarray(data[0]["embedding"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise EmbeddingError("embedding response shape is invalid") from error
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise EmbeddingError("embedding vector dimension or values are invalid")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if abs(norm - 1.0) > 0.0001:
        raise EmbeddingError("embedding vector is not L2 normalized")
    return vector
