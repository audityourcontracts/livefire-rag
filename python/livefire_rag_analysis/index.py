"""Read and validate the language-neutral Rust fast-index artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


MAGIC = b"LFRAGV1\0"
HEADER_BYTES = 64
VERSION = 1
DTYPE_F32_LE = 1
_HEADER = struct.Struct("<8sIHBBQII32s")
_INDEX_SCHEMA = "livefire.rag.fast-index/1"


class FastIndexError(RuntimeError):
    """A fast index does not satisfy its cross-language physical contract."""


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - optional dependency
        raise FastIndexError(
            "DuckDB is required; install livefire-rag[analysis]"
        ) from error
    return duckdb


def document_order_sha256(document_ids: Sequence[str]) -> str:
    """Digest UTF-8 document IDs in vector order, each terminated by NUL."""

    digest = hashlib.sha256()
    for document_id in document_ids:
        if not isinstance(document_id, str) or not document_id or "\0" in document_id:
            raise FastIndexError("document IDs must be non-empty NUL-free strings")
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class VectorHeader:
    count: int
    dimensions: int
    document_order_sha256: str
    header_bytes: int = HEADER_BYTES
    version: int = VERSION
    dtype: int = DTYPE_F32_LE
    flags: int = 0

    @classmethod
    def read(cls, path: Path) -> "VectorHeader":
        try:
            raw = path.read_bytes()[:HEADER_BYTES]
        except OSError as error:
            raise FastIndexError("vectors.f32 is unreadable") from error
        if len(raw) != HEADER_BYTES:
            raise FastIndexError("vectors.f32 has a truncated header")
        magic, header_bytes, version, dtype, flags, count, dimensions, reserved, order = (
            _HEADER.unpack(raw)
        )
        if magic != MAGIC:
            raise FastIndexError("vectors.f32 magic is invalid")
        if header_bytes != HEADER_BYTES or version != VERSION:
            raise FastIndexError("vectors.f32 header version is unsupported")
        if dtype != DTYPE_F32_LE or flags != 0 or reserved != 0:
            raise FastIndexError("vectors.f32 dtype, flags, or reserved field is invalid")
        if count < 1 or dimensions < 1:
            raise FastIndexError("vectors.f32 count and dimensions must be positive")
        expected_bytes = HEADER_BYTES + count * dimensions * 4
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise FastIndexError("vectors.f32 metadata is unreadable") from error
        if actual_bytes != expected_bytes:
            raise FastIndexError(
                f"vectors.f32 length mismatch: expected {expected_bytes}, got {actual_bytes}"
            )
        return cls(
            count=count,
            dimensions=dimensions,
            document_order_sha256=order.hex(),
        )


def _closed_object(value: Any, fields: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise FastIndexError(f"index.json {what} has unknown or missing fields")
    return value


def _safe_child(root: Path, value: Any, what: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FastIndexError(f"index.json {what} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise FastIndexError(f"index.json {what} path is not root-relative")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise FastIndexError(f"index.json {what} path escapes the index")
    if what == "lexical":
        if not path.exists():
            raise FastIndexError("index.json lexical artifact is missing")
    elif not path.is_file():
        raise FastIndexError(f"index.json {what} artifact is missing")
    return path


class FastIndex:
    """Validated zero-copy view over a Rust fast experimental index."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        header: VectorHeader,
        document_ids: list[str],
        metadata: list[dict[str, Any]],
        vectors: np.memmap,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.header = header
        self.document_ids = document_ids
        self.metadata = metadata
        self.vectors = vectors

    @classmethod
    def open(cls, root: Path, *, validate_vectors: bool = True) -> "FastIndex":
        root = Path(root).resolve()
        try:
            manifest = json.loads((root / "index.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FastIndexError("index.json is unreadable") from error
        _validate_manifest(manifest)
        documents_spec = manifest["documents"]
        occurrences_spec = manifest["occurrences"]
        vectors_spec = manifest["vectors"]
        documents_path = _safe_child(root, documents_spec["path"], "documents")
        occurrences_path = _safe_child(root, occurrences_spec["path"], "occurrences")
        vectors_path = _safe_child(root, vectors_spec["path"], "vectors")
        lexical_path = _safe_child(root, manifest["lexical"]["path"], "lexical")

        header = VectorHeader.read(vectors_path)
        _validate_vector_bindings(manifest, header)
        metadata = _load_documents(documents_path)
        document_ids = [str(row["document_id"]) for row in metadata]
        if len(metadata) != documents_spec["rows"] or len(metadata) != header.count:
            raise FastIndexError("document, manifest, and vector counts differ")
        if [row["vector_ordinal"] for row in metadata] != list(range(len(metadata))):
            raise FastIndexError("document vector ordinals are not contiguous and ordered")
        if len(document_ids) != len(set(document_ids)):
            raise FastIndexError("document IDs are not unique")
        order_sha = document_order_sha256(document_ids)
        if order_sha != documents_spec["order_sha256"] or order_sha != header.document_order_sha256:
            raise FastIndexError("document order digest mismatch")
        occurrence_rows = _parquet_rows(occurrences_path)
        if occurrence_rows != occurrences_spec["rows"]:
            raise FastIndexError("occurrence Parquet row count differs from index.json")
        _validate_occurrence_closure(
            occurrences_path, metadata, manifest["source"]
        )
        _validate_lexical_association(lexical_path, document_ids)

        vectors = np.memmap(
            vectors_path,
            dtype="<f4",
            mode="r",
            offset=header.header_bytes,
            shape=(header.count, header.dimensions),
            order="C",
        )
        if validate_vectors:
            _validate_vector_values(vectors, manifest["embedding_profile"]["normalization"])
        return cls(root, manifest, header, document_ids, metadata, vectors)

    def iter_rows(self) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
        for ordinal, metadata in enumerate(self.metadata):
            yield metadata, self.vectors[ordinal]

    def close(self) -> None:
        mmap = getattr(self.vectors, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "FastIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_manifest(manifest: Any) -> None:
    required = {
        "schema_version", "source", "build_scope", "complete", "embedding_profile",
        "documents", "occurrences", "vectors", "lexical",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise FastIndexError("index.json has unknown or missing fields")
    if manifest["schema_version"] != _INDEX_SCHEMA:
        raise FastIndexError("index.json schema version is unsupported")
    _closed_object(manifest["source"], {"snapshot_sha256", "mapping_sha256"}, "source")
    scope = manifest["build_scope"]
    complete = manifest["complete"]
    if scope not in ("full", "sample") or not isinstance(complete, bool):
        raise FastIndexError("index.json build scope is invalid")
    if complete != (scope == "full"):
        raise FastIndexError("index.json build scope completeness is contradictory")
    profile = manifest["embedding_profile"]
    profile_required = {"id", "version", "sha256", "model", "dimensions", "normalization"}
    profile_optional = {"query_instruction", "query_composition"}
    if (
        not isinstance(profile, dict)
        or not profile_required <= set(profile)
        or not set(profile) <= profile_required | profile_optional
    ):
        raise FastIndexError("index.json embedding_profile has unknown or missing fields")
    query_fields = profile_optional & set(profile)
    if query_fields not in (set(), profile_optional) or any(
        not isinstance(profile[field], str) or not profile[field]
        for field in query_fields
    ):
        raise FastIndexError(
            "index.json query instruction and composition must be absent or non-empty strings"
        )
    if (
        not isinstance(profile["dimensions"], int) or isinstance(profile["dimensions"], bool)
        or profile["dimensions"] < 1 or profile["normalization"] not in ("l2", "none")
        or any(
            not isinstance(profile[field], str) or not profile[field]
            for field in ("id", "version", "model")
        )
    ):
        raise FastIndexError("index.json embedding profile is invalid")
    documents = _closed_object(
        manifest["documents"], {"path", "rows", "order_sha256"}, "documents"
    )
    occurrences = _closed_object(
        manifest["occurrences"], {"path", "rows", "order_sha256"}, "occurrences"
    )
    if occurrences["order_sha256"] is not None:
        raise FastIndexError("index.json occurrences order digest must be null in format v1")
    _closed_object(
        manifest["vectors"],
        {"path", "count", "dimensions", "dtype", "header_bytes", "document_order_sha256"},
        "vectors",
    )
    lexical = _closed_object(
        manifest["lexical"], {"path", "document_count", "tokenizer", "k1", "b"}, "lexical"
    )
    if (
        lexical["tokenizer"] != "ascii_camel_lower_v1"
        or not isinstance(lexical["k1"], (int, float)) or isinstance(lexical["k1"], bool)
        or not isinstance(lexical["b"], (int, float)) or isinstance(lexical["b"], bool)
        or not math.isfinite(lexical["k1"]) or not math.isfinite(lexical["b"])
        or lexical["k1"] <= 0 or not 0 <= lexical["b"] <= 1
    ):
        raise FastIndexError("index.json lexical tokenizer is unsupported")
    for digest in (
        manifest["source"]["snapshot_sha256"], manifest["source"]["mapping_sha256"],
        profile["sha256"], documents["order_sha256"],
        manifest["vectors"]["document_order_sha256"],
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise FastIndexError("index.json contains an invalid SHA-256")
    counts = (
        manifest["documents"]["rows"], manifest["occurrences"]["rows"],
        manifest["vectors"]["count"], lexical["document_count"],
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise FastIndexError("index.json contains an invalid row count")
    if counts[0] < 1 or counts[0] != counts[2] or counts[0] != counts[3]:
        raise FastIndexError("index.json document counts do not agree")


def _validate_vector_bindings(manifest: dict[str, Any], header: VectorHeader) -> None:
    vectors = manifest["vectors"]
    profile = manifest["embedding_profile"]
    if (
        vectors["dtype"] != "f32le"
        or vectors["header_bytes"] != HEADER_BYTES
        or vectors["count"] != header.count
        or vectors["dimensions"] != header.dimensions
        or profile["dimensions"] != header.dimensions
        or vectors["document_order_sha256"] != header.document_order_sha256
    ):
        raise FastIndexError("vector header, profile, and index.json bindings differ")


def _load_documents(path: Path) -> list[dict[str, Any]]:
    connection = _duckdb().connect()
    try:
        columns = [
            str(row[0])
            for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        ]
        required = {"document_id", "vector_ordinal"}
        if not required <= set(columns):
            raise FastIndexError("documents.parquet lacks document_id or vector_ordinal")
        selected = ["document_id", "vector_ordinal"] + [
            name
            for name in (
                "document_kind", "relation", "relations_json", "semantic_text",
                "occurrence_count",
            )
            if name in columns
        ]
        quoted = ", ".join(f'"{name}"' for name in selected)
        rows = connection.execute(
            f"SELECT {quoted} FROM read_parquet(?) ORDER BY vector_ordinal", [str(path)]
        ).fetchall()
    finally:
        connection.close()
    output = [dict(zip(selected, row, strict=True)) for row in rows]
    for row in output:
        if (
            not isinstance(row["document_id"], str) or not row["document_id"]
            or not isinstance(row["vector_ordinal"], int) or row["vector_ordinal"] < 0
        ):
            raise FastIndexError("documents.parquet contains an invalid identity or ordinal")
    return output


def _parquet_rows(path: Path) -> int:
    connection = _duckdb().connect()
    try:
        return int(connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0])
    finally:
        connection.close()


def _validate_occurrence_closure(
    path: Path, documents: list[dict[str, Any]], source: dict[str, str]
) -> None:
    connection = _duckdb().connect()
    try:
        summary = connection.execute(
            """SELECT count(*), count(DISTINCT occurrence_id),
                      count(*) FILTER (WHERE event_id = '' OR support_ref = ''),
                      count(*) FILTER (WHERE snapshot_sha256 <> ? OR mapping_sha256 <> ?)
                 FROM read_parquet(?)""",
            [source["snapshot_sha256"], source["mapping_sha256"], str(path)],
        ).fetchone()
        actual = dict(
            connection.execute(
                "SELECT document_id, count(*) FROM read_parquet(?) GROUP BY document_id",
                [str(path)],
            ).fetchall()
        )
    finally:
        connection.close()
    expected = {row["document_id"]: row.get("occurrence_count") for row in documents}
    if (
        summary[0] != summary[1]
        or summary[2] != 0
        or summary[3] != 0
        or set(actual) != set(expected)
        or any(expected[document_id] != count for document_id, count in actual.items())
    ):
        raise FastIndexError("occurrence source/document closure is invalid")


def _validate_lexical_association(path: Path, document_ids: list[str]) -> None:
    try:
        lexical = json.loads(path.read_text(encoding="utf-8"))
        lexical_ids = [row["document_id"] for row in lexical["documents"]]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise FastIndexError("lexical index is unreadable") from error
    if (
        lexical.get("document_count") != len(document_ids)
        or len(lexical_ids) != len(set(lexical_ids))
        or set(lexical_ids) != set(document_ids)
    ):
        raise FastIndexError("lexical document association is invalid")


def _validate_vector_values(vectors: np.ndarray, normalization: str) -> None:
    block = 2048
    for start in range(0, vectors.shape[0], block):
        values = np.asarray(vectors[start : start + block], dtype=np.float32)
        if not np.isfinite(values).all():
            raise FastIndexError("vectors.f32 contains a non-finite value")
        if normalization == "l2":
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-4):
                raise FastIndexError("vectors.f32 contains a vector that is not L2 normalized")


__all__ = [
    "DTYPE_F32_LE", "FastIndex", "FastIndexError", "HEADER_BYTES", "MAGIC",
    "VERSION", "VectorHeader", "document_order_sha256",
]
