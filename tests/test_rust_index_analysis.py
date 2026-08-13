from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rfc8785

from livefire_rag_analysis import (
    FastIndex,
    FastIndexError,
    document_order_sha256,
    evaluate_retrieval_run,
    write_pca_report,
)


HEADER = struct.Struct("<8sIHBBQII32s")


def _fixture(root: Path) -> Path:
    import duckdb

    index = root / "index"
    index.mkdir()
    (index / "lexical").mkdir()
    document_ids = ["doc-a", "doc-b", "doc-c", "doc-d"]
    semantic_texts = ["alpha", "alpha beta", "gamma", "distant delta"]
    relations = ["process", "process", "api", "api"]
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE documents(document_id VARCHAR, vector_ordinal INTEGER, "
            "document_kind VARCHAR, relation VARCHAR, semantic_text VARCHAR, occurrence_count BIGINT)"
        )
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            [
                (document_id, ordinal, "activity", relations[ordinal], semantic_texts[ordinal], 1)
                for ordinal, document_id in enumerate(document_ids)
            ],
        )
        connection.execute(
            "COPY (SELECT * FROM documents ORDER BY vector_ordinal) TO ? (FORMAT PARQUET)",
            [str(index / "documents.parquet")],
        )
        connection.execute(
            "CREATE TABLE occurrences(occurrence_id VARCHAR, document_id VARCHAR, event_id VARCHAR, "
            "support_ref VARCHAR, snapshot_sha256 VARCHAR, mapping_sha256 VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?)",
            [
                (f"occ-{ordinal}", document_id, f"evt-{ordinal}", f"sup-{ordinal}",
                 "a" * 64, "b" * 64)
                for ordinal, document_id in enumerate(document_ids)
            ],
        )
        connection.execute(
            "COPY occurrences TO ? (FORMAT PARQUET)", [str(index / "occurrences.parquet")]
        )
    finally:
        connection.close()

    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.98, 0.2, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype="<f4",
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    order_sha = document_order_sha256(document_ids)
    header = HEADER.pack(
        b"LFRAGV1\0", 64, 1, 1, 0, len(vectors), vectors.shape[1], 0, bytes.fromhex(order_sha)
    )
    (index / "vectors.f32").write_bytes(header + vectors.tobytes(order="C"))
    digest = "a" * 64
    lexical_path = index / "lexical/index.json"
    lexical_path.write_text(
        json.dumps({
            "document_count": 4,
            "documents": [{"document_id": document_id} for document_id in document_ids],
        }) + "\n",
        encoding="utf-8",
    )
    lookup_path = index / "occurrence-index.sqlite3"
    lookup = sqlite3.connect(lookup_path)
    try:
        lookup.executescript(
            """CREATE TABLE occurrences (
                 occurrence_id TEXT PRIMARY KEY NOT NULL,
                 document_id TEXT NOT NULL,
                 event_time_ms INTEGER,
                 relation TEXT NOT NULL,
                 snapshot_sha256 TEXT NOT NULL,
                 mapping_sha256 TEXT NOT NULL,
                 event_id TEXT NOT NULL,
                 support_ref TEXT NOT NULL
               ) WITHOUT ROWID;
               CREATE TABLE metadata(
                 key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL
               ) WITHOUT ROWID;"""
        )
        lookup.executemany(
            "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f"occ-{ordinal}", document_id, None, relations[ordinal],
                    "a" * 64, "b" * 64, f"evt-{ordinal}", f"sup-{ordinal}",
                )
                for ordinal, document_id in enumerate(document_ids)
            ],
        )
        lookup.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("schema", "sqlite-occurrence-lookup-v1"),
                ("rows", "4"),
                ("snapshot_sha256", "a" * 64),
                ("mapping_sha256", "b" * 64),
            ],
        )
        lookup.commit()
    finally:
        lookup.close()

    def artifact(path: Path) -> dict[str, object]:
        raw = path.read_bytes()
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    manifest = {
        "schema_version": "livefire.rag.fast-index/2",
        "component_sha256": "",
        "source": {"snapshot_sha256": digest, "mapping_sha256": "b" * 64},
        "build_scope": "sample",
        "complete": False,
        "embedding_profile": {
            "id": "test.embedding",
            "version": "1",
            "sha256": "c" * 64,
            "model": "synthetic",
            "dimensions": 3,
            "normalization": "l2",
        },
        "documents": {
            "path": "documents.parquet",
            "rows": 4,
            **artifact(index / "documents.parquet"),
            "order_sha256": order_sha,
        },
        "occurrences": {
            "path": "occurrences.parquet",
            "rows": 4,
            **artifact(index / "occurrences.parquet"),
            "order_sha256": None,
        },
        "vectors": {
            "path": "vectors.f32",
            "count": 4,
            **artifact(index / "vectors.f32"),
            "dimensions": 3,
            "dtype": "f32le",
            "header_bytes": 64,
            "document_order_sha256": order_sha,
        },
        "lexical": {
            "path": "lexical/index.json",
            "document_count": 4,
            **artifact(lexical_path),
            "tokenizer": "ascii_camel_lower_v1",
            "k1": 1.2,
            "b": 0.75,
        },
        "occurrence_lookup": {
            "path": "occurrence-index.sqlite3",
            "rows": 4,
            **artifact(lookup_path),
            "schema": "sqlite-occurrence-lookup-v1",
        },
    }
    material = {key: value for key, value in manifest.items() if key != "component_sha256"}
    manifest["component_sha256"] = hashlib.sha256(rfc8785.dumps(material)).hexdigest()
    (index / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index


class RustIndexAnalysisTests(unittest.TestCase):
    def test_reader_validates_and_memmaps_cross_language_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = _fixture(Path(directory))
            with FastIndex.open(index) as opened:
                self.assertEqual(opened.header.count, 4)
                self.assertEqual(opened.header.dimensions, 3)
                self.assertEqual(opened.document_ids, ["doc-a", "doc-b", "doc-c", "doc-d"])
                self.assertIsInstance(opened.vectors, np.memmap)
                self.assertAlmostEqual(float(np.linalg.norm(opened.vectors[1])), 1.0, places=5)

    def test_reader_rejects_order_and_length_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = _fixture(Path(directory))
            manifest_path = index / "index.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["documents"]["order_sha256"] = hashlib.sha256(b"wrong").hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FastIndexError, "component identity"):
                FastIndex.open(index)

        with tempfile.TemporaryDirectory() as directory:
            index = _fixture(Path(directory))
            with (index / "vectors.f32").open("ab") as handle:
                handle.write(b"\0")
            with self.assertRaisesRegex(FastIndexError, "artifact content digest"):
                FastIndex.open(index)

    def test_pca_report_marks_original_space_outliers_and_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _fixture(root)
            report = write_pca_report(index, root / "analysis", seed=7, mark_count=2)
            self.assertEqual(report["population"]["documents"], 4)
            self.assertEqual(report["markers"]["marked_count"], 2)
            self.assertEqual(report["markers"]["documents"][0]["document_id"], "doc-d")
            self.assertEqual(
                report["pca"]["purpose"], "visualization_only_not_retrieval_or_anomaly_space"
            )
            self.assertTrue((root / "analysis/pca.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(
                json.loads((root / "analysis/report.json").read_text()), report
            )

    def test_evaluator_reports_metrics_only_when_qrels_are_supplied(self) -> None:
        run = [
            {"query_id": "q1", "document_id": "d1", "rank": 1, "score": 0.9},
            {"query_id": "q1", "document_id": "d2", "rank": 2, "score": 0.8},
            {"query_id": "q2", "document_id": "d3", "rank": 1, "score": 0.7},
            {"query_id": "q2", "document_id": "d4", "rank": 2, "score": 0.6},
        ]
        no_qrels = evaluate_retrieval_run(run, cutoffs=(1, 2))
        self.assertEqual(no_qrels["metrics_status"], "unavailable_without_qrels")
        self.assertNotIn("macro", no_qrels)
        qrels = [
            {"query_id": "q1", "document_id": "d2", "relevance": 2},
            {"query_id": "q1", "document_id": "d1", "relevance": 1},
            {"query_id": "q2", "document_id": "d3", "relevance": 1},
        ]
        report = evaluate_retrieval_run(run, qrels=qrels, cutoffs=(1, 2))
        self.assertEqual(report["metrics_status"], "available")
        self.assertAlmostEqual(report["macro"]["recall@1"], 0.75)
        self.assertAlmostEqual(report["macro"]["reciprocal_rank"], 1.0)
        self.assertLess(report["per_query"][0]["ndcg@1"], 1.0)
        self.assertEqual(report["per_query"][1]["ndcg@1"], 1.0)

        missing_run = [row for row in run if row["query_id"] == "q1"]
        zero_hit = evaluate_retrieval_run(
            missing_run,
            qrels=qrels,
            cutoffs=(1, 2),
            planned_query_ids=("q1", "q2"),
        )
        self.assertEqual(zero_hit["queries"], 2)
        self.assertEqual(zero_hit["per_query"][1]["query_id"], "q2")
        self.assertEqual(zero_hit["per_query"][1]["reciprocal_rank"], 0.0)
        self.assertEqual(zero_hit["per_query"][1]["recall@2"], 0.0)


if __name__ == "__main__":
    unittest.main()
