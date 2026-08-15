from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import rfc8785

from tools.build_catalogue_review_pool import ReviewPoolError, build_review_pool


SHA_A = "a" * 64
SHA_B = "b" * 64


def _canonical(value: object, *, newline: bool = False) -> bytes:
    return rfc8785.dumps(value) + (b"\n" if newline else b"")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(value: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(value)
    value["component_sha256"] = "0" * 64
    material = dict(value)
    del material["component_sha256"]
    value["component_sha256"] = _sha(_canonical(material))
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value, newline=True))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical(row, newline=True) for row in rows))


def _artifact(path: Path, relative: str, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha(path.read_bytes()),
    }
    if rows is not None:
        value["rows"] = rows
    return value


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_dir = root / "index"
        self.index_dir.mkdir()
        self.run_dir = root / "raw-run"
        self.run_dir.mkdir()
        self.out = root / "review"
        self.query_fixture = root / "queries.json"
        self.catalogue_path = root / "catalogue.json"
        self.source_snapshot = {"id": "snapshot", "version": "1", "sha256": SHA_A}
        self.mapping = {"id": "mapping", "version": "1", "sha256": SHA_B}
        self.projection_policy = {"id": "projection", "version": "1", "sha256": "5" * 64}
        self.profile_ref = {"id": "profile", "version": "1", "sha256": "2" * 64}
        self.embedding_profile = {
            **self.profile_ref,
            "model": "local-model",
            "dimensions": 4,
            "normalization": "l2",
            "query_instruction": "Retrieve relevant test evidence.",
            "query_composition": "Instruct: {query_instruction}\nQuery: {query}",
        }
        self.dataset = {
            "id": "dataset-neutral-name",
            "version": "1",
            "source_snapshot": self.source_snapshot,
            "mapping": self.mapping,
            "included_relations": ["ocsf_process_activity"],
            "excluded_relations": [],
            "structured_only_relations": [],
        }
        self.dataset_sha = _sha(_canonical(self.dataset))
        self.documents = {
            "document-a": "interpreter telemetry suppression activity",
            "document-b": "ordinary browser launch",
        }
        self.occurrences = {
            "document-a": {
                "event_time_ms": 1_700_000_000_000,
                "relation": "ocsf_process_activity",
                "snapshot_sha256": SHA_A,
                "mapping_sha256": SHA_B,
                "event_id": "event-a",
                "support_ref": "support-a",
            },
            "document-b": {
                "event_time_ms": None,
                "relation": "ocsf_process_activity",
                "snapshot_sha256": SHA_A,
                "mapping_sha256": SHA_B,
                "event_id": "event-b",
                "support_ref": "support-b",
            },
        }
        self._write_index()
        self._write_catalogue()
        self._write_fixture()
        self._write_run()

    def _write_index(self) -> None:
        lexical_path = self.index_dir / "lexical.sqlite"
        connection = sqlite3.connect(lexical_path)
        connection.execute(
            "CREATE TABLE documents(document_ordinal INTEGER PRIMARY KEY, document_id TEXT UNIQUE, semantic_text TEXT, length INTEGER)"
        )
        for ordinal, (document_id, text) in enumerate(self.documents.items()):
            connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?)", (ordinal, document_id, text, 3))
        connection.commit()
        connection.close()

        occurrence_path = self.index_dir / "occurrence-lookup.sqlite"
        connection = sqlite3.connect(occurrence_path)
        connection.execute(
            "CREATE TABLE occurrences(occurrence_id TEXT, document_id TEXT, event_time_ms INTEGER, relation TEXT, snapshot_sha256 TEXT, mapping_sha256 TEXT, event_id TEXT, support_ref TEXT)"
        )
        for ordinal, (document_id, pointer) in enumerate(self.occurrences.items()):
            connection.execute(
                "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"occurrence-{ordinal}", document_id, *pointer.values()),
            )
        connection.commit()
        connection.close()

        lexical = _artifact(lexical_path, "lexical.sqlite")
        lexical.update({
            "document_count": 2,
            "schema": "sqlite-inverted-bm25-v1",
            "tokenizer": "ascii_camel_lower_v1",
            "k1": 1.2,
            "b": 0.75,
        })
        lookup = _artifact(occurrence_path, "occurrence-lookup.sqlite", 2)
        lookup["schema"] = "sqlite-occurrence-lookup-v1"
        index = _seal({
            "schema_version": "livefire.rag.fast-index/3",
            "component_sha256": "0" * 64,
            "source": {"snapshot_sha256": SHA_A, "mapping_sha256": SHA_B},
            # Portable per-dataset indexes cover their declared dataset scope,
            # not the whole source snapshot. The fast-index v3 compatibility
            # fields therefore remain sample/false while pipeline_provenance
            # binds the complete prepared/plan/result dataset chain.
            "build_scope": "sample",
            "complete": False,
            "documents": {
                "path": "documents.parquet", "rows": 2, "bytes": 1,
                "sha256": "c" * 64, "order_sha256": "6" * 64,
            },
            "occurrences": {
                "path": "occurrences.parquet", "rows": 2, "bytes": 1,
                "sha256": "d" * 64, "order_sha256": None,
            },
            "vectors": {
                "path": "vectors.f32", "count": 2, "bytes": 96, "sha256": "7" * 64,
                "dimensions": 4, "dtype": "f32le", "header_bytes": 64,
                "document_order_sha256": "6" * 64,
            },
            "lexical": lexical,
            "occurrence_lookup": lookup,
            "embedding_profile": self.embedding_profile,
            "pipeline_provenance": {
                "dataset_sha256": self.dataset_sha,
                "prepared_corpus_sha256": "e" * 64,
                "embedding_plan_sha256": "f" * 64,
                "embedding_result_set_sha256": "1" * 64,
            },
        })
        self.index = index
        _write_json(self.index_dir / "index.json", index)

    def _write_catalogue(self) -> None:
        artifact_component = lambda path, name, sha: {
            "path": path, "id": name, "version": "1", "sha256": sha,
        }
        entry = {
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha,
            "projection_policy": self.projection_policy,
            "prepared_corpus": artifact_component("prepared/manifest.json", "prepared", "e" * 64),
            "embedding_plan": artifact_component("plan/plan.json", "plan", "f" * 64),
            "embedding_result_set": artifact_component("results/manifest.json", "results", "1" * 64),
            "embedding_profile": self.profile_ref,
            "searchable_document_count": 2,
            "searchable_reference_count": 2,
            "final_index": {
                "path": "index/index.json",
                "id": "index",
                "version": "1",
                "sha256": self.index["component_sha256"],
            },
            "test_only": False,
        }
        self.catalogue = _seal({
            "schema_version": "livefire.rag.dataset-catalogue/1",
            "component_sha256": "0" * 64,
            "mode": "normal",
            "source_snapshot": self.source_snapshot,
            "mapping": self.mapping,
            "projection_policy": self.projection_policy,
            "embedding_profile": self.profile_ref,
            "query_compatibility": "single_embedding_profile",
            "rank_merge": "reciprocal_rank_fusion_v1",
            "datasets": [entry],
            "allowed_relation_overlaps": [],
        })
        _write_json(self.catalogue_path, self.catalogue)

    def _write_fixture(self) -> None:
        _write_json(self.query_fixture, {
            "schema_version": "livefire.rag.generic-evidence-pilot-queries/1",
            "status": "frozen",
            "queries": [{
                "query_id": "telemetry-suppression",
                "query": "Find attempts to suppress execution telemetry.",
                "expected_relation_families": ["secret_expected_relation_hint"],
            }],
        })

    def _hit(self, document_id: str, rank: int, mode: str) -> dict[str, object]:
        pointer = self.occurrences[document_id]
        return {
            "rank": rank,
            "reciprocal_rank_score": 1.0 / (60 + rank),
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha,
            "index_sha256": self.index["component_sha256"],
            "index_rank": rank,
            "hit": {
                "rank": rank,
                "document_id": document_id,
                "semantic_text": self.documents[document_id],
                "score": 0.9 / rank,
                "dense_score": None if mode == "lexical" else 0.7 / rank,
                "lexical_score": None if mode == "dense" else 0.8 / rank,
                "eligible_occurrence_count": 1,
                "occurrences_exhausted": True,
                "occurrences": [pointer],
            },
        }

    def _write_run(self) -> None:
        query_id = "telemetry-suppression"
        query = "Find attempts to suppress execution telemetry."
        requests = [
            {"query_id": query_id, "query": query, "mode": mode, "top_n": 2, "relations": []}
            for mode in ("dense", "lexical", "fused")
        ]
        results = []
        for mode in ("dense", "lexical", "fused"):
            hits = [self._hit("document-a", 1, mode)]
            if mode == "lexical":
                hits.append(self._hit("document-b", 2, mode))
            results.append({
                "schema_version": "livefire.rag.catalogue-batch-search-result/1",
                "query_id": query_id,
                "catalogue_sha256": self.catalogue["component_sha256"],
                "query": query,
                "mode": mode,
                "top_n": 2,
                "relations": [],
                "rank_merge": "reciprocal_rank_fusion_v1",
                "hits": hits,
            })
        requests_path = self.run_dir / "requests.jsonl"
        results_path = self.run_dir / "results.jsonl"
        _write_jsonl(requests_path, requests)
        _write_jsonl(results_path, results)
        manifest = _seal({
            "schema_version": "livefire.rag.catalogue-batch-search-run/1",
            "component_sha256": "0" * 64,
            "status": "complete",
            "catalogue_sha256": self.catalogue["component_sha256"],
            "embedding_profile": self.profile_ref,
            "requests": _artifact(requests_path, "requests.jsonl", 3),
            "results": _artifact(results_path, "results.jsonl", 3),
            "request_count": 3,
            "result_count": 3,
            "modes": ["dense", "lexical", "fused"],
            "top_n_values": [2],
            "relation_filters": [[]],
            "request_shapes": [
                {"mode": "dense", "top_n": 2, "relations": [], "rows": 1},
                {"mode": "lexical", "top_n": 2, "relations": [], "rows": 1},
                {"mode": "fused", "top_n": 2, "relations": [], "rows": 1},
            ],
            "model": {
                "status": "used", "configured_model": "local-model",
                "returned_model": "local-model", "calls": 1,
            },
            "query_vectors": [
                {
                    "composed_query_sha256": _sha(
                        b"Instruct: Retrieve relevant test evidence.\nQuery: "
                        b"Find attempts to suppress execution telemetry."
                    ),
                    "vector_sha256": "4" * 64,
                    "dimensions": 4,
                }
            ],
            "rank_merge": {"policy": "reciprocal_rank_fusion_v1", "k": 60},
        })
        _write_json(self.run_dir / "manifest.json", manifest)

    def reseal_run(self) -> None:
        results_path = self.run_dir / "results.jsonl"
        results = [json.loads(line) for line in results_path.read_text().splitlines()]
        _write_jsonl(results_path, results)
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        manifest["results"] = _artifact(results_path, "results.jsonl", len(results))
        _write_json(self.run_dir / "manifest.json", _seal(manifest))

    def mutate_run_manifest(self, mutate: object) -> None:
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        mutate(manifest)
        _write_json(self.run_dir / "manifest.json", _seal(manifest))

    def mutate_index_and_rebind(self, mutate: object) -> None:
        index_path = self.index_dir / "index.json"
        index = json.loads(index_path.read_text())
        mutate(index)
        self.index = _seal(index)
        _write_json(index_path, self.index)

        catalogue = json.loads(self.catalogue_path.read_text())
        catalogue["datasets"][0]["final_index"]["sha256"] = self.index["component_sha256"]
        self.catalogue = _seal(catalogue)
        _write_json(self.catalogue_path, self.catalogue)

        results_path = self.run_dir / "results.jsonl"
        results = [json.loads(line) for line in results_path.read_text().splitlines()]
        for result in results:
            result["catalogue_sha256"] = self.catalogue["component_sha256"]
            for hit in result["hits"]:
                hit["index_sha256"] = self.index["component_sha256"]
        _write_jsonl(results_path, results)
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        manifest["catalogue_sha256"] = self.catalogue["component_sha256"]
        manifest["results"] = _artifact(results_path, "results.jsonl", len(results))
        _write_json(self.run_dir / "manifest.json", _seal(manifest))

    def build(self, out: Path | None = None, snapshot: Path | None = None) -> dict[str, object]:
        return build_review_pool(
            run_dir=self.run_dir,
            catalogue_path=self.catalogue_path,
            query_fixture=self.query_fixture,
            out_dir=out or self.out,
            snapshot_root=snapshot,
        )


class CatalogueReviewPoolTests(unittest.TestCase):
    def test_pool_is_deterministic_deduplicated_and_has_no_system_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            first = fixture.root / "review-one"
            second = fixture.root / "review-two"
            manifest = fixture.build(first)
            fixture.build(second)
            first_files = {
                path.relative_to(first): path.read_bytes()
                for path in first.rglob("*") if path.is_file()
            }
            second_files = {
                path.relative_to(second): path.read_bytes()
                for path in second.rglob("*") if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(manifest["unique_candidate_count"], 2)
            pool_rows = [json.loads(line) for line in (first / "review-pool.jsonl").read_text().splitlines()]
            self.assertEqual(len(pool_rows), 2)
            expected_keys = {
                "schema_version", "candidate_id", "query_id", "query", "dataset_id",
                "document_id", "semantic_text", "eligible_occurrence_count", "occurrences",
            }
            self.assertTrue(all(set(row) == expected_keys for row in pool_rows))
            self.assertTrue(all(row["candidate_id"].startswith("candidate-") for row in pool_rows))
            public = (first / "review-pool.jsonl").read_text()
            for forbidden in (
                '"mode"', '"rank"', '"score"', '"index_sha256"',
                '"catalogue_sha256"', "secret_expected_relation_hint",
            ):
                self.assertNotIn(forbidden, public)
            private = [
                json.loads(line)
                for line in (first / "audit/system-provenance.jsonl").read_text().splitlines()
            ]
            systems = {row["document_id"]: row["systems"] for row in private}
            self.assertEqual(len(systems["document-a"]), 3)
            self.assertEqual(len(systems["document-b"]), 1)
            universe = json.loads((first / "audit/candidate-universe.json").read_text())
            self.assertEqual(universe["document_count"], 2)
            snapshot_receipt = json.loads(
                (first / "audit/snapshot-validation.json").read_text()
            )
            self.assertEqual(snapshot_receipt["status"], "not_requested")
            self.assertEqual(snapshot_receipt["checked_unique_pointer_count"], 0)

    def test_fused_pool_accepts_candidates_returned_by_only_one_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            results_path = fixture.run_dir / "results.jsonl"
            results = [json.loads(line) for line in results_path.read_text().splitlines()]
            fused = next(result for result in results if result["mode"] == "fused")
            fused["hits"][0]["hit"]["lexical_score"] = None
            _write_jsonl(results_path, results)
            fixture.reseal_run()
            manifest = fixture.build()
            self.assertGreater(manifest["unique_candidate_count"], 0)

    def test_raw_byte_tamper_fails_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            with (fixture.run_dir / "results.jsonl").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ReviewPoolError, "byte receipt mismatch"):
                fixture.build()
            self.assertFalse(fixture.out.exists())

    def test_raw_jsonl_files_require_a_final_lf(self) -> None:
        for artifact_name in ("requests", "results"):
            with self.subTest(artifact=artifact_name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                path = fixture.run_dir / f"{artifact_name}.jsonl"
                raw = path.read_bytes()
                self.assertTrue(raw.endswith(b"\n"))
                path.write_bytes(raw[:-1])
                manifest = json.loads((fixture.run_dir / "manifest.json").read_text())
                manifest[artifact_name] = _artifact(path, f"{artifact_name}.jsonl", 3)
                _write_json(fixture.run_dir / "manifest.json", _seal(manifest))
                with self.assertRaisesRegex(ReviewPoolError, "lacks a final LF"):
                    fixture.build()
                self.assertFalse(fixture.out.exists())

    def test_every_frozen_query_requires_all_three_search_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            requests_path = fixture.run_dir / "requests.jsonl"
            results_path = fixture.run_dir / "results.jsonl"
            requests = [
                json.loads(line)
                for line in requests_path.read_text().splitlines()
                if json.loads(line)["mode"] != "fused"
            ]
            results = [
                json.loads(line)
                for line in results_path.read_text().splitlines()
                if json.loads(line)["mode"] != "fused"
            ]
            _write_jsonl(requests_path, requests)
            _write_jsonl(results_path, results)
            manifest = json.loads((fixture.run_dir / "manifest.json").read_text())
            manifest["requests"] = _artifact(requests_path, "requests.jsonl", 2)
            manifest["results"] = _artifact(results_path, "results.jsonl", 2)
            manifest["request_count"] = 2
            manifest["result_count"] = 2
            manifest["modes"] = ["dense", "lexical"]
            manifest["request_shapes"] = [
                shape for shape in manifest["request_shapes"] if shape["mode"] != "fused"
            ]
            _write_json(fixture.run_dir / "manifest.json", _seal(manifest))
            with self.assertRaisesRegex(ReviewPoolError, "all three modes"):
                fixture.build()
            self.assertFalse(fixture.out.exists())

    def test_request_shape_receipt_must_close_over_exact_grouped_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.mutate_run_manifest(
                lambda manifest: manifest["request_shapes"][0].update({"rows": 2})
            )
            with self.assertRaisesRegex(ReviewPoolError, "request-shape closure"):
                fixture.build()
            self.assertFalse(fixture.out.exists())

    def test_embedding_profile_model_and_vector_receipts_are_bound(self) -> None:
        cases = (
            (
                "profile",
                lambda manifest: manifest["embedding_profile"].update({"sha256": "8" * 64}),
                "different embedding profile",
            ),
            (
                "configured_model",
                lambda manifest: manifest["model"].update(
                    {"configured_model": "other-model", "returned_model": "other-model"}
                ),
                "model receipt does not match",
            ),
            (
                "returned_model",
                lambda manifest: manifest["model"].update({"returned_model": "other-model"}),
                "semantic model-call closure",
            ),
            (
                "dimensions",
                lambda manifest: manifest["query_vectors"][0].update({"dimensions": 5}),
                "dimensions do not match",
            ),
            (
                "composed_query",
                lambda manifest: manifest["query_vectors"][0].update(
                    {"composed_query_sha256": "9" * 64}
                ),
                "do not match the composed queries",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.mutate_run_manifest(mutate)
                with self.assertRaisesRegex(ReviewPoolError, message):
                    fixture.build()
                self.assertFalse(fixture.out.exists())

    def test_admitted_index_profile_and_vector_dimensions_are_exact(self) -> None:
        cases = (
            (
                "dataset_scope_completion",
                lambda index: index.update({"complete": True}),
                "build-scope declaration is inconsistent",
            ),
            (
                "model",
                lambda index: index["embedding_profile"].update({"model": "other-model"}),
                "model receipt does not match",
            ),
            (
                "vector_dimensions",
                lambda index: index["vectors"].update({"dimensions": 3}),
                "vectors use different dimensions",
            ),
            (
                "prepared_provenance",
                lambda index: index["pipeline_provenance"].update(
                    {"prepared_corpus_sha256": "8" * 64}
                ),
                "dataset binding failed",
            ),
            (
                "source_snapshot",
                lambda index: index["source"].update({"snapshot_sha256": "8" * 64}),
                "source binding failed",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.mutate_index_and_rebind(mutate)
                with self.assertRaisesRegex(ReviewPoolError, message):
                    fixture.build()
                self.assertFalse(fixture.out.exists())

    def test_pointer_tamper_with_resealed_raw_run_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            results_path = fixture.run_dir / "results.jsonl"
            results = [json.loads(line) for line in results_path.read_text().splitlines()]
            results[0]["hits"][0]["hit"]["occurrences"][0]["event_id"] = "forged-event"
            _write_jsonl(results_path, results)
            fixture.reseal_run()
            with self.assertRaisesRegex(ReviewPoolError, "occurrence pointers"):
                fixture.build()
            self.assertFalse(fixture.out.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.out.mkdir()
            sentinel = fixture.out / "owned-by-user.txt"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(ReviewPoolError, "refusing to overwrite"):
                fixture.build()
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertEqual(list(fixture.out.iterdir()), [sentinel])

    def test_optional_snapshot_validation_checks_event_and_support_membership(self) -> None:
        try:
            import duckdb
        except ImportError:
            self.skipTest("DuckDB optional dependency is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            snapshot = fixture.root / "snapshot"
            semantic = snapshot / "semantic"
            semantic.mkdir(parents=True)
            _write_json(
                snapshot / "build-receipt.json",
                {"runnable_snapshot": {"component": fixture.source_snapshot}},
            )
            parquet = semantic / "ocsf_process_activity.parquet"
            connection = duckdb.connect(":memory:")
            connection.execute(
                "COPY (SELECT * FROM (VALUES ('event-a','support-a'),('event-b','support-b')) "
                "v(event_id,support_ref)) TO ? (FORMAT parquet)",
                [str(parquet)],
            )
            connection.close()
            passed = fixture.root / "snapshot-pass"
            fixture.build(passed, snapshot)
            self.assertTrue((passed / "review-pool.jsonl").is_file())
            receipt = json.loads((passed / "audit/snapshot-validation.json").read_text())
            self.assertEqual(receipt["status"], "exact_typed_parquet_membership_passed")
            self.assertEqual(receipt["checked_unique_pointer_count"], 2)
            audit = json.loads((passed / "audit/manifest.json").read_text())
            self.assertEqual(
                audit["snapshot_validation"]["component_sha256"],
                receipt["component_sha256"],
            )

            connection = duckdb.connect(":memory:")
            connection.execute(
                "COPY (SELECT 'event-a' event_id, 'support-a' support_ref) TO ? (FORMAT parquet)",
                [str(parquet)],
            )
            connection.close()
            failed = fixture.root / "snapshot-fail"
            with self.assertRaisesRegex(ReviewPoolError, "absent from typed"):
                fixture.build(failed, snapshot)
            self.assertFalse(failed.exists())


if __name__ == "__main__":
    unittest.main()
