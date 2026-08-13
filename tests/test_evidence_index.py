from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livefire_rag.canonical import sha256_file
from livefire_rag.evidence_builder import RelationSource, build_evidence_pack
from livefire_rag.evidence_index import (
    EvidenceIndex,
    EvidenceIndexCorrupt,
    EvidenceIndexError,
    promote_evidence_pack,
    verify_promoted_evidence_index,
)
from livefire_rag.evidence_derivation import _build_derivation_pack_for_test
from livefire_rag.evidence_projection import projection_policy_ref
from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"
PROFILE = json.loads(
    (REPOSITORY / "profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json")
    .read_text(encoding="utf-8")
)
SNAPSHOT = {"id": "test.ocsf.snapshot", "version": "1", "sha256": "1" * 64}
SOURCE_ADMISSION = {
    "id": "test.ocsf.source-admission", "version": "1", "sha256": "2" * 64
}


def fake_embed(texts: list[str] | tuple[str, ...]) -> np.ndarray:
    result = np.zeros((len(texts), PROFILE["dimensions"]), dtype=np.float32)
    for row, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        first = int.from_bytes(digest[:2], "big") % PROFILE["dimensions"]
        second = (first + 1 + int.from_bytes(digest[2:4], "big")) % PROFILE["dimensions"]
        result[row, first] = 0.8
        result[row, second] = 0.6
    return result


def fake_preflight(profile: dict, fixture_path: Path) -> dict:
    return {
        "schema_version": "livefire.rag.embedding-execution-preflight/1",
        "fixture_sha256": profile["conformance"]["fixture_sha256"],
        "fixture_path": fixture_path.name,
        "test_double": True,
    }


fake_embed.preflight = fake_preflight


def event(time_ms: int, command: str) -> str:
    return json.dumps(
        {
            "ocsf": {"time": time_ms, "activity_id": 1, "class_uid": 1007, "category_uid": 1},
            "process": {"cmd_line": command, "name": command.split()[0]},
            "actor": {"user": {"name": "alice"}},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class EvidencePromotionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, list[RelationSource]]:
        import duckdb

        definitions = {
            "ocsf_process_activity": [
                ("event-a", event(1_700_000_000_000, "curl https://example.test/a"), "support:a"),
                ("event-b", event(1_700_000_001_000, "curl https://example.test/a"), "support:b"),
                ("event-c", event(1_700_000_002_000, "uname -a"), "support:c"),
            ],
            "ocsf_ext_livefire_system_metric": [
                ("metric-a", json.dumps({"ocsf": {"time": 1_700_000_003_000}, "metric": {"name": "cpu", "value": 7}}), "support:m"),
            ],
        }
        sources = []
        connection = duckdb.connect()
        try:
            for relation, rows in definitions.items():
                path = root / f"{relation}.parquet"
                connection.execute(
                    "CREATE OR REPLACE TABLE fixture(event_id VARCHAR, typed_event_json VARCHAR, support_ref VARCHAR)"
                )
                connection.executemany("INSERT INTO fixture VALUES (?, ?, ?)", rows)
                connection.execute("COPY fixture TO ? (FORMAT PARQUET)", [str(path)])
                sources.append(RelationSource(
                    relation, path, expected_sha256=sha256_file(path), expected_rows=len(rows)
                ))
        finally:
            connection.close()
        pack = root / "projection"
        build_evidence_pack(
            pack, sources, index_id="test.evidence.projection", version="1",
            source_snapshot=SNAPSHOT, projection_policy=projection_policy_ref(), batch_size=2,
        )
        return pack, sources

    def _promote(
        self, root: Path, pack: Path, sources: list[RelationSource], name: str = "index",
        derivation_pack: Path | None = None,
    ):
        return promote_evidence_pack(
            pack, root / name, relation_sources=sources, source_snapshot=SNAPSHOT,
            projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
            embedding_profile=PROFILE,
            embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
            embedding_profile_version="1", embedder=fake_embed,
            embedding_conformance_fixture=(
                REPOSITORY / "fixtures/generic-evidence-embedding-conformance.v1.json"
            ),
            source_admission_receipt=SOURCE_ADMISSION,
            index_id="test.evidence.index", version="1",
            derivation_pack=derivation_pack,
            resume_dir=root / "resume", batch_size=2,
        )

    def _derivation(self, root: Path, pack: Path) -> Path:
        occurrences = []
        event_ids = []
        for line in (pack / "occurrences.jsonl").read_text(encoding="utf-8").splitlines():
            occurrence = json.loads(line)
            if occurrence["relation_identity"]["relation"] != "ocsf_process_activity":
                continue
            event_id = occurrence["source_pointer"]["record_id"]
            event_ids.append(event_id)
            occurrences.append({
                "event_id": event_id,
                "occurrence_id": occurrence["occurrence_id"],
                "relation_name": "ocsf_process_activity",
            })
        typed_rows = {"ocsf_process_activity": [
            {"event_id": event_id, "typed_event_json": json.loads(event(
                1_700_000_000_000 + offset * 1_000,
                "curl https://example.test/a" if offset < 2 else "uname -a",
            )), "support_ref": f"support:{event_id}"}
            for offset, event_id in enumerate(event_ids)
        ]}
        participants = [{
            "event_id": event_id, "entity_id": "ent-alice", "role": "actor",
            "support_ref": f"support:{event_id}",
        } for event_id in event_ids]
        destination = root / "derivation"
        base_manifest = json.loads((pack / "manifest.json").read_text())
        _build_derivation_pack_for_test(
            destination, typed_rows=typed_rows, occurrences=occurrences,
            participants=participants, entities=[{
                "entity_id": "ent-alice", "kind": "user", "display_name": "alice",
                "canonical_value": "alice", "support_ref": "support:entity-alice",
            }], relationships=[{
                "relationship_id": "rel-alice", "kind": "observed_as",
                "source_id": "ent-alice", "target_id": "ent-alice",
                "event_id": event_ids[0], "support_ref": "support:rel-alice",
            }], component_id="test.derivation", version="1",
            source_snapshot=SNAPSHOT, base_projection_pack=base_manifest["component"],
        )
        return destination

    def test_promotes_resumes_verifies_and_searches_occurrences_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, sources = self._fixture(root)
            manifest = self._promote(root, pack, sources)
            self.assertEqual(manifest["coverage"], {
                "source_record_count": 4,
                "terminal_disposition_count": 4,
                "document_count": 2,
                "searchable_document_count": 2,
                "unaccounted_record_count": 0,
                "unresolved_pointer_count": 0,
            })
            self.assertEqual(
                verify_promoted_evidence_index(root / "index", projection_pack=pack, sdk_specs=SDK_SPECS),
                manifest,
            )

            index = EvidenceIndex.open(root / "index", sdk_specs=SDK_SPECS)
            try:
                curl_document = index.connection.execute(
                    "SELECT semantic_projection.text FROM evidence_documents "
                    "WHERE semantic_projection.text LIKE '%curl%'"
                ).fetchone()[0]
                query_vector = fake_embed([curl_document])[0]
                request = {
                    "schema_version": "livefire.rag.evidence-search.input/1",
                    "query": "download with curl",
                    "top_n": 2,
                    "time_range": {
                        "start": "2023-11-14T22:13:20.500Z",
                        "end_exclusive": "2023-11-14T22:13:21.500Z",
                    },
                    "retrieval": {"methods": ["dense"], "fusion": "none"},
                    "filters": {
                        "relations": [{"namespace": "ocsf", "relation": "ocsf_process_activity"}],
                        "ocsf_class_uids": [1007],
                        "attribute_predicates": [{
                            "namespace": "ocsf", "path": "/ocsf/activity_id", "operator": "eq", "value": 1,
                        }],
                    },
                }
                output = index.search_dense(request, query_vector, max_occurrences=1)
                self.assertEqual(output["kind"], "pointer")
                self.assertEqual(output["coverage"]["eligible_occurrences"], 1)
                self.assertEqual(output["coverage"]["eligible_documents"], 1)
                self.assertIn("curl", output["candidates"][0]["preview"])
                self.assertEqual(output["candidates"][0]["matching_occurrence_count"], 1)
                self.assertTrue(output["candidates"][0]["occurrences_exhausted"])

                all_time = dict(request)
                all_time.pop("time_range")
                all_time["top_n"] = 1
                repeated = index.search_dense(all_time, query_vector, max_occurrences=1)
                self.assertEqual(repeated["candidates"][0]["matching_occurrence_count"], 2)
                self.assertEqual(repeated["candidates"][0]["returned_occurrence_count"], 1)
                self.assertFalse(repeated["candidates"][0]["occurrences_exhausted"])

                _, schemas = _offline_registry(None, SDK_SPECS)
                from jsonschema import Draft202012Validator, FormatChecker
                from livefire_rag.evidence_schema import _offline_registry as registry_factory
                registry, _ = registry_factory(None, SDK_SPECS)
                Draft202012Validator(
                    schemas["evidence-search.output.v1.schema.json"], registry=registry,
                    format_checker=FormatChecker(),
                ).validate(output)
            finally:
                index.close()

            calls = []
            def must_use_cache(texts):
                calls.append(texts)
                raise AssertionError("content-bound resume cache was not used")
            must_use_cache.preflight = fake_preflight
            promote_evidence_pack(
                pack, root / "index-2", relation_sources=sources, source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
                embedding_profile=PROFILE,
                embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
                embedding_profile_version="1", embedder=must_use_cache,
                embedding_conformance_fixture=(
                    REPOSITORY / "fixtures/generic-evidence-embedding-conformance.v1.json"
                ),
                source_admission_receipt=SOURCE_ADMISSION,
                index_id="test.evidence.index", version="1", resume_dir=root / "resume", batch_size=1,
            )
            self.assertEqual(calls, [])
            for name in ("documents.parquet", "occurrences.parquet", "embeddings.parquet"):
                self.assertEqual((root / "index" / name).read_bytes(), (root / "index-2" / name).read_bytes())

    def test_verifier_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, sources = self._fixture(root)
            self._promote(root, pack, sources)
            with (root / "index" / "embeddings.parquet").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(EvidenceIndexCorrupt, "artifact digest mismatch"):
                verify_promoted_evidence_index(
                    root / "index", projection_pack=pack, sdk_specs=SDK_SPECS
                )

    def test_promotion_requires_preflight_and_enforces_profile_batch_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, sources = self._fixture(root)
            arguments = dict(
                relation_sources=sources, source_snapshot=SNAPSHOT,
                projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
                embedding_profile=PROFILE,
                embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
                embedding_profile_version="1",
                embedding_conformance_fixture=(
                    REPOSITORY / "fixtures/generic-evidence-embedding-conformance.v1.json"
                ),
                source_admission_receipt=SOURCE_ADMISSION,
                index_id="test.evidence.index", version="1",
            )
            with self.assertRaisesRegex(EvidenceIndexError, "mandatory execution preflight"):
                promote_evidence_pack(
                    pack, root / "no-preflight", embedder=lambda texts: fake_embed(texts),
                    batch_size=2,
                    **arguments,
                )
            with self.assertRaisesRegex(ValueError, "profile maximum"):
                promote_evidence_pack(
                    pack, root / "large-batch", embedder=fake_embed,
                    batch_size=PROFILE["batching"]["maximum_batch_items"] + 1,
                    **arguments,
                )

    def test_promotes_derivation_overlay_and_filters_entity_memberships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, sources = self._fixture(root)
            derivation = self._derivation(root, pack)
            manifest = self._promote(root, pack, sources, derivation_pack=derivation)
            self.assertEqual(manifest["coverage"]["derived_document_count"], 1)
            self.assertEqual(manifest["coverage"]["derivation_membership_count"], 3)
            self.assertEqual(len(manifest["derivation_packs"]), 1)
            self.assertTrue((root / "index" / "derivation-documents.parquet").is_file())
            self.assertTrue((root / "index" / "derivation-memberships.parquet").is_file())

            index = EvidenceIndex.open(
                root / "index", sdk_specs=SDK_SPECS,
            )
            try:
                derived = index.connection.execute(
                    "SELECT document_id, semantic_projection.text FROM evidence_documents "
                    "WHERE document_id LIKE 'ddoc-%'"
                ).fetchone()
                request = {
                    "schema_version": "livefire.rag.evidence-search.input/1",
                    "query": "activity by alice", "top_n": 5,
                    "retrieval": {"methods": ["dense"], "fusion": "none"},
                    "filters": {"entity_ids": ["ent-alice"]},
                }
                output = index.search_dense(request, fake_embed([derived[1]])[0])
                self.assertEqual(output["kind"], "pointer")
                self.assertEqual(output["candidates"][0]["document_id"], derived[0])
                self.assertEqual(output["candidates"][0]["matching_occurrence_count"], 3)
                # Entity identity is occurrence-level filter metadata: every
                # base or derived document linked to Alice's occurrences stays
                # eligible, not only the entity-derived document carrying it.
                self.assertEqual(output["coverage"]["eligible_documents"], 3)
            finally:
                index.close()

            replay = EvidenceIndex.open(
                root / "index", projection_pack=pack, derivation_pack=derivation,
                replay_verify=True, sdk_specs=SDK_SPECS,
            )
            replay.close()


if __name__ == "__main__":
    unittest.main()
