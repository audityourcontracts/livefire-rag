from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from livefire_rag.evidence_builder import _build_evidence_pack_for_test
from livefire_rag.evidence_geometry import (
    CLAIM_STATUS,
    _neighbor_geometry,
    build_evidence_pilot_geometry,
    geometry_policy_ref,
    verify_evidence_pilot_geometry,
)
from livefire_rag.evidence_index import promote_evidence_pack
from livefire_rag.evidence_pilot import build_evidence_pilot_sample
from livefire_rag.evidence_projection import projection_policy_ref
from tests.test_evidence_index import PROFILE, SOURCE_ADMISSION, fake_embed
from tests.test_evidence_pilot import SDK_SPECS, SNAPSHOT, _rows


ROOT = Path(__file__).resolve().parents[1]


class EvidenceGeometryTests(unittest.TestCase):
    def _pilot_index(self, root: Path) -> tuple[Path, Path]:
        pack = root / "projection"
        _build_evidence_pack_for_test(
            pack,
            row_sources={
                "ocsf_process_activity": _rows(
                    4, relation="ocsf_process_activity"
                ),
                "ocsf_api_activity": _rows(4, relation="ocsf_api_activity"),
            },
            index_id="test.geometry.projection", version="1",
            source_snapshot=SNAPSHOT, projection_policy=projection_policy_ref(),
            batch_size=2,
        )
        pilot = root / "pilot"
        build_evidence_pilot_sample(
            pack, pilot, component_id="test.geometry.pilot", version="1",
            sdk_specs=SDK_SPECS,
        )
        index = root / "index"
        promote_evidence_pack(
            pack, index, relation_sources=(), source_snapshot=SNAPSHOT,
            projection_policy=projection_policy_ref(), sdk_specs=SDK_SPECS,
            embedding_profile=PROFILE,
            embedding_profile_id="livefire.rag.embedding.generic-evidence.qwen3-8b-q4",
            embedding_profile_version="1", embedder=fake_embed,
            embedding_conformance_fixture=(
                ROOT / "fixtures/generic-evidence-embedding-conformance.v1.json"
            ),
            source_admission_receipt=SOURCE_ADMISSION,
            index_id="test.geometry.pilot-index", version="1",
            pilot_sample=pilot, resume_dir=root / "resume", batch_size=2,
        )
        return index, pilot

    def test_original_space_neighbors_use_relation_local_exact_cosine(self) -> None:
        vectors = np.asarray([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.1],
        ], dtype=np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        metadata = [
            {
                "document_id": f"doc-{index}",
                "relation": "relation-a" if index < 2 else "relation-b",
                "occurrence_count": 1,
                "sampling_weight_numerator": 1,
                "sampling_weight_denominator": 1,
            }
            for index in range(4)
        ]
        rows, summaries, confusion = _neighbor_geometry(metadata, vectors)
        k10 = [row for row in rows if row["requested_k"] == 10]
        self.assertTrue(all(row["effective_k"] == 1 for row in k10))
        self.assertEqual(k10[0]["neighbor_document_ids"], ["doc-1"])
        self.assertEqual(k10[2]["neighbor_document_ids"], ["doc-3"])
        self.assertTrue(all(row["reciprocal_neighbor_rate"] == 1.0 for row in k10))
        self.assertEqual({row["nearest_global_relation"] for row in k10}, {
            "relation-a", "relation-b"
        })
        self.assertEqual(len(summaries), 6)
        self.assertTrue(all(not row["cross_relation"] for row in confusion))

    def test_seals_deterministic_geometry_with_bound_inputs_and_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, pilot = self._pilot_index(root)
            first = build_evidence_pilot_geometry(
                index, pilot, root / "geometry-a", sdk_specs=SDK_SPECS,
                component_id="test.geometry", version="1", seed=17,
            )
            second = build_evidence_pilot_geometry(
                index, pilot, root / "geometry-b", sdk_specs=SDK_SPECS,
                component_id="test.geometry", version="1", seed=17,
            )
            self.assertEqual(first, second)
            self.assertEqual(
                verify_evidence_pilot_geometry(root / "geometry-a"), first
            )
            report = json.loads((root / "geometry-a/report.json").read_text())
            self.assertEqual(report["claim_status"], CLAIM_STATUS)
            self.assertEqual(report["neighbor_geometry"]["requested_k"], [10, 25, 50])
            self.assertEqual(report["neighbor_geometry"]["space"], "original_l2_embedding")
            self.assertEqual(report["population"]["documents"], 8)
            self.assertEqual(
                report["inputs"]["selection"],
                json.loads((pilot / "manifest.json").read_text())["objects"]["selection"],
            )
            for name in ("pca-pc1-pc2.png", "pca-pc1-pc3.png"):
                self.assertEqual(
                    (root / "geometry-a" / name).read_bytes(),
                    (root / "geometry-b" / name).read_bytes(),
                )
                self.assertTrue(
                    (root / "geometry-a" / name).read_bytes().startswith(
                        b"\x89PNG\r\n\x1a\n"
                    )
                )
            self.assertEqual(
                (root / "geometry-a/report.json").read_bytes(),
                (root / "geometry-b/report.json").read_bytes(),
            )
            import duckdb

            connection = duckdb.connect()
            try:
                rows = connection.execute(
                    "SELECT requested_k, min(effective_k), max(effective_k), count(*) "
                    "FROM read_parquet(?) GROUP BY requested_k ORDER BY requested_k",
                    [str(root / "geometry-a/neighbors.parquet")],
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(rows, [(10, 3, 3, 8), (25, 3, 3, 8), (50, 3, 3, 8)])

    def test_policy_and_public_api_have_closed_index_only_boundary(self) -> None:
        policy = json.loads(
            (ROOT / "specs/evidence-pilot-geometry-policy.v1.json").read_text()
        )
        self.assertEqual(geometry_policy_ref(policy), geometry_policy_ref())
        forbidden = set(policy["input_boundary"]["forbidden"])
        parameters = set(inspect.signature(build_evidence_pilot_geometry).parameters)
        self.assertTrue({"index_root", "pilot_root", "output_dir"} <= parameters)
        self.assertFalse(forbidden & parameters)
        source = inspect.getsource(build_evidence_pilot_geometry)
        self.assertNotIn("query_fixture", source)
        self.assertNotIn("expected_relation", source)


if __name__ == "__main__":
    unittest.main()
