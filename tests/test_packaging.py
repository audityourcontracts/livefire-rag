from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from livefire_rag.bundle import package_bundle


REPOSITORY = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_provider_package_refuses_test_only_index_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index"
            index.mkdir()
            (index / "index.json").write_text(
                json.dumps({"schema_version": "livefire.rag.fast-index/4", "test_only": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "test-only indexes cannot be packaged"):
                package_bundle(REPOSITORY, root / "bundle", index, root / "sdk")
            self.assertFalse((root / "bundle").exists())

    def test_wheel_contains_apache_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(output)],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            )
            [wheel] = output.glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                self.assertIn(
                    "livefire_rag/evidence_specs/corpus-census.v1.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/embedding-plan.v2.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/benchmark-selection-manifest.v1.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/benchmark-selection-row.v1.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/tokenizer-parity-fixture.v1.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/dataset-catalogue.v1.schema.json",
                    archive.namelist(),
                )
                for catalogue_run_schema in [
                    "catalogue-batch-search-request.v1.schema.json",
                    "catalogue-batch-search-result.v1.schema.json",
                    "catalogue-batch-search-run.v1.schema.json",
                    "catalogue-review-pool-row.v1.schema.json",
                    "catalogue-review-pool-manifest.v1.schema.json",
                ]:
                    self.assertIn(
                        f"livefire_rag/evidence_specs/{catalogue_run_schema}",
                        archive.namelist(),
                    )
                self.assertIn(
                    "livefire_rag/evidence_specs/fast-index-manifest.v3.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/fast-index-manifest.v4.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/fast-build-report.v2.schema.json",
                    archive.namelist(),
                )
                self.assertIn(
                    "livefire_rag/evidence_specs/embedding-result-set.v2.schema.json",
                    archive.namelist(),
                )
                for reduced_schema in [
                    "embedding-policy.v2.schema.json",
                    "embedding-task-receipt.v2.schema.json",
                    "embedding-result-set.v3.schema.json",
                    "projection-parity-report.v1.schema.json",
                ]:
                    self.assertIn(
                        f"livefire_rag/evidence_specs/{reduced_schema}",
                        archive.namelist(),
                    )
                self.assertIn(
                    "livefire_rag/evidence_specs/fast-lexical-profile.v2.json",
                    archive.namelist(),
                )
                for report_schema in [
                    "embedding-task-run-report.v1.schema.json",
                    "embedding-run-summary.v1.schema.json",
                    "query-benchmark.v1.schema.json",
                    "index-overlap.v1.schema.json",
                ]:
                    self.assertIn(
                        f"livefire_rag/evidence_specs/{report_schema}",
                        archive.namelist(),
                    )
                license_paths = [
                    name for name in archive.namelist() if name.endswith(".dist-info/licenses/LICENSE")
                ]
                self.assertEqual(len(license_paths), 1)
                self.assertEqual(
                    archive.read(license_paths[0]),
                    (REPOSITORY / "LICENSE").read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
