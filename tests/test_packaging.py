from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from livefire_rag.bundle import package_bundle
from livefire_rag.evidence_schema import GENERIC_EVIDENCE_SCHEMA_NAMES, _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_offline_schema_registry_names_are_unique(self) -> None:
        self.assertEqual(
            len(GENERIC_EVIDENCE_SCHEMA_NAMES),
            len(set(GENERIC_EVIDENCE_SCHEMA_NAMES)),
        )

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
                self.assertFalse(
                    any(
                        name.endswith(".dist-info/entry_points.txt")
                        for name in archive.namelist()
                    ),
                    "the analysis/test wheel must not publish the retired Python CLI",
                )
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
                for cloud_embedding_schema in [
                    "embedding-policy.v3.schema.json",
                    "tei-model-artifact-set.v1.schema.json",
                    "runpod-embedding-bundle.v1.schema.json",
                    "runpod-worker-attempt.v1.schema.json",
                    "runpod-worker-runtime-event.v1.schema.json",
                    "runpod-run-report.v1.schema.json",
                    "runpod-tei-conformance-candidate.v1.schema.json",
                    "runpod-tei-conformance-result.v1.schema.json",
                    "runpod-executor-image-build-receipt.v1.schema.json",
                    "runpod-worker-observation.v1.schema.json",
                ]:
                    packaged_path = (
                        f"livefire_rag/evidence_specs/{cloud_embedding_schema}"
                    )
                    self.assertEqual(
                        archive.namelist().count(packaged_path),
                        1,
                    )
                self.assertIn(
                    "livefire_rag/evidence_specs/fast-lexical-profile.v2.json",
                    archive.namelist(),
                )
                for report_schema in [
                    "embedding-task-run-report.v1.schema.json",
                    "embedding-run-summary.v1.schema.json",
                    "embedding-task-run-report.v2.schema.json",
                    "embedding-run-summary.v2.schema.json",
                    "tei-worker-report-context.v1.schema.json",
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
                unpacked = output / "unpacked"
                archive.extractall(unpacked)

            _, packaged_schemas = _offline_registry(
                unpacked / "livefire_rag" / "evidence_specs",
                REPOSITORY.parent / "livefire-sdk" / "specs",
            )
            for schema_name in GENERIC_EVIDENCE_SCHEMA_NAMES:
                self.assertIn(schema_name, packaged_schemas)

    def test_source_distribution_contains_external_schema_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                ["uv", "build", "--sdist", "--out-dir", str(output)],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            )
            [source_distribution] = output.glob("*.tar.gz")
            with tarfile.open(source_distribution, "r:gz") as archive:
                names = archive.getnames()
            for schema_path in [
                "crates/rag-pipeline/schema/runpod-executor-image-build-receipt.v1.schema.json",
                "crates/rag-runpod-worker/schema/runpod-worker-observation.v1.schema.json",
            ]:
                self.assertEqual(
                    sum(name.endswith(schema_path) for name in names),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
