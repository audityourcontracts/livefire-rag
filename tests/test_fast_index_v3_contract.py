from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from livefire_rag.canonical import canonical_sha256_omitting
from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"
FIXTURE = REPOSITORY / "rust-fixtures/index/fast-index-manifest.v3.json"


class FastIndexV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        cls.v2 = Draft202012Validator(
            schemas["fast-index-manifest.v2.schema.json"], registry=registry
        )
        cls.v3 = Draft202012Validator(
            schemas["fast-index-manifest.v3.schema.json"], registry=registry
        )
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_real_rust_v3_manifest_validates_offline_with_exact_identity(self) -> None:
        self.v3.validate(self.fixture)
        self.assertEqual(
            self.fixture["component_sha256"],
            canonical_sha256_omitting(self.fixture, ("component_sha256",)),
        )
        self.assertEqual(
            self.fixture["lexical"]["schema"], "sqlite-inverted-bm25-v1"
        )

    def test_v2_and_v3_remain_distinct_contracts(self) -> None:
        with self.assertRaises(ValidationError):
            self.v2.validate(self.fixture)

        v2_fixture = copy.deepcopy(self.fixture)
        v2_fixture["schema_version"] = "livefire.rag.fast-index/2"
        v2_fixture["lexical"].pop("schema")
        with self.assertRaises(ValidationError):
            self.v3.validate(v2_fixture)

    def test_v3_requires_the_exact_lexical_storage_schema(self) -> None:
        missing = copy.deepcopy(self.fixture)
        missing["lexical"].pop("schema")
        with self.assertRaises(ValidationError):
            self.v3.validate(missing)

        wrong = copy.deepcopy(self.fixture)
        wrong["lexical"]["schema"] = "sqlite-inverted-bm25-v2"
        with self.assertRaises(ValidationError):
            self.v3.validate(wrong)

    def test_v3_accepts_exact_optional_pipeline_provenance(self) -> None:
        bound = copy.deepcopy(self.fixture)
        bound["pipeline_provenance"] = {
            "dataset_sha256": "1" * 64,
            "prepared_corpus_sha256": "2" * 64,
            "embedding_plan_sha256": "3" * 64,
            "embedding_result_set_sha256": "4" * 64,
        }
        bound["component_sha256"] = canonical_sha256_omitting(
            bound, ("component_sha256",)
        )
        self.v3.validate(bound)

        incomplete = copy.deepcopy(bound)
        incomplete["pipeline_provenance"].pop("embedding_result_set_sha256")
        with self.assertRaises(ValidationError):
            self.v3.validate(incomplete)


if __name__ == "__main__":
    unittest.main()
