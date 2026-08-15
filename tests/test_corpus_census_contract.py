from __future__ import annotations

import copy
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from livefire_rag.canonical import canonical_sha256_omitting
from livefire_rag.evidence_schema import _offline_registry


REPOSITORY = Path(__file__).resolve().parents[1]
SDK_SPECS = REPOSITORY.parent / "livefire-sdk" / "specs"


def _component(component_id: str, fill: str) -> dict[str, str]:
    return {"id": component_id, "version": "1", "sha256": fill * 64}


def _report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "livefire.rag.corpus-census/1",
        "component_sha256": "0" * 64,
        "source_snapshot": _component("test.snapshot", "1"),
        "mapping": _component("test.mapping", "2"),
        "projection_policy": _component("test.projection-policy", "3"),
        "relations_counted": ["ocsf_process_activity"],
        "source_rows": 4,
        "semantic_occurrences": 3,
        "structured_only_occurrences": 1,
        "distinct_documents": 2,
        "document_order_sha256": "4" * 64,
        "document_kinds": {"activity": 2},
        "relations": {
            "ocsf_process_activity": {
                "source_rows": 4,
                "semantic_occurrences": 3,
                "structured_only_occurrences": 1,
                "distinct_documents": 2,
                "document_kinds": {"activity": 2},
            }
        },
    }
    report["component_sha256"] = canonical_sha256_omitting(
        report, ("component_sha256",)
    )
    return report


class CorpusCensusContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        registry, schemas = _offline_registry(REPOSITORY / "specs", SDK_SPECS)
        cls.validator = Draft202012Validator(
            schemas["corpus-census.v1.schema.json"], registry=registry
        )

    def test_complete_report_validates_and_uses_standard_component_digest(self) -> None:
        report = _report()
        self.validator.validate(report)
        self.assertEqual(
            report["component_sha256"],
            canonical_sha256_omitting(report, ("component_sha256",)),
        )

    def test_report_rejects_old_non_reproducible_digest_field(self) -> None:
        report = _report()
        report["report_material_sha256"] = report.pop("component_sha256")
        with self.assertRaises(ValidationError):
            self.validator.validate(report)

    def test_report_rejects_unknown_document_kind_and_extra_relation_field(self) -> None:
        bad_kind = _report()
        bad_kind["document_kinds"] = {"structured_only": 2}
        with self.assertRaises(ValidationError):
            self.validator.validate(bad_kind)

        extra_field = copy.deepcopy(_report())
        relation = extra_field["relations"]["ocsf_process_activity"]
        relation["embedded_documents"] = 2
        with self.assertRaises(ValidationError):
            self.validator.validate(extra_field)


if __name__ == "__main__":
    unittest.main()
