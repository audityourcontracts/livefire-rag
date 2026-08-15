from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


class TokenizerParityContractTests(unittest.TestCase):
    def test_pinned_gguf_parity_fixture_validates(self) -> None:
        schema = json.loads(
            (ROOT / "specs/tokenizer-parity-fixture.v1.schema.json").read_text()
        )
        fixture = json.loads(
            (ROOT / "fixtures/qwen3-embedding-8b-tokenizer-parity.v1.json").read_text()
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(fixture)

        names = [case["name"] for case in fixture["cases"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("decomposed_unicode", names)
        [boundary] = fixture["generated_cases"]
        self.assertEqual(boundary["count"], 16_384)


if __name__ == "__main__":
    unittest.main()
