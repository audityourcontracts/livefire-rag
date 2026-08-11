#!/usr/bin/env python3
"""Validate evidence benchmark schemas, fixtures, and a comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-specs",
        type=Path,
        default=Path("../livefire-sdk/specs"),
        help="directory containing livefire-sdk JSON schemas",
    )
    parser.add_argument("--report", type=Path, help="optional compact comparison report")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--candidate-universes", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--hard-negatives", type=Path)
    parser.add_argument("--candidate-rankings", type=Path)
    parser.add_argument("--baseline-rankings", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sdk_specs = args.sdk_specs.resolve()
    schema_paths = [*sorted((root / "specs").glob("*.json")), *sorted(sdk_specs.glob("*.json"))]
    if not sdk_specs.is_dir():
        raise ValueError(f"SDK schema directory does not exist: {sdk_specs}")

    registry = Registry()
    schemas: dict[Path, dict] = {}
    for path in schema_paths:
        schema = load_json(path)
        Draft202012Validator.check_schema(schema)
        schemas[path.resolve()] = schema
        if schema.get("$id"):
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))

    fixture_dir = root / "fixtures/fact-evidence-synthetic"
    inventory_path = args.inventory or fixture_dir / "inventory.json"
    ledger_schema = schemas[(root / "specs/evidence-eligibility-ledger.v1.schema.json").resolve()]
    Draft202012Validator(ledger_schema, registry=registry).validate(
        load_json(inventory_path)
    )
    checks = [
        ("evidence-query-row.v1.schema.json", args.queries or fixture_dir / "queries.jsonl"),
        ("evidence-candidate-universe-row.v1.schema.json", args.candidate_universes or fixture_dir / "candidate-universes.jsonl"),
        ("evidence-qrel-row.v1.schema.json", args.qrels or fixture_dir / "qrels.jsonl"),
        ("evidence-hard-negative-row.v1.schema.json", args.hard_negatives or fixture_dir / "hard-negatives.jsonl"),
        ("evidence-ranking-row.v1.schema.json", args.candidate_rankings or fixture_dir / "candidate-rankings.jsonl"),
        ("evidence-ranking-row.v1.schema.json", args.baseline_rankings or fixture_dir / "baseline-rankings.jsonl"),
    ]
    for schema_name, fixture_path in checks:
        schema = schemas[(root / "specs" / schema_name).resolve()]
        validator = Draft202012Validator(schema, registry=registry)
        for line_number, line in enumerate(fixture_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    validator.validate(json.loads(line))
                except Exception as error:
                    raise ValueError(f"{fixture_path}:{line_number}: {error}") from error

    plan = load_json(root / "fixtures/fact-evidence-coverage-plan.v1.json")
    if len(plan["atoms"]) != 76:
        raise ValueError("coverage plan must contain exactly 76 atoms")
    if sum(plan["summary"]["by_benchmark"].values()) != 76:
        raise ValueError("coverage plan benchmark cohorts do not reconcile to 76")
    if sum(plan["summary"]["by_preliminary_eligibility"].values()) != 76:
        raise ValueError("coverage plan eligibility classes do not reconcile to 76")

    if args.report:
        comparison = schemas[(root / "specs/evidence-benchmark-comparison.v1.schema.json").resolve()]
        Draft202012Validator(comparison, registry=registry).validate(load_json(args.report))

    print(f"validated {len(schemas)} schemas, the eligibility ledger, 6 fixture sets, and the 76-atom coverage plan")
    if args.report:
        print(f"validated comparison report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
