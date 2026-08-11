#!/usr/bin/env python3
"""Check frozen standalone-provider POC behaviors without model dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class CheckError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckError(f"{path}: {error}") from error


def provider_calls(value: Any) -> list[dict[str, Any]]:
    calls = value.get("calls") if isinstance(value, dict) else value
    if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
        raise CheckError("results must be an array, or an object with a calls array")
    return calls


def provider_output(call: dict[str, Any]) -> dict[str, Any]:
    response = call.get("response", call.get("output"))
    if not isinstance(response, dict):
        raise CheckError(f"{call.get('case_id')}: missing response/output object")
    for _ in range(4):
        if isinstance(response.get("tool"), str):
            return response
        if isinstance(response.get("output"), dict):
            response = response["output"]
            continue
        if isinstance(response.get("result"), dict):
            response = response["result"]
            continue
        break
    raise CheckError(f"{call.get('case_id')}: cannot locate provider tool output")


def candidates_for(call: dict[str, Any], expected_tool: str) -> list[dict[str, Any]]:
    response = provider_output(call)
    if response.get("tool") != expected_tool:
        raise CheckError(
            f"{call.get('case_id')}: expected tool {expected_tool}, got {response.get('tool')}"
        )
    if response.get("kind") == "miss":
        candidates: list[dict[str, Any]] = []
    elif isinstance(response.get("pointers"), list):
        candidates = response["pointers"]
    else:
        candidates = legacy_ranking_candidates(call, response, expected_tool)
    ranks = [row.get("rank") for row in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        raise CheckError(f"{call.get('case_id')}: candidate ranks must be contiguous from 1")
    command_ids = [row.get("command_id") for row in candidates]
    if any(not isinstance(value, str) or not value for value in command_ids):
        raise CheckError(f"{call.get('case_id')}: every candidate needs command_id")
    if len(command_ids) != len(set(command_ids)):
        raise CheckError(f"{call.get('case_id')}: duplicate command_id")
    return candidates


def legacy_ranking_candidates(
    call: dict[str, Any], response: dict[str, Any], expected_tool: str
) -> list[dict[str, Any]]:
    expected_ranking = "semantic_search" if expected_tool == "cli.search" else "similar_command"
    rankings = response.get("rankings")
    if not isinstance(rankings, list):
        raise CheckError(
            f"{call.get('case_id')}: provider output needs pointers or legacy rankings"
        )
    matches = [row for row in rankings if row.get("ranking") == expected_ranking]
    if len(matches) != 1 or not isinstance(matches[0].get("candidates"), list):
        raise CheckError(f"{call.get('case_id')}: expected one {expected_ranking} ranking")
    return matches[0]["candidates"]


def candidate_text(candidate: dict[str, Any], fields: list[str]) -> str:
    return "\n".join(str(candidate.get(field, "")) for field in fields).lower()


def matcher_matches(text: str, matcher: dict[str, Any]) -> bool:
    all_terms = [str(term).lower() for term in matcher.get("all", [])]
    any_terms = [str(term).lower() for term in matcher.get("any", [])]
    none_terms = [str(term).lower() for term in matcher.get("none", [])]
    return (
        all(term in text for term in all_terms)
        and (not any_terms or any(term in text for term in any_terms))
        and not any(term in text for term in none_terms)
    )


def evaluate_rule(
    rule: dict[str, Any], candidates: list[dict[str, Any]], fields: list[str]
) -> dict[str, Any]:
    max_rank = int(rule.get("max_rank", len(candidates)))
    hits = [
        candidate["rank"]
        for candidate in candidates
        if candidate["rank"] <= max_rank
        and matcher_matches(candidate_text(candidate, fields), rule["matcher"])
    ]
    minimum = int(rule.get("min_matches", 1))
    return {
        "label": rule.get("label"),
        "passed": len(hits) >= minimum,
        "matched_ranks": hits,
        "observed_matches": len(hits),
        "required_matches": minimum,
        "max_rank": max_rank,
    }


def evaluate_case(
    case: dict[str, Any], call: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    candidates = candidates_for(call, case["tool"])
    behaviors = [evaluate_rule(rule, candidates, fields) for rule in case.get("behaviors", [])]
    negatives = [evaluate_rule(rule, candidates, fields) for rule in case.get("hard_negatives", [])]
    required_negatives = [
        result
        for rule, result in zip(case.get("hard_negatives", []), negatives)
        if rule.get("mandatory", False)
    ]
    boundary_result = None
    checks = [row["passed"] for row in behaviors] + [row["passed"] for row in required_negatives]
    if case.get("boundary"):
        boundary = case["boundary"]
        present = evaluate_rule(
            {"label": "required_present", **boundary["required_present"]}, candidates, fields
        )
        absent_probe = evaluate_rule(
            {"label": "required_absent", **boundary["required_absent"]}, candidates, fields
        )
        absent = {**absent_probe, "passed": absent_probe["observed_matches"] == 0}
        boundary_result = {
            "label": boundary["label"],
            "passed": present["passed"] and absent["passed"],
            "required_present": present,
            "required_absent": absent,
            "follow_up": boundary["follow_up"],
        }
        checks.append(boundary_result["passed"])
    passed = all(checks)
    if case["expected_disposition"] == "expected_boundary_failure":
        outcome = "expected_boundary_failure" if passed else "boundary_not_reproduced"
    else:
        outcome = "pass" if passed else "fail"
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "tool": case["tool"],
        "expected_disposition": case["expected_disposition"],
        "outcome": outcome,
        "passed": passed,
        "returned_candidates": len(candidates),
        "behaviors": behaviors,
        "hard_negatives": negatives,
        "boundary": boundary_result,
    }


def build_report(
    suite: dict[str, Any], raw_results: Any, scope: str = "all"
) -> dict[str, Any]:
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise CheckError("suite.cases must be an array")
    all_declared_ids = {case["case_id"] for case in cases}
    if scope not in {"all", "acceptance", "diagnostic"}:
        raise CheckError(f"unsupported scope: {scope}")
    if scope != "all":
        cases = [case for case in cases if case.get("kind") == scope]
    declared = {case["case_id"]: case for case in cases}
    calls = provider_calls(raw_results)
    observed: dict[str, dict[str, Any]] = {}
    for call in calls:
        case_id = call.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise CheckError("every call needs a case_id")
        if case_id in observed:
            raise CheckError(f"duplicate result for {case_id}")
        observed[case_id] = call
    missing = sorted(set(declared) - set(observed))
    unknown = sorted(set(observed) - all_declared_ids)
    ignored_out_of_scope = sorted((set(observed) & all_declared_ids) - set(declared))
    fields = suite.get("policy", {}).get("candidate_text_fields", ["preview", "command_id"])
    results = [
        evaluate_case(case, observed[case["case_id"]], fields)
        for case in cases
        if case["case_id"] in observed
    ]
    primary = [row for row in results if row["kind"] == "acceptance"]
    diagnostics = [row for row in results if row["kind"] == "diagnostic"]
    passed = not missing and not unknown and all(row["passed"] for row in results)
    return {
        "schema_version": "livefire.rag.provider-poc-acceptance-report/1",
        "suite_id": suite.get("suite_id"),
        "scope": scope,
        "status": "pass" if passed else "fail",
        "non_cherry_pick": {
            "declared_cases": len(cases),
            "observed_cases": len(observed),
            "missing_case_ids": missing,
            "unknown_case_ids": unknown,
            "ignored_out_of_scope_case_ids": ignored_out_of_scope,
            "complete": not missing and not unknown,
        },
        "summary": {
            "acceptance_passed": sum(row["passed"] for row in primary),
            "acceptance_total": len([case for case in cases if case["kind"] == "acceptance"]),
            "diagnostics_passed": sum(row["passed"] for row in diagnostics),
            "diagnostics_total": len([case for case in cases if case["kind"] == "diagnostic"]),
            "expected_boundary_failures_observed": sum(
                row["outcome"] == "expected_boundary_failure" for row in results
            ),
            "mandatory_hard_negatives_observed": sum(
                item["passed"] for row in results for item in row["hard_negatives"]
            ),
        },
        "cases": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Standalone provider POC acceptance",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "| Case | Kind | Expected disposition | Observed outcome | Pass |",
        "|---|---|---|---|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['kind']} | {row['expected_disposition']} | "
            f"{row['outcome']} | {'yes' if row['passed'] else 'no'} |"
        )
    coverage = report["non_cherry_pick"]
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Declared cases: {coverage['declared_cases']}",
            f"- Observed cases: {coverage['observed_cases']}",
            f"- Missing: {', '.join(coverage['missing_case_ids']) or 'none'}",
            f"- Unknown: {', '.join(coverage['unknown_case_ids']) or 'none'}",
            "",
            "Q9 is successful only when its predeclared flat-query facet-collapse boundary "
            "is reproduced and explicitly reported as `expected_boundary_failure`. It is "
            "not counted as successful evidence retrieval.",
            "",
        ]
    )
    lines.extend(
        [
            "## Observed checks",
            "",
            "| Case | Check class | Label | Matched ranks | Pass |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in report["cases"]:
        for check_class, checks in (
            ("behavior", row["behaviors"]),
            ("hard negative", row["hard_negatives"]),
        ):
            for check in checks:
                ranks = ", ".join(str(rank) for rank in check["matched_ranks"]) or "none"
                lines.append(
                    f"| {row['case_id']} | {check_class} | {check['label']} | {ranks} | "
                    f"{'yes' if check['passed'] else 'no'} |"
                )
        if row["boundary"]:
            boundary = row["boundary"]
            present = ", ".join(
                str(rank) for rank in boundary["required_present"]["matched_ranks"]
            ) or "none"
            absent = ", ".join(
                str(rank) for rank in boundary["required_absent"]["matched_ranks"]
            ) or "none"
            lines.append(
                f"| {row['case_id']} | boundary present | {boundary['label']} | {present} | "
                f"{'yes' if boundary['required_present']['passed'] else 'no'} |"
            )
            lines.append(
                f"| {row['case_id']} | boundary absent | {boundary['label']} | {absent} | "
                f"{'yes' if boundary['required_absent']['passed'] else 'no'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--scope", choices=("all", "acceptance", "diagnostic"), default="all")
    args = parser.parse_args()
    try:
        report = build_report(load_json(args.suite), load_json(args.results), args.scope)
    except CheckError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
