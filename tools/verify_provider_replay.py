#!/usr/bin/env python3
"""Prove SDK-replayed call outputs exactly equal the in-process demo outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from livefire_rag.schema_validation import validate_semantic_result


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_output(response: dict[str, Any], label: str) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, dict) or result.get("response_kind") != "call":
        raise ValueError(f"{label}: expected a successful call response")
    output = result.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"{label}: call output is not an object")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-results", type=Path, required=True)
    parser.add_argument("--sdk-wire", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotate-demo", action="store_true")
    args = parser.parse_args()

    demo_digest = sha256_file(args.demo_results)
    sdk_digest = sha256_file(args.sdk_wire)
    demo = json.loads(args.demo_results.read_text(encoding="utf-8"))
    calls = demo.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("demo results contain no calls")
    wire = [json.loads(line) for line in args.sdk_wire.read_text(encoding="utf-8").splitlines() if line.strip()]
    wire_calls = [response for response in wire if response.get("result", {}).get("response_kind") == "call"]
    if len(wire_calls) != len(calls):
        raise ValueError(f"call count mismatch: demo={len(calls)} sdk={len(wire_calls)}")

    comparisons = []
    all_equal = True
    for position, (demo_call, sdk_response) in enumerate(zip(calls, wire_calls), 1):
        case_id = demo_call.get("case_id")
        demo_output = call_output(demo_call.get("response", {}), f"demo {case_id}")
        sdk_output = call_output(sdk_response, f"SDK response {sdk_response.get('id')}")
        validate_semantic_result(demo_output)
        validate_semantic_result(sdk_output)
        equal = demo_output == sdk_output
        all_equal &= equal
        comparisons.append(
            {
                "position": position,
                "case_id": case_id,
                "sdk_request_id": sdk_response.get("id"),
                "equal": equal,
                "demo_output_sha256": hashlib.sha256(canonical(demo_output)).hexdigest(),
                "sdk_output_sha256": hashlib.sha256(canonical(sdk_output)).hexdigest(),
            }
        )
    report = {
        "schema_version": "livefire.rag.provider-replay-verification/1",
        "status": "pass" if all_equal else "fail",
        "comparison": "exact parsed JSON equality plus canonical-output SHA-256 equality",
        "demo_results": {"path": str(args.demo_results), "sha256_before_annotation": demo_digest},
        "sdk_wire": {"path": str(args.sdk_wire), "sha256": sdk_digest},
        "expected_case_order": [call["case_id"] for call in calls],
        "compared_calls": len(comparisons),
        "all_outputs_equal": all_equal,
        "schema_validation": {
            "status": "pass",
            "registry": "offline livefire-sdk plus livefire-rag schemas",
            "validated_outputs": len(comparisons) * 2,
        },
        "calls": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(report) + b"\n")
    if args.annotate_demo and all_equal:
        demo["sdk_replay_verification"] = {
            "status": "pass",
            "comparison": report["comparison"],
            "compared_calls": len(comparisons),
            "report": str(args.out),
            "report_sha256": sha256_file(args.out),
            "sdk_wire_sha256": sdk_digest,
        }
        args.demo_results.write_bytes(canonical(demo) + b"\n")
    print(json.dumps({"status": report["status"], "compared_calls": len(comparisons), "out": str(args.out)}, sort_keys=True))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
