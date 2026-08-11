#!/usr/bin/env python3
"""Build answer-free preparation artifacts for the 23-cloud/53-BOTS suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ELIGIBLE_ASSIST_MODES = {
    "direct_single",
    "direct_multi",
    "retrieve_then_compute",
    "exact_metadata",
}
REQUIRED_SURFACES = ["analyst_question", "terse_soc", "entity_light_paraphrase"]
NEXT_RECEIPTS = [
    "query_text_lock",
    "leakage_audit",
    "candidate_universe",
    "qrel_adjudication",
    "hard_negative_adjudication",
]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_ledger(plan: dict) -> dict:
    atoms = []
    for source in plan["atoms"]:
        assist_mode = source["preliminary_eligibility"]
        if assist_mode == "needs_review":
            raise ValueError(f"{source['atom_id']}: needs_review is not terminal")

        if source["benchmark_id"] == "botsv3-cloud-incident-chain":
            cohort = "cloud"
        elif assist_mode == "external_enrichment":
            cohort = "external"
        else:
            cohort = "bots_native"

        if assist_mode in ELIGIBLE_ASSIST_MODES:
            eligibility = "eligible_native"
            reason_codes = None
        elif assist_mode == "external_enrichment":
            eligibility = "external_source_unbound"
            reason_codes = ["external_snapshot_unbound"]
        elif assist_mode == "outside_current_index":
            eligibility = "outside_index_domain"
            reason_codes = ["outside_command_script_cloud_action_domain"]
        else:
            raise ValueError(f"{source['atom_id']}: unsupported assist mode {assist_mode}")

        atom = {
            "benchmark_id": source["benchmark_id"],
            "atom_id": source["atom_id"],
            "eligibility": eligibility,
            "cohort": cohort,
            "resampling_cluster_id": f"incident:{source['domain']}",
            "incident_id": source["domain"],
        }
        if reason_codes:
            atom["reason_codes"] = reason_codes
        atoms.append(atom)

    counts = Counter(atom["benchmark_id"] for atom in atoms)
    return {
        "schema_version": "livefire.rag.evidence-eligibility-ledger/1",
        "suite_contract": "livefire-23-cloud-53-bots-v1",
        "summary": {
            "total_atoms": len(atoms),
            "by_benchmark": dict(sorted(counts.items())),
        },
        "atoms": atoms,
    }


def build_query_worklist(plan: dict, plan_digest: str) -> dict:
    atoms = []
    for source in plan["atoms"]:
        assist_mode = source["preliminary_eligibility"]
        if assist_mode not in ELIGIBLE_ASSIST_MODES:
            continue
        cohort = "cloud" if source["benchmark_id"] == "botsv3-cloud-incident-chain" else "bots_native"
        atoms.append(
            {
                "benchmark_id": source["benchmark_id"],
                "atom_id": source["atom_id"],
                "question_id": source["question_id"],
                "domain": source["domain"],
                "cohort": cohort,
                "incident_id": source["domain"],
                "resampling_cluster_id": f"incident:{source['domain']}",
                "phase_id": source["phase_id"],
                "scope_ids": source["scopes"],
                "assist_mode": assist_mode,
                "safe_summary": source["title"],
                "answer_free_atom_sha256": sha256_bytes(canonical_bytes(source)),
                "required_surfaces": REQUIRED_SURFACES,
                "authoring_status": "pending_blinded_author",
                "next_receipts": NEXT_RECEIPTS,
            }
        )

    by_cohort = Counter(atom["cohort"] for atom in atoms)
    by_assist_mode = Counter(atom["assist_mode"] for atom in atoms)
    return {
        "schema_version": "livefire.rag.evidence-query-authoring-worklist/1",
        "suite_contract": "livefire-23-cloud-53-bots-v1",
        "status": "awaiting_independent_authoring",
        "coverage_plan_sha256": plan_digest,
        "policy": {
            "required_surfaces": REQUIRED_SURFACES,
            "answer_values_visible_to_author": False,
            "candidate_rankings_visible_to_author": False,
            "forbidden_literal_classes": [
                "event_id",
                "exact_timestamp",
                "principal_or_host",
                "resource_identifier",
                "hash",
                "answer_only_command_fragment",
                "evaluator_conclusion",
            ],
            "activation_requirements": NEXT_RECEIPTS,
        },
        "summary": {
            "eligible_atoms": len(atoms),
            "required_surfaces": len(atoms) * len(REQUIRED_SURFACES),
            "by_cohort": dict(sorted(by_cohort.items())),
            "by_assist_mode": dict(sorted(by_assist_mode.items())),
        },
        "atoms": atoms,
    }


def write_object(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("fixtures/fact-evidence-coverage-plan.v1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("fixtures/fact-evidence-real"))
    args = parser.parse_args()

    plan_bytes = args.plan.read_bytes()
    plan = load_object(args.plan)
    declared = plan["summary"]["total_atoms"]
    if declared != 76 or len(plan["atoms"]) != 76:
        raise ValueError("livefire-23-cloud-53-bots-v1 requires exactly 76 atoms")
    if sum(plan["summary"]["by_preliminary_eligibility"].values()) != declared:
        raise ValueError("preliminary eligibility summary does not reconcile")

    ledger = build_ledger(plan)
    worklist = build_query_worklist(plan, sha256_bytes(plan_bytes))
    write_object(args.out_dir / "eligibility-ledger.json", ledger)
    write_object(args.out_dir / "query-authoring-worklist.json", worklist)
    print(
        f"wrote {len(ledger['atoms'])} terminal ledger atoms and "
        f"{worklist['summary']['required_surfaces']} pending query surfaces"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
