"""Deterministic, scenario-blind sampling of a sealed evidence projection pack.

The artifact produced here is an evaluation sample, never a projection pack or
an admission receipt.  Selection observes structural metadata only and retains
every occurrence for each selected semantic document group.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from .canonical import (
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256_omitting,
    component_ref,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)
from .evidence_builder import evidence_manifest_identity


MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
OCCURRENCES_NAME = "occurrences.jsonl"
SELECTION_NAME = "selection.jsonl"
POLICY_NAME = "sampling-policy.json"
COVERAGE_NAME = "coverage-report.json"
LOCK_NAME = "objects.lock.json"
SCOPE_STATUS = "sample_only_not_corpus_coverage"
ADMISSION_STATUS = "local_evaluation_only_not_sdk_admitted"
COUNT_SEMANTICS = {
    "source_documents": "all_searchable_semantic_document_groups_in_projection_pack",
    "source_occurrences": "all_projection_pack_occurrences_including_structured_only",
    "selected_occurrences": "all_occurrences_attached_to_selected_documents",
}


class EvidencePilotError(RuntimeError):
    """A pilot sample cannot be built or verified faithfully."""


def _repository_policy() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "specs" / "evidence-pilot-sampling-policy.v1.json"
    if not path.is_file():
        path = Path(__file__).resolve().parent / "evidence_specs" / path.name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidencePilotError("pilot sampling policy is invalid")
    return value


def sampling_policy_ref(policy: dict[str, Any] | None = None) -> dict[str, str]:
    return component_ref(
        "livefire.rag.evidence-pilot-sampling-policy", "1", policy or _repository_policy()
    )


def _canonical_rows(path: Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidencePilotError(f"{path.name}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict) or raw != canonical_json_bytes(value, newline=True):
                raise EvidencePilotError(f"{path.name}:{line_number}: non-canonical JSON")
            yield raw, value


def _bucket(count: int) -> str:
    if count < 1:
        raise EvidencePilotError("document occurrence_count must be positive")
    lower = 1 << (count.bit_length() - 1)
    return f"{lower}-{lower * 2 - 1}"


def _structure(document: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    identities = document.get("relation_identities")
    if not isinstance(identities, list) or len(identities) != 1:
        raise EvidencePilotError("pilot requires each base document to bind exactly one relation")
    relation = identities[0].get("relation") if isinstance(identities[0], dict) else None
    facets = document.get("semantic_projection", {}).get("facets")
    if not isinstance(relation, str) or not relation or not isinstance(facets, list):
        raise EvidencePilotError("document lacks structural pilot fields")
    names = tuple(sorted({facet.get("name") for facet in facets if isinstance(facet, dict)}))
    if any(not isinstance(name, str) or not name for name in names):
        raise EvidencePilotError("document contains an invalid facet name")
    count = document.get("occurrence_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise EvidencePilotError("document occurrence_count is invalid")
    return relation, names, _bucket(count)


def _largest_remainder(populations: dict[tuple[str, tuple[str, ...]], int], total: int) -> dict[tuple[str, tuple[str, ...]], int]:
    size = sum(populations.values())
    if total >= size:
        return dict(populations)
    keys = sorted(populations, key=lambda item: (item[0], item[1]))
    quotas = {key: (populations[key] * total) // size for key in keys}
    # Every structural stratum remains represented when the relation budget permits.
    if len(keys) <= total:
        for key in keys:
            quotas[key] = max(1, quotas[key])
    while sum(quotas.values()) > total:
        candidates = [key for key in keys if quotas[key] > 1]
        if not candidates:
            raise EvidencePilotError("pilot relation budget cannot represent all strata")
        key = min(candidates, key=lambda item: (
            (populations[item] * total) % size, item[0], item[1]
        ))
        quotas[key] -= 1
    while sum(quotas.values()) < total:
        candidates = [key for key in keys if quotas[key] < populations[key]]
        key = max(candidates, key=lambda item: (
            (populations[item] * total) % size, tuple(-ord(c) for c in repr(item))
        ))
        quotas[key] += 1
    return quotas


def build_evidence_pilot_sample(
    projection_pack: Path,
    output_dir: Path,
    *,
    component_id: str,
    version: str,
    component_uri: str | None = None,
    sdk_specs: Path | None = None,
) -> dict[str, Any]:
    """Seal a bounded structural sample without mutating or relabelling its source."""

    pack = Path(projection_pack)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite pilot sample: {out}")
    pack_manifest = json.loads((pack / MANIFEST_NAME).read_text(encoding="utf-8"))
    if evidence_manifest_identity(pack_manifest) != pack_manifest.get("component", {}).get("sha256"):
        raise EvidencePilotError("projection pack component identity is invalid")
    for role, name in (("documents", DOCUMENTS_NAME), ("occurrences", OCCURRENCES_NAME)):
        ref = pack_manifest.get("objects", {}).get(role, {})
        path = pack / name
        if ref.get("path") != name or not path.is_file() or ref.get("bytes") != path.stat().st_size or ref.get("sha256") != sha256_file(path):
            raise EvidencePilotError(f"projection pack object is not sealed: {name}")

    policy = _repository_policy()
    policy_ref = sampling_policy_ref(policy)
    census_limit = int(policy["relation_frame"]["census_at_or_below"])
    sample_limit = int(policy["relation_frame"]["sample_above"])
    records: list[tuple[str, tuple[str, tuple[str, ...]], str, bytes, dict[str, Any]]] = []
    populations: dict[str, dict[tuple[str, tuple[str, ...]], int]] = defaultdict(lambda: defaultdict(int))
    source_occurrences = pack_manifest.get("closure", {}).get("source_record_count")
    if isinstance(source_occurrences, bool) or not isinstance(source_occurrences, int):
        raise EvidencePilotError("projection pack source occurrence count is invalid")
    for raw, document in _canonical_rows(pack / DOCUMENTS_NAME):
        relation, facets, bucket = _structure(document)
        stratum = (bucket, facets)
        populations[relation][stratum] += 1
        rank = sha256_bytes(canonical_json_bytes({
            "schema_version": "livefire.rag.evidence-pilot-rank/1",
            "projection_pack": pack_manifest["component"],
            "sampling_policy": policy_ref,
            "relation": relation,
            "occurrence_count_bucket": bucket,
            "facet_name_pattern": list(facets),
            "document_id": document["document_id"],
        }))
        records.append((relation, stratum, rank, raw, document))

    quotas: dict[str, dict[tuple[str, tuple[str, ...]], int]] = {}
    for relation, strata in populations.items():
        relation_total = sum(strata.values())
        quotas[relation] = (
            dict(strata) if relation_total <= census_limit
            else _largest_remainder(dict(strata), min(sample_limit, relation_total))
        )
    by_stratum: dict[tuple[str, tuple[str, tuple[str, ...]]], list[tuple[str, bytes, dict[str, Any]]]] = defaultdict(list)
    for relation, stratum, rank, raw, document in records:
        by_stratum[(relation, stratum)].append((rank, raw, document))

    selected: dict[str, tuple[bytes, dict[str, Any], dict[str, Any]]] = {}
    relation_stats: dict[str, dict[str, Any]] = {}
    for relation in sorted(populations):
        relation_total = sum(populations[relation].values())
        census = relation_total <= census_limit
        relation_stats[relation] = {
            "relation": relation, "source_documents": relation_total,
            "selected_documents": 0, "selected_occurrences": 0,
            "frame": "census" if census else "stratified_hash_min",
        }
        for stratum in sorted(populations[relation], key=lambda item: (item[0], item[1])):
            population = populations[relation][stratum]
            quota = quotas[relation][stratum]
            rows = sorted(by_stratum[(relation, stratum)], key=lambda item: (item[0], item[2]["document_id"]))[:quota]
            for rank, raw, document in rows:
                selection = {
                    "schema_version": "livefire.rag.evidence-pilot-selection-row/1",
                    "document_id": document["document_id"], "relation": relation,
                    "stratum": {"occurrence_count_bucket": stratum[0], "facet_name_pattern": list(stratum[1])},
                    "selection_reason": "relation_census" if census else "relation_stratified_hash_min",
                    "rank_sha256": rank, "stratum_population": population, "stratum_quota": quota,
                    "inclusion_probability": {"numerator": quota, "denominator": population},
                    "sampling_weight": {"numerator": population, "denominator": quota},
                }
                selected[document["document_id"]] = (raw, document, selection)
                relation_stats[relation]["selected_documents"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        with (staging / DOCUMENTS_NAME).open("wb") as docs, (staging / SELECTION_NAME).open("wb") as selections:
            for document_id in sorted(selected):
                raw, _, selection = selected[document_id]
                docs.write(raw)
                selections.write(canonical_json_bytes(selection, newline=True))
        selected_occurrences = 0
        seen_counts: dict[str, int] = defaultdict(int)
        with (staging / OCCURRENCES_NAME).open("wb") as occurrences:
            for raw, occurrence in _canonical_rows(pack / OCCURRENCES_NAME):
                ids = occurrence.get("document_ids")
                if not isinstance(ids, list):
                    raise EvidencePilotError("occurrence document_ids is invalid")
                retained = [document_id for document_id in ids if document_id in selected]
                if not retained:
                    continue
                if retained != ids:
                    raise EvidencePilotError("base occurrence contains mixed selected/unselected groups")
                occurrences.write(raw)
                selected_occurrences += 1
                relation = occurrence["relation_identity"]["relation"]
                relation_stats[relation]["selected_occurrences"] += 1
                for document_id in retained:
                    seen_counts[document_id] += 1
        for document_id, (_, document, _) in selected.items():
            if seen_counts[document_id] != document["occurrence_count"]:
                raise EvidencePilotError(f"selected occurrence closure failed: {document_id}")

        write_canonical_json(staging / POLICY_NAME, policy)
        source_counts = {
            "documents": len(records),
            # This is the full source projection-pack occurrence universe. The
            # document count is the searchable semantic-document universe.
            "occurrences": source_occurrences,
        }
        selected_counts = {"documents": len(selected), "occurrences": selected_occurrences}
        coverage = {
            "schema_version": "livefire.rag.evidence-pilot-coverage/1",
            "scope_status": SCOPE_STATUS, "admission_status": ADMISSION_STATUS,
            "count_semantics": COUNT_SEMANTICS,
            "source": source_counts, "selected": selected_counts,
            "by_relation": [relation_stats[key] for key in sorted(relation_stats)],
            "closure": {
                "preserves_all_selected_documents": True,
                "preserves_all_selected_document_occurrences": True,
                "unresolved_selected_document_count": 0,
                "unresolved_selected_occurrence_count": 0,
                "corpus_miss_definitive": False,
            },
        }
        write_canonical_json(staging / COVERAGE_NAME, coverage)
        artifacts = [
            artifact_ref(staging / DOCUMENTS_NAME, DOCUMENTS_NAME, "application/x-ndjson"),
            artifact_ref(staging / OCCURRENCES_NAME, OCCURRENCES_NAME, "application/x-ndjson"),
            artifact_ref(staging / SELECTION_NAME, SELECTION_NAME, "application/x-ndjson"),
            artifact_ref(staging / POLICY_NAME, POLICY_NAME, "application/json"),
            artifact_ref(staging / COVERAGE_NAME, COVERAGE_NAME, "application/json"),
        ]
        artifacts.sort(key=lambda item: item["path"])
        write_canonical_json(staging / LOCK_NAME, {"schema_version": "livefire.object-lock/1", "objects": artifacts})
        role = {DOCUMENTS_NAME: "documents", OCCURRENCES_NAME: "occurrences", SELECTION_NAME: "selection", POLICY_NAME: "sampling_policy", COVERAGE_NAME: "coverage_report"}
        objects = {role[item["path"]]: item for item in artifacts}
        objects["object_lock"] = artifact_ref(staging / LOCK_NAME, LOCK_NAME, "application/json")
        component = {"id": component_id, "version": version, "sha256": ""}
        if component_uri:
            component["uri"] = component_uri
        manifest = {
            "schema_version": "livefire.rag.evidence-pilot-sample/1", "component": component,
            "stage": "evaluation_sampling_pre_embedding", "scope_status": SCOPE_STATUS,
            "admission_status": ADMISSION_STATUS, "projection_pack": pack_manifest["component"],
            "sampling_policy": policy_ref, "selection_unit": "semantic_document_group",
            "count_semantics": COUNT_SEMANTICS,
            "objects": objects, "source_counts": source_counts, "selected_counts": selected_counts,
            "closure": {"preserves_all_selected_documents": True, "preserves_all_selected_document_occurrences": True, "corpus_miss_definitive": False},
        }
        manifest["component"]["sha256"] = canonical_sha256_omitting(manifest, ("component", "sha256"))
        write_canonical_json(staging / MANIFEST_NAME, manifest)
        verify_evidence_pilot_sample(staging, projection_pack=pack, sdk_specs=sdk_specs)
        os.rename(staging, out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_evidence_pilot_sample(
    root: Path, *, projection_pack: Path | None = None, sdk_specs: Path | None = None
) -> dict[str, Any]:
    """Verify sample identity, object bytes, membership, and selected closure."""

    root = Path(root)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("scope_status") != SCOPE_STATUS or manifest.get("admission_status") != ADMISSION_STATUS:
        raise EvidencePilotError("pilot scope/admission declaration is missing")
    if manifest.get("count_semantics") != COUNT_SEMANTICS:
        raise EvidencePilotError("pilot count semantics are missing")
    if canonical_sha256_omitting(manifest, ("component", "sha256")) != manifest.get("component", {}).get("sha256"):
        raise EvidencePilotError("pilot component identity mismatch")
    expected = {"documents": DOCUMENTS_NAME, "occurrences": OCCURRENCES_NAME, "selection": SELECTION_NAME, "sampling_policy": POLICY_NAME, "coverage_report": COVERAGE_NAME, "object_lock": LOCK_NAME}
    if set(manifest.get("objects", {})) != set(expected):
        raise EvidencePilotError("pilot object set mismatch")
    for role, name in expected.items():
        ref = manifest["objects"][role]
        path = root / name
        if ref.get("path") != name or not path.is_file() or ref.get("bytes") != path.stat().st_size or ref.get("sha256") != sha256_file(path):
            raise EvidencePilotError(f"pilot artifact mismatch: {name}")
    locked = [manifest["objects"][role] for role in expected if role != "object_lock"]
    locked.sort(key=lambda item: item["path"])
    if json.loads((root / LOCK_NAME).read_text()) != {"schema_version": "livefire.object-lock/1", "objects": locked}:
        raise EvidencePilotError("pilot object lock mismatch")
    policy = json.loads((root / POLICY_NAME).read_text())
    if sampling_policy_ref(policy) != manifest.get("sampling_policy"):
        raise EvidencePilotError("pilot sampling policy binding mismatch")
    selected: dict[str, int] = {}
    prior = ""
    for _, document in _canonical_rows(root / DOCUMENTS_NAME):
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or document_id <= prior or document_id in selected:
            raise EvidencePilotError("pilot documents are not unique and sorted")
        prior = document_id
        selected[document_id] = int(document["occurrence_count"])
    selection_ids = [row["document_id"] for _, row in _canonical_rows(root / SELECTION_NAME)]
    if selection_ids != sorted(selected):
        raise EvidencePilotError("pilot selection does not exactly cover selected documents")
    occurrence_counts: dict[str, int] = defaultdict(int)
    prior = ""
    occurrence_total = 0
    for _, occurrence in _canonical_rows(root / OCCURRENCES_NAME):
        occurrence_id = occurrence.get("occurrence_id")
        if not isinstance(occurrence_id, str) or occurrence_id <= prior:
            raise EvidencePilotError("pilot occurrences are not unique and sorted")
        prior = occurrence_id
        for document_id in occurrence.get("document_ids", []):
            if document_id not in selected:
                raise EvidencePilotError("pilot occurrence references an unselected document")
            occurrence_counts[document_id] += 1
        occurrence_total += 1
    if occurrence_counts != selected:
        raise EvidencePilotError("pilot selected-document occurrence closure mismatch")
    coverage = json.loads((root / COVERAGE_NAME).read_text())
    if coverage.get("scope_status") != SCOPE_STATUS or coverage.get("count_semantics") != COUNT_SEMANTICS or coverage.get("closure", {}).get("corpus_miss_definitive") is not False:
        raise EvidencePilotError("pilot coverage scope is misleading")
    if manifest.get("selected_counts") != {"documents": len(selected), "occurrences": occurrence_total} or coverage.get("selected") != manifest["selected_counts"]:
        raise EvidencePilotError("pilot selected counts do not reconcile")
    if projection_pack is not None:
        source_root = Path(projection_pack)
        pack_manifest = json.loads((source_root / MANIFEST_NAME).read_text())
        if evidence_manifest_identity(pack_manifest) != pack_manifest.get("component", {}).get("sha256"):
            raise EvidencePilotError("projection pack component identity is invalid")
        if manifest.get("projection_pack") != pack_manifest.get("component"):
            raise EvidencePilotError("pilot projection-pack binding mismatch")
        for role, name in (("documents", DOCUMENTS_NAME), ("occurrences", OCCURRENCES_NAME)):
            ref = pack_manifest.get("objects", {}).get(role, {})
            path = source_root / name
            if ref.get("path") != name or not path.is_file() or ref.get("bytes") != path.stat().st_size or ref.get("sha256") != sha256_file(path):
                raise EvidencePilotError(f"projection pack object is not sealed: {name}")
        expected_documents = {document_id: raw for raw, document in _canonical_rows(root / DOCUMENTS_NAME) for document_id in [document["document_id"]]}
        expected_occurrences = {occurrence["occurrence_id"]: raw for raw, occurrence in _canonical_rows(root / OCCURRENCES_NAME)}
        policy_ref = sampling_policy_ref(policy)
        populations: dict[str, dict[tuple[str, tuple[str, ...]], int]] = defaultdict(lambda: defaultdict(int))
        ranked: dict[tuple[str, tuple[str, tuple[str, ...]]], list[tuple[str, str]]] = defaultdict(list)
        seen_documents: set[str] = set()
        source_document_count = 0
        for raw, document in _canonical_rows(source_root / DOCUMENTS_NAME):
            document_id = document["document_id"]
            relation, facets, bucket = _structure(document)
            stratum = (bucket, facets)
            populations[relation][stratum] += 1
            rank = sha256_bytes(canonical_json_bytes({
                "schema_version": "livefire.rag.evidence-pilot-rank/1",
                "projection_pack": pack_manifest["component"],
                "sampling_policy": policy_ref, "relation": relation,
                "occurrence_count_bucket": bucket, "facet_name_pattern": list(facets),
                "document_id": document_id,
            }))
            ranked[(relation, stratum)].append((rank, document_id))
            source_document_count += 1
            if document_id in expected_documents:
                if raw != expected_documents[document_id]:
                    raise EvidencePilotError("pilot document differs from sealed projection pack")
                seen_documents.add(document_id)
        seen_occurrences: set[str] = set()
        for raw, occurrence in _canonical_rows(source_root / OCCURRENCES_NAME):
            occurrence_id = occurrence["occurrence_id"]
            belongs_to_selected = any(
                document_id in expected_documents
                for document_id in occurrence.get("document_ids", [])
            )
            if belongs_to_selected and occurrence_id not in expected_occurrences:
                raise EvidencePilotError("pilot omitted an occurrence for a selected document")
            if occurrence_id in expected_occurrences:
                if raw != expected_occurrences[occurrence_id]:
                    raise EvidencePilotError("pilot occurrence differs from sealed projection pack")
                seen_occurrences.add(occurrence_id)
        if seen_documents != set(expected_documents) or seen_occurrences != set(expected_occurrences):
            raise EvidencePilotError("pilot contains rows absent from sealed projection pack")
        expected_selection: dict[str, tuple[str, tuple[str, tuple[str, ...]], int, int]] = {}
        census_limit = int(policy["relation_frame"]["census_at_or_below"])
        sample_limit = int(policy["relation_frame"]["sample_above"])
        for relation, strata in populations.items():
            relation_total = sum(strata.values())
            quotas = dict(strata) if relation_total <= census_limit else _largest_remainder(dict(strata), min(sample_limit, relation_total))
            for stratum, population in strata.items():
                quota = quotas[stratum]
                for rank, document_id in sorted(ranked[(relation, stratum)])[:quota]:
                    expected_selection[document_id] = (rank, stratum, population, quota)
        actual_rows = {row["document_id"]: row for _, row in _canonical_rows(root / SELECTION_NAME)}
        if set(actual_rows) != set(expected_selection):
            raise EvidencePilotError("pilot selection does not replay from bound sampling policy")
        for document_id, (rank, stratum, population, quota) in expected_selection.items():
            row = actual_rows[document_id]
            expected_reason = "relation_census" if sum(populations[row["relation"]].values()) <= census_limit else "relation_stratified_hash_min"
            if (
                row.get("rank_sha256") != rank
                or row.get("stratum") != {"occurrence_count_bucket": stratum[0], "facet_name_pattern": list(stratum[1])}
                or row.get("stratum_population") != population
                or row.get("stratum_quota") != quota
                or row.get("selection_reason") != expected_reason
                or row.get("inclusion_probability") != {"numerator": quota, "denominator": population}
                or row.get("sampling_weight") != {"numerator": population, "denominator": quota}
            ):
                raise EvidencePilotError("pilot selection metadata does not replay")
        if manifest.get("source_counts") != {
            "documents": source_document_count,
            "occurrences": pack_manifest.get("closure", {}).get("source_record_count"),
        }:
            raise EvidencePilotError("pilot source counts differ from projection pack")
    if sdk_specs is not None:
        from jsonschema import Draft202012Validator, FormatChecker
        from .evidence_schema import _offline_registry

        registry, schemas = _offline_registry(None, Path(sdk_specs))
        try:
            Draft202012Validator(schemas["evidence-pilot-sample.v1.schema.json"], registry=registry, format_checker=FormatChecker()).validate(manifest)
            Draft202012Validator(schemas["evidence-pilot-coverage.v1.schema.json"], registry=registry, format_checker=FormatChecker()).validate(coverage)
            selection_validator = Draft202012Validator(schemas["evidence-pilot-selection-row.v1.schema.json"], registry=registry, format_checker=FormatChecker())
            for _, row in _canonical_rows(root / SELECTION_NAME):
                selection_validator.validate(row)
        except Exception as error:
            raise EvidencePilotError(f"pilot schema validation failed: {error}") from error
    return manifest


def pilot_index_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": manifest["component"], "scope_status": SCOPE_STATUS,
        "admission_status": ADMISSION_STATUS, "selection_unit": "semantic_document_group",
        "selected_document_count": manifest["selected_counts"]["documents"],
        "selected_occurrence_count": manifest["selected_counts"]["occurrences"],
        "source_document_count": manifest["source_counts"]["documents"],
        "source_occurrence_count": manifest["source_counts"]["occurrences"],
        "preserves_all_selected_document_occurrences": True,
        "corpus_miss_definitive": False,
    }


def pilot_projection_coverage(root: Path, projection_pack: Path) -> dict[str, Any]:
    """Express closure over the selected sample in the standard coverage row shape."""

    root = Path(root)
    source = json.loads((Path(projection_pack) / COVERAGE_NAME).read_text(encoding="utf-8"))
    dispositions: Counter[str] = Counter()
    reasons: Counter[tuple[str, str]] = Counter()
    relations: dict[str, Counter[str]] = defaultdict(Counter)
    occurrence_total = 0
    for _, occurrence in _canonical_rows(root / OCCURRENCES_NAME):
        disposition = occurrence["terminal_disposition"]
        relation = occurrence["relation_identity"]["relation"]
        dispositions[disposition] += 1
        relations[relation][disposition] += 1
        relations[relation]["total"] += 1
        for reason in occurrence.get("reason_codes", []):
            reasons[(disposition, reason)] += 1
        occurrence_total += 1
    kinds: Counter[str] = Counter()
    searchable = 0
    for _, document in _canonical_rows(root / DOCUMENTS_NAME):
        kinds[document["document_kind"]] += 1
        searchable += int(document["searchable"])
    disposition_names = (
        "direct_semantic_document", "semantic_group_occurrence", "derived_document_input",
        "structured_only_occurrence", "rejected",
    )
    kind_names = (
        "activity", "state", "state_transition", "metric_window", "network_window",
        "entity", "detection", "structured_only",
    )
    return {
        "schema_version": "livefire.rag.evidence-coverage-report/1",
        "source_snapshots": source["source_snapshots"],
        "projection_policy": source["projection_policy"],
        "derivation_policies": [],
        "closure": {
            "source_record_count": occurrence_total,
            "terminal_disposition_count": occurrence_total,
            "unaccounted_record_count": 0, "multiply_dispositioned_record_count": 0,
            "all_source_records_dispositioned": True,
            "by_terminal_disposition": {name: dispositions[name] for name in disposition_names},
        },
        "documents": {
            "total": sum(kinds.values()), "searchable": searchable,
            "by_kind": {name: kinds[name] for name in kind_names},
        },
        "relation_coverage": [{
            "relation_identity": {"namespace": "ocsf", "relation": relation},
            "source_record_count": counts["total"],
            "terminal_disposition_count": counts["total"],
            "by_terminal_disposition": {name: counts[name] for name in disposition_names},
        } for relation, counts in sorted(relations.items())],
        "pointer_resolution": {
            "pointer_count": occurrence_total, "resolved_count": occurrence_total,
            "unresolved_count": 0, "all_pointers_resolved": True,
        },
        "reason_counts": [{"terminal_disposition": disposition, "reason_code": reason, "count": count}
            for (disposition, reason), count in sorted(reasons.items())],
    }


__all__ = ["EvidencePilotError", "build_evidence_pilot_sample", "verify_evidence_pilot_sample", "pilot_index_binding", "pilot_projection_coverage", "sampling_policy_ref"]
