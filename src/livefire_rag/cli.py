"""Historical Python prototype CLI retained for tests and comparisons.

The project does not publish this module as a console entry point. Current M44
preparation, embedding, indexing, query, packaging, and provider execution use
the Rust binaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .evidence_builder import build_evidence_pack, verify_evidence_pack
from .evidence_projection import projection_policy_ref
from .evidence_source import admit_typed_snapshot


def _load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="livefire-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-fixture", help="build a deterministic development index")
    build.add_argument("--fixture", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)

    promote = subparsers.add_parser("promote-prototype", help="rebuild and bind the exploratory 3,806-document corpus")
    promote.add_argument("--repository-root", type=Path, default=Path.cwd())
    promote.add_argument("--prototype-dir", type=Path, default=Path("reports/prototype-rag-demo"))
    promote.add_argument("--out", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify every immutable index object")
    verify.add_argument("--index", type=Path, required=True)

    inspect = subparsers.add_parser("inspect", help="print the verified index manifest")
    inspect.add_argument("--index", type=Path, required=True)

    similar = subparsers.add_parser("similar", help="run cli.similar standalone")
    similar.add_argument("--index", type=Path, required=True)
    similar.add_argument("--request", type=Path, required=True)
    similar.add_argument("--deadline-ms", type=int, default=5000)

    search = subparsers.add_parser("search", help="run cli.search standalone")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--request", type=Path, required=True)
    search.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    search.add_argument("--deadline-ms", type=int, default=5000)

    provider = subparsers.add_parser("provider", help="serve the Livefire SDK JSONL protocol")
    provider.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")

    package = subparsers.add_parser("package-bundle", help="build a content-inventoried SDK POC bundle")
    package.add_argument("--repository-root", type=Path, default=Path.cwd())
    package.add_argument("--index", type=Path, required=True)
    package.add_argument("--sdk-specs", type=Path, default=Path("../livefire-sdk/specs"))
    package.add_argument("--out", type=Path, required=True)

    demo = subparsers.add_parser("demo-provider-poc", help="run frozen Q1-Q9 and S1-S2 through the provider")
    demo.add_argument("--suite", type=Path, default=Path("fixtures/provider-poc/acceptance-suite.v1.json"))
    demo.add_argument("--index", type=Path, required=True)
    demo.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    demo.add_argument("--out", type=Path, default=Path("reports/provider-poc/provider-results.json"))
    demo.add_argument("--requests-out", type=Path, default=Path("reports/provider-poc/provider-requests.jsonl"))
    demo.add_argument("--deadline-ms", type=int, default=30000)

    evidence_build = subparsers.add_parser(
        "build-evidence-projection",
        help="build an occurrence-complete generic pre-embedding projection pack",
    )
    evidence_build.add_argument("--snapshot-root", type=Path, required=True)
    evidence_build.add_argument("--source-build-receipt", type=Path)
    evidence_build.add_argument("--snapshot-id", help="optional expected receipt component id")
    evidence_build.add_argument(
        "--snapshot-version", help="optional expected receipt component version"
    )
    evidence_build.add_argument("--index-id", required=True)
    evidence_build.add_argument("--index-version", default="1")
    evidence_build.add_argument(
        "--index-uri", help="optional identity-bearing URI for the projection-pack component"
    )
    evidence_build.add_argument("--out", type=Path, required=True)
    evidence_build.add_argument("--batch-size", type=int, default=4096)
    evidence_build.add_argument("--temp-directory", type=Path)

    evidence_verify = subparsers.add_parser(
        "verify-evidence-projection",
        help="verify a generic evidence projection pack and its occurrence closure",
    )
    evidence_verify.add_argument("--pack", type=Path, required=True)
    evidence_verify.add_argument("--snapshot-root", type=Path, required=True)
    evidence_verify.add_argument("--source-build-receipt", type=Path)
    evidence_verify.add_argument(
        "--rag-specs",
        type=Path,
        help="optional generic RAG schema directory; defaults to packaged schemas",
    )
    evidence_verify.add_argument(
        "--sdk-specs", type=Path, required=True, help="host SDK schema directory"
    )

    pilot_build = subparsers.add_parser(
        "build-evidence-pilot", help="build a scenario-blind non-corpus sample from a sealed projection pack"
    )
    pilot_build.add_argument("--pack", type=Path, required=True)
    pilot_build.add_argument("--component-id", required=True)
    pilot_build.add_argument("--component-version", default="1")
    pilot_build.add_argument("--component-uri")
    pilot_build.add_argument("--sdk-specs", type=Path, required=True)
    pilot_build.add_argument("--out", type=Path, required=True)

    pilot_verify = subparsers.add_parser(
        "verify-evidence-pilot", help="verify a sealed scenario-blind pilot sample"
    )
    pilot_verify.add_argument("--pilot", type=Path, required=True)
    pilot_verify.add_argument("--pack", type=Path)
    pilot_verify.add_argument("--sdk-specs", type=Path, required=True)

    evidence_inspect = subparsers.add_parser(
        "inspect-evidence-projection",
        help="print the verified generic evidence projection-pack manifest",
    )
    evidence_inspect.add_argument("--pack", type=Path, required=True)
    evidence_inspect.add_argument("--snapshot-root", type=Path, required=True)
    evidence_inspect.add_argument("--source-build-receipt", type=Path)
    evidence_inspect.add_argument(
        "--rag-specs",
        type=Path,
        help="optional generic RAG schema directory; defaults to packaged schemas",
    )
    evidence_inspect.add_argument(
        "--sdk-specs", type=Path, required=True, help="host SDK schema directory"
    )

    evidence_derive = subparsers.add_parser(
        "derive-evidence-overlay",
        help="build deterministic metric, network, state, and entity evidence documents",
    )
    evidence_derive.add_argument("--snapshot-root", type=Path, required=True)
    evidence_derive.add_argument("--source-build-receipt", type=Path)
    evidence_derive.add_argument("--pack", type=Path, required=True)
    evidence_derive.add_argument("--component-id", required=True)
    evidence_derive.add_argument("--component-version", default="1")
    evidence_derive.add_argument("--component-uri")
    evidence_derive.add_argument("--out", type=Path, required=True)

    evidence_derive_verify = subparsers.add_parser(
        "verify-evidence-overlay",
        help="verify a deterministic evidence-derivation overlay",
    )
    evidence_derive_verify.add_argument("--overlay", type=Path, required=True)

    evidence_promote = subparsers.add_parser(
        "promote-evidence-index", help="promote a verified projection pack to a searchable Parquet index"
    )
    evidence_promote.add_argument("--pack", type=Path, required=True)
    evidence_promote.add_argument("--derivation-pack", type=Path)
    evidence_promote.add_argument("--snapshot-root", type=Path, required=True)
    evidence_promote.add_argument("--source-build-receipt", type=Path)
    evidence_promote.add_argument("--source-admission-component", type=Path, required=True)
    evidence_promote.add_argument("--embedding-profile", type=Path, required=True)
    evidence_promote.add_argument(
        "--embedding-conformance-fixture", type=Path, required=True
    )
    evidence_promote.add_argument("--embedding-profile-id", required=True)
    evidence_promote.add_argument("--embedding-profile-version", default="1")
    evidence_promote.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    evidence_promote.add_argument("--index-id", required=True)
    evidence_promote.add_argument("--index-version", default="1")
    evidence_promote.add_argument("--index-uri")
    evidence_promote.add_argument("--sdk-specs", type=Path, required=True)
    evidence_promote.add_argument("--resume-dir", type=Path)
    evidence_promote.add_argument("--batch-size", type=int, default=32)
    evidence_promote.add_argument("--out", type=Path, required=True)

    pilot_promote = subparsers.add_parser(
        "promote-evidence-pilot-index", help="embed a sealed pilot sample as a local-only searchable index"
    )
    pilot_promote.add_argument("--pack", type=Path, required=True)
    pilot_promote.add_argument("--pilot", type=Path, required=True)
    pilot_promote.add_argument("--source-admission-component", type=Path, required=True)
    pilot_promote.add_argument("--embedding-profile", type=Path, required=True)
    pilot_promote.add_argument("--embedding-conformance-fixture", type=Path, required=True)
    pilot_promote.add_argument("--embedding-profile-id", required=True)
    pilot_promote.add_argument("--embedding-profile-version", default="1")
    pilot_promote.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    pilot_promote.add_argument("--index-id", required=True)
    pilot_promote.add_argument("--index-version", default="1")
    pilot_promote.add_argument("--index-uri")
    pilot_promote.add_argument("--sdk-specs", type=Path, required=True)
    pilot_promote.add_argument("--resume-dir", type=Path)
    pilot_promote.add_argument("--batch-size", type=int, default=32)
    pilot_promote.add_argument("--out", type=Path, required=True)

    pilot_evaluate = subparsers.add_parser(
        "evaluate-evidence-pilot",
        help="run every frozen pilot query through lexical, dense, and fused retrieval",
    )
    pilot_evaluate.add_argument("--index", type=Path, required=True)
    pilot_evaluate.add_argument("--query-fixture", type=Path, required=True)
    pilot_evaluate.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    pilot_evaluate.add_argument("--component-id", required=True)
    pilot_evaluate.add_argument("--component-version", default="1")
    pilot_evaluate.add_argument("--top-n", type=int, default=20)
    pilot_evaluate.add_argument("--deadline-seconds", type=int, default=300)
    pilot_evaluate.add_argument("--sdk-specs", type=Path, required=True)
    pilot_evaluate.add_argument("--out", type=Path, required=True)

    pilot_geometry = subparsers.add_parser(
        "report-evidence-pilot-geometry",
        help="seal index-only PCA and original-embedding kNN diagnostics",
    )
    pilot_geometry.add_argument("--index", type=Path, required=True)
    pilot_geometry.add_argument("--pilot", type=Path, required=True)
    pilot_geometry.add_argument("--component-id", required=True)
    pilot_geometry.add_argument("--component-version", default="1")
    pilot_geometry.add_argument("--seed", type=int, default=0)
    pilot_geometry.add_argument("--sdk-specs", type=Path, required=True)
    pilot_geometry.add_argument("--out", type=Path, required=True)

    evidence_index_verify = subparsers.add_parser(
        "verify-evidence-index", help="verify a promoted index against its bound projection pack"
    )
    evidence_index_verify.add_argument("--index", type=Path, required=True)
    evidence_index_verify.add_argument("--pack", type=Path)
    evidence_index_verify.add_argument("--pilot-sample", type=Path)
    evidence_index_verify.add_argument("--derivation-pack", type=Path)
    evidence_index_verify.add_argument("--sdk-specs", type=Path, required=True)

    evidence_search = subparsers.add_parser(
        "search-evidence", help="run occurrence-filtered exact dense evidence.search"
    )
    evidence_search.add_argument("--index", type=Path, required=True)
    evidence_search.add_argument("--pack", type=Path)
    evidence_search.add_argument("--derivation-pack", type=Path)
    evidence_search.add_argument("--replay-verify", action="store_true")
    evidence_search.add_argument("--sdk-specs", type=Path, required=True)
    evidence_search.add_argument("--request", type=Path, required=True)
    evidence_search.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    evidence_search.add_argument("--max-occurrences", type=int, default=20)

    evidence_provider = subparsers.add_parser(
        "evidence-provider", help="serve the generic evidence.search SDK JSONL protocol"
    )
    evidence_provider.add_argument("--sdk-specs", type=Path)

    evidence_package = subparsers.add_parser(
        "package-evidence-bundle",
        help="package the source/schema-closed development provider without index data",
    )
    evidence_package.add_argument("--repository-root", type=Path, default=Path.cwd())
    evidence_package.add_argument("--sdk-specs", type=Path, required=True)
    evidence_package.add_argument("--out", type=Path, required=True)

    evidence_loadout = subparsers.add_parser(
        "prepare-evidence-loadout",
        help="emit a deterministic local-test receipt, binding lock, and SDK transcript",
    )
    evidence_loadout.add_argument("--index", type=Path, required=True)
    evidence_loadout.add_argument("--bundle", type=Path, required=True)
    evidence_loadout.add_argument("--sdk-specs", type=Path, required=True)
    evidence_loadout.add_argument("--request", type=Path, action="append", required=True)
    evidence_loadout.add_argument("--embedding-endpoint", default="http://127.0.0.1:1234")
    evidence_loadout.add_argument("--deadline-unix-ms", type=int, default=4_102_444_800_000)
    evidence_loadout.add_argument("--out", type=Path, required=True)

    evidence_wire = subparsers.add_parser(
        "validate-evidence-wire",
        help="validate SDK wire call outputs and export pointers for authoritative hydration",
    )
    evidence_wire.add_argument("--wire", type=Path, required=True)
    evidence_wire.add_argument("--loadout", type=Path, required=True)
    evidence_wire.add_argument("--sdk-specs", type=Path, required=True)
    evidence_wire.add_argument("--report", type=Path, required=True)
    evidence_wire.add_argument("--hydration-requests", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-fixture":
        from .builder import build_fixture

        _print(build_fixture(args.fixture, args.out))
    elif args.command == "promote-prototype":
        from .builder import promote_prototype

        _print(promote_prototype(args.repository_root.resolve(), args.prototype_dir.resolve(), args.out))
    elif args.command in {"verify", "inspect"}:
        from .index import SemanticIndex

        index = SemanticIndex.open(args.index)
        _print(index.manifest if args.command == "inspect" else {"verified": True, "index": index.manifest["component"], "documents": len(index.documents)})
    elif args.command == "similar":
        from .index import SemanticIndex
        from .service import SemanticService

        service = SemanticService(SemanticIndex.open(args.index))
        _print(service.similar(_load_request(args.request), int(time.time() * 1000) + args.deadline_ms))
    elif args.command == "search":
        from .index import SemanticIndex
        from .service import SemanticService

        service = SemanticService(SemanticIndex.open(args.index), args.embedding_endpoint)
        _print(service.search(_load_request(args.request), int(time.time() * 1000) + args.deadline_ms))
    elif args.command == "provider":
        from .provider import serve

        return serve(embedding_endpoint=args.embedding_endpoint)
    elif args.command == "package-bundle":
        from .bundle import package_bundle

        _print(package_bundle(args.repository_root.resolve(), args.out, args.index, args.sdk_specs.resolve()))
    elif args.command == "demo-provider-poc":
        from .demo import run_demo

        _print(
            run_demo(
                args.suite.resolve(),
                args.index.resolve(),
                args.out,
                args.requests_out,
                embedding_endpoint=args.embedding_endpoint,
                per_call_deadline_ms=args.deadline_ms,
            )
        )
    elif args.command == "build-evidence-projection":
        snapshot_root = args.snapshot_root.resolve()
        receipt = (
            args.source_build_receipt.resolve()
            if args.source_build_receipt
            else snapshot_root / "build-receipt.json"
        )
        admitted = admit_typed_snapshot(
            snapshot_root,
            receipt,
            snapshot_id=args.snapshot_id,
            snapshot_version=args.snapshot_version,
        )
        manifest = build_evidence_pack(
            args.out,
            relations=admitted.relations,
            index_id=args.index_id,
            version=args.index_version,
            index_uri=args.index_uri,
            source_snapshot=admitted.component,
            projection_policy=projection_policy_ref(),
            batch_size=args.batch_size,
            temp_directory=args.temp_directory,
        )
        _print(
            {
                "manifest": manifest,
                "source_build_receipt_sha256": admitted.receipt_sha256,
                "typed_relation_count": len(admitted.relations),
                "expected_source_record_count": sum(admitted.expected_rows.values()),
            }
        )
    elif args.command in {"verify-evidence-projection", "inspect-evidence-projection"}:
        snapshot_root = args.snapshot_root.resolve()
        receipt = (
            args.source_build_receipt.resolve()
            if args.source_build_receipt
            else snapshot_root / "build-receipt.json"
        )
        admitted = admit_typed_snapshot(snapshot_root, receipt)
        manifest = verify_evidence_pack(
            args.pack,
            source_snapshot=admitted.component,
            relation_sources=admitted.relations,
            projection_policy=projection_policy_ref(),
            rag_specs=args.rag_specs.resolve() if args.rag_specs else None,
            sdk_specs=args.sdk_specs.resolve(),
        )
        _print(
            manifest
            if args.command == "inspect-evidence-projection"
            else {
                "verified": True,
                "component": manifest["component"],
                "closure": manifest["closure"],
            }
        )
    elif args.command == "build-evidence-pilot":
        from .evidence_pilot import build_evidence_pilot_sample

        manifest = build_evidence_pilot_sample(
            args.pack.resolve(), args.out, component_id=args.component_id,
            version=args.component_version, component_uri=args.component_uri,
            sdk_specs=args.sdk_specs.resolve(),
        )
        _print({"manifest": manifest, "scope_status": manifest["scope_status"], "sdk_admission": False})
    elif args.command == "verify-evidence-pilot":
        from .evidence_pilot import verify_evidence_pilot_sample

        manifest = verify_evidence_pilot_sample(
            args.pilot, projection_pack=args.pack.resolve() if args.pack else None,
            sdk_specs=args.sdk_specs.resolve(),
        )
        _print({"verified": True, "component": manifest["component"], "scope_status": manifest["scope_status"], "sdk_admission": False})
    elif args.command == "derive-evidence-overlay":
        from .evidence_derivation import build_evidence_derivation_pack

        snapshot_root = args.snapshot_root.resolve()
        receipt = (
            args.source_build_receipt.resolve()
            if args.source_build_receipt
            else snapshot_root / "build-receipt.json"
        )
        manifest = build_evidence_derivation_pack(
            args.out,
            snapshot_root=snapshot_root,
            receipt_path=receipt,
            base_projection_pack=args.pack.resolve(),
            component_id=args.component_id,
            version=args.component_version,
            component_uri=args.component_uri,
        )
        _print({"manifest": manifest, "stage": "pre_embedding_derivation_overlay"})
    elif args.command == "verify-evidence-overlay":
        from .evidence_derivation import verify_evidence_derivation_pack

        manifest = verify_evidence_derivation_pack(args.overlay)
        _print({"verified": True, "component": manifest["component"], "closure": manifest["closure"]})
    elif args.command == "promote-evidence-index":
        from .evidence_index import loopback_embedder, promote_evidence_pack

        snapshot_root = args.snapshot_root.resolve()
        receipt = args.source_build_receipt.resolve() if args.source_build_receipt else snapshot_root / "build-receipt.json"
        admitted = admit_typed_snapshot(snapshot_root, receipt)
        profile = _load_request(args.embedding_profile)
        source_admission = _load_request(args.source_admission_component)
        embedder = loopback_embedder(
            args.embedding_endpoint, profile["api_model_key"]
        )
        manifest = promote_evidence_pack(
            args.pack, args.out, relation_sources=admitted.relations,
            source_snapshot=admitted.component, projection_policy=projection_policy_ref(),
            sdk_specs=args.sdk_specs, embedding_profile=profile,
            embedding_profile_id=args.embedding_profile_id,
            embedding_profile_version=args.embedding_profile_version, embedder=embedder,
            embedding_conformance_fixture=args.embedding_conformance_fixture,
            source_admission_receipt=source_admission, index_id=args.index_id,
            version=args.index_version, derivation_pack=args.derivation_pack,
            index_uri=args.index_uri,
            resume_dir=args.resume_dir, batch_size=args.batch_size,
        )
        _print({"manifest": manifest, "sdk_admission": False})
    elif args.command == "promote-evidence-pilot-index":
        from .evidence_index import loopback_embedder, promote_evidence_pack
        from .evidence_pilot import verify_evidence_pilot_sample

        pack = args.pack.resolve()
        pilot = args.pilot.resolve()
        pilot_manifest = verify_evidence_pilot_sample(
            pilot, projection_pack=pack, sdk_specs=args.sdk_specs.resolve()
        )
        pack_manifest = _load_request(pack / "manifest.json")
        profile = _load_request(args.embedding_profile)
        source_admission = _load_request(args.source_admission_component)
        manifest = promote_evidence_pack(
            pack, args.out, relation_sources=(),
            source_snapshot=pack_manifest["source_snapshots"][0],
            projection_policy=pack_manifest["projection_policy"], sdk_specs=args.sdk_specs,
            embedding_profile=profile, embedding_profile_id=args.embedding_profile_id,
            embedding_profile_version=args.embedding_profile_version,
            embedder=loopback_embedder(args.embedding_endpoint, profile["api_model_key"]),
            embedding_conformance_fixture=args.embedding_conformance_fixture,
            source_admission_receipt=source_admission, index_id=args.index_id,
            version=args.index_version, index_uri=args.index_uri,
            resume_dir=args.resume_dir, batch_size=args.batch_size, pilot_sample=pilot,
        )
        _print({"manifest": manifest, "pilot_sample": pilot_manifest["component"], "sdk_admission": False})
    elif args.command == "evaluate-evidence-pilot":
        from .evidence_index import EvidenceIndex, loopback_embedder
        from .evidence_pilot_eval import run_evidence_pilot_evaluation

        with EvidenceIndex.open(args.index, sdk_specs=args.sdk_specs.resolve()) as index:
            profile = index.profile
        batch_embed = loopback_embedder(
            args.embedding_endpoint, profile["api_model_key"]
        )

        def query_embed(query: str, deadline_unix_ms: int):
            del deadline_unix_ms
            text = profile["query_composition"].format(
                query_instruction=profile["query_instruction"], query=query
            )
            return batch_embed([text])[0]

        manifest = run_evidence_pilot_evaluation(
            args.index, args.query_fixture, args.out,
            sdk_specs=args.sdk_specs, embed_query=query_embed,
            component_id=args.component_id, version=args.component_version,
            top_n=args.top_n, deadline_seconds=args.deadline_seconds,
        )
        _print({
            "manifest": manifest,
            "scope_status": "sample_only_not_corpus_coverage",
            "quality_claim": False,
            "sdk_admission": False,
        })
    elif args.command == "report-evidence-pilot-geometry":
        from .evidence_geometry import build_evidence_pilot_geometry

        manifest = build_evidence_pilot_geometry(
            args.index,
            args.pilot,
            args.out,
            sdk_specs=args.sdk_specs,
            component_id=args.component_id,
            version=args.component_version,
            seed=args.seed,
        )
        _print({
            "manifest": manifest,
            "claim_status": "geometric_isolation_not_maliciousness_or_retrieval_quality",
            "sdk_admission": False,
        })
    elif args.command == "verify-evidence-index":
        from .evidence_index import verify_promoted_evidence_index

        manifest = verify_promoted_evidence_index(
            args.index, projection_pack=args.pack, derivation_pack=args.derivation_pack,
            pilot_sample=args.pilot_sample,
            sdk_specs=args.sdk_specs
        )
        _print({"verified": True, "sdk_admission": False, "component": manifest["component"]})
    elif args.command == "search-evidence":
        from .evidence_index import EvidenceIndex, loopback_embedder

        index = EvidenceIndex.open(
            args.index, projection_pack=args.pack, derivation_pack=args.derivation_pack,
            replay_verify=args.replay_verify, sdk_specs=args.sdk_specs,
        )
        try:
            request = _load_request(args.request)
            profile = index.profile
            query_text = profile["query_composition"].format(
                query_instruction=profile["query_instruction"], query=request["query"]
            )
            vector = loopback_embedder(
                args.embedding_endpoint, profile["api_model_key"]
            )([query_text])[0]
            _print(index.search_dense(request, vector, max_occurrences=args.max_occurrences))
        finally:
            index.close()
    elif args.command == "evidence-provider":
        from .evidence_provider import serve

        return serve(sdk_specs=args.sdk_specs)
    elif args.command == "package-evidence-bundle":
        from .evidence_bundle import package_evidence_provider_bundle

        _print(
            package_evidence_provider_bundle(
                args.repository_root.resolve(), args.out, args.sdk_specs.resolve()
            )
        )
    elif args.command == "prepare-evidence-loadout":
        from .evidence_loadout import prepare_evidence_loadout

        _print(prepare_evidence_loadout(
            args.index, args.bundle, args.out,
            sdk_specs=args.sdk_specs,
            queries=[_load_request(path) for path in args.request],
            embedding_endpoint=args.embedding_endpoint,
            deadline_unix_ms=args.deadline_unix_ms,
        ))
    elif args.command == "validate-evidence-wire":
        from .evidence_loadout import validate_evidence_wire

        _print(validate_evidence_wire(
            args.wire, args.loadout, sdk_specs=args.sdk_specs,
            report_path=args.report,
            hydration_requests_path=args.hydration_requests,
        ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"livefire-rag: {error}", file=sys.stderr)
        raise SystemExit(1) from error
