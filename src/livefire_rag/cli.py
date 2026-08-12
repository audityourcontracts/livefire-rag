"""Command-line entrypoint for the standalone semantic index and provider."""

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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"livefire-rag: {error}", file=sys.stderr)
        raise SystemExit(1) from error
