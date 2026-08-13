"""Small command line interface for Rust fast-index analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evaluate import evaluate_retrieval_run
from .geometry import write_pca_report
from .index import FastIndex


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m livefire_rag_analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate and summarize an index")
    inspect.add_argument("--index", required=True, type=Path)
    pca = commands.add_parser("pca", help="write PCA PNG and geometry report")
    pca.add_argument("--index", required=True, type=Path)
    pca.add_argument("--out", required=True, type=Path)
    pca.add_argument("--seed", type=int, default=0)
    pca.add_argument("--mark-count", type=int, default=12)
    evaluate = commands.add_parser("evaluate", help="evaluate a retrieval JSONL run")
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--qrels", type=Path)
    evaluate.add_argument("--out", type=Path)
    evaluate.add_argument("--cutoff", action="append", type=int)
    evaluate.add_argument(
        "--planned-query-id",
        action="append",
        help="frozen query ID; repeat to score zero-hit queries as empty rankings",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "inspect":
        with FastIndex.open(arguments.index) as index:
            result = {
                "schema_version": "livefire.rag.fast-index-analysis-inspection/1",
                "index": str(index.root),
                "documents": index.header.count,
                "dimensions": index.header.dimensions,
                "document_order_sha256": index.header.document_order_sha256,
                "source": index.manifest["source"],
                "build_scope": index.manifest["build_scope"],
                "complete": index.manifest["complete"],
                "embedding_profile": index.manifest["embedding_profile"],
            }
    elif arguments.command == "pca":
        result = write_pca_report(
            arguments.index,
            arguments.out,
            seed=arguments.seed,
            mark_count=arguments.mark_count,
        )
    else:
        result = evaluate_retrieval_run(
            arguments.run,
            qrels=arguments.qrels,
            cutoffs=arguments.cutoff or (5, 10, 20),
            planned_query_ids=arguments.planned_query_id,
        )
        if arguments.out is not None:
            if arguments.out.exists():
                raise FileExistsError(f"refusing to overwrite evaluation: {arguments.out}")
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            arguments.out.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
