# Command retrieval and anomaly architecture

Historical design record. This document describes the earlier multi-adapter
command and anomaly design. It is not the active source or runtime boundary.
Current production work reads only admitted normalized M45 Parquet with Rust;
see [`../README.md`](../README.md) and
[`runpod-embedding.md`](runpod-embedding.md).

## Components

```text
                 OFFLINE SNAPSHOT AND INDEX BUILD

Splunk API  -> livefire-splunk  --\
Panther API -> livefire-panther ---+-> immutable command snapshot(s)
OCSF files  -> livefire-ocsf    --/                 |
                                                    v
                                      decode + parse + normalize
                                                    |
                                      embed + materialize scores
                                                    |
                                                    v
                                        immutable RAG index pack


                  WASMTIME-FIRST INVESTIGATION

import-free Livefire runner WASM
        |
        | call_tool("cli.outliers", ...)
        v
native Livefire capability host
        |
        v
livefire-rag provider -> retrieval engine -> immutable index pack
        |
        v
candidate pointers and comparisons -> exact source hydration tool
```

Livefire owns the investigative brain. The RAG provider does not create
hypotheses, decide that activity is malicious, submit evidence, or write a
finding. It performs bounded retrieval and derived ranking.

## Immutable boundaries

There are two separately sealed artifacts:

1. A **source snapshot** is a bounded export from OCSF, Splunk, Panther, or
   another adapter. It binds the source scope, time range, extraction policy,
   adapter, record schema, coverage, object digests, and source pointers.
2. A **RAG index snapshot** is built only from sealed source snapshots. It binds
   projection, parser/decoder, embedding model, runtime, dimensions, scoring
   policy, objects, coverage, and source-snapshot identities.

Neither artifact is updated in place. A refresh creates a new identity. An old
investigation continues to resolve the exact index snapshot it originally used.

## Canonical index pack

```text
index/
  manifest.json
  commands.parquet
  powershell-asts.parquet
  embeddings.parquet
  outlier-scores.parquet
  comparisons.parquet
  objects.lock.json
```

`commands.parquet` contains canonical command documents, typed metadata, parse
features, and source pointers. `powershell-asts.parquet` contains the portable
AST documents addressed by digest from command rows. `embeddings.parquet` contains fixed-dimension
vectors keyed by command ID. `outlier-scores.parquet` contains scores computed
chronologically at build time. `comparisons.parquet` contains the nearest prior
commands and distances used to explain those scores.

Parquet plus the manifest is the portable, authoritative representation. A
DuckDB database file, an HNSW graph, a full-text index, or another engine cache is
derived, deletable, and reproducible from the canonical pack. Derived caches have
their own policy and digest and never change the pack's identity.

## Retrieval engine boundary

The public tools and index manifest contain no SQL or DuckDB types. The provider
uses an internal engine-neutral interface:

```text
open(index binding) -> read-only session
exact_vector_search(query vector, closed filters, top_n) -> ranked command IDs
materialized_outliers(scope, closed filters, top_n) -> ranked command IDs
get_explanation(command ID, scope) -> score components + comparisons
close()
```

The first implementation uses native DuckDB:

- `cli.outliers` and `cli.explain` query materialized Parquet tables.
- `cli.search` and `cli.similar` use exact cosine distance over stored vectors.
- Prepared queries receive closed typed filters; model-authored SQL is forbidden.
- Exact vector results are the correctness oracle.

DuckDB's experimental HNSW extension may be evaluated only as a derived native
cache. It is admitted only after Recall@N is measured against exact search. Its
persistent database is never canonical.

## Embedding boundary

The embedder is another private provider interface inside `livefire-rag`:

```text
describe() -> exact model/runtime/prompt/dimension identity
embed_documents(canonical documents) -> normalized vectors
embed_query(query, task instruction) -> normalized vector
```

For development, both the native builder and provider may call a loopback LM
Studio embeddings endpoint. The endpoint address is configuration, not identity;
the served model artifact, revision, tokenizer, quantization, dimension, prompt,
runtime version, and output conformance digest are identity-bearing.

`cli.outliers` and `cli.explain` need no runtime model because their scores and
comparisons are materialized. `cli.search` needs the exact bound query embedder.
The capability host grants loopback access explicitly; the runner WASM never
performs that call.

## Portability

V1 targets the native capability host around a Wasmtime runner. It does not claim
that the provider executable, DuckDB native engine, or LM Studio adapter runs in
a browser.

Later, a browser capability host can implement the same tool schemas using
DuckDB-Wasm exact scans, a Rust/Wasm vector scanner, or an authorized remote
provider. Browser work must consume the same canonical index semantics and pass
the same golden requests, filter checks, pointer checks, and ranking tolerances.
No browser-specific behavior is added to the runner.
