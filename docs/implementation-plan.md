# Implementation plan

## Milestone 1: immutable command snapshot

1. Implement the `livefire-sdk` source-snapshot reader and canonical command
   record schema.
2. Build an OCSF/OpenBOTS adapter fixture with command, principal, host, parent,
   process, cloud-action, and source-pointer coverage.
3. Verify and seal the snapshot; prove the builder runs with no source adapter or
   credentials present.

Exit gate: all records and pointers validate and snapshot accounting closes.

## Milestone 2: deterministic index and scores

1. Implement bounded decoding and `powershell_ast_document.v1`.
2. Implement canonical action, target, structure, and obfuscation projections.
3. Materialize principal and population rolling-30-day scores and comparisons.
4. Write canonical Parquet objects and a content-addressed manifest.

Exit gate: no future leakage, stable rebuilds, and correct top-N results using
only deterministic test vectors.

## Milestone 3: local embedding bake-off

1. Download and pin the official Qwen3-Embedding-8B reference checkpoint.
2. Run it locally through PyTorch MPS.
3. Serve an exact compatible embedding artifact through LM Studio and implement
   the loopback adapter.
4. Benchmark dimensions, quantization, Qwen3 size ablations, EmbeddingGemma, and
   lexical baselines on held-out command queries.
5. Freeze the winning embedding policy and conformance vectors.

Exit gate: selected policy wins the declared quality/cost rule and every model
artifact/runtime/output is content bound.

## Milestone 4: DuckDB reference provider

1. Implement prepared exact queries over canonical Parquet.
2. Implement `cli.outliers`, `cli.search`, `cli.similar`, and `cli.explain`.
3. Add pointer, filter, coverage, deadline, request/result-size, and corruption
   tests.
4. Benchmark exact search over the OpenBOTS-scale index.

Exit gate: provider works standalone with vendor endpoints unavailable and exact
search meets the initial latency budget.

## Milestone 5: SDK packaging and Wasmtime-host integration contract

1. Package native builder/provider artifacts, schemas, policies, model
   references, SBOM, provenance, and conformance report.
2. Run the SDK harness without a Livefire checkout.
3. Define the future Livefire loadout binding for descriptor/provider/index/model
   identities and loopback embedding permission.

No Livefire repository change is part of these milestones.

## Milestone 6: measured acceleration and portability

1. If exact search is insufficient, build a derived HNSW cache and require
   Recall@20 against the exact oracle.
2. Add `livefire-splunk` and `livefire-panther` snapshot exporters using the same
   canonical record schema.
3. Prototype browser host parity using DuckDB-Wasm exact scans or a Rust/Wasm
   scanner over the same index pack.

Each item is independently gated and cannot change v1 tool semantics.
