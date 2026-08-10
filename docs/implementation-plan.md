# Implementation plan

## Milestone 1: deterministic corpus

1. Import only the public OCSF snapshot interchange contract needed by the
   builder; do not import Livefire internals.
2. Implement `ocsf_event_document.v1` with golden projection fixtures.
3. Emit canonical documents and an index manifest with complete lineage.
4. Add a deterministic fake embedder for protocol and reproducibility tests.

Exit gate: identical input and policy produce identical document and manifest
digests, and forbidden fields cannot leak.

## Milestone 2: reference retrieval

1. Add a pinned local embedding model behind an embedder interface.
2. Add exact cosine scoring with stable integer score serialization and event-ID
   tie-breaking.
3. Implement closed filters, bounded `top_k`, coverage, and pointer validation.
4. Expose build, inspect, search, and evaluate CLI commands.

Exit gate: all pointers hydrate, all filters are exact, and standalone replay is
deterministic.

## Milestone 3: baselines and production index

1. Add BM25 and hybrid reciprocal-rank fusion over the same documents.
2. Add an ANN backend, keeping the exact scorer as an oracle.
3. Benchmark public fixtures and held-out/counterfactual variants.
4. Select a default only from measured quality, latency, memory, and index size.

Exit gate: the chosen backend meets declared Recall@k and operational budgets.

## Milestone 4: SDK provider

1. Implement `handshake`, `open`, `call`, `health`, and `close` using the
   released `livefire-sdk` package.
2. Publish a bundle containing provider executable, builder executable, schemas,
   manifest, checksums, and SBOM; do not bundle telemetry or model cache data.
3. Run SDK conformance and the same standalone evaluation suite against the
   provider process.

Exit gate: a generic SDK harness can build and query the provider without a
Livefire checkout.

## Milestone 5: later Livefire loadout

In a separate Livefire change, pin the signed bundle and compatible index
manifest, grant read-only index access, translate tool calls into runner effects,
and hydrate returned event pointers before they can support evidence. This repo
must remain independently buildable and testable.
