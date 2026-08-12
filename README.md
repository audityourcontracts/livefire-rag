# Livefire RAG

`livefire-rag` builds immutable command-intelligence indexes and serves typed
retrieval and anomaly tools. Its first domain is command-line activity,
including PowerShell, shell commands, process ancestry, and cloud CLI/API
invocations.

It is not Livefire's investigative brain. It returns ranked leads, score
decompositions, comparisons, and exact source pointers. Livefire owns
hypotheses, evidence selection, conclusions, and stopping.

## Boundary

- `livefire-ocsf`, `livefire-splunk`, and `livefire-panther` export bounded,
  immutable command snapshots through `livefire-sdk` contracts.
- `livefire-rag build` consumes one or more sealed snapshots. It never queries a
  live SIEM while building an admitted index.
- `livefire-rag-provider` opens one immutable index read-only. It has no Splunk
  or Panther credentials and makes no vendor calls at query time.
- DuckDB is the first exact retrieval engine, not the public interface or the
  canonical index format.
- The Wasmtime-first integration is the existing Livefire pattern: an
  import-free runner emits `call_tool`; the native capability host executes the
  RAG provider. Browser host parity is later work.

## V1 tools

```text
cli.outliers  rank commands against the principal's own prior history,
              the prior population, or both
cli.search    retrieve commands from a natural-language security query
cli.similar   find commands semantically similar to one indexed command
cli.explain   return materialized score components and prior comparisons
```

All tools return candidate pointers, never a malicious/benign verdict or
authoritative evidence. `top_n` is honored up to the declared bound; there is no
hidden alert threshold.

## Implemented standalone POC commands

```text
livefire-rag build-fixture --fixture FIXTURE --out INDEX
livefire-rag promote-prototype --prototype-dir CACHE --out INDEX
livefire-rag verify --index INDEX
livefire-rag inspect --index INDEX
livefire-rag search --index INDEX --request REQUEST
livefire-rag similar --index INDEX --request REQUEST
livefire-rag provider
livefire-rag package-bundle --index INDEX --sdk-specs SDK_SPECS --out BUNDLE
livefire-rag demo-provider-poc --index INDEX --suite SUITE --out RESULTS
livefire-rag build-evidence-projection --snapshot-root SNAPSHOT --snapshot-id ID \
  --index-id ID [--index-uri URI] --out PACK
livefire-rag verify-evidence-projection --pack PACK --snapshot-root SNAPSHOT \
  --sdk-specs SDK_SPECS
livefire-rag inspect-evidence-projection --pack PACK --snapshot-root SNAPSHOT \
  --sdk-specs SDK_SPECS
```

The immutable POC pack contains canonical `documents.jsonl`, row-major
little-endian float32 L2 vectors, an object lock, and a content-addressed
manifest. Exact search accumulates in float64 and breaks equal distances by
ascending command ID. The SDK provider implements the complete JSONL lifecycle
and returns typed pointer or miss results.

The generic evidence projection command admits every typed relation from a
completed normalized-snapshot build receipt, verifies each Parquet object and
row count, and emits one terminal occurrence for every source row. It is a
pre-embedding projection pack, not yet a searchable SDK index. Remaining
planned production work includes promotion of those generic documents into the
canonical Parquet/embedding index, deterministic metric/state/network derived
windows, `cli.outliers`, and `cli.explain`. The fact-evidence evaluator remains
the separate `tools/evaluate_fact_evidence.py` command documented below.

Generic RAG schemas plus the projection policy and typed-Parquet pointer profile
are included in the wheel and discovered automatically by projection-pack
verification. SDK schemas remain a caller-selected host input (`--sdk-specs`);
the generic verifier loads only evidence schemas and their transitive SDK
dependencies, never scenario benchmark contracts.

The fixture builder, immutable-index verifier, provider, bundle packager, and
SDK replay are testable without a Livefire checkout. The one-off prototype
promotion command reads the pinned local M21/OpenBOTS output paths used to build
the exploratory corpus; it does not import or modify Livefire source. The
provider uses the adjacent `livefire-sdk` protocol and standalone harness.

A runnable development-only implementation of the immutable semantic pack,
standalone CLI, JSONL provider, SDK bundle, and frozen real-data demonstration is
documented in [`docs/standalone-provider-poc.md`](docs/standalone-provider-poc.md).
The focused provider suite covers reproducible builds, object corruption,
document/vector pairing, filters, misses, deadlines, exact ranking, loopback
search, and both in-process and subprocess lifecycle execution. The real-data
demo freezes all Q1-Q9 and S1/S2 calls; its SDK replay verifier requires exact
output equality for request IDs 3 through 13 and records per-case canonical
digests.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/source-snapshots.md`](docs/source-snapshots.md),
[`docs/command-index.md`](docs/command-index.md),
[`docs/physical-formats.md`](docs/physical-formats.md),
[`docs/model-selection.md`](docs/model-selection.md), and
[`docs/implementation-plan.md`](docs/implementation-plan.md). The complete
source-fidelity, conformance, quality, performance, and reporting program is in
[`docs/test-program.md`](docs/test-program.md).

The scenario-blind evidence indexing boundary, closure rules, document
families, and promotion contract are specified in
[`docs/generic-evidence-index.md`](docs/generic-evidence-index.md).
The first complete 13.9-million-row M21 projection build and its closure,
artifact, verification, and performance results are recorded in
[`docs/generic-evidence-m21-v1-build-report.md`](docs/generic-evidence-m21-v1-build-report.md).

The evaluator-only 23-cloud/53-BOTS fact-to-evidence benchmark is specified in
[`docs/fact-evidence-benchmark.md`](docs/fact-evidence-benchmark.md). Its metric
calculator is runnable today without a model or Livefire checkout:

```sh
python3 tools/evaluate_fact_evidence.py \
  --inventory fixtures/fact-evidence-synthetic/inventory.json \
  --queries fixtures/fact-evidence-synthetic/queries.jsonl \
  --candidate-universes fixtures/fact-evidence-synthetic/candidate-universes.jsonl \
  --qrels fixtures/fact-evidence-synthetic/qrels.jsonl \
  --hard-negatives fixtures/fact-evidence-synthetic/hard-negatives.jsonl \
  --candidate fixtures/fact-evidence-synthetic/candidate-rankings.jsonl \
  --baseline fixtures/fact-evidence-synthetic/baseline-rankings.jsonl \
  --gates fixtures/fact-evidence-synthetic/gates.json \
  --out reports/fact-evidence-synthetic/report.json
```

Macro nDCG@20 is the primary selection metric. Recall@20, eligible-fact
coverage, hard-negative discrimination, and pointer/filter correctness are
required gates. Ranking inputs declare their score kind and direction, so dense
cosine, BM25, exact-field, and reranker systems share rank-based metrics while
raw hard-negative margins remain comparable only within one score family. This
command produces a local comparison and gate report; formal promotion also
requires the sealed leakage, control, repeatability, and statistical receipts
specified by the benchmark contract.

Run the evaluator tests and validate every schema/fixture against the adjacent
private SDK checkout with:

```sh
uv run --extra test python -m unittest discover -s tests -v
uv run --with jsonschema python tools/validate_evidence_fixtures.py \
  --sdk-specs ../livefire-sdk/specs \
  --report reports/fact-evidence-synthetic/report.json
```

## Repository status

This is a private specification and implementation repository. Its GitHub
remote is private. No model weights, credentials, source telemetry, or built
indexes are tracked.
