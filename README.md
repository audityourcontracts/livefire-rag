# Livefire RAG

`livefire-rag` builds immutable, generic OCSF evidence-candidate indexes and
serves typed retrieval tools. Process and command activity are one relation
family among API, identity, configuration, cloud, file, network, email,
detection, and other normalized OCSF relations.

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
- Livefire runners emit `call_tool`; the implemented Node-side SDK adapter owns
  provider spawn/open/call/health/close and validates the exact declared
  read-only mount bindings. OS-level immutability is enforced externally.
  Browser/WASM code receives only an injected typed host client and never
  spawns a process.

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

## Rust experimental vertical slice

The fast experimental path now has a working OCSF fixture-to-tool vertical
slice. It streams typed Parquet through the Rust projection, batches embeddings
through LM Studio, writes the language-neutral fast index, queries dense,
lexical, or fused retrieval, and opens that index in the standalone Rust JSONL
provider. Python reads the same Parquet and `vectors.f32` artifacts for
evaluation against human relevance labels (often called qrels) and for PCA; it
is not in the build or query path.

With LM Studio serving the model named by
`profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json`, run the
six-document interface smoke with:

```sh
uv run --extra analysis python tools/run_rust_smoke.py \
  --work /tmp/livefire-rag-smoke --mode fused
```

The command refuses to overwrite its work directory. It writes the source
fixture, index, per-query results, retrieval run, qrel metrics, PCA PNG/report,
direct provider JSONL transcript, and a `smoke-report.json`. It also repeats the
build with the embedding endpoint deliberately unreachable, requires zero model
calls, and compares the stable index artifacts byte-for-byte. The bundled qrels
are generated from the same six synthetic scenarios, so a perfect score proves
only that the interfaces and document/vector/pointer bindings work. It is not a
retrieval-quality benchmark.

The isolated Livefire adapter includes
`tools/prepare-external-evidence-search.mjs`, which converts the packaged RAG
bundle and prepared local-test transcript into a validated external-tool
loadout. A desktop/server composition root must inject that Node host into the
runner; deterministic hunt scheduling is intentionally unchanged.

The individual commands are:

```sh
cargo run -p rag-builder --bin rag -- build \
  --snapshot SNAPSHOT --out INDEX \
  --embedding-profile profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json \
  --embedding-endpoint http://127.0.0.1:1234 \
  --resume CACHE.sqlite3 --embedding-batch-size 16 \
  --representative-sample

cargo run -p rag-builder --bin rag -- query \
  --index INDEX --query 'encoded PowerShell download' --mode fused --top-n 20 \
  --embedding-endpoint http://127.0.0.1:1234

# Open and verify the index once, then execute a frozen JSONL experiment plan.
cargo run -p rag-builder --bin rag -- batch-query \
  --index INDEX --requests QUERIES.jsonl \
  --embedding-endpoint http://127.0.0.1:1234 > RESULTS.jsonl

cargo run -p rag-provider --bin rag-package-tool -- \
  --provider target/release/rag-provider \
  --sdk-specs ../livefire-sdk/specs --out PROVIDER_BUNDLE

cargo run -p rag-provider --bin rag-prepare-local-tool -- \
  --index INDEX --bundle PROVIDER_BUNDLE \
  --source-receipt SNAPSHOT/build-receipt.json \
  --embedding-profile profiles/qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json \
  --out LOCAL_TEST_LOADOUT

cargo run -p rag-builder --bin rag -- inspect --index INDEX
uv run --extra analysis python -m livefire_rag_analysis inspect --index INDEX
uv run --extra analysis python -m livefire_rag_analysis pca \
  --index INDEX --out REPORT_DIR
uv run --extra analysis python -m livefire_rag_analysis evaluate \
  --run RUN.jsonl --qrels QRELS.jsonl --out REPORT.json \
  --planned-query-id q-1 --planned-query-id q-2
```

Tool preparation does not modify `INDEX`. It atomically creates a separate
`LOCAL_TEST_LOADOUT/evidence-index` wrapper using hard links to the verified
physical files, so the loadout must be created on the same filesystem. The
external host remains responsible for keeping both paths read-only while the
provider runs.

`--representative-sample` bounds retained documents and embedding calls per
searchable relation, but still scans and projects the typed snapshot twice: the
first pass selects documents and the second spills all occurrences for them.
Do not start the next released `livefire-ocsf` build until its verified source
snapshot is available. The Rust provider is packaged with every file and digest
it needs, then exercised through the SDK as an explicitly local test. Search
returns pointers to OCSF events, not verified evidence: the released OCSF query
service must resolve and check those pointers before their fields are used as
facts. A production host must still enforce file, network, memory, and process
limits, and Livefire must still connect its desktop or server runner to this
provider. Wasmtime guest support is also separate future work. The exact
contracts, implemented status, and remaining gaps are in
[`docs/rust-experimental-rag-spec.md`](docs/rust-experimental-rag-spec.md).
The next large-index design separates parallel Rust preparation, replaceable
local or cloud embedding, and streaming final assembly. Its implementation
contract and staged test plan are in
[`docs/portable-embedding-pipeline.md`](docs/portable-embedding-pipeline.md).
The local scale path is now measured through a 10,000-document LM Studio build,
version-3 query, recovery checks, and 2,048- and 1,024-dimensional comparison
indexes. The measured non-network target is 422,566 documents and 5,325,200
event references. All 16 M41 datasets in that scope are prepared, verified, and
planned for 92,466,199 exact tokens. Twelve datasets are real embedded and
indexed: the ten small datasets plus API and HTTP activity, for 23,636
documents and 165,186 event references. Their sealed catalogue completed 45
searches across 15 queries with 15 model calls and produced a label-hidden pool
of 690 unique candidates whose 1,275 unique event pointers were checked against
typed Parquet. The four largest datasets
also passed deterministic test-vector assembly as explicitly test-only indexes,
covering another 398,930 documents and 5,160,014 event references. People still
need to review pooled search results and mark which are relevant. Returned
event references still need confirmation through the released OCSF query
service. A freshly packaged provider also passed the complete local SDK
handshake/open/search/health/close lifecycle against the HTTP index.
Runpod is deferred and is not part of the current goal. See
[`docs/local-first-embedding-scale-plan.md`](docs/local-first-embedding-scale-plan.md).

The first modular dataset path is implemented. For example, one relation can be
prepared, embedded, assembled, and queried without rebuilding any other index:

```sh
cargo run -p rag-builder --bin rag -- prepare \
  --snapshot SNAPSHOT --dataset-id DATASET --relation ocsf_detection_finding \
  --out PREPARED
cargo run -p rag-builder --bin rag -- verify-prepared --prepared PREPARED
cargo run -p rag-builder --bin rag -- plan-embeddings \
  --prepared PREPARED --embedding-profile PROFILE \
  --tokenizer-json TOKENIZER_JSON --tokenizer-ref TOKENIZER_REF \
  --maximum-task-tokens 262144 --maximum-task-documents 2048 --out PLAN
cargo run -p rag-builder --bin rag -- embed \
  --prepared PREPARED --plan PLAN --embedding-profile PROFILE \
  --embedding-endpoint http://127.0.0.1:1234 --out EMBEDDINGS
cargo run -p rag-builder --bin rag -- finalize-embeddings \
  --prepared PREPARED --plan PLAN --embedding-profile PROFILE \
  --embeddings EMBEDDINGS
cargo run -p rag-builder --bin rag -- assemble \
  --prepared PREPARED --plan PLAN --embeddings EMBEDDINGS \
  --embedding-profile PROFILE --out INDEX
cargo run -p rag-builder --bin rag -- query \
  --index INDEX --mode fused --query 'encoded PowerShell' \
  --embedding-endpoint http://127.0.0.1:1234
```

`plan-embeddings` now emits only the exact-token, token-balanced v2 plan. A v1
plan is rejected with a migration message instead of being interpreted using
the new rules. Several `embed` processes may write disjoint task ranges to the
same output. Each publishes one vector part, a content-bound receipt, and a
sanitized timing report. `finalize-embeddings` is deliberately separate: it
refuses partial or extra output and writes `manifest.json` only after every
planned task is present and valid.
Planning and embedding verify the prepared manifest and document files, but do
not read occurrence files that those stages cannot use. `verify-prepared` and
`assemble` perform the full check, including every occurrence file.
For larger plans, give separate workers non-overlapping ranges such as
`--task-range 0..8` and `--task-range 8..16`; each range is checked against the
actual number of tasks before any model request.

Occurrence rows and vectors stream in bounded chunks. Preparation writes
bounded sorted document runs and merges them by document ID, so the former
600,000-document in-memory limit is removed. Parallel Arrow preparation and
streaming version-3 SQLite assembly are implemented. LM Studio is the only
implemented model executor in this branch; the documented Runpod design is
future work and would reuse the same prepared Parquet and binary vector format.

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
livefire-rag build-evidence-pilot --pack PACK --component-id ID \
  --sdk-specs SDK_SPECS --out PILOT
livefire-rag verify-evidence-pilot --pilot PILOT --pack PACK \
  --sdk-specs SDK_SPECS
livefire-rag promote-evidence-pilot-index --pack PACK --pilot PILOT \
  --source-admission-component RECEIPT_REF --embedding-profile PROFILE \
  --embedding-conformance-fixture FIXTURE --embedding-profile-id ID \
  --index-id ID --sdk-specs SDK_SPECS --out PILOT_INDEX
livefire-rag verify-evidence-index --index PILOT_INDEX --pilot-sample PILOT \
  --sdk-specs SDK_SPECS
livefire-rag evaluate-evidence-pilot --index PILOT_INDEX \
  --query-fixture fixtures/generic-evidence-pilot-queries.v1.json \
  --embedding-endpoint http://127.0.0.1:1234 --component-id ID \
  --sdk-specs SDK_SPECS --out PILOT_EVALUATION
livefire-rag report-evidence-pilot-geometry --index PILOT_INDEX --pilot PILOT \
  --component-id ID --sdk-specs SDK_SPECS --out PILOT_GEOMETRY
livefire-rag derive-evidence-overlay --pack PACK --snapshot-root SNAPSHOT \
  --component-id ID --out OVERLAY
livefire-rag verify-evidence-overlay --overlay OVERLAY
livefire-rag promote-evidence-index --pack PACK --derivation-pack OVERLAY \
  --snapshot-root SNAPSHOT --source-admission-component RECEIPT_REF \
  --embedding-profile PROFILE --embedding-conformance-fixture FIXTURE \
  --embedding-profile-id ID --index-id ID \
  --sdk-specs SDK_SPECS --out INDEX
livefire-rag verify-evidence-index --index INDEX --pack PACK \
  --derivation-pack OVERLAY --sdk-specs SDK_SPECS
livefire-rag search-evidence --index INDEX --pack PACK \
  --derivation-pack OVERLAY --sdk-specs SDK_SPECS --request REQUEST
livefire-rag evidence-provider --sdk-specs SDK_SPECS
livefire-rag package-evidence-bundle --sdk-specs SDK_SPECS --out BUNDLE
livefire-rag prepare-evidence-loadout --index INDEX --bundle BUNDLE \
  --sdk-specs SDK_SPECS --request REQUEST [--request REQUEST...] --out LOADOUT
livefire-rag validate-evidence-wire --wire WIRE --loadout LOADOUT \
  --sdk-specs SDK_SPECS --report REPORT --hydration-requests POINTERS
```

The immutable POC pack contains canonical `documents.jsonl`, row-major
little-endian float32 L2 vectors, an object lock, and a content-addressed
manifest. Exact search accumulates in float64 and breaks equal distances by
ascending command ID. The SDK provider implements the complete JSONL lifecycle
and returns typed pointer or miss results.

The generic evidence path admits every typed relation from a completed
normalized-snapshot build receipt, verifies each Parquet object and row count,
and emits one terminal occurrence for every source row. A separate immutable
overlay derives fixed-policy metric/network windows, state transitions, and
entity summaries without rewriting source occurrences. Promotion converts the
base and derived documents to canonical Parquet, embeds every and only
searchable document, and emits a locally verified index. `evidence.search`
then applies source filters to occurrences before ranking documents and returns
only source pointers. Local verification and the SDK bundle are implemented;
the repository deliberately does not claim production host admission or an
authority signature.

Before full-corpus derivation and embedding, the pilot commands can seal and
embed a deterministic structural sample of an already verified projection
pack. Selection is scenario-blind, binds the fixed sampling policy, and retains
every occurrence for each selected semantic document group. The resulting
index has the normal physical/query interface, but its manifest and build
report declare `sample_only_not_corpus_coverage` and
`local_evaluation_only_not_sdk_admitted`. Every pointer or miss returned from
that index has partial coverage with
`pilot_sample_not_corpus_coverage`; a miss is explicitly not a corpus-wide
absence claim. Pilot promotion does not accept a derivation overlay.

`evaluate-evidence-pilot` freezes the complete execution plan before its first
search, runs every predeclared query through lexical, dense, and fused retrieval,
and seals every top-N output plus fixed pairwise ranking comparisons. It verifies
partial sample scope and exact occurrence-pointer closure. Expected relation
families are reported only as answer-neutral diagnostics. Until people review
the pooled results and mark which are relevant, the report makes no
retrieval-quality claim. PCA/kNN corpus geometry is a
separate index-only analysis so query metadata cannot influence it; see
[`docs/evidence-pilot-evaluation.md`](docs/evidence-pilot-evaluation.md).

`prepare-evidence-loadout` creates a deterministic local-test admission receipt,
exact binding lock, and SDK lifecycle transcript for a promoted index and
development bundle. It never creates a production authority receipt. After
`livefire-sdk invoke`, `validate-evidence-wire` validates every successful call
output and its request/index/lock identities, then exports deduplicated immutable
pointer requests. It does not hydrate source data; an authoritative OCSF/source
adapter must resolve and verify those pointers before Livefire treats fields as
facts.

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
The many-to-many derivation boundary and its scenario-blind policies are in
[`docs/evidence-derivation-overlay.md`](docs/evidence-derivation-overlay.md).
The Rust-first fast experimental workflow and its future `livefire-ocsf`
adapter are specified in
[`docs/rust-experimental-rag-spec.md`](docs/rust-experimental-rag-spec.md).
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
