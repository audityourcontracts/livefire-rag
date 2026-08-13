# Rust experimental RAG specification

Status: draft implementation specification

Scope: fast local RAG experimentation over `livefire-ocsf` snapshots

Production admission: explicitly deferred

## 1. Decision

The next implementation is a Rust execution rewrite of the useful RAG
contracts, not another expansion of the current promotion and proof pipeline.
The normal operator loop is:

```text
rag build -> rag query -> rag eval -> rag visualize -> rag serve
```

It must be possible to change a projection, embedding profile, retrieval method,
or corpus sample and obtain a comparable result in minutes or hours. Full source
replay, authority admission, content-closed packaging, and production signing
are separate explicit operations for configurations worth preserving.

The existing Python implementation remains the semantic oracle during the
migration. Its schemas, projection policy, frozen query fixtures, source
identity rules, and SDK wire behavior are retained. Its row-at-a-time execution
path is not retained.

## 2. Goals

1. Stream released `livefire-ocsf` typed Parquet relations without importing
   private implementation crates or depending on a sibling checkout.
2. Build activity, state, and detection documents directly into the searchable
   index with no mandatory JSONL projection-pack or promotion stage.
3. Batch local embedding requests, resume safely after interruption, and write
   vectors without converting them into millions of Python objects.
4. Compare dense, lexical, and reciprocal-rank-fused retrieval on the same
   frozen corpus and query suite.
5. Return snapshot-scoped OCSF event references that the structured runner can
   hydrate through the released OCSF query host.
6. Implement the Livefire SDK JSONL provider in Rust and preserve a plausible
   WASI/Wasmtime path.
7. Keep PCA, corpus plots, qrel authoring, statistical analysis, and research
   reports in Python over stable index files.

The primary success criterion is retrieval effectiveness. Throughput matters
because it determines how many experiments can be run, but a faster inferior
retriever does not win.

## 3. Non-goals for the experimental path

The default build does not:

- independently replay every parent or sampled source row;
- mint an SDK admission receipt or authority signature;
- prove the native source record behind an OCSF event;
- run JSON Schema validation over every output row;
- derive metric, network, entity, and state-transition overlays;
- build an ANN index before exact search is shown to be insufficient;
- package a production-content-closed runtime;
- depend on Livefire source, hunt fixtures, qrels, expected evidence, or
  scenario indicators.

Those capabilities are not discarded. They are moved behind explicit strict
commands:

```text
rag verify --strict
rag package
rag admit
```

## 4. `livefire-ocsf` input boundary

### 4.1 What is stable enough now

The extracted repository currently publishes a snapshot rooted at a
`build-receipt.json`. The embedded snapshot manifest enumerates content-addressed
objects with:

```text
relation, path, rows, sha256, logical_sha256
```

Runner-facing data is under `semantic/`. Typed class relations have three UTF-8
columns:

```text
event_id, typed_event_json, support_ref
```

Core graph relations include events, facets, entities, observables,
participants, event-observables, and relationships. Typed relations must be
discovered from the manifest rather than hard-coded because a zero-row class is
not necessarily materialized.

An OCSF `event_id` is stable only within the bound snapshot and mapping
revision. A RAG evidence reference is therefore:

```rust
pub struct EvidenceRef {
    pub snapshot_sha256: Sha256,
    pub mapping_sha256: Sha256,
    pub event_id: String,
    pub support_ref: String,
}
```

Normal RAG results do not need SDK-native Parquet row locators. The structured
runner can pass the returned event IDs to `ocsf.hydrate_event_envelopes`, which
returns normalized details, participants, observables, relationships, and
support references. Native-source authority resolution remains the OCSF data
plane's responsibility.

### 4.2 Adapter contract

Only `rag-ocsf` knows the snapshot layout:

```rust
pub struct OcsfSnapshot {
    pub schema_version: u8,
    pub snapshot_sha256: Sha256,
    pub dataset_sha256: Sha256,
    pub mapping_sha256: Sha256,
    pub ocsf_schema_sha256: Sha256,
    pub extension_pack_sha256: Sha256,
    pub relation_contract_sha256: Sha256,
    pub normalized_events: u64,
    pub relations: Vec<OcsfRelation>,
}

pub struct OcsfRelation {
    pub name: String,
    pub kind: RelationKind,
    pub path: PathBuf,
    pub rows: u64,
    pub object_sha256: Sha256,
    pub logical_sha256: Sha256,
}

pub trait SnapshotReader {
    fn identity(&self) -> &OcsfSnapshot;
    fn typed_relations(&self) -> impl Iterator<Item = &OcsfRelation>;
    fn scan(&self, relation: &OcsfRelation)
        -> Result<impl Iterator<Item = Result<RecordBatch>>>;
}
```

Fast admission checks receipt shape, root-relative safe paths, required
objects, Parquet metadata row counts, required columns, and retained snapshot,
mapping, schema, and relation identities. It does not hash every multi-gigabyte
object again.

The first adapter may read the present local `build-receipt.json` shape for
parity. A released adapter must consume the eventual E9 release manifest and
must not use path dependencies on private `livefire-ocsf` implementation
crates. `typed_event_json` is treated as schema-bound JSON and walked by the
RAG-owned generic projection policy.

### 4.3 What is not stable yet

`livefire-ocsf` extraction remains in progress. Its E6 full-corpus work is
currently reconciling differences between the accepted M41 snapshot and the
new extracted build, including facets, provenance, graph rows, process rows,
and typed-relation digests. The E9 release manifest and final digests do not yet
exist.

Consequently we may implement the adapter and use fixtures or the accepted M41
format now, but we must not begin the next full-corpus RAG build until E6 closes
and a candidate snapshot is published.

## 5. Workspace and ownership

```text
Cargo.toml
crates/
  rag-contracts/    schemas, DTOs, canonical identities, SDK messages
  rag-ocsf/         snapshot receipt and streaming Parquet adapter
  rag-projection/   generic flattening, redaction, facets, grouping
  rag-embedding/    LM Studio client, scheduler, resumable cache
  rag-index/        index writer/reader, filters, dense, lexical, fusion
  rag-builder/      `rag build` executable
  rag-provider/     SDK JSONL provider and `rag serve`
  rag-testkit/      fixtures, Python-parity oracle, benchmarks
python/
  livefire_rag_analysis/
    evaluate.py
    geometry.py
    visualize.py
```

The first Rust provider is native. Core projection, filters, exact vector scan,
fusion, and wire DTOs must avoid dependencies that prevent a later WASI build.
DuckDB, Tantivy, LM Studio, and OS filesystem details sit behind adapters and
never appear in public tool contracts.

The 1,900-line derivation engine is not ported first. Its document/membership
contract is retained for a later phase after direct activity/state/detection
retrieval is measured.

## 6. Fast index format

```text
index/
  index.json
  documents.parquet
  occurrences.parquet
  vectors.f32
  lexical/
  build-report.json
```

`documents.parquet` stores document ID, digest, kind, semantic text, facets,
relation identities, occurrence count, and vector ordinal.

`occurrences.parquet` stores document ordinal, event time, relation identity,
bounded exact attributes, and `EvidenceRef`.

`vectors.f32` contains a versioned header followed by contiguous row-major
little-endian float32 vectors. The header binds dimensions, count, embedding
profile, and document-order digest. Python opens it directly with
`numpy.memmap`; Rust scans it without JSON or Arrow list conversion.

The lexical directory is derived and replaceable. The initial native adapter
may use Tantivy, but BM25 tokenization/profile identity and output semantics are
RAG contracts. A portable postings implementation can replace it for WASI.

The default builder streams Arrow batches, projects records, groups documents,
writes Parquet in batches, submits only missing embeddings, appends contiguous
vectors, computes inexpensive digests while writing, writes a report, and
atomically renames staging into place. It never loads all occurrences into a
database or replays the completed index back through its parent.

## 7. Commands

```text
rag build --snapshot SNAPSHOT --out INDEX \
  --embedding-endpoint http://127.0.0.1:1234 \
  --embedding-profile PROFILE [--sample-documents 20000]

rag query --index INDEX --mode dense|lexical|fused --query TEXT
rag eval --index INDEX --suite SUITE --out REPORT
rag visualize --index INDEX --out REPORT
rag serve --index INDEX
rag verify --index INDEX [--strict --snapshot SNAPSHOT]
```

Sampling is deterministic and scenario-blind. It can inspect structural fields
such as relation, document kind, occurrence-count bucket, and facet names, but
never semantic values, queries, qrels, incident labels, or known indicators.
The sample manifest records source identity, selection policy, relation budgets,
selected counts, and selected-document digest. A normal sample build does not
rescan the parent after selection.

Index open performs only constant- or metadata-scale checks: compatible format,
required files, vector header/profile/dimension/count, Parquet metadata, and
document/vector ordinal bounds. It never performs a corpus-wide verification
scan. Strict verification is a separate operator action and emits a reusable
receipt.

## 8. Embedding scheduler and measured LM Studio behavior

The development host currently runs LM Studio 0.4.20+1 with
`text-embedding-qwen3-embedding-8b` Q4_K_M, 4,096 dimensions, and no configured
parallel model workers.

A fixed benchmark on 13 August 2026 found:

| Mode | Result |
|---|---:|
| 8 short requests, concurrency 1 | 15.11 requests/s |
| concurrency 2 | 16.19 requests/s |
| concurrency 4 | 16.25 requests/s |
| concurrency 8 | 16.34 requests/s |
| batch 8 | 15.56 embeddings/s |
| batch 16 | 16.19 embeddings/s |
| batch 32 | 15.77 embeddings/s |
| batch 8, concurrency 4 | 16.58 embeddings/s |

All responses were HTTP 200 and repeated float32 vectors were byte-identical.
Increasing client concurrency increased latency but did not materially improve
throughput; the server accepted and queued concurrent work rather than
executing it in parallel.

Rust defaults are therefore:

```text
embedding_batch_size = 16
embedding_max_in_flight = 1
```

Both remain configurable, but a value greater than one is enabled only by a
recorded benchmark. The scheduler uses stable batch sequence numbers, validates
response order/cardinality/dimensions/finiteness, retries only failed batches,
applies bounded backpressure, and persists every successful batch immediately.
The cache key binds embedding profile, document ID/digest, and semantic-text
digest. Concurrency must never change document/vector association or ranking.

LM Studio's model-worker `--parallel` option is a separate controlled
experiment because it changes external model state and memory consumption. It
is not assumed by the client.

## 9. Current Python run postmortem

The former promotion process is no longer running. It left no published index.
Its durable SQLite cache is healthy and contains all 18,592 vectors. The
abandoned staging database contains:

| Table | Rows |
|---|---:|
| documents | 18,592 |
| occurrences | 1,915,260 |
| embedding_source_documents | 18,592 |
| embeddings | 585 |

The staging DuckDB is 12.77 GB; the reusable embedding cache is approximately
304 MB. The job completed source JSONL transfer and then stopped while Python
converted cached 4,096-element vectors to lists and inserted them with DuckDB
`executemany`. Model inference was not the failed retry's bottleneck.

The existing signed and pushed Python branch remains the behavioral oracle.
Rust work begins on `feature/rust-experimental-indexer-spec`. The abandoned
staging directory may be deleted after recording these counts; the embedding
cache is retained for conversion/parity testing.

## 10. Fast versus strict assurance

| Concern | `rag build` | `rag verify --strict` / admission |
|---|---|---|
| Source receipt | parse and retain | independently validate all identities |
| Parquet objects | metadata/count/columns | recompute object and logical digests |
| Projection | online invariants and counts | deterministic golden/replay sampling or full replay |
| Pointers | required snapshot-scoped event reference | hydrate selected/all references through OCSF host |
| Vectors | count, dimension, finite, norm | conformance fixture and full pairing/digest audit |
| Schemas | typed Rust construction | offline schema validation |
| Index open | metadata only | never substitutes for admission |
| Signing | none | host/test/production authority as applicable |

Strict verification results are reusable receipts keyed by exact source,
projection, embedding profile, and index identities. They are not rerun on
every query or cached rebuild.

## 11. Evaluation and visualization

Rust produces retrieval runs as canonical JSONL/Parquet. Python consumes only
the stable index and run artifacts. It owns:

- macro nDCG@20, Recall@5/10/20, MRR, and worst-paraphrase recall;
- hard-negative win rate and margin;
- dense/lexical/fused paired comparisons;
- PCA plots and exact original-space nearest-neighbor diagnostics;
- relation/source overlays and outlier review tables;
- bootstrap intervals and report rendering.

PCA/UMAP coordinates are visualization aids, never the retrieval distance or
anomaly statistic. An embedding-space outlier is not called malicious.

## 12. Migration plan

1. **Contracts and fixture reader.** Add the Rust workspace, deserialize the
   present RAG contracts, and implement an OCSF receipt/Parquet fixture reader.
2. **Projection parity.** Port activity/state/detection projection and compare
   exact semantic fields, group membership, and identifier/secret redaction to
   the Python oracle on varied frozen records.
3. **Direct writer.** Write document/occurrence Parquet and `vectors.f32`
   directly; demonstrate a cached rebuild without JSONL, DuckDB insertion, or
   parent replay.
4. **Embedding client.** Add resumable LM Studio batches with the measured
   defaults and concurrency equivalence tests.
5. **Retrieval.** Implement exact dense search, cached BM25, filters, and bound
   reciprocal-rank fusion.
6. **Provider.** Implement SDK handshake/open/call/health/close and replay the
   existing golden lifecycle and output-schema fixtures.
7. **Analysis package.** Point Python evaluation and visualization at the new
   index format.
8. **Settled OCSF adapter.** Update only `rag-ocsf` to the E9 release manifest
   after E6 reconciliation; then run a representative corpus and the frozen
   quality suite.
9. **Strict assurance.** Port or wrap strict verification and packaging only
   after the experimental loop demonstrates value.
10. **Portability.** Replace native-only lexical/filter adapters where needed
    and add a WASI provider variant. Query embedding becomes a host capability
    rather than direct network access from the Wasm guest.

## 13. Acceptance gates

### Correctness

- Rust/Python projection parity on every golden record.
- Every searchable document has exactly one vector and every vector maps to
  the intended document.
- Vectors have the bound dimension, are finite, and meet the normalization
  contract.
- Occurrence filters are applied before document eligibility; zero filter
  violations are tolerated.
- Every returned event reference hydrates through the bound OCSF host.
- Repeated runs and concurrency settings preserve deterministic tie order and
  document/vector association.
- A cached rebuild makes zero embedding calls.
- Normal index open performs no corpus-wide scan.

### Effectiveness

- Report macro nDCG@20, Recall@5/10/20, MRR, hard-negative wins/margins, and
  worst-paraphrase recall for dense, lexical, and fused retrieval.
- Demonstrate semantic wins over lexical retrieval without regressing exact
  controls or increasing unsupported conclusions in the structured runner.
- Keep structured OCSF operations authoritative for counts, chronology,
  joins, exhaustive populations, exact polarity, and negative evidence.

### Engineering

- A cached representative build completes in minutes, not hours.
- Projection/materialization throughput improves by at least 10x over the
  measured Python path on identical input.
- No JSONL bulk staging, Python `executemany`, parent replay, or full
  verification occurs in the default build.
- The Rust provider passes the existing SDK lifecycle and output-schema tests.
- The analysis package can produce the same report and PCA inputs from
  `documents.parquet` and `vectors.f32` without importing builder code.

## 14. Open decisions

These decisions are deliberately deferred until measurements exist:

1. Tantivy versus a portable custom postings format for lexical retrieval.
2. Qwen3 8B versus 4B and native 4,096 versus supported Matryoshka dimensions.
3. LM Studio single-worker versus explicitly loaded parallel model workers.
4. Exact scan versus HNSW after measuring exact-search latency and recall needs.
5. The final E9 release-manifest adapter shape.
6. The SDK host-capability shape for query embedding in Wasmtime.

None of these blocks the first Rust fixture-to-query vertical slice.
