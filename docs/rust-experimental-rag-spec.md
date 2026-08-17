# Rust experimental RAG specification

Historical vertical-slice specification. Its migration decisions remain useful
context, but M41 fixtures and Python parity work are not the active source path.
Current work uses normalized M45 data and the Rust/RunPod workflow in
[`runpod-embedding.md`](runpod-embedding.md).

For the next full-size index, the performance plan has moved to
[`portable-embedding-pipeline.md`](portable-embedding-pipeline.md). That design
keeps this document's query and tool boundary while splitting preparation,
embedding, and assembly into independently resumable stages.

Status: implemented experimental vertical slice; representative-corpus build
and production authority admission remain

Scope: fast local RAG experimentation over `livefire-ocsf` snapshots

Production admission: explicitly deferred

## 1. Decision

The next implementation is a Rust execution rewrite of the useful RAG
contracts, not another expansion of the current promotion and proof pipeline.
The implemented operator loop is:

```text
rag build -> rag query -> Python evaluate/PCA -> rag-provider JSONL lifecycle
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
- mint a production SDK authority receipt or signature (the packaging flow may
  mint an explicitly local-test receipt for conformance);
- prove the native source record behind an OCSF event;
- run JSON Schema validation over every output row;
- derive metric, network, entity, and state-transition overlays;
- build an ANN index before exact search is shown to be insufficient;
- claim the locally packaged native bundle is a production-admitted runtime;
- depend on Livefire source, hunt fixtures, qrels, expected evidence, or
  scenario indicators.

Those capabilities are not discarded. The intended later operator surface is:

```text
rag verify --strict
rag package
rag admit
```

These strict commands are design targets, not commands in the current Rust CLI.

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
mapping, schema, and relation identities. It also reconciles the runnable
snapshot, closure and completeness receipts, requires clean closure metrics,
and requires typed-relation rows to sum exactly to the normalized event count.
It does not hash every multi-gigabyte object again.

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
  rag-provider/     standalone SDK JSONL provider executable
  rag-testkit/      fixtures, Python-parity oracle, benchmarks
python/
  livefire_rag_analysis/
    index.py
    evaluate.py
    geometry.py
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
  lexical/index.json
  occurrence-index.sqlite3
  build-report.json
```

`documents.parquet` stores `document_id`, `document_sha256`, `document_kind`,
`semantic_text`, `facets_json`, `relations_json`, `occurrence_count`, and
`vector_ordinal`.

`occurrences.parquet` stores `occurrence_id`, `document_id`, optional
`event_time_ms`, `relation`, `exact_attributes_json`, `snapshot_sha256`,
`mapping_sha256`, `event_id`, and `support_ref`.

Provider candidates return at most 50 matching OCSF hydration references per
semantic document and always report the full eligible occurrence count plus an
exhaustion flag. Exact attributes remain index-side filter material and are not
copied into tool responses.

`vectors.f32` contains a 64-byte versioned header followed by contiguous
row-major little-endian float32 vectors. The header is `LFRAGV1\0`, header size,
version, f32-LE dtype, flags, count, dimensions, a reserved field, and the raw
32-byte document-order SHA-256. `index.json` binds that file and order to the
embedding profile. Python opens it directly with
`numpy.memmap`; Rust scans it without JSON or Arrow list conversion.

The lexical directory is derived and replaceable. The initial native adapter
may use Tantivy, but BM25 tokenization/profile identity and output semantics are
RAG contracts. A portable postings implementation can replace it for WASI.

The current builder streams Arrow batches and projects records without DuckDB
staging. Its first pass retains document identities and the bounded selected
document set. A second pass writes every selected occurrence to a temporary
JSONL spill, and the index writer consumes that spill in Parquet-sized chunks.
Only missing embeddings are submitted through the persistent SQLite cache;
vectors are read back and written one at a time rather than materialized as a
matrix in builder memory. The writer stages and atomically publishes the bound
Parquet, vector, lexical, occurrence-lookup, manifest, and build-report files.
The enriched JSON build report is also printed to stdout. It does not replay
the completed index through its parent.

Representative-build memory is proportional to the distinct-document census,
the selected documents, and lexical state, not occurrence fan-out or the vector
matrix. Full-build memory still grows with the searchable document set and the
initial JSON lexical index. Query open is metadata-scale and lazy. A compact
portable postings index remains an explicit scale and WASI gap.

## 7. Commands

```text
rag build --snapshot SNAPSHOT --out INDEX \
  --embedding-endpoint http://127.0.0.1:1234 \
  --embedding-profile PROFILE [--representative-sample]

rag query --index INDEX --mode dense|lexical|fused --query TEXT
rag batch-query --index INDEX --requests QUERIES.jsonl > RESULTS.jsonl
rag inspect --index INDEX

python -m livefire_rag_analysis inspect --index INDEX
python -m livefire_rag_analysis evaluate --run RUN [--qrels QRELS] \
  [--planned-query-id ID ...] [--out REPORT]
python -m livefire_rag_analysis pca --index INDEX --out REPORT_DIR

rag-provider  # JSONL requests on stdin, responses on stdout

rag-package-tool --provider PROVIDER --sdk-specs SDK_SPECS --out BUNDLE

rag-prepare-local-tool --index INDEX --bundle BUNDLE \
  --source-receipt SNAPSHOT/build-receipt.json \
  --embedding-profile PROFILE --out LOADOUT
```

When a frozen experiment can legitimately return no hits, pass every planned
query ID to the evaluator. Missing ranking rows then remain explicit zero-score
queries instead of disappearing from macro metrics.

The package command emits a content-closed SDK plugin bundle. The loadout
command leaves the physical index unchanged and atomically creates a separate
same-filesystem wrapper with hard links to its verified objects. It binds
the exact snapshot and mapping-pack components from its admitted build receipt,
and emits a local-test admission receipt, tool-binding lock, four declared
read-only mount bindings, and a provider transcript. OS immutability is enforced
externally. This local receipt is deliberately not a production
authority claim. The packaged provider returns only
`livefire.ocsf-hydration-ref/1` candidate handoffs; semantic previews and raw
source records are excluded. A returned event becomes evidence only after an
authoritative OCSF hydration operation resolves and verifies its snapshot,
mapping, event ID, and support reference.

Sampling is deterministic and scenario-blind. The current
`--representative-sample` policy discovers every searchable typed relation,
declares a census at or below 1,000 documents, and applies a 2,000-document
snapshot-bound hash-min cap above that threshold. Because the cap exceeds the
census threshold, relations with 1,001 through 2,000 documents are also fully
retained; only larger relations are reduced. The report records both the
declared threshold/cap and this effective full-retention boundary.
The report explicitly records the source-document census, per-relation budget,
selected composition, and policy identity. It also reconciles source rows by
relation and terminally accounts for non-searchable metric rows as
structured-only observations. Its separate semantic-source coverage flag is
false for every representative build and for a full searchable build whenever
structured-only source rows exist; physical index completeness is not a claim
that all source rows are searchable. Empty and structured-only relations cannot
consume quota. Selection never reads semantic values, queries, qrels, incident
labels, or known indicators. `index.json` records
`build_scope: "sample"` and `complete: false`. The builder scans and projects
the typed snapshot once to select documents and a second time to spill every
occurrence for the final selection, bounding retained memory without weakening
membership closure. Sampling therefore bounds embedding and retained state but
not source scan time.

Python index open validates the closed manifest shape, required files, vector
header/profile/dimension/count, Parquet row counts, contiguous vector ordinals,
document-order digest, and vector finiteness/normalization. The Rust reader
performs metadata checks at open and initializes query data lazily, then checks
document/vector ordinals, lexical membership, occurrence counts and exact
snapshot/mapping bindings before returning a hit. It rechecks canonical path
containment at lazy reads. The OCSF adapter streams and rehashes each typed
Parquet object against its admitted receipt immediately before every scan; the
SDK wrapper and provider likewise stream-check every index artifact. Production
authority admission remains a host action.

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
This paragraph describes the legacy single-command `rag build` path. Increasing
client concurrency there increased latency without materially improving
throughput; the server accepted and queued concurrent work.

The implemented Rust client therefore defaults to:

```text
embedding_batch_size = 16
embedding_requests_in_flight = 1 (fixed)
```

Batch size is configurable from 1 through 32. Request concurrency is not
configurable: missing vectors are processed as sequential batches with one HTTP
request in flight. The client validates response indices and cardinality, then
validates every vector's dimensions, finiteness, and configured normalization
before writing it to the SQLite cache. It preserves input order and the cache
key binds embedding profile, document ID/digest, and semantic-text digest.

That legacy path has no retry scheduler. The preferred portable `rag embed`
path does implement bounded concurrent batches, temporary-failure retries,
crash-safe task parts, explicit recovery, and restart from sealed receipts.

LM Studio 0.4.20+1 accepted an embedding-model load request with `--parallel
2`, but neither `lms ps` nor the local REST model configuration exposed the
setting. The experiment therefore did not claim a parallel-2 or parallel-4
embedding run. The supported measured default remains four inputs and one
request in flight.

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

The Rust query command emits one JSON result. The smoke runner converts a query
set into a stable JSONL retrieval run, and Python consumes only the index/run
artifacts. The implemented analysis package provides configurable-cutoff macro
nDCG and Recall, MRR, per-query metrics when human relevance labels (qrels) are supplied, explicit
diagnostics when they are not, and PCA plots with original-space centroid
distance markers.

Worst-paraphrase recall, hard-negative wins/margins, dense/lexical/fused paired
statistics, relation/source overlays, nearest-neighbor review tables, bootstrap
intervals, and richer report rendering remain planned work for the blinded
representative suite.

PCA/UMAP coordinates are visualization aids, never the retrieval distance or
anomaly statistic. An embedding-space outlier is not called malicious.

## 12. Migration plan

1. **Implemented — contracts and current snapshot reader.** The Rust workspace
   reads the present embedded build receipt and scans manifested typed Parquet.
2. **Implemented for the admitted relation set — generic projection.** Direct
   activity, state, and detection projection, grouping, redaction, and
   accounting have Rust fixtures. A deterministic 4,128-row comparison across
   all 17 searchable relation types matched the Python implementation exactly.
3. **Implemented for the experiment — direct writer.** The builder emits
   document/occurrence Parquet, `vectors.f32`, lexical JSON, and bound manifests
   without JSONL, DuckDB insertion, or parent replay. Build memory remains
   proportional to the retained set.
4. **Implemented — embedding clients.** The legacy `rag build` path has LM
   Studio batching, strict response validation, and a persistent resumable
   cache, but remains sequential. The preferred portable `rag embed` path adds
   bounded requests in flight, retries temporary failures with backoff, validates
   returned model identity and vector order, publishes one restart-safe shard
   per task, and finalizes only complete result sets. The 512-, 2,000-, and
   10,000-document local runs measure throughput and show that extra client
   requests did not help while LM Studio had one prediction slot.
5. **Implemented — retrieval.** Exact streamed dense scoring, BM25, relation/time
   filters, deterministic ordering, and bound reciprocal-rank fusion exist.
6. **Implemented and SDK-tested locally — provider.** The native executable
   implements handshake/open/call/health/close. A content-closed plugin bundle,
   SDK index wrapper, exact binding lock, four declared read-only mount
   bindings, and an
   explicitly local-test admission receipt are generated and validated through
   the adjacent SDK harness.
   Search output is always non-definitive partial candidate coverage: even a
   physically complete full-scan index excludes structured-only observations
   from semantic retrieval, and every returned occurrence requires hydration.
   Its provider, descriptor, schema, profile, and physical-index identities are
   content-bound; no production authority admission is claimed. The direct
   executable trusts the operator to provide
   an OS-enforced immutable index mount for the session. The provider retains an
   open vector-file handle, while documents and the occurrence SQLite lookup are
   opened lazily by path. Local digest and association checks cannot prevent
   path replacement or in-place mutation by a writable peer; OS immutability is
   therefore enforced externally as a production host admission/sandbox gate,
   not provided by this local harness.
7. **Implemented — analysis package.** Python strictly reads the Rust manifest,
   Parquet, and vector header; it provides run evaluation and PCA PNG/report.
8. **Blocked on upstream qualification — settled OCSF adapter.** Update only
   `rag-ocsf` for the final release manifest, then run a representative corpus
   and the frozen blinded quality suite.
9. **Partly implemented — strict assurance.** Bundle closure, exact physical
   artifact digests, SDK schema validation, local-test binding, output schema,
   and synthetic hydration-reference closure are automated. Production signing,
   host sandbox admission, and an authoritative released-snapshot hydration run
   remain operator/integration gates.
10. **Deferred — portability.** Replace native filesystem/HTTP/lexical details
    where necessary and add a WASI provider. Query embedding should become a
    host capability rather than guest network access.

### 12.1 Reproducible vertical-slice smoke

`tools/run_rust_smoke.py` generates six generic typed OCSF records, builds a
real 4,096-dimensional index through the locally served Qwen3 embedding model,
runs six queries, evaluates a JSONL run, writes PCA artifacts, and exercises the
standalone provider's complete JSONL lifecycle over the generated index:

```sh
uv run --extra analysis python tools/run_rust_smoke.py \
  --work /tmp/livefire-rag-smoke --mode fused
```

On 13 August 2026 the fused smoke produced six documents, six occurrences, six
vectors, 36 run rows, a provider pointer response, and rank-one retrieval for
all six synthetic relevance labels (MRR, Recall@1, and nDCG@1 all `1.0`). This is an
interface and binding sanity check only. The runner also performs a cached
rebuild against an unreachable model endpoint, requires zero embedding calls,
compares stable index artifacts byte-for-byte, and validates the actual provider
pointer output against the packaged fast-search schema. The fixture and relevance labels
are generated together, the corpus is tiny, and the result must not be presented
as evidence that retrieval improved. The effectiveness decision requires a
representative result pool whose system labels are hidden from reviewers,
human-reviewed relevance labels, and lexical/dense/fused comparison.

## 13. Acceptance gates

### Correctness

| Gate | Status |
|---|---|
| Rust/Python parity for the checked-in representative projection records | Achieved by the golden projection tests |
| One bound vector per searchable document, with stable document order | Achieved by writer/reader and cross-language tests |
| Dimension, finiteness, and normalization validation | Achieved in the embedding client and Python reader |
| Occurrence-first relation/time filtering | Achieved by focused Rust tests |
| Every returned reference hydrates through the qualified OCSF host | Pending the released OCSF snapshot/host |
| Rebuild preserves document/vector association and stable artifacts | Achieved by the cached smoke rebuild; concurrent scheduling is not implemented |
| Cached rebuild makes zero embedding calls | Achieved by the unreachable-endpoint smoke rebuild |
| Normal Rust index open performs no corpus-wide scan | Achieved by the metadata-open test |

### Effectiveness

All effectiveness gates remain pending. The synthetic smoke exercises metric
calculation but cannot satisfy them:

- report macro nDCG@20, Recall@5/10/20, MRR, hard-negative wins/margins, and
  worst-paraphrase recall for dense, lexical, and fused retrieval on the blinded
  representative suite;
- demonstrate semantic wins over lexical retrieval without regressing exact
  controls or increasing unsupported conclusions in the structured runner;
- keep structured OCSF operations authoritative for counts, chronology, joins,
  exhaustive populations, exact polarity, and negative evidence.

### Engineering

| Gate | Status |
|---|---|
| Cached representative build completes in minutes | Pending representative snapshot measurement |
| Projection/materialization is at least 10x the measured Python path | Pending identical-input benchmark |
| Default build avoids JSONL staging, Python `executemany`, parent replay, and full verification | Achieved by the Rust vertical slice |
| Provider SDK bundle/lifecycle and pointer/miss schema validation | Achieved with explicit local-test admission; production authority admission remains pending |
| Python evaluation/PCA reads Parquet and `vectors.f32` without builder imports | Achieved by focused tests and the real smoke |

## 14. Open decisions

These decisions are deliberately deferred until measurements exist:

1. Tantivy versus a portable custom postings format for lexical retrieval.
2. Qwen3 8B versus 4B and native 4,096 versus supported Matryoshka dimensions.
3. LM Studio single-worker versus explicitly loaded parallel model workers.
4. Exact scan versus HNSW after measuring exact-search latency and recall needs.
5. The final E9 release-manifest adapter shape.
6. The SDK host-capability shape for query embedding in Wasmtime.

None of these blocks the first Rust fixture-to-query vertical slice.
