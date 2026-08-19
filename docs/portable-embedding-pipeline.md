# Portable embedding pipeline specification

Status: Rust preparation, local embedding, recovery, scalable assembly, and
modular search implemented; M45 RunPod execution is the active cloud phase

This document replaces the single-command build path as the preferred path for
large indexes. The existing `rag build` command remains useful for small smoke
tests. Earlier sections record the completed M41 local run and fixed
10,000-document benchmark as historical measurements. Bounded document runs
and an external merge removed the former 600,000-document in-memory limit. The
active source is now the normalized `livefire-ocsf` M45 build; direct M21,
M41, and OpenBOTS inputs are not supported by the active builder.

The detailed execution order is in
[`local-first-embedding-scale-plan.md`](local-first-embedding-scale-plan.md).
That plan records the completed local LM Studio measurements and recovery tests.
It is a historical phase record. Current cloud execution is specified in
[`runpod-embedding.md`](runpod-embedding.md).

## 1. Outcome

Prepare searchable text once per dataset, then generate embeddings with any
supported compute backend and assemble an independently queryable Rust index.
Small and large datasets use the same contracts.

The pipeline has three independent stages:

```text
OCSF snapshot
    |
    v
rag prepare       CPU work; no model or network required
    |
    v
prepared corpus   portable, immutable Parquet shards
    |
    v
rag embed         local LM Studio, or sealed RunPod worker output
    |
    v
embedding set     portable binary vector shards
    |
    v
rag assemble      CPU and disk work; no model required
    |
    v
fast RAG index
```

The prepared corpus is independent of the embedding model. An embedding set is
not: it binds one exact model and execution profile.

Each dataset is a complete build unit:

```text
dataset source
    -> prepared dataset
    -> embedding set
    -> assembled dataset index
    -> CLI search
```

The local implementation proved this path across the earlier M41 non-network
scope. The active M45 build repeats the same model-independent preparation and
assembly contracts with a separately identified cloud embedding profile.

## 2. Goals

1. Project the source data once rather than once per embedding experiment.
2. Prepare every searchable relation in bounded memory using Rust concurrency.
3. Use the same prepared documents with LM Studio, Runpod, or future workers.
4. Let multiple workers process independent embedding shards safely.
5. Resume after interruption without repeating completed shards.
6. Avoid JSON representations of occurrence rows and vectors.
7. Assemble the final index without loading the complete vector matrix or
   occurrence set into memory.
8. Preserve exact document-to-vector and document-to-event associations.
9. State excluded data plainly in the prepared corpus and final index.
10. Make small experiments quick without weakening the full-build contracts.
11. Build, replace, verify, and query datasets independently.
12. Add datasets without re-embedding or reassembling existing dataset indexes.

## 3. Non-goals for the first version

- Preparing network-flow activity does not require committing to embed it in a
  later backend run; each relation remains an independent build unit.
- Individual system metrics are not embedded as ordinary text documents.
- The first Runpod implementation does not automatically provision arbitrary
  cloud infrastructure.
- Different model weights or numeric formats are not mixed in one embedding
  set.
- The first implementation does not promise a one-hour build. It measures the
  real rate before scheduling the full job.
- Prepared semantic text is not assumed safe for public storage. Cloud storage
  remains private and access-controlled.

## 4. Current full-corpus scope

The active preparation uses the accepted M45 OCSF snapshot and publishes every
searchable relation as an independent dataset, including
`ocsf_network_activity`. A later embedding run may select a smaller set without
re-preparing the source.

DNS, HTTP, API, authentication, process, file, email, detection, event-log,
configuration, inventory, datastore, cloud, and identity-related relations
remain included. Excluding raw network activity must not accidentally exclude
these higher-level relations.

Expected M45 accounting from the unchanged normalized Parquet bytes, confirmed
by the current Rust census:

| Item | Count |
|---|---:|
| Source rows examined by the existing complete accounting | 13,905,577 |
| Searchable occurrences | 6,367,276 |
| Distinct searchable documents | 560,842 |
| Network occurrences in their own dataset | 1,042,076 |
| Network documents in their own dataset | 138,276 |
| Metric rows recorded as structured-only | 7,538,301 |
| Searchable relation types prepared | 18 |

The prepared manifest records both included and excluded relations. The final
tool must never describe a miss as proof that an event is absent from excluded
raw network or structured metric data.

### 4.1 Problems this design removes

The current combined builder is useful as a working reference, but it is not
the full-scale execution design. It currently:

- hashes and scans searchable source objects twice;
- parses and projects the same searchable JSON twice;
- retains the selected document set in a corpus-sized ordered map;
- writes occurrences as JSONL and parses them again during index creation;
- sends one embedding request at a time through one SQLite cache connection;
- creates a corpus-sized JSON lexical index; and
- maintains occurrence lookup indexes while inserting millions of rows.

The portable pipeline removes those costs stage by stage while retaining the
existing projection and identity rules.

## 5. Commands

The implemented first-milestone commands are:

```text
rag prepare --snapshot SNAPSHOT --dataset-id ID --relation RELATION --out PREPARED
rag verify-prepared --prepared PREPARED
rag plan-embeddings --prepared PREPARED --embedding-profile PROFILE \
  --tokenizer-json TOKENIZER_JSON --tokenizer-ref TOKENIZER_REF \
  --maximum-task-tokens 131072 --maximum-task-documents 256 --out PLAN
rag embed --prepared PREPARED --plan PLAN --embedding-profile PROFILE --out EMBEDDINGS
rag finalize-embeddings --prepared PREPARED --plan PLAN \
  --embedding-profile PROFILE --embeddings EMBEDDINGS
rag assemble --prepared PREPARED --plan PLAN --embeddings EMBEDDINGS \
  --embedding-profile PROFILE --index-format sqlite-v3 --out INDEX
rag inspect --index INDEX
rag query --index INDEX --mode fused --query QUERY
```

Each command checks the files it can use. Planning and embedding check the
prepared manifest and document files without opening occurrence files. The
read-only `verify-prepared` command and assembly check every document and
occurrence file. Local embedding uses LM Studio. Cloud embedding uses the Rust
RunPod controller and `rag-runpod-worker`; both publish the same portable
vector-shard and receipt chain under different exact embedding profiles.

The checked-in first scope file is preferable to a loosely typed set of CLI
flags. It declares the 16 included relations, raw network exclusion, and metric
handling. The CLI may also accept explicit relation overrides for development,
but the resolved scope is always written to the manifest.

### 5.1 Dataset build unit

A dataset build unit is one admitted source dataset plus one resolved relation
scope and projection policy. It has a stable dataset ID and produces its own:

- prepared-corpus manifest;
- embedding plan and result set;
- assembled fast index;
- build and performance report; and
- optional Livefire tool package.

The dataset ID is descriptive metadata. Dataset identity comes from the bound
source component, mapping, projection policy, and resolved scope. Two snapshots
with the same friendly name remain different datasets when their source
identity differs.

No dataset build may modify another dataset's prepared files, embeddings, or
index. Rebuilding one dataset publishes a new directory atomically, after which
a catalogue or loadout may choose the new version.

The implemented catalogue commands register, validate, and search several
completed dataset chains. Each repeated `--dataset` group supplies that
dataset's prepared corpus, plan, embedding result, and assembled index:

```text
rag catalogue build \
  --dataset PREPARED_A PLAN_A RESULTS_A INDEX_A \
  --dataset PREPARED_B PLAN_B RESULTS_B INDEX_B \
  --out CATALOGUE
rag catalogue validate --catalogue CATALOGUE
rag catalogue search --catalogue CATALOGUE --mode fused --query QUERY
```

The catalogue is a list of immutable index references, not a merged corpus. A
query searches compatible indexes independently and merges their returned hits
with a declared stable ranking rule. Every returned hit includes the dataset ID
and index component identity. The stable hit identity is therefore
`(dataset_component, document_id)`, so the same projected text in two source
datasets does not lose either source history.

The catalogue validates each complete prepared/plan/result/index chain and
binds the dataset and index identities in its own sealed manifest. Scalable
fast-index v3 is implemented with SQLite lexical postings; legacy v2 remains
available for compatibility. Dataset identity is still catalogue-level rather
than inferred from source and mapping hashes alone.

Existing dataset indexes are not rebuilt. Indexes may share one query embedding
only when they bind the exact same embedding profile. Different profiles
require separate query embeddings. Raw lexical scores are never compared
across indexes because each dataset has different document-frequency
statistics. Catalogue search merges per-index ranks with a fixed reciprocal-rank
fusion rule and uses dataset component plus document ID as the final tie-break.
Raw dense scores may be reported for inspection but are not silently mixed
across incompatible profiles.

## 6. Prepared corpus contract

### 6.1 Layout

```text
prepared-corpus/
  manifest.json
  accounting.json
  scope.json
  documents/
    part-000000.parquet
    part-000001.parquet
    ...
  occurrences/
    ocsf_api_activity/
      part-000000.parquet
    ocsf_authentication/
      part-000000.parquet
    ...
  objects.sha256
```

There is no JSONL staging file and no vector data in this artifact.

The dataset identity binds the source snapshot and mapping. M45 manifests also
carry a sorted `source_admission` list containing the exact relation-contract
and capability-sidecar identities. Older prepared manifests omit that optional
field and retain their original bytes; an M45 manifest without both entries is
not accepted as the current copy-ready corpus.

### 6.2 Document rows

Each document shard uses a fixed Arrow schema:

| Column | Type | Meaning |
|---|---|---|
| `document_ordinal` | non-null unsigned 64-bit | Position in canonical prepared-corpus order |
| `document_id` | non-null UTF-8 | Stable projection identity |
| `document_sha256` | non-null 64-char lowercase hex | Hash of the complete projected document |
| `semantic_text_sha256` | non-null 64-char lowercase hex | Hash of the exact text sent to a model |
| `semantic_text` | non-null UTF-8 | Model-independent document text |
| `document_kind` | non-null UTF-8 | `activity`, `state`, or `detection` |
| `primary_relation` | non-null UTF-8 | OCSF relation that owns the document |
| `facets_json` | non-null UTF-8 | Canonical JSON facets for filters and inspection |
| `relations_json` | non-null UTF-8 | Canonical JSON relation list |
| `occurrence_count` | non-null unsigned 64-bit | Number of retained event references |

`document_ordinal` is assigned after the deterministic merge and is contiguous
from zero. Assembly maps it to the final index's `vector_ordinal`.

### 6.3 Occurrence rows

Occurrence shards retain the current fast occurrence fields and add a stable
source position:

- occurrence ID;
- document ID;
- event time in milliseconds when available;
- relation;
- source row ordinal within that relation;
- exact projected attributes;
- snapshot hash;
- mapping hash;
- event ID;
- support reference.

Occurrence files remain local during cloud embedding. GPU workers never need
them.

The source row ordinal is independent of Arrow batch size, Parquet row-group
scheduling, and worker count. It makes `(relation, source_row_ordinal)` a cheap
deterministic occurrence order without a corpus-wide sort.

### 6.4 Ordering and sharding

Canonical document order is ascending unsigned UTF-8 `document_id` bytes. The whole-corpus
document-order digest retains the current rule: SHA-256 over each document ID
followed by a zero byte.

Prepared document shards close at 2,048 rows or 32 MiB of semantic UTF-8 text,
whichever comes first. A single oversized valid row forms its own shard. This
produces about 248 shards for the first non-network build when the row limit is
dominant.

Physical preparation shards and model-specific embedding tasks are separate.
An embedding plan may split or combine consecutive ordinal ranges without
regenerating the prepared corpus.

Each shard entry records:

- ordinal;
- relative path;
- row count;
- byte count;
- file SHA-256;
- first and last document IDs;
- document-order SHA-256;
- embedding-input-order SHA-256.

The embedding-input digest is SHA-256 over a fixed domain separator followed by
the ordered document ID, document hash, and semantic-text hash, with zero-byte
separators. This binds the exact work without binding an embedding model.

Occurrence fragments are canonically ordered by relation ordinal, source row
group, and source row ordinal. The manifest lists the fragments in that order.
This avoids sorting more than five million rows while remaining deterministic.
Their logical order digest permits a future Parquet re-encoding without
changing the meaning of the corpus.

### 6.5 Manifest identity

`manifest.json` uses schema `livefire.rag.prepared-corpus/1` and binds:

- dataset ID and version;
- exact source snapshot and mapping components;
- dataset, OCSF schema, extension-pack, and relation-contract identities when
  the accepted source receipt provides them;
- projection policy;
- resolved include/exclude scope;
- document and occurrence schemas;
- every object path, size, row count, and SHA-256;
- whole-corpus counts and order digests;
- per-relation source, searchable, selected, and excluded counts;
- terminal metric handling;
- preparation implementation version;
- its own canonical component digest.

The component digest uses the repository's canonical JSON convention. It is
not computed over pretty-printed bytes.

One later cleanup is to narrow the preparation implementation identity. It
currently hashes the whole `portable.rs` module, so changes confined to later
planning or embedding stages give newly prepared datasets a new implementation
digest even when preparation behavior is unchanged. Existing prepared
artifacts remain valid because each carries its own sealed identity.

## 7. Implemented parallel Rust preparation

Preparation scans admitted Parquet row groups in bounded parallel waves and
flushes occurrence Parquet every 8,192 rows. It writes bounded sorted document
runs and merges them in document-ID order, so it no longer retains the complete
deduplicated document table in memory.

### 7.1 Work units

The unit of CPU work is an admitted Parquet row group, not an entire relation.
This matters because configuration data is much larger than most relations.

Full-scale preparation performs these steps:

1. Read and validate the source receipt.
2. Resolve the included relation objects.
3. Stream-hash every included source object once.
4. Enumerate its Parquet row groups.
5. Submit `(relation, row_group)` work items to a bounded worker pool.
6. Project each source row once into a document candidate and occurrence.
7. Write deterministic occurrence fragments and bounded, sorted document runs.
8. Merge document runs, remove identical duplicates, and reject conflicts.
9. Create canonical document and occurrence shards.
10. Verify accounting and atomically publish the prepared corpus.

Metrics and explicitly excluded raw network data are accounted from the
accepted source receipt and scope. They are not parsed merely to prove that the
pipeline intentionally excluded them.

### 7.2 Crates and execution model

Use the existing Arrow 58 and Parquet 58 crates directly.

- `ArrowReaderMetadata::load` reads each Parquet footer once.
- `ParquetRecordBatchReaderBuilder::new_with_metadata` creates independent
  readers from the shared metadata.
- `with_row_groups` and `ProjectionMask::roots` read only the assigned row
  groups and columns.
- `parquet::arrow::ArrowWriter` writes Zstandard-compressed fragments and final
  shards.
- A private Rayon thread pool runs CPU-heavy parsing, projection, hashing, and
  sorting. It does not use the global Rayon pool.
- `crossbeam-channel` provides bounded blocking backpressure where a dedicated
  writer is required.
- Tokio is not used for projection. It remains responsible for HTTP and cloud
  orchestration.
- `rusqlite` stores mutable build progress only when a durable work ledger is
  needed; immutable Parquet and manifests remain the portable contract.

The implemented local scale path uses these crates:

| Purpose | Crate |
|---|---|
| Columnar input and output | Arrow and Parquet 58 |
| CPU scheduling | Rayon bounded thread pool |
| HTTP scheduling | Tokio tasks plus bounded in-flight requests |
| Lexical index | deterministic SQLite inverted BM25 index |
| Occurrence lookup and work ledger | rusqlite 0.38 |
| Hashing | sha2 with SHA-256 |
| Timings and build reports | machine-readable stage reports |

The implementation uses a bounded Rayon pool and deterministic source-order
merge. A clean one-row-group fixture produced identical output at every worker
count. Median projection time improved from 1,792,597 microseconds with one
worker to 836,900 with four workers and 705,621 with eight workers, or 2.14 and
2.54 times faster. These fixture measurements prove useful CPU concurrency;
they are not a whole-corpus forecast.

Do not add Polars to the core path initially. Polars is valuable for analysis
and ad hoc reports, but the core work consists of custom JSON projection,
identity hashing, ordered deduplication, Parquet writing, and binary-vector
assembly. Direct Arrow gives clearer row-group control and avoids a second
execution engine. A benchmark may reconsider Polars for the external merge
only if it demonstrates a material improvement on the fixed corpus.

Preparation buffers at most a configured number of unique document candidates,
sorts them by `document_id`, combines identical candidates, and writes a
temporary sorted JSONL run. A deterministic heap merge opens the completed runs,
deduplicates them, and streams the final Parquet document shards. The current
implementation bounds rows per run but does not yet bound merge fan-in; building
each M41 relation separately keeps the number of runs manageable.

### 7.3 Bounded memory

No stage may retain all source occurrences or all vectors in memory.

The implemented public controls with hard bounds are the preparation worker
count and `LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS` for sorted-run rows. Other
batch and shard bounds are internal constants. Future tuning may expose more of
them, but the following are design targets rather than current CLI flags:

- worker threads;
- Arrow batch rows;
- open relation writers;
- channel capacity;
- fragment rows and bytes;
- merge fan-in;
- embedding requests in flight;
- embedding batch items and tokens.

The build report records peak resident memory when the platform exposes it.
In particular, byte-based candidate limits, Parquet row-group sizing, merge
fan-in, and verification worker counts are not public controls yet.

## 8. Embedding plan contract

### 8.1 Semantic profile and executor settings

Keep semantic-space identity separate from executor tuning. The sealed semantic
profile binds exact model weights and revision, tokenizer, document formatting,
query instruction and formatting, maximum input tokens, overflow behavior,
pooling, dimensions, optional reduced dimension, normalization and tolerance,
output dtype, and similarity calculation. The initial compatibility rule is
`exact_profile_only`.

Backend aliases, endpoint URLs, credentials, GPU type, request batch limits,
requests in flight, retry policy, and lease duration are executor settings. They
do not change plan identity. The task receipt records the executor and actual
loaded artifacts so a mismatch is detectable.

A local Q4 GGUF model and a cloud Safetensors model served at a different
verified runtime dtype use different semantic profiles unless a separately
versioned compatibility test proves that they may share an index. Model names
alone never establish compatibility.

### 8.2 Plan contents

`rag plan-embeddings` validates the prepared manifest and document objects
against one exact embedding profile and emits `livefire.rag.embedding-plan/2`.
Occurrence objects are left to full verification and assembly because planning
cannot use them.

The plan binds:

- prepared-corpus component digest;
- embedding profile digest;
- model artifact and revision;
- tokenizer and maximum input length;
- pooling and normalization;
- output dimensions and dtype;
- document input formatting;
- every expected preparation shard and order digest;
- every consecutive embedding task, its input slices, ordinal range, and order
  digest;
- the expected result path and identity for every task.

Token counts are model-specific and are calculated here rather than during
model-independent preparation. Tasks target one to five warm GPU minutes and
obey configured item, text-byte, and token ceilings. Endpoint locations,
credentials, batch size, worker count, and retry settings are execution details,
so the same plan can run locally or in the cloud. Over-length documents fail
planning unless the profile explicitly defines a deterministic truncation
rule. The first profile continues to reject over-length input.

## 9. Embedding result contract

### 9.1 Layout

```text
embedding-set/
  manifest.json
  summary.json
  embedding-profile.json
  parts/
    part-000000.f32
    part-000001.f32
    ...
  receipts/
    part-000000.json
    part-000001.json
    ...
  reports/
    part-000000.json
    part-000001.json
    ...
```

One result part corresponds to exactly one planned embedding task.
The finalized directory keeps the exact embedding-profile input so later
catalogue checks can reproduce every summary field without relying on a file
outside the result set. The summary reports both the full calendar span and
the union of active task intervals; pauses between resumed ranges do not lower
the published active-throughput rate.

### 9.2 Binary vector part

Each part has a 64-byte little-endian header followed by contiguous row-major
float32 vectors:

```text
magic[8]                 = "LFREMB01"
header_bytes u32         = 64
version u16              = 1
dtype u8                 = 1 (float32 little-endian)
flags u8                 = 0
row_count u64
dimensions u32
reserved u32             = 0
embedding_task_order_sha[32]
```

Document IDs are not repeated beside every vector. The planned task and its
order digest provide the association.

The implemented portable embedding shards use the `LFREMB01` header shown
above. Readers validate that header, dimensions, row count, task order, and
payload length before accepting a shard.

For 422,566 documents at 4,096 dimensions, the vector payload is 6,923,321,344
bytes, approximately 6.92 GB before headers and filesystem overhead. JSON
vectors are forbidden as durable output.

### 9.3 Per-task receipt

A completed task receipt binds:

- plan and prepared-corpus digests;
- task ID, ordinal range, and every input slice's path, SHA-256, rows, and
  input-order digest;
- embedding profile digest;
- executor implementation and container/runtime digest;
- returned model identity;
- vector path, SHA-256, bytes, rows, dimensions, and dtype;
- validation of finiteness and configured normalization;
- request, retry, token, time, and throughput measurements;
- conformance-test result.

Credentials and endpoint secrets never appear in a plan, receipt, or log.

### 9.4 Completion and retries

A task becomes complete only after the worker validates the entire output,
flushes it, atomically publishes it, and publishes its receipt.

Partial files are ignored on restart. A scheduler resubmits only tasks without
valid receipts. Duplicate attempts are acceptable only when their vector
digests agree. Different valid outputs for the same input and exact profile are
a hard failure, not a last-writer-wins condition.

## 10. Embedding backends

### 10.1 LM Studio

The LM Studio adapter retains the existing strict loopback-only network policy
and calls the OpenAI-compatible `/v1/embeddings` endpoint.

It adds:

- bounded requests in flight;
- item and token batch limits;
- exponential backoff with jitter for temporary transport, timeout, 408, 429,
  and selected 5xx failures;
- terminal handling for invalid profile or vector responses;
- atomic completion per prepared shard;
- optional import and export through the existing SQLite content cache.

The measured local default is one request in flight and four inputs per
request. On the 512-document screen it reached 2.737 documents and 775.17
tokens per second. Larger batches were slower, and two or four requests in
flight only queued behind LM Studio's single prediction slot.

The executor uses Tokio rather than Rayon: a producer reads prepared Parquet,
a semaphore limits requests in flight, a `JoinSet` owns request tasks, and a
bounded reorder buffer restores task order before one atomic vector-shard
writer. The queue holds no more than twice the configured in-flight request
count.

### 10.2 Generic OpenAI-compatible client boundary

The Rust TEI client supports the bounded OpenAI-compatible wire format required
inside the worker. It does not weaken the local provider's loopback-only
policy or create a public inference endpoint.

The adapter accepts credentials only through an environment variable or host
secret provider. The endpoint origin, allowed path, certificate validation,
timeouts, response limits, and model ID are explicit policy.

Sending the complete corpus and returning all vectors through ordinary host-to-
cloud JSON HTTP is not the RunPod bulk path.

### 10.3 Active RunPod bulk-worker design

The current cloud implementation uses a dedicated RunPod Pod:

1. Upload the prepared manifest and document shards to private object storage or
   a network volume.
2. Start the digest-pinned custom executor image containing the Rust worker and
   a separately pinned Text Embeddings Inference base.
3. Process every unfinished shard near the GPU.
4. Write binary result parts and receipts directly to persistent storage.
5. Download the completed embedding set and stop the Pod.

The inference engine is Hugging Face Text Embeddings Inference with the pinned
upstream Qwen3-Embedding-8B checkpoint. It listens only on loopback within the
worker container.

Multiple workers write unique attempt-scoped shard paths. They never append
concurrently to one shared output file. The host fetches deterministic
completion markers and only the exact objects named by those markers.

RunPod Serverless remains future work. In that design, one idempotent job would
process one task. A job would receive only corpus/profile digests, a task ID,
and object locations. A worker would write vectors and its receipt to storage;
the job response would contain only the receipt location. Retried jobs could
accept an existing result only when its digest matches.

### 10.4 Model compatibility

Prepared documents are portable; embeddings are not automatically portable
between model builds.

The present local profile uses a Q4 GGUF model through LM Studio. A RunPod TEI
deployment uses Safetensors with a runtime dtype that must be observed and
recorded rather than assumed. These create separate embedding profiles and,
initially, separate indexes. Queries must use the profile that matches the
indexed documents.

An exact GGUF worker on RunPod may later be declared compatible with local LM
Studio only after a fixed comparison proves acceptable vector and ranked-result
agreement. Matching the model name alone is insufficient.

## 11. Assembly

`rag assemble` verifies the prepared corpus and embedding set before writing
the final index.

It:

1. Requires exactly one completed vector part per planned document shard.
2. Rejects missing, extra, overlapping, or differently profiled parts.
3. Streams documents in canonical order and assigns vector ordinals.
4. Concatenates validated vector payloads without creating a vector matrix.
5. Streams occurrence Parquet into the event-reference lookup.
6. Builds the selected v2 or scalable v3 lexical index.
7. Writes the final manifest and build report.
8. Atomically publishes the index.

The implemented assembly streams documents, occurrences, and vectors. Version
2 retains the original lexical JSON for compatibility. Version 3 builds a
deterministic SQLite inverted index with the same `ascii_camel_lower_v1`
tokenizer, BM25 formula, filtering, tie order, and hit results. Queries read
only matching postings through independent read-only connections. The v3
manifest binds the exact SQLite storage schema and object digest.

Preparation's bounded sorted runs and external merge removed the former
600,000-document in-memory ceiling. Assembly and version-3 query already stream
their corpus-sized inputs.

Occurrence lookup remains SQLite. Assembly bulk-loads rows in one exclusive
transaction without secondary indexes, creates unique and query indexes after
the load, then runs `ANALYZE` and `PRAGMA optimize`. This avoids maintaining
several B-trees during all 5.3 million inserts.

## 12. Initial implementation test

The first tests rebuild CLI search around independently assembled dataset
indexes. They are deliberately small but exercise the final architecture.

### Step A: six-record synthetic snapshot

Run `prepare -> embed -> assemble -> query` using the existing synthetic OCSF
fixture and a fake deterministic embedder. Prove exact schemas, shard identities,
restart, corruption rejection, and pointer lookup without using a model.
Prepare it with one and four workers and require identical logical manifests,
document order, counts, and artifact bytes. Run dense, lexical, fused,
relation-filtered, and time-filtered queries.

### Step B: one real dataset with LM Studio

Choose one bounded real dataset that contains enough documents to exercise
grouping, repeated occurrences, filtering, and non-trivial dense ranking. Run:

```text
rag prepare
rag plan-embeddings
rag embed --backend lmstudio
rag assemble
rag inspect
rag query
```

This is the first user-facing milestone. It must prove:

- the commands work independently and can restart;
- LM Studio produces one valid vector per prepared document;
- the assembled index opens through the Rust CLI;
- lexical, dense, and fused searches work;
- returned event references belong to that dataset;
- an embedding retry does not repeat completed vector tasks; and
- rebuilding this dataset does not touch another completed dataset index.

Record the real dataset, model profile, document count, occurrence count,
runtime, and throughput in a small sanitized report. This proves the complete
CLI path, not search quality across all datasets.

#### Completed real proof — 2026-08-14

The accepted M41 `ocsf_detection_finding` relation passed the implemented path:

- 53 prepared document groups and all 2,240 event references;
- four document shards and four independently resumable embedding tasks;
- Qwen3 Embedding 8B Q4 through LM Studio, 4,096 dimensions;
- about 31.2 seconds of measured task execution, plus the fixed conformance
  check;
- 18.84 seconds to prepare, 1.35 seconds to plan, and 1.74 seconds to assemble
  on the local development machine with warm filesystem caches;
- lexical, dense, and fused CLI searches all returned results;
- a complete rerun with `http://127.0.0.1:9` succeeded without network access
  and left every vector-shard digest unchanged;
- an independently prepared entity-management dataset did not change the
  detection dataset's manifests; and
- prepared manifests, plans, task receipts, and the result set passed the
  packaged offline JSON Schema registry.

These numbers prove that the commands and restart boundaries work. They do not
measure search quality or predict full-corpus GPU throughput.

The assembled build report keeps dataset coverage separate from whole-source
coverage. Source counts come from the accepted OCSF receipt; only the selected
dataset relation is projected and indexed. The report lists dataset-scope and
structured-only exclusions separately. Because embedding is an earlier stage,
the assembly report records zero new embeddings and binds the reused vector
objects through their task receipts.

### Step C: current 18,791-document representative corpus

Prepare the same relation-balanced sample through the new path. Import matching
vectors from the current SQLite cache. Assemble a new index and require:

- the same document IDs and semantic text;
- the same occurrence membership;
- the same vectors for imported cache entries;
- equivalent lexical, dense, and fused results for the frozen 45-query plan;
- no model calls when all cache entries are reusable.

This is the migration gate. The old combined builder remains available until it
passes.

### Step D: completed fixed 10,000-document performance corpus

The sealed scenario-blind corpus contains every included relation and covers
the observed length strata. Its implemented per-relation allocation is:

| Relation | Documents |
|---|---:|
| API activity | 1,059 |
| Application lifecycle | 235 |
| Authentication | 695 |
| Cloud inventory | 256 |
| Datastore activity | 1,059 |
| Detection findings | 53 |
| DNS activity | 76 |
| Email activity | 927 |
| Entity management | 58 |
| Event log activity | 1,059 |
| Configuration snapshot | 1,059 |
| File activity | 251 |
| HTTP activity | 1,058 |
| Inventory information | 1,057 |
| Process activity | 1,057 |
| User inventory | 41 |

The selected IDs, quotas, length strata, source identities, and order digest
are sealed before backend measurement. The prepared corpus contains 10,000
documents, 192,011 event references, and 2,863,810 exact tokens.

LM Studio produced 10,000 4,096-dimensional vectors in 2,500 requests with zero
retries. Summed executor time corresponds to 2.282 documents and 653.64 tokens
per second. The longer wall interval includes host-session pauses and is kept
separate from model throughput. Repeating the frozen 15-query plan across
lexical, dense, and fused search produced byte-identical result files.

Reduced 2,048- and 1,024-dimensional profiles and indexes were derived locally
without another model call. Mean top-20 dense/fused overlap with 4,096 dimensions
was 75.67%/81.67% at 2,048 dimensions and 56%/71% at 1,024 dimensions. These are
ranking-overlap diagnostics only. People have not yet reviewed the pooled
results and marked which documents are relevant, so no search-quality or
reduced-dimension default claim is made.

A paid RunPod measurement has now completed on the full M45 corpus. The cloud
build used a separately sealed FP16 profile rather than relabelling this older
LM Studio input or its vectors. See [the RunPod embedding guide](runpod-embedding.md)
for the measured throughput, cost, artifact transfer, and local index result.

### Step E: one complete large dataset or relation

Build one large dataset, or one complete relation-scoped dataset such as process
activity or event-log activity, without sampling. This tests hundreds of
thousands of occurrences and catches assembly problems that a 10,000-document
corpus cannot expose.

All 16 non-network relation datasets are prepared, fully verified, bound to
exact-token plans containing 92,466,199 tokens, embedded by LM Studio, and
assembled as real version-3 indexes. Together they contain 422,566 documents
and 5,325,200 event references. Earlier deterministic test-vector builds of the
four largest datasets remain useful scale and refusal tests, but the real model
indexes now supersede them as the authoritative local result.

The 16 embedding summaries close over 1,658 tasks and 105,647 HTTP requests
with zero retries. Their monotonic task timers sum to 149,425 seconds (41.51
hours), or 2.828 documents and 618.81 exact tokens per active second. The build
used batches of four with at most two client requests in flight. This is a
measurement of the completed resumable run, not evidence that LM Studio used
two internal prediction slots; its model status did not expose an internal
parallel setting.

### Step F: additional datasets and catalogue search

Build at least two dataset indexes independently, register them in a catalogue,
and prove that the CLI:

- opens both without rewriting either index;
- searches them concurrently;
- returns dataset identity with every hit;
- embeds the query once when their exact embedding profiles match;
- refuses or separately handles incompatible profiles;
- produces deterministic reciprocal-rank-fused ordering and tie handling; and
- can add or remove one dataset by changing only the catalogue.

Sixteen independent real indexes have passed validation and search. Their
sealed catalogue completed a frozen 45-search run over 15 queries with 15 model
calls. A separate transformer deduplicated the results into 709 label-hidden
review candidates, checked 1,244 unique returned event pointers against typed
Parquet, and sealed the snapshot identity and count in a private receipt beside
the modes, ranks, scores, and system identities. This proves modular local
construction and a reviewer handoff; it
does not claim corpus-wide search quality because people have not marked
relevance yet.

Stored-document similarity is also implemented in Rust. `rag similar` reads a
seed vector from one index, while `rag catalogue similar` reuses that exact
profile-bound vector across every admitted dataset. Neither command contacts LM
Studio. Runtime catalogue queries verify every final index and its sealed
prepared/plan/result provenance without replaying all intermediate embedding
parts; the separate `rag catalogue validate` command still performs the full
offline chain check.

An exact census across all 422,566 formatted document inputs found 418,930
distinct values, with 3,636 duplicate rows beyond the first occurrence
(approximately 0.86%). This does not justify adding a cross-dataset embedding
cache to the current implementation.

### Step G: full non-network corpus

Start only after the 10,000-document benchmark predicts time and cost and the
complete-dataset index passes. The full target may remain a set of independently
assembled dataset indexes behind one catalogue rather than one physical index.
Across the catalogue, require all 422,566 documents and 5,325,200 event
references, then run the existing inspect, query, result-review, and event-open
checks.

## 13. Performance gates

The first full build requires these measured gates:

| Gate | Requirement |
|---|---|
| Preparation determinism | Two runs have identical logical manifests and order digests |
| Preparation memory | Peak resident memory below 4 GiB on M41 preparation |
| Preparation concurrency | Parallel mode is at least 1.5x single-worker throughput, targets 2x, or defaults back to fewer workers |
| Embedding restart | Interrupted run repeats no completed shard |
| Worker merge | Disjoint workers produce exact-one coverage with no duplicates |
| Result validation | Missing, corrupt, non-finite, wrong-sized, and misordered vectors are rejected |
| Assembly memory | No complete vector matrix or occurrence corpus in memory |
| Migration | Representative index retains document/vector/event associations and frozen-query behavior |
| Dataset isolation | Rebuilding or replacing one dataset does not change another dataset index |
| Multi-index CLI | Catalogue search reports dataset identity and handles embedding-profile compatibility explicitly |
| Cloud forecast | Warm 10,000-document rate, transfer time, and expected cost recorded before full execution |
| Full corpus | Exactly 422,566 documents and 5,325,200 retained event references for the frozen M41 scope |

One-hour embedding would require about 117.4 documents per second. This is a
measurement target, not a promise.

## 14. Estimated storage

Initial estimates extrapolated from the current M41 representative index:

| Artifact | Estimate |
|---|---:|
| Prepared document text | 0.1-0.6 GB compressed |
| Prepared occurrences | 1-2 GB compressed |
| Float32 vectors at 4,096 dimensions | 6.92 GB |
| Final occurrence lookup | about 6 GB |
| Final lexical data | below 1 GB target |
| Final index | about 16 GB |
| Recommended temporary free space | at least 40 GB beyond the source snapshot |

Actual sizes from the 10,000-document and complete-relation tests replace these
estimates before the full build.

## 15. Delivery sequence and Git strategy

The completed local-first work remains recorded in
[`local-first-embedding-scale-plan.md`](local-first-embedding-scale-plan.md).
The current M45 and RunPod implementation belongs on
`feature/livefire-ocsf-runpod-rag`. Generated containers, model objects,
prepared corpora, vectors, indexes, credentials, and raw cloud reports remain
outside Git; source, schemas, deterministic fixtures, and sanitized evidence
are reviewed together at the end of the goal.

One integrator owns shared contracts and final commits. Parallel agents may own
disjoint implementation or review areas but do not create overlapping contract
commits.

Generated corpora, vectors, indexes, credentials, and benchmark raw output stay
ignored and are never committed. Specifications, schemas, source, deterministic
small fixtures, and sanitized reports are committed.

## 16. Source references

- LM Studio OpenAI-compatible embeddings:
  <https://lmstudio.ai/docs/developer/openai-compat/embeddings>
- LM Studio model loading:
  <https://lmstudio.ai/docs/developer/rest/load>
- Qwen3-Embedding-8B model and supported runtimes:
  <https://huggingface.co/Qwen/Qwen3-Embedding-8B>
- Hugging Face Text Embeddings Inference:
  <https://huggingface.co/docs/text-embeddings-inference/en/index>
- Text Embeddings Inference batching controls:
  <https://huggingface.co/docs/text-embeddings-inference/en/cli_arguments>
- Runpod endpoint types and worker scaling:
  <https://docs.runpod.io/serverless/endpoints/overview>
- Runpod job payload limits and handlers:
  <https://docs.runpod.io/serverless/workers/handler-functions>
- Runpod persistent network storage:
  <https://docs.runpod.io/storage/network-volumes>
- Runpod S3-compatible storage:
  <https://docs.runpod.io/storage/s3-api>
