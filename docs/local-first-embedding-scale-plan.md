# Local-first RAG scale plan

Historical phase record: local implementation and evidence completed through
the 16-index M41 non-network catalogue, Rust similarity/provider checks, and
reviewer-safe pool. It is not the current source or execution plan. The active
work rebuilds from normalized M45 data and implements RunPod; see
[`runpod-embedding.md`](runpod-embedding.md).

## 1. Outcome

The completed local goal proved and measured the modular Rust RAG pipeline with
LM Studio. It stopped before RunPod; statements below about the "current goal"
describe that historical phase, not the active M45 cloud goal.

The order is deliberate:

```text
reconcile corpus counts
    -> freeze local benchmark datasets
    -> measure and tune LM Studio
    -> test interruption and recovery
    -> prepare every non-network dataset locally
    -> build and query representative and small local dataset indexes
    -> prove multi-index CLI search
    -> stop with a documented local handoff
```

If Runpod work is approved later, it is an embedding accelerator rather than a
different indexing system. It must consume the same prepared document shards
and produce the same portable vector-shard format and validation contract that
passed locally. Cloud vectors belong to a separate cloud profile.

## 2. Decisions already made

1. The unit of work is a dataset, initially one OCSF relation from one source
   snapshot. Each dataset has its own prepared corpus, embedding set, assembled
   index, measurements, and immutable identity.
2. The first full target contains all searchable M41 data except raw
   `ocsf_network_activity`. System metrics remain counted but are not embedded.
3. The measured M41 target is 422,566 documents and 5,325,200 event references
   across 16 relation datasets. It is not the old 1,319,974-document Python
   number. The document count comes from the corrected Rust projection; the
   earlier 505,835 estimate included a camel-case key-normalization bug.
4. Prepared documents use Parquet. Vectors use the `LFREMB01` binary format.
   Provider request JSON is never the durable interchange format.
5. Local LM Studio Q4 and a future cloud Safetensors profile are different
   embedding profiles and produce different indexes. They are not relabelled
   as equivalent because their vectors look similar.
6. Only prepared document shards need to go to an embedding machine.
   Occurrence rows and source references remain local.
7. Direct Arrow and Parquet remain the core data path. Polars may be used for
   analysis, but it is not added to the build path unless a benchmark shows a
   clear improvement.
8. Generated corpora, vectors, indexes, and raw search results are not
   committed. Credentials are supplied only through the environment or the
   approved secret store and never written into the repository. Code, schemas,
   deterministic fixtures, plans, and sanitized measurement summaries are
   committed.

## 3. Current baseline

The portable commands are implemented:

```text
rag prepare
rag plan-embeddings
rag embed
rag assemble
rag inspect
rag query
```

The real detection dataset has already proved the complete command chain with
LM Studio:

- 53 document groups;
- 2,240 event references;
- four resumable embedding tasks;
- Qwen3 Embedding 8B Q4_K_M at 4,096 dimensions;
- lexical, dense, and fused CLI search; and
- a restart with the endpoint unavailable that reused every completed task.

This proves the interfaces. It is too small to forecast a full build.

The current embedding executor already supports:

- request batches from 1 to 32 documents;
- bounded concurrent requests within one task;
- retries for temporary HTTP failures;
- ordered output when requests finish out of order;
- model-name, dimension, finite-value, and normalization checks; and
- atomic vector-part publication with receipt-based restart.

The completed scale work provides deterministic cross-relation benchmark
selection, an offline-verified executable tokenizer, exact token counts,
token-balanced tasks, task-range execution, recovery commands, detailed
performance reports, parallel row-group census and benchmark preparation, and
a disk-backed lexical index format. Parallel preparation, bounded document
runs and their external merge, streaming version-3 assembly, and multi-index
catalogue search are implemented. They have contract tests and real M41 runs.

The executor does not automatically lease tasks between independent processes.
Explicit non-overlapping task ranges are the current safe multi-process
mechanism.

## 4. Phase L0: reconcile the corpus counts

Do this before forecasting storage, duration, or cloud cost.

The available builds are different:

| Build | All searchable documents | Excluding network |
|---|---:|---:|
| Python M21, projection policy v1 | 1,319,974 | 432,894 |
| Python M21, projection policy v2 | 560,574 | 422,550 |
| Rust M41 before camel-case normalization fix | 638,216 | 505,835 |
| Rust M41, corrected current policy | 560,842 | 422,566 |

The old 1,319,974 figure is not the correct current cost baseline. Most of its
difference came from an older network grouping policy. The useful comparison
is Python policy v2 against the current Rust policy.

The corrected M41 result is only 16 documents above Python M21 policy v2 after
excluding network data. The former 83,285-document difference was not a real
corpus increase. Rust's regular-expression replacement used `$1_$2`; Rust
interpreted `$1_` as a capture name, so camel-case keys such as
`hostIdentifier` and `calendarTime` lost a character and were not recognized
as identity or time fields. Using `${1}_${2}` restores the intended grouping.

The corrected M41 census records 13,905,577 source rows, 6,367,276 searchable
event references, 7,538,301 structured-only metric rows, and 560,842 document
groups. Raw network data accounts for 138,276 document groups and 1,042,076
event references, leaving the stated non-network target.

The identities, complete per-relation table, reproduction result, and
concurrency measurement are recorded in [M41 corpus census](m41-corpus-census.md).

### Completed evidence

1. The read-only census command emits source rows,
   searchable rows, structured-only rows, and distinct document groups by
   relation.
2. The historical Python M21 policy-v2 artifact contains 560,574 documents
   overall and 422,550 without network. Focused corrected Rust censuses exactly
   reproduce its authentication, HTTP, network, and process-activity counts;
   a fresh full Rust M21 census was not run.
3. The final Python/Rust comparison covers 4,128 rows across all 17 searchable
   relations and matches on every row.
4. The accepted M41 census and its exact identities are recorded in the linked
   report.

### Exit gate

- Every material per-relation difference has a written cause.
- Two Rust census runs produce the same counts and order hashes.
- The accepted M41 table is committed as a sanitized report.
- The confirmed non-network target is 422,566 documents.

## 5. Phase L1: recheck the existing local LM Studio proof

Build the current release binary and repeat the detection dataset in a fresh
work directory.

### Tests

1. Run `prepare -> plan-embeddings -> embed -> assemble -> inspect`.
2. Run lexical, dense, and fused searches.
3. Verify that every returned event reference belongs to the detection
   dataset.
4. Repeat `embed` with `http://127.0.0.1:9` and require zero model work and
   unchanged vector-part hashes.
5. Rebuild a second small dataset and verify that neither dataset modifies the
   other's artifacts.

### Exit gate

The current machine, LM Studio version, loaded model, profile, and checked-in
code still reproduce the accepted command chain. This is a compatibility
check, not a performance or search-quality claim.

## 6. Phase L2: build the local measurement tools

### 6.1 Deterministic benchmark selection

Add a Rust benchmark-preparation command that projects directly from the
admitted snapshot and produces a standard prepared corpus. It performs a
deterministic relation-and-length census first, freezes the selected document
IDs, then materializes only their document and occurrence rows. It must call
the same projection implementation as ordinary preparation rather than invent
a benchmark-only projector.

The command derives three nested benchmark corpora:

| Corpus | Purpose |
|---|---|
| 512 documents | quick batch and concurrency screening |
| 2,000 documents | stable local tuning and failure tests |
| 10,000 documents | final local forecast and cloud comparison input |

Selection is relation-balanced and includes short, median, p90, p95, p99, and
maximum document lengths. It is based only on source identity, relation,
document ID, and declared length buckets. It cannot use hunt answers or search
results. It therefore does not depend on the complete prepared relation
datasets built later in phase L7.

The command publishes a selection manifest containing exact document IDs,
input hashes, relation quotas, length buckets, and source prepared-corpus
identities. Re-running it must produce the same logical selection.

### 6.2 Exact token counting

The current code records UTF-8 bytes as a conservative upper bound. That is
safe for rejecting obviously oversized inputs, but it is not enough for
tokens-per-second measurements or balanced GPU tasks.

Add the exact Qwen tokenizer to planning and bind its files and revision in the
embedding profile. The current profile identifies the tokenizer embedded in
the GGUF but does not provide executable tokenizer bytes. Implement and prove
one of these before measurement: read that GGUF tokenizer directly, or bind a
`tokenizer.json` from the exact Qwen revision and prove token-ID parity against
the GGUF/llama.cpp tokenizer on a fixed hostile and length-boundary fixture.
Planning records:

- tokens per formatted document;
- tokens per task;
- p50, p90, p95, p99, and maximum tokens;
- documents rejected for length; and
- the chosen maximum-token policy.

Inputs that exceed the profile limit fail during planning. No backend may
silently truncate them.

### 6.3 Token-balanced tasks

Extend planning so tasks have both a maximum document count and a target token
count. Use consecutive document ordinals so vector order remains simple.

For local LM Studio, target tasks that take about two to five minutes. For the
first tuning run, start with at most 256 documents per task. The current
2,048-document default is too large for convenient local recovery at the
observed real-document rate.

### 6.4 Performance report

Add a machine-readable local report with:

- Git commit and all source, projection, prepared-corpus, plan, and profile
  identities;
- LM Studio version, model key, GGUF file, quantization, context, and configured
  parallel predictions;
- hardware and available memory;
- document, byte, and exact token distributions;
- batch size, requests in flight, and task sizing;
- cold-load time and warm embedding time kept separate;
- documents per second and tokens per second;
- request p50 and p95 latency;
- retries, errors, request bytes, and response bytes;
- Rust and LM Studio peak memory;
- prepared, vector, temporary, and final-index disk sizes; and
- vector dimensions, norms, and output hashes.

Add a query benchmark path that accepts a precomputed, profile-bound query
vector. Report LM Studio query-embedding latency separately from index-only
dense search and from complete end-to-end query latency. Lexical search does
not need the model.

Python may turn this report into charts. Rust remains responsible for the
measurements and artifact validation.

### 6.5 Recovery tools

Add a task-scoped verify and quarantine command. A corrupt or incomplete task
must never be overwritten silently. The command moves only the invalid part
and receipt into a clearly named quarantine directory, after verifying that
they belong to the selected embedding set.

### 6.6 Task partition and finalization

Add an explicit worker command that accepts exact task IDs or a deterministic
task range and publishes only those parts and receipts. Add a separate finalize
command that verifies exactly one valid result for every planned task before
writing the embedding-set manifest.

Prove this locally with two worker processes using disjoint task IDs and a fake
backend. Two processes never write the same task path. This is the correctness
foundation for later multi-GPU assignment; it is not expected to make one
locally loaded LM Studio model faster.

### Exit gate

- The three benchmark corpora are deterministic.
- Exact token counts are reproducible.
- Task boundaries are deterministic and token-balanced.
- The sanitized performance report contains no document text or credentials.
- A corrupt task has a safe, documented recovery path.
- Disjoint local workers can complete one plan and the finalizer rejects
  missing, duplicated, or conflicting task results.

## 7. Phase L3: tune LM Studio locally

LM Studio currently has one loaded Qwen3 Q4 model with no explicit parallel
prediction setting. Earlier short-text tests found that concurrent requests
mostly queued. The new measurements must use real prepared documents.

### 7.1 Quick screen on 512 documents

Warm the model, then test batch sizes 4, 8, 16, and 32 with one request in
flight. Run each setting at least twice in a new embedding output directory.

Choose the smallest batch size within 10% of the fastest median warm
throughput.

### 7.2 Request concurrency

With the selected batch size, test one, two, and four requests in flight. This
uses the existing bounded Tokio scheduler inside one embedding task.

Do not run multiple indexer processes against the same embedding output. The
current pipeline does not lease tasks between processes. Separate dataset
processes with separate outputs are safe, but they will contend for one loaded
model and are unlikely to improve throughput.

### 7.3 LM Studio model parallelism

Reload the model separately with parallel prediction counts one, two, and four.
For each load setting, test the useful request counts again. Record model-load
time separately from embedding time.

Adopt model parallelism only when:

- sustained throughput improves by at least 15%;
- the improvement repeats on the 2,000-document confirmation;
- vectors remain byte-identical for the same local profile, or the changed
  execution setting is assigned a distinct profile;
- retries and failures do not increase; and
- memory remains within the machine's safe operating range.

### 7.4 Confirmation on 2,000 documents

Run the best two configurations at least twice. This produces the first useful
local time forecast and the task-size measurement for recovery testing.

### Exit gate

- One local default batch and request-concurrency setting is selected.
- LM Studio parallel predictions are either adopted with evidence or left off.
- The 2,000-document forecast predicts a repeated run within 20%.
- Results include both documents per second and tokens per second.

### Completed measurement

The real 512-document screen selected four inputs per request and one request
in flight. It measured 2.737 documents and 775.17 tokens per second. Batches of
8, 16, and 32 were slower, and raising the in-flight limit to two or four also
reduced throughput because the one LM Studio prediction slot queued the work.
The clean 2,000-document confirmation completed with zero retries. Its summed
executor time corresponds to 2.552 documents and 722.69 tokens per second.

A later warmed repeat at the selected four-input, one-request setting produced
byte-identical vectors but ran at 1.417 documents and 401.40 tokens per second.
This exposes substantial host/runtime variance, so the earlier executor rate is
not a wall-clock promise. LM Studio 0.4.20+1 accepted `--parallel 2` for the
embedding-model load but both `lms ps` and the local REST model configuration
reported no parallel setting. No run was labelled parallel 2 or 4 without an
observable server binding. The model was restored to its normal 8,192-token
load. The supported local default remains four inputs and one request in
flight; previously measured client in-flight limits of two and four were
slower.

## 8. Phase L4: interruption and failure tests

Run these tests on copies of the 2,000-document plan:

1. Stop the Rust process after one or more task receipts exist. Restart and
   prove completed task hashes do not change.
2. Stop it during an active task. Restart and prove only the unfinished task is
   recomputed.
3. Stop LM Studio temporarily. Verify bounded retries and a safe resumable
   output directory.
4. Delete a copied receipt while retaining its vector part. Verify the orphan
   is not trusted and the recovery command handles it explicitly.
5. Corrupt a copied vector part. Verify assembly rejects it.
6. Use a local fault server to return a wrong model identifier. Verify the
   task fails.
7. Simulate 408, 429, selected 5xx responses, timeout, duplicate response
   indexes, missing indexes, wrong dimensions, NaN, and a non-normalized vector.
8. Run a completed embedding set with an unreachable endpoint. Verify zero
   network work.

### Exit gate

No completed task is repeated, no invalid vector is accepted, temporary
failures retry only within the declared bound, and every interrupted state has
a safe restart or quarantine path.

## 9. Phase L5: completed fixed 10,000-document local run

The selected local settings ran over the complete benchmark:

```text
select -> plan -> LM Studio embed -> assemble -> inspect -> query
```

The frozen 15-query plan also ran through lexical, dense, and fused search, for
45 mode-specific executions.

### Required measurements

- total and warm embedding time;
- documents and tokens per second overall and by length bucket;
- retries and errors;
- preparation and assembly time;
- peak memory and all artifact sizes;
- exact dense, lexical, and fused query latency; and
- result stability across repeated runs.

Record index-only dense latency using the precomputed query-vector path,
LM Studio query-embedding latency on its own, and full dense/fused latency. Do
not report the combined CLI time as if it were only index search.

### Required validation

- Exactly 10,000 documents and one vector per ordinal.
- Every event reference returned by search belongs to the selected source
  data.
- A completed restart performs no model work.
- Batch and request scheduling do not change vector identities or rankings for
  the same profile.
- The 2,000-document forecast is within 20% or the length-distribution effect
  is explained.

The completed result contains 10,000 vectors and 192,011 event references for
2,863,810 exact input tokens. LM Studio served 2,500 requests with zero retries.
Summed executor time was 4,381.30 seconds, or 2.282 documents and 653.64 tokens
per second. The report's wall-interval union was 7,917.25 seconds, or 1.263
documents per second, because the host session paused during the run. The two
numbers are kept separate; the paused wall interval is not presented as model
throughput. The raw 4,096-dimensional vector file is 163,840,064 bytes.

The version-3 index has 10,000 documents and 192,011 occurrence rows. For the
recorded query, median index-only latency was 24.23 ms lexical, 42.51 ms dense,
and 69.16 ms fused. Median query embedding was 101.11 ms. Two complete frozen
query executions produced byte-identical result files.

The frozen runs establish result stability and latency, not search quality.
People still need to review the pooled results without knowing which search
mode produced them, then mark which documents are relevant. The run confirmed
1,347 distinct returned pointers against the authoritative typed Parquet. It
did not confirm returned event references through the released OCSF query
service; that separate check is still pending.

## 10. Phase L6: preparation and assembly scale locally

This work is implemented. Timed CPU preparation still must not share the same
machine with a timed LM Studio run because CPU, disk, and memory contention
would invalidate both measurements.

### 10.1 Parallel preparation

Use the existing Arrow and Parquet crates directly:

```text
Parquet row groups
    -> bounded Rayon projection workers
    -> ordered occurrence fragments
    -> sorted document runs
    -> bounded external merge by document ID
    -> prepared document shards
```

Implementation rules:

- Load each Parquet footer once.
- Assign row groups, not whole relations, to a private Rayon pool.
- Read only required columns.
- Keep worker results ordered by source row within each row group.
- Bound result channels and document-candidate memory.
- Sort and deduplicate document candidates into temporary runs.
- Merge runs with a bounded file count and binary heap.
- Never retain all occurrences or all documents in memory.
- Use Tokio only for HTTP and remote orchestration, not CPU projection.

The `rag-ocsf` adapter now has an admitted-object API that verifies an object,
loads its footer once, and creates independent projected row-group readers.
Preparation uses bounded row-group waves. When a large file has too few row
groups to fill the worker pool, it borrows the active Arrow block and projects
deterministic row ranges in parallel, then merges them in source order without
copying the Arrow buffers.

Test one, two, four, and eight workers; Arrow batches 2,048, 4,096, and 8,192;
and Zstandard levels one and three.

Keep parallel preparation only when it is at least 1.5 times as fast as serial
preparation, uses no more than twice the memory, and produces identical logical
manifests and order hashes. Choose the smallest worker count within 10% of the
fastest result. Peak memory must remain below 4 GiB; below 2 GiB is the target.

The old 600,000-document in-memory limit has been removed. Preparation writes
bounded sorted document runs and merges them by document ID. The generated
750,000-unique-document acceptance test exercises that boundary separately from
ordinary unit tests.

### 10.2 Stage-specific validation

Do not decode occurrence shards during planning or embedding:

- planning verifies the prepared manifest and document shards;
- embedding verifies only the planned document slices and profile;
- assembly validates document, vector, and occurrence streams as it consumes
  them; and
- an explicit offline verify command performs the expensive complete check.

This preserves integrity without rereading millions of occurrence rows for a
stage that cannot use them.

### 10.3 Assembly without model calls

Generate clearly marked deterministic, normalized test vectors with the real
4,096-dimensional shape. They are performance fixtures and must never be
published as searchable evidence indexes. Their manifests and assembled test
indexes carry a machine-readable `test_only` marker, and normal `rag query` and
tool packaging refuse to open them.

Use them to measure full-size vector I/O, Parquet output, occurrence lookup,
lexical construction, memory, and exact dense scan without waiting for LM
Studio.

### 10.4 Lexical index

The scalable version-3 index replaces corpus-sized lexical JSON with a
deterministic read-only SQLite inverted index. Tests prove exact version-2
tokenization, BM25 scores, filtering, ties, and hit parity. Version 2 remains
available as the compatibility oracle, while portable full-size assembly uses
version 3 explicitly.

### Exit gate

- Parallel and serial preparation are logically identical.
- The 750,000-document test completes without a corpus-sized document map.
- Planning and embedding do not read occurrence shards.
- Assembly holds neither the vector matrix nor occurrence corpus in memory.
- Fixture lexical results remain stable.
- Generated stress datasets and their deterministic test-vector indexes can be
  produced without a memory failure.

## 11. Phase L7: complete datasets and multi-index CLI

All 16 non-network relations are prepared and fully verified locally. Their
exact-token plans contain 92,466,199 tokens in total. This provides actual
document, occurrence, and upload sizes without depending on a cloud service.

After all relation datasets are prepared, run the full non-network assembly
measurement with the machine-marked test vectors from phase L6. This is the
first point at which a full-size synthetic assembly is possible; it must not be
run before the prepared inputs exist.

Current document counts guide the order:

| Dataset group | Documents |
|---|---:|
| Separate relation indexes with at most 3,000 documents each | 5,060 total |
| API activity | 6,531 |
| HTTP activity | 12,045 |
| Datastore activity | 29,600 |
| Process activity | 21,471 |
| Configuration snapshots | 148,110 |
| Event-log activity | 199,749 |

All 16 non-network relation datasets are now real embedded and independently
indexed: 422,566 documents and 5,325,200 event references in total.
Across the 16 finalized embedding summaries, LM Studio processed 92,466,199
exact tokens in 1,658 resumable tasks and 105,647 HTTP requests with zero
retries. The sum of Rust's monotonic task timers was 149,425 seconds (41.51
hours), while the sum of the per-dataset timestamp intervals was 165,085
seconds (45.86 hours). The monotonic total is the best measure of active local
executor work; neither total is a promise about uninterrupted elapsed time
because the laptop slept and the work was resumed in separate ranges.
API required 2,738,612 exact tokens, 1,633 requests, and zero retries. Its summed
executor time was 4,215.78 seconds, or 649.53 tokens per second, and its
version-3 assembly took 3.74 seconds. The small row is a scheduling group, not
one merged physical dataset. HTTP required 4,341,069 exact tokens, 3,012
requests, and zero retries. Its version-3 index contains 12,045 documents and
25,114 event references and passed inspect, lexical search, and fused search.

Build a catalogue that opens completed dataset indexes without rewriting
them. The CLI must:

- search indexes concurrently;
- embed a query once when exact profiles match;
- keep incompatible profiles separate;
- identify the dataset and relation on every hit;
- merge per-index ranks using the declared stable rule rather than comparing
  raw lexical scores; and
- add or remove one dataset without rebuilding the rest.

The catalogue has one source-level accounting record for the M41 snapshot and
one non-overlapping coverage entry per relation dataset. It must not sum each
relation index's copy of whole-snapshot metric or excluded-relation counts;
doing that would count the same source rows 16 times.

### Current status and remaining checks

- Complete: all 16 relation datasets are prepared, verified, and exactly
  token-planned.
- Complete: all 16 datasets are embedded, assembled as real SQLite-v3 indexes,
  inspected, and admitted into one sealed catalogue. The largest event-log
  dataset contains 199,749 documents and 407,729 event references. Its 50,135,863
  tokens were embedded in 781 tasks and 49,938 requests with zero retries.
- Measured: the full local embedding work used batches of four and at most two
  client requests in flight. It completed all 422,566 documents with zero
  retries in 41.51 hours of summed monotonic task time, an aggregate 2.828
  documents or 618.81 tokens per active second. This proves bounded client-side
  concurrency. It does not prove that LM Studio ran two model predictions at
  once: LM Studio continued to report no parallel-prediction setting for this
  embedding model.
- Complete: the sealed 16-index catalogue covers 422,566 documents and
  5,325,200 event references. A frozen 15-query plan ran lexical, dense, and
  fused modes in 45 searches with 15 model calls. It produced 709 unique
  label-hidden review candidates. A sealed private receipt binds the M41
  snapshot and records that 1,244 unique event pointers across all 16 relations
  passed exact typed Parquet membership checks. System names, modes, ranks, and
  scores remain in the private audit directory rather than the reviewer files.
- Complete: deterministic test-vector builds previously exercised the four
  largest datasets without model calls and proved that normal consumers refuse
  synthetic vectors. Those scale tests are superseded by the real LM Studio
  indexes, but remain useful assembly and safety evidence.
- Complete: native Rust stored-document similarity works for one index and for
  the complete catalogue without a model call. The denied IAM access-key seed
  retrieves denied CreateUser, DeleteAccessKey, and ListAccessKeys neighbours
  from the API index.
- Complete: separately granted Rust `evidence.search` and `evidence.similar`
  tools passed SDK handshake, open, call, health, and close against the real
  process-activity index. The similarity session used an offline contract.
- Complete for forecasting: the exact token distribution and measured
  length-sensitive local throughput are available.
- Pending: people review the 709-candidate local pool and mark which documents are
  relevant before it is used as a search-quality baseline.

The historical 3,806-document Python command prototype is a comparison oracle,
not the production path. Its old Q1-Q9 questions execute through the new Rust
catalogue, but the literal acceptance rules do not pass because the admitted
M41 OCSF records do not contain several raw PowerShell script bodies, full
command arguments, or unredacted identities used by that prototype. Dataset
routing does not change that result. Reproducing those exact command behaviors
requires a separate Rust command projection backed by an admitted raw
command/script source; changing the general evidence projection or restoring
sensitive identifiers merely to satisfy the old fixture would be incorrect.

## 12. Phase L8: local representation and search experiments

The 10,000-document 4,096-dimensional vectors were used to derive 2,048- and
1,024-dimensional Qwen prefixes locally, followed by normalization. This did
not call the model a second time.

Each dimension has a separate profile, result set, and version-3 index. The
vector files are 81,920,064 bytes at 2,048 dimensions and 40,960,064 bytes at
1,024 dimensions, compared with 163,840,064 bytes at 4,096 dimensions. Across
the 15 frozen queries, mean top-20 dense/fused overlap with the full vectors was
75.67%/81.67% at 2,048 dimensions and 56%/71% at 1,024 dimensions. These are
ranking-overlap measurements, not relevance judgements. Do not make a reduced
dimension the default until people have reviewed the results and marked which
documents are relevant.

The exact formatted-input census over all 422,566 prepared documents found
418,930 distinct inputs. There are 2,405 duplicate groups and 3,636 document
rows beyond the first copy, a reuse rate of about 0.86%; the largest group has
61 documents. That is too little reuse to justify adding a cross-dataset cache
to the current pipeline. If a future corpus changes that decision, the cache
identity remains:

```text
exact embedding-profile digest + formatted-input digest
```

It is never keyed by semantic text alone.

Measure exact dense search on the largest locally assembled index. Add an
approximate nearest-neighbour index only if exact search is too slow. Treat the
approximate structure as a derived artifact and compare it with exact search
for every reviewed query, including rare relations.

## 13. Local handoff and deferred Runpod gate

The current goal stops at the local handoff. Runpod-specific implementation and
data upload are deferred future work. If that work is approved, it must not
start until all of the following are true:

1. Historical and current corpus counts are reconciled.
2. The 512, 2,000, and 10,000 document corpora are frozen and reproducible.
3. Exact Qwen token counts and length distributions are recorded.
4. The best supported local batch and request settings are known; LM Studio's
   embedding load does not expose an attestable server-parallel setting.
5. Restart, timeout, retry, corruption, wrong-model, and order tests pass.
6. The 10,000-document build is complete and its time forecast is recorded.
7. Every non-network relation is prepared locally with measured artifact size.
8. All 16 non-network relation indexes and their catalogue work locally.
9. Human-reviewed local search results provide a comparison baseline.
10. Only prepared document shards, the plan, and the cloud profile are in the
    reviewed upload set; occurrence data remains local.
11. A matching cloud-profile query-embedding strategy is documented and can be
    tested during the one-GPU phase.

The local pipeline, benchmark, prepared datasets, HTTP index, catalogue, and
reviewer handoff do not imply that this future cloud gate has passed. Human
relevance review is still pending, so the work remains local.

## 14. Deferred future design: Runpod

This section records design constraints only. It is outside the current goal.
If approved later, Runpod work begins on a separate branch after the local
branch is merged.

### R1. Freeze a cloud profile

Create a new profile binding:

- the exact Qwen Hugging Face revision and every model/tokenizer file digest;
- the exact verified runtime dtype;
- last-token pooling;
- document format and query instruction;
- maximum input length with truncation disabled;
- L2 normalization and output dimension;
- inference engine and version; and
- container image digest.

Do not call the runtime BF16 merely because that was requested. Record the
dtype actually loaded. The cloud profile never inherits the LM Studio Q4
identity.

### R2. Use one Pod and one GPU first

Use TEI first because it is designed for embedding, supports Qwen3, and offers
token-based dynamic batching. Test vLLM only if TEI is unstable or misses the
measured goal. SentenceTransformers is a useful correctness reference and may
become the worker only if measurements justify it.

Run, in order:

1. the existing conformance fixture;
2. 128 documents spanning relations and length buckets;
3. one repeated task to measure numeric repeatability;
4. the local 2,000-document tuning corpus; and
5. the exact local 10,000-document corpus.

Download the resulting vector shards, validate them locally, assemble the
index locally, and query it through the exact cloud-profile query service while
the Pod is available, unless a compatible local query runtime has already been
proved. This requires a profile-aware remote HTTPS query adapter or a
profile-bound imported-query-vector path; the current loopback-only LM Studio
client is not sufficient.

### R3. Storage and worker boundary

Upload only:

```text
prepared manifest
embedding plan
cloud profile
prepared document Parquet shards
```

The worker writes one vector part and one receipt per planned task into its own
output prefix. It stages the complete result on the Pod, flushes and hashes it,
uploads it under its final unique object key, verifies that object, and only
then publishes the receipt. The design does not rely on object-store rename
being atomic. An existing result is accepted only when its receipt and object
digest match. Workers never append to one shared output file or concurrently
modify the same object path.

The result receipt adds exact token counts, GPU model, engine settings, loaded
model identity, dtype, timings, and output digest. It contains no document
text, credentials, or secret-bearing request logs.

### R4. One, two, and four GPU measurements

Start with static disjoint task assignment. Do not add Serverless leases or an
elastic scheduler to the first experiment. Use the task-ID/range worker and
separate finalizer already proved locally.

The one, two, and four-GPU runs use the identical 10,000-document plan and task
boundaries, identical GPU models, and the same profile, container image, engine
settings, and batch tuning. Measure warm throughput separately from Pod
provisioning and model loading.

Measure model load, warm documents and tokens per second, GPU use, memory,
upload/download, retries, duplicate work, total elapsed time, and total cost.

Required scaling gates:

- two GPUs reach at least 1.7 times one-GPU throughput;
- four GPUs reach at least 3.4 times one-GPU throughput if cost must stay
  within 20% of ideal linear scaling;
- retries remain below 5%;
- duplicate work remains below 1%; and
- no task is missing or invalid.

If four GPUs miss the gate, compare one faster GPU instead of adding scheduler
complexity.

### R5. Cloud promotion gate

Build one complete dataset with Runpod only when:

- every task validates and assembles locally;
- no input was silently truncated;
- interruption loses no completed task;
- a matching query embedder is available;
- the cloud profile's reviewed search results meet the agreed tolerance
  against the local baseline, including rare relations;
- real throughput and transfer measurements support an accepted forecast; and
- the sanitized report contains no document text or credentials.

The selected setup must reach about 117.4 documents per second for a one-hour
non-network embedding pass, or the team must record and explicitly accept a
different maximum duration and cost before processing a complete dataset.

Run one complete dataset first. Verify and query it locally before submitting
the remaining datasets. Runpod Serverless and automatic provisioning remain a
later optimisation.

## 15. Feature and commit plan

The completed portable-pipeline branch is already merged into `main`. This
planning work starts from that clean commit on:

```text
feature/local-first-embedding-scale-plan
```

Implementation should continue on a separate local scale branch created from
the merged plan, for example:

```text
feature/local-rag-scale
```

Keep concurrent implementation changes uncommitted while the shared contracts
are moving. At the final integration point, after every vertical slice has
passed its tests, the integrator creates these reviewable commits:

1. Corpus count reconciliation and sanitized report.
2. Deterministic benchmark selection and exact tokenizer support.
3. Token-balanced plans, local measurements, and task recovery tools.
4. Parallel Arrow preparation and bounded external document merge.
5. Stage-specific verification and assembly measurements.
6. SQLite lexical index with parity tests.
7. Dataset catalogue and multi-index CLI search.
8. Sanitized local LM Studio benchmark and result pool for later human review.

One integrator owns shared contracts and final commits. Parallel agents may
work on disjoint crates or audits but do not commit overlapping contract
changes independently.

Merge the local scale branch after its local implementation and evidence are
reviewed. A later Runpod goal may then create:

```text
feature/runpod-embedding-workers
```

Runpod code, container files, cloud credentials, storage adapters, and cloud
reports stay off the local branch. The prepared-corpus and vector-shard formats
must not change merely to suit Runpod.

## 16. What each phase proves

| Phase | What it proves |
|---|---|
| L0-L2 | We understand the corpus and can measure it correctly |
| L3-L5 | LM Studio behaviour, rate, recovery, and local CLI indexing |
| L6-L8 | Rust preparation/assembly scale and modular multi-index search |
| R1-R3 | One cloud worker can consume and return the portable artifacts |
| R4 | More GPUs reduce elapsed time at an acceptable cost |
| R5 | A cloud-built dataset remains useful and verifiable locally |

Passing file and command tests proves the pipeline works. It does not prove
that search results are good. Search quality is claimed only after people have
reviewed the results and the measured local and cloud profiles satisfy the
same declared tests.
