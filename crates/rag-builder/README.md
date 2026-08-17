# rag-builder

Native `rag` CLI for direct `livefire-ocsf` snapshot projection, resumable local
embedding, fast-index construction, inspection, and dense/lexical/fused query.
The modular pipeline uses a pinned `tokenizer.json` plus its tracked reference
to build exact-token, token-balanced embedding plan v2 files. `embed` can run a
checked `START..END` task range, so independent workers can safely take
disjoint ranges. Run `finalize-embeddings` after all ranges finish; assembly
accepts only that complete, validated result set. Legacy plan v1 files must be
replanned and are never silently reinterpreted.

`prepare` and `prepare-benchmark` project admitted Parquet row groups
concurrently. Use `--workers N` to set the bound; the default uses at most eight
available CPU threads and the hard maximum is 64. Worker fragments are limited
to one bounded wave of row groups and are merged in original source order.
Ordinary preparation feeds that ordered merge through its existing bounded
occurrence-shard buffer. Unique documents are accumulated in sorted temporary
runs and merged in document-ID order, so the complete document set is no
longer held in memory or limited to 600,000 rows. Each run holds at most
100,000 documents by default. Machines with a tighter memory budget can set
`LIVEFIRE_RAG_PREPARE_DOCUMENT_RUN_ROWS` to a value from 1,024 through 600,000;
this changes temporary work only, not the published document order. Temporary
runs are removed on success and after a failed preparation. Benchmark
preparation also retains only selected occurrences in its second pass. One
worker and many workers therefore publish the same manifests and Parquet
bytes; benchmark selection is also unchanged.

Completed per-dataset pipelines can be registered without rebuilding their
indexes. Each `--dataset` supplies the prepared corpus, embedding plan,
finalized embedding results, and assembled index in that order.

For a fast local check of the complete file and assembly chain without calling
LM Studio, use `rag test-embed` with a compatible 4096-dimensional profile.
It writes deterministic normalized vectors and marks the result and index as
test-only. Normal `rag query`, provider, and packaging paths refuse that index;
only `rag inspect --allow-test-only` and an explicitly test-only catalogue may
open it.

```sh
rag catalogue build \
  --dataset datasets/process/prepared datasets/process/plan \
            datasets/process/embeddings datasets/process/index \
  --dataset datasets/http/prepared datasets/http/plan \
            datasets/http/embeddings datasets/http/index \
  --out datasets/catalogue.json
rag catalogue validate --catalogue datasets/catalogue.json
rag catalogue search --catalogue datasets/catalogue.json \
  --query 'encoded PowerShell' --mode fused --workers 4
rag similar --index datasets/process/index \
  --document-id sha256:SEED --top-n 20
rag catalogue similar --catalogue datasets/catalogue.json \
  --dataset-id DATASET_ID --document-id sha256:SEED \
  --top-n 20 --workers 4
rag catalogue batch-search --catalogue datasets/catalogue.json \
  --requests queries.jsonl --embedding-endpoint http://127.0.0.1:1234 \
  --workers 4 --out catalogue-run
rag catalogue batch-search --catalogue datasets/catalogue.json \
  --requests queries.jsonl --query-vector-set sealed-query-vectors \
  --workers 4 --out cloud-profile-catalogue-run
```

All artifact paths must be below the catalogue file's parent directory. Build
and `catalogue validate` re-open the prepared objects, exact-token plan, result
parts, receipts, reports, and index files before accepting an entry. Runtime
search and similarity open and verify each final index and compare its sealed
prepared/plan/result provenance with the catalogue; they do not reread every
intermediate embedding part. Overlapping
relations fail unless named with `--allow-relation-overlap RELATION=REASON`.
Dense and fused searches embed the query once, search compatible indexes in
parallel, then merge their per-index ranks. Every hit retains its dataset and
index identity. Synthetic catalogues require `--test-only` when built and
`--allow-test-only` each time they are searched.

Similarity resolves the seed by its dataset ID and document ID, reads its
stored vector, and searches every compatible index without contacting LM
Studio. The exact seed is excluded by default; `--include-seed` reverses that
choice. Relation and half-open time filters are applied to candidate event
references before ranking.

Each `queries.jsonl` row uses the same closed request shape as `batch-query`:

```json
{"query_id":"q-001","query":"encoded PowerShell","mode":"fused","top_n":20,"relations":["ocsf_process_activity"]}
```

The file is UTF-8 JSON Lines. Every row, including the final row, must end with
an LF byte (`0A`). The command rejects a missing final LF instead of silently
changing the file, so the published `requests.jsonl` remains an exact byte copy.

`catalogue batch-search` validates every request and opens the catalogue once.
It preserves request order, reuses one vector when dense and fused rows contain
the same query, and never calls the embedding endpoint for lexical rows. The
required `--out` path must not exist. The command stages an exact
`requests.jsonl` copy, content-bearing `results.jsonl`, and a self-digested
`manifest.json`, then publishes the directory with one rename only after every
request succeeds. A failure publishes nothing. Standard output contains only a
content-free completion summary.

For an index built with the cloud profile, `--query-vector-set` replaces
`--embedding-endpoint`. The set must contain the byte-exact same
`queries.jsonl`, profile, returned model, dimensions, and normalization. Each
dense or fused row is checked against its query ID, raw query hash, and the
profile's recomputed query composition before its packed vector is used. The
run receipt records zero model calls and the sealed set's component digest.
The two vector sources are mutually exclusive; an all-lexical plan uses
neither. A single-index frozen query uses the same boundary:

```sh
rag query --index datasets/process/index \
  --query-id q-001 --query 'encoded PowerShell' --mode fused \
  --query-vector-set sealed-query-vectors
```

The raw run is intentionally ignored by relevance reviewers: it contains
retrieval modes, ranks, scores, and system identities. For the final reviewer
workflow, `queries.jsonl` must contain exactly one lexical, one dense, and one
fused row for every query in the frozen query fixture. All three rows for one
`query_id` must repeat the same query text, `top_n`, and relation filters. Review
comparisons must use an empty `relations` array so the request does not reveal
an expected relation. After the batch search completes, build the separate
label-hidden pool:

```sh
uv run --extra analysis python tools/build_catalogue_review_pool.py \
  --run-dir catalogue-run \
  --catalogue datasets/catalogue.json \
  --queries fixtures/generic-evidence-pilot-queries.v1.json \
  --snapshot-root SNAPSHOT \
  --out catalogue-review
```

The tool verifies that every frozen query has all three modes and that both raw
JSONL files have an LF after their final row. Give reviewers only
`catalogue-review/review-pool.jsonl` and `catalogue-review/manifest.json`. Keep
the `catalogue-review/audit/` directory private because it retains system
provenance for later analysis. The public pool conforms to
`catalogue-review-pool-row.v1`; it hides system labels and ranking details.
Unknown request fields, ambiguous query IDs, unsorted or unknown relation
filters, invalid rows, and more than 10,000 rows fail before staging or calling
the model. Use `--allow-test-only` only for an explicitly test-only catalogue.

Before planning local LM Studio work, verify the derived tokenizer offline
against the token IDs captured from the exact llama.cpp GGUF model:

```sh
rag verify-tokenizer \
  --tokenizer-json indexes/tokenizers/qwen3-embedding-8b-gguf-q4-k-m-69d0e58a13e463cd99a9b83e3f5fee7c10265fab/tokenizer.json \
  --tokenizer-ref profiles/qwen3-embedding-8b-gguf-q4-k-m-tokenizer.ref.json \
  --fixture fixtures/qwen3-embedding-8b-tokenizer-parity.v1.json
```

The command checks the tokenizer and fixture identities, every captured token
ID sequence, and the token count and little-endian token-ID digest at the
16,384-byte input boundary. It prints a JSON success report without including
fixture inputs. Any identity or token difference stops verification. It does
not contact LM Studio or another model server.

Use `rag verify-prepared --prepared PREPARED` for a read-only check of every
prepared document and occurrence file. Planning and embedding check only the
manifest and document files they can use; finalization and assembly repeat the
full document-and-occurrence check.

Use the separate benchmark command when model and index timings must not be
mixed:

```sh
rag benchmark-query --index INDEX --query 'encoded PowerShell' \
  --query-id q-powershell --embedding-warmups 1 --embedding-repeats 5 \
  --warmups 3 --repeats 20 \
  --embedding-endpoint http://127.0.0.1:1234
```

It measures query embedding with separate `--embedding-warmups` and
`--embedding-repeats`, validates and binds one returned vector to the index's
embedding profile, then reuses it for dense and fused index-only measurements.
Lexical search is timed separately. Set
`--end-to-end-repeats N` only when additional model calls are wanted. The JSON
report contains the query ID and SHA-256, never the query text.

Local embedding tasks, the final embedding run, and query benchmarks use the
strict report schemas in `specs/`. They record the Git revision and whether the
working tree had changes, source and model identities, machine details,
artifact sizes, and measurements that were actually available. HTTP request
and response body sizes, process memory peaks, LM Studio version, and cold-load
time remain explicit `null` values unless they were measured. A benchmark
wrapper may supply the last four values with these non-secret environment
variables:

- `LIVEFIRE_LM_STUDIO_VERSION`
- `LIVEFIRE_LM_STUDIO_COLD_LOAD_MICROS`
- `LIVEFIRE_RUST_PEAK_RSS_BYTES`
- `LIVEFIRE_LM_STUDIO_PEAK_RSS_BYTES`

The final run summary reports exact-token length buckets. Each bucket uses the
complete run wall time, so it is a comparable share-of-run rate rather than a
claim that the bucket was timed in isolation.

Assembly keeps the existing JSON lexical index as the compatibility default.
Choose the disk-backed format explicitly for larger datasets:

```sh
rag assemble --prepared PREPARED --plan PLAN --embeddings EMBEDDINGS \
  --embedding-profile PROFILE --out INDEX --index-format sqlite-v3
```

`legacy-json-v2` writes fast-index version 2. `sqlite-v3` writes version 3 with
the same tokenizer, BM25 scores, result ordering, and occurrence-first filters,
while keeping lexical postings on disk. The provider can open both versions;
older packaged provider bundles should continue using version 2 until rebuilt.

A completed result containing the model's original 4,096-value vectors can
also produce smaller vector sets entirely from local files:

```sh
rag derive-embeddings --prepared PREPARED --plan PLAN \
  --embedding-profile PROFILE --embeddings EMBEDDINGS \
  --dimensions 2048 --out DERIVED_2048
rag assemble --prepared PREPARED --plan DERIVED_2048/plan \
  --embeddings DERIVED_2048/results \
  --embedding-profile DERIVED_2048/embedding-profile.json \
  --out INDEX_2048 --index-format sqlite-v3
```

The command keeps the first 2,048 or 1,024 values and normalizes the shorter
vector again. It does not contact LM Studio. The new profile, plan, receipts,
and result set have new identities and name the exact parent profile and result
set, so they cannot be presented as the original 4,096-value output. Compare
search results before choosing a smaller vector size:

```sh
rag compare-index-overlap --full-index INDEX_4096 \
  --reduced-index INDEX_2048 --query 'encoded PowerShell' --top-n 20
```

The comparison reports shared documents and overlap fractions for vector-only
and combined text-plus-vector search. It embeds the query once, then applies
the same local shortening step used for stored vectors.

Task recovery never contacts the model. Select the exact task ID from
`plan.json` and choose one explicit action:

```sh
rag recover-embedding-task --plan PLAN --embedding-profile PROFILE \
  --embeddings EMBEDDINGS --task-id TASK_SHA256 --action verify
rag recover-embedding-task --plan PLAN --embedding-profile PROFILE \
  --embeddings EMBEDDINGS --task-id TASK_SHA256 --action quarantine
rag recover-embedding-task --plan PLAN --embedding-profile PROFILE \
  --embeddings EMBEDDINGS --task-id TASK_SHA256 --action restore
```

Verification checks the plan, profile, receipt, vector bytes, and sanitized
task report together. Quarantine preserves corrupt or orphan files under
deterministic sibling names. Restore happens only when explicitly requested;
restored bytes must still pass verification before use.
Use `--representative-sample` for the fixed scenario-blind experiment path.
It declares a census for searchable relations with at most 1,000 documents and
a 2,000-document snapshot-bound hash-min cap above that threshold. Consequently
relations with 1,001 through 2,000 documents are also fully retained; only
larger relations are reduced. A second source scan spills every occurrence for
the final selected documents, so high-fanout membership is complete without
being held in memory.
