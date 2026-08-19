# Command RAG schemas

The active production contracts cover normalized M45 preparation, portable
embedding, RunPod execution, fast indexes, catalogues, query-vector sets, and
the Rust provider. Older command-record, anomaly, M21 projection, and Python
promotion schemas that remain in this directory are historical test fixtures;
they do not define an OpenBOTS compatibility path.

- `corpus-census.v1`: source-snapshot and projection-policy-bound counts of
  semantic documents, their source occurrences, and structured-only rows. It
  proves counting and projection closure only; it does not prove embedding,
  indexing, or search quality.
- `portable-dataset-identity.v1`: dataset scope plus source snapshot, mapping,
  and optional sorted source-admission components. M45 uses the latter to bind
  its relation contract and capability receipt into every prepared corpus.
- `command-record.v1`: adapter-neutral command, script-block, or cloud-action
  record with an immutable SDK source pointer.
- `comparison-universe.v1`: tenant/scope, identity policy, population policy,
  and exact source-snapshot membership for anomaly baselines.
- `command-snapshot-profile.v1`: canonical command-record Parquet profile,
  writer identity, ordering, and pointer coordinates.
- `powershell-analysis-policy.v1`: parser, AST, decoder, and execution-denial
  contract.
- `embedding-policy.v1`: exact model, artifact, runtime, prompt, dimension, and
  conformance identity.
- `embedding-policy.v2`, `embedding-task-receipt.v2`, and
  `embedding-result-set.v3`: a separately identified 2,048- or 1,024-value
  vector set derived from completed 4,096-value model output. These artifacts
  name the exact parent profile, result set, receipt, vector, and local
  prefix-and-normalize rule; their executor counters prove that derivation made
  no model requests.
- `embedding-policy.v3`: a complete upstream Qwen3-Embedding-8B checkpoint,
  tokenizer, TEI container, local-only load policy, request limits, vector
  behavior, and measured conformance identity. It is distinct from the local
  GGUF profile and from reduced vectors derived after embedding.
- `tei-model-artifact-set.v1`: the complete, ordered set of files for the
  pinned upstream Qwen3 checkpoint. Each file has an exact byte count and
  SHA-256 digest.
- `runpod-embedding-bundle.v1`, `runpod-worker-attempt.v1`, and
  `runpod-run-report.v1`: immutable remote-embedding inputs, one bounded worker
  attempt, and exact successful task coverage. The source files live with the
  Rust pipeline contracts and are included in the offline Python schema bundle;
  they contain no credentials or mutable provider state.
- `runpod-tei-conformance-candidate.v1` and
  `runpod-tei-conformance-result.v1`: the input and measured output for proving
  that the same pinned checkpoint, container, worker, and GPU class reproduce
  the same normalized vectors on two separate Pods. The candidate deliberately
  contains no expected vector digest.
- `runpod-executor-image-build-receipt.v1`: the exact custom executor image,
  official TEI base image, `linux/amd64` platform, Dockerfile, and Rust worker
  binary used by conformance and later embedding runs.
- `runpod-worker-observation.v1`: the attempt-scoped machine and accelerator
  identity written by the Rust host and verified by the Rust worker. This is an
  active internal host-to-worker wire contract, so it is included in the
  offline schema bundle; it is not a public provider tool API.
- `runpod-worker-runtime-event.v1`: immutable startup checkpoints and a sealed
  terminal failure for each embedding-worker attempt. These records make slow
  input verification distinguishable from model startup or inference failure
  without SSH or access to container standard error.
- `runpod-storage-challenge-response.v1`,
  `runpod-storage-challenge-failure.v1`, and
  `runpod-storage-challenge-receipt.v1`: the exact image-and-object-bound reply
  or bounded startup failure produced after the worker reads a fresh host
  upload through the mounted volume, and the host's Pod, price-cap, watchdog,
  response-verification, and termination evidence. These contain no credential
  values.
- `embedding-plan.v2`: exact tokenizer-bound, token-balanced embedding tasks
  over consecutive prepared-document ranges. It keeps the portable vector
  receipt/result formats while making task sizes and token counts reproducible.
- `benchmark-selection-manifest.v1` and `benchmark-selection-row.v1`:
  scenario-blind, relation-and-text-length-balanced selection of the nested
  512, 2,000, and 10,000-document local benchmark corpora.
- `projection-parity-report.v1`: content-free proof that the Rust and Python
  projectors produced the same identities for a fixed sampled row set.
- `tokenizer-parity-fixture.v1`: fixed token IDs from the tokenizer embedded in
  the pinned GGUF, including Unicode and maximum-input boundary cases.
- `dataset-catalogue.v1`: a deterministic set of modular dataset indexes that
  share one source, projection policy, and embedding profile.
- `catalogue-batch-search-request.v1`, `catalogue-batch-search-result.v1`, and
  `catalogue-batch-search-run.v1`: a bounded multi-query plan, its raw
  content-bearing catalogue results, and an atomically published run receipt.
  Both JSONL files require an LF after every row, including the final row; the
  published request file is an exact byte copy of the admitted plan.
- `query-vector-set.v1`: packed float32 query vectors for that exact frozen
  JSONL plan. It binds every raw and composed query hash, vector and order
  hash, embedding profile and policy, cloud execution identity, accelerator,
  and executor-image build receipt. Search selects by query ID; request bodies
  never contain caller-supplied vectors.
- `catalogue-review-pool-row.v1` and `catalogue-review-pool-manifest.v1`: a
  separate label-hidden candidate pool for people to mark search relevance,
  with retrieval modes, ranks, scores, system identities, and expected hints
  excluded.
- `embedding-task-run-report.v1`: content-free timing, byte, host, and identity
  evidence for one local embedding task.
- `embedding-run-summary.v1`: final local embedding totals, throughput,
  artifact sizes, and exact-token length buckets.
- `embedding-task-run-report.v2`: backend-neutral task evidence for TEI or
  other sealed executors. It binds the exact image, runtime, Rust worker,
  model, embedding profile, and certified single-GPU class while recording
  each worker machine separately.
- `embedding-run-summary.v2`: final totals for v2 task reports. It accepts
  different machine identities only when the sealed execution and accelerator
  identities are identical, and preserves every task's worker provenance.
- `tei-worker-report-context.v1`: the exact sealed execution identity and
  observed machine/backend fields required by the local/internal TEI command.
- `query-benchmark.v1`: content-free model and index-only query timings with
  the local machine and model setup recorded.
- `index-overlap.v1`: dense and fused top-result overlap between a full vector
  index and a locally derived 2,048- or 1,024-value index.
- `command-index-manifest.v1`: canonical Parquet objects, lineage, policies,
  coverage, and engine-independent query contract.
- `evaluation-suite-manifest.v1`: sealed corpus, split, catalogue, qrel, metric,
  and gate identities for repeatable experiments.
- `evaluation-run-report.v1`: independent conformance, quality, operational,
  audit-artifact, violation, waiver, and receipt reporting.
- `evidence-benchmark-manifest.v1`, `evidence-query-row.v1`,
  `evidence-qrel-row.v1`, and `evidence-hard-negative-row.v1`: evaluator-only,
  post-index fact-to-evidence queries, human relevance labels (qrels), and
  vector-blind controls.
- `evidence-eligibility-ledger.v1`: closed terminal source/index disposition,
  cohort, incident, and resampling-cluster binding for every declared fact atom.
- `evidence-candidate-universe-row.v1`: independently counted filtered candidate
  universe and expected top-20 cardinality receipt for each active query.
- `evidence-benchmark-run.v1`: retrieval, geometry, leakage, repeatability, and
  promotion reporting for a sealed evidence benchmark overlay.
- `evidence-benchmark-comparison.v1`: compact deterministic candidate/baseline
  comparison emitted by the standalone fact-to-evidence evaluator.
- `evidence-ranking-row.v1`: ranked provider output consumed by the
  fact-to-evidence evaluator, with an explicit cosine/native score contract and
  pointer/filter correctness flags.
- `cli-outliers.input.v1`: principal/population top-N anomaly request.
- `cli-search.input.v1`: semantic query with closed filters.
- `cli-similar.input.v1`: indexed-command similarity request.
- `cli-explain.input.v1` and `cli-explain.output.v1`: materialized component and
  prior-comparison explanation.
- `cli-candidates.output.v1`: ranked candidate pointers for outlier, search, and
  similarity tools.
- `fast-evidence-search.input.v1` and `fast-evidence-search.output.v1`:
  development contract for dense/lexical/fused queries over the Rust fast
  index. Results contain only snapshot/mapping-bound OCSF event references;
  indexed exact attributes are not tool evidence and are not returned.
- `fast-evidence-similar.input.v1` and `fast-evidence-similar.output.v1`:
  Rust stored-document similarity contract. It uses the seed vector already
  stored in the index, makes no model request, and returns the same bounded
  event references as search.
- `fast-index-manifest.v3`: the scalable Rust index format. It keeps the v2
  document, vector, occurrence, tokenizer, BM25, and result-order behavior but
  replaces the corpus-sized lexical JSON object with a content-bound SQLite
  inverted index. Its lexical `schema` field names the exact table contract;
  the overall manifest version and a filename extension alone do not safely
  identify that storage layout. Version 2 remains a separate supported format.
- `embedding-result-set.v2`, `fast-index-manifest.v4`, and
  `fast-build-report.v2`: diagnostic-only artifacts made with deterministic
  4096-dimensional test vectors and no model calls. Their required
  `test_only: true` marker prevents normal query, catalogue, packaging, and
  provider paths from treating them as model-produced embeddings.
- `fast-lexical-profile.v2`: the SQLite lexical object used by fast-index v3.
  It explicitly binds the storage tables while retaining the v1 tokenizer,
  BM25 score, document ID, and result-order behavior.
- `livefire.plugin.example`: native provider/builder packaging example for the
  SDK capability host.

These schemas are versioned draft sources of truth. They intentionally contain no SQL,
DuckDB types, vendor credentials, mutable vendor cursors, model conclusions, or
evidence claims.
