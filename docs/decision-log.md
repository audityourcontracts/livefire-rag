# Decision log

## Locked for v1

- Source ingestion produces immutable snapshots; there is no continuous index.
- OCSF, Splunk, and Panther integrations are source adapters and exact evidence
  tools. They are not retrieval backends and are not runtime RAG dependencies.
- The first index covers command lines, including PowerShell, shell commands,
  process ancestry, and cloud CLI/API actions represented as canonical commands.
- Compare a command with the principal's own strictly prior history, the strictly
  prior population, or both.
- Baseline horizon is the previous 30 days. For one-day OpenBOTS data, this is
  naturally all events strictly before the candidate.
- Return requested top N up to the contract maximum. There is no hidden alert
  threshold and no claim that an outlier is malicious.
- Four reported components are action novelty, target novelty, structural
  novelty, and obfuscation novelty.
- Scores, baseline coverage, and nearest prior comparisons are materialized at
  build time. Queries never rewrite history.
- Static decoding supports bounded Base64 UTF-8/UTF-16LE, URL/escape decoding,
  identified gzip/deflate, constant concatenation, backtick/case normalization,
  at most three layers, and a policy-bounded expanded size. Nothing is executed.
- Native indexing may use Microsoft's PowerShell parser. The stored contract is
  the stable `powershell_ast_document.v1`, never a serialized .NET object graph.
- Tools are `cli.outliers`, `cli.search`, `cli.similar`, and `cli.explain`.
- Tool results are candidate pointers/derived comparisons, not evidence or
  generated answers.
- Canonical storage is Parquet plus a content-addressed manifest.
- Native DuckDB exact search is the first query engine and correctness oracle.
- HNSW/ANN and full-text indexes are optional rebuildable caches.
- V1 integration uses the native capability host around Livefire's Wasmtime
  runner. Browser host parity is a later conformance target.
- The quality-first embedding candidate is the official
  `Qwen/Qwen3-Embedding-8B` checkpoint. Selection of dimension, runtime artifact,
  and quantization is gated by local domain evaluation.
- LM Studio is the first local embedding-server adapter. No external embedding
  API is needed.

## Versioned policy decisions

The following are not protocol constants. Each selected value is recorded in an
immutable policy and creates a new index identity when changed:

- command projection and tokenizer;
- PowerShell parser/decoder versions and limits;
- model revision, artifact, runtime, quantization, prompts, and dimensions;
- score feature definitions, calibration, missing-history behavior, and weights;
- exact retrieval tie-breaking;
- any ANN or lexical cache parameters.

## Deferred

- Continuous or incremental indexing.
- Browser-side embedding inference and ANN.
- A generated `rag.answer` tool.
- Cross-encoder reranking.
- Knowledge, ATT&CK, playbook, or prior-case indexes; those are separately
  governed corpora and tool bindings.
- Configuration-delta and numeric-metric anomaly indexes.
