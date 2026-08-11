# Standalone evaluation plan

This document summarizes the gates. The normative implementation programme,
fixture ladder, benchmark construction, report artifacts, and provisional
promotion thresholds are in `docs/test-program.md`.

Evaluation runs through the CLI and provider without the Livefire runner.
Labels and expected findings live only in the evaluation harness.

## Snapshot and build gates

- Every source snapshot and object digest verifies before indexing.
- Source plus rejected counts close exactly.
- Every indexed command has one resolvable local source pointer.
- Scores use only events strictly before the candidate and within 30 days.
- Equal timestamps never leak into one another's history.
- Rebuilding with identical inputs, model/runtime profile, and policy produces
  identical projections, counts, and canonical manifests.
- Parser/decoder limits are enforced and no command is executed.
- Indexing succeeds after vendor credentials are revoked and endpoints blocked.

## Retrieval correctness

- Time, principal, host, source, and scope filters have zero violations.
- Requested top N is honored up to eligible cardinality and contract limits.
- Every returned pointer resolves in the bound source snapshot.
- Stable requests have stable rankings, integer scores, tie-breaking, coverage,
  and result digests.
- Snapshot, model, policy, vector dimension, and object mismatches fail closed.
- Empty, corrupt, partial, left-censored, and insufficient-history conditions are
  distinguished.
- The provider passes golden tests with Splunk/Panther unavailable and no vendor
  credentials mounted.

## Anomaly quality

Evaluate principal and population rankings separately:

- nDCG@5/10/20 and Recall@5/10/20 for labelled unusual commands;
- rank of encoded/obfuscated PowerShell;
- rank of anomalous process ancestry;
- rank of new action and new target combinations;
- rank of sensitive cloud actions by a principal without prior use;
- benign-admin false-candidate rate;
- cold-start and sparse-principal behavior;
- ablation of action, target, structural, and obfuscation components.

The model may request any `top_n` within the schema. Evaluation reports the full
curve rather than promoting a fixed alert cutoff.

## Semantic retrieval quality

- Recall@1/5/10/20, MRR, and nDCG@20.
- Worst-paraphrase Recall@20 and rank variance.
- Relevant-versus-benign score margin.
- Command-to-command similarity quality.
- Exact token/BM25, EmbeddingGemma, Qwen3 0.6B/4B/8B, dimension, and
  quantization ablations.
- Exact DuckDB scan is the vector oracle; ANN Recall@20 must be at least 0.98
  before an ANN cache is admitted.

## Operational measures

- Snapshot/export and index build duration, rows/second, bytes, and peak memory.
- Embedding throughput for reference and LM Studio profiles.
- Provider startup, cold/warm query p50/p95, queries/second, and resident memory.
- Exact scan latency at realistic principal/time filters and full population.
- Browser feasibility is measured later and is not a v1 promotion gate.

## Anti-overfit suite

Hold out whole hosts/principals and scenarios. Rename users/hosts/resources,
shift timestamps, add benign decoys, vary background volume, paraphrase queries,
and delete labelled events. Never ship evaluation answers in projection prompts,
policies, fixtures used by developers, or index metadata.
