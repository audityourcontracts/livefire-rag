# Command RAG schemas

- `command-record.v1`: adapter-neutral command, script-block, or cloud-action
  record with an immutable SDK source pointer.
- `comparison-universe.v1`: tenant/scope, identity policy, population policy,
  and exact source-snapshot membership for anomaly baselines.
- `command-snapshot-profile.v1`: canonical command-record Parquet profile,
  writer identity, ordering, and pointer coordinates.
- `powershell-analysis-policy.v1`: parser, AST, decoder, and execution-denial
  contract.
- `command-anomaly-policy.v1`: rolling history, scopes, four components, weights,
  and cold-start behavior.
- `embedding-policy.v1`: exact model, artifact, runtime, prompt, dimension, and
  conformance identity.
- `command-index-manifest.v1`: canonical Parquet objects, lineage, policies,
  coverage, and engine-independent query contract.
- `evaluation-suite-manifest.v1`: sealed corpus, split, catalogue, qrel, metric,
  and gate identities for repeatable experiments.
- `evaluation-run-report.v1`: independent conformance, quality, operational,
  audit-artifact, violation, waiver, and receipt reporting.
- `evidence-benchmark-manifest.v1`, `evidence-query-row.v1`,
  `evidence-qrel-row.v1`, and `evidence-hard-negative-row.v1`: evaluator-only,
  post-index fact-to-evidence queries, judgments, and vector-blind controls.
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
- `livefire.plugin.example`: native provider/builder packaging example for the
  SDK capability host.

These schemas are draft v1 sources of truth. They intentionally contain no SQL,
DuckDB types, vendor credentials, mutable vendor cursors, model conclusions, or
evidence claims.
