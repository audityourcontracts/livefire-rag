# Livefire RAG

`livefire-rag` builds immutable command-intelligence indexes and serves typed
retrieval and anomaly tools. Its first domain is command-line activity,
including PowerShell, shell commands, process ancestry, and cloud CLI/API
invocations.

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
- The Wasmtime-first integration is the existing Livefire pattern: an
  import-free runner emits `call_tool`; the native capability host executes the
  RAG provider. Browser host parity is later work.

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

## Planned commands

```text
livefire-rag build --source SNAPSHOT --policy POLICY --out INDEX
livefire-rag verify --index INDEX
livefire-rag inspect --index INDEX
livefire-rag outliers --index INDEX --request REQUEST
livefire-rag search --index INDEX --request REQUEST
livefire-rag evaluate --index INDEX --suite SUITE
livefire-rag evaluate-facts --queries QUERIES --qrels QRELS \
  --candidate RANKINGS --baseline BASELINE --out REPORT
livefire-rag provider --index INDEX
```

Every command is testable without a Livefire checkout. The provider uses the
released `livefire-sdk` protocol; the repository never imports Livefire source.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/source-snapshots.md`](docs/source-snapshots.md),
[`docs/command-index.md`](docs/command-index.md),
[`docs/physical-formats.md`](docs/physical-formats.md),
[`docs/model-selection.md`](docs/model-selection.md), and
[`docs/implementation-plan.md`](docs/implementation-plan.md). The complete
source-fidelity, conformance, quality, performance, and reporting program is in
[`docs/test-program.md`](docs/test-program.md).

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
python3 -m unittest discover -s tests -v
uv run --with jsonschema python tools/validate_evidence_fixtures.py \
  --sdk-specs ../livefire-sdk/specs \
  --report reports/fact-evidence-synthetic/report.json
```

## Repository status

This is a private specification and implementation repository. Its GitHub
remote is private. No model weights, credentials, source telemetry, or built
indexes are tracked.
