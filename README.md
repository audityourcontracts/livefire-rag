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
livefire-rag provider --index INDEX
```

Every command is testable without a Livefire checkout. The provider uses the
released `livefire-sdk` protocol; the repository never imports Livefire source.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/source-snapshots.md`](docs/source-snapshots.md),
[`docs/command-index.md`](docs/command-index.md),
[`docs/physical-formats.md`](docs/physical-formats.md),
[`docs/model-selection.md`](docs/model-selection.md), and
[`docs/implementation-plan.md`](docs/implementation-plan.md).

## Repository status

This is a local, private-by-default specification repository. It has no remote
configured. No model weights, credentials, source telemetry, or built indexes
are tracked.
