# Livefire RAG

A standalone semantic index builder and retrieval provider for Livefire-compatible
OCSF snapshots.

The first version is retrieval-only: it embeds deterministic OCSF event
projections, searches them, and returns stable event pointers. It does not create
evidence, infer relationships, or generate an investigation answer. Livefire can
hydrate the returned pointers through its authoritative OCSF tools.

## Intended dependency

This repository depends on released `livefire-sdk` protocol packages. It does not
depend on the Livefire monorepo. During local development, a path dependency may
be used, but release artifacts must pin an SDK version.

## Planned commands

```text
livefire-rag build --snapshot SNAPSHOT --policy embedding-policy.json --out INDEX
livefire-rag inspect --index INDEX
livefire-rag search --index INDEX --request request.json
livefire-rag evaluate --index INDEX --suite fixtures/evaluation-suite.json
livefire-rag provider --index INDEX
```

The first four commands run completely outside the Livefire runner. The final
command exposes the same tested implementation using `livefire.tool/1` JSONL.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/investigation-use-cases.md`](docs/investigation-use-cases.md), and
[`docs/implementation-plan.md`](docs/implementation-plan.md). Data access and
unsettled design choices are explicit in [`docs/data-boundary.md`](docs/data-boundary.md)
and [`docs/decision-log.md`](docs/decision-log.md).

## Repository status

This is a local, private-by-default repository. It has no remote configured and
contains specifications only; no model weights or indexed telemetry are tracked.
