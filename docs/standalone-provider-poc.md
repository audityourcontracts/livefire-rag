# Standalone semantic provider POC

This repository now contains a runnable, development-only semantic index and
Livefire SDK tool provider. It is independent of the Livefire repository. The
provider implements `handshake`, `open`, `call`, `health`, and `close` over JSON
Lines and exposes `cli.search` and `cli.similar`.

The POC does not admit evidence. Both tools return immutable source-pointer
candidates or an explicit miss. The promoted M21/OpenBOTS pack remains
`development_only`: it mixes normalized and authority data, retains
representative metadata after semantic deduplication, and uses non-admitted
`record_id_only` provenance hints.

## Environment and fixture smoke test

```sh
uv sync --extra test --extra prototype

uv run livefire-rag build-fixture \
  --fixture fixtures/semantic-index/small.v1.json \
  --out indexes/semantic-small-poc

uv run livefire-rag verify --index indexes/semantic-small-poc

uv run livefire-rag similar \
  --index indexes/semantic-small-poc \
  --request path/to/cli-similar-request.json
```

Index construction refuses to overwrite an existing path. Remove or choose a
new versioned `indexes/` path for another build. `indexes/`, `dist/`, and
`reports/` are ignored generated-output roots.

## Promote the exploratory corpus without re-embedding

The promotion command rebuilds the corpus from the pinned local M21/OpenBOTS
inputs, requires its digest and embedding-profile digest to match the existing
cache, checks vector shape and L2 norms, fills required representative metadata,
and then writes a new immutable development pack.

```sh
uv run --extra prototype livefire-rag promote-prototype \
  --repository-root . \
  --prototype-dir reports/prototype-rag-demo \
  --out indexes/prototype-m21-poc

uv run livefire-rag verify --index indexes/prototype-m21-poc
```

This is promotion into a queryable POC format, not SDK admission.

## Run the frozen real-data demonstration

LM Studio must serve the model key pinned by the index embedding profile on the
loopback endpoint.

```sh
uv run livefire-rag demo-provider-poc \
  --index indexes/prototype-m21-poc \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --embedding-endpoint http://127.0.0.1:1234 \
  --out reports/provider-poc/provider-results.json \
  --requests-out reports/provider-poc/provider-requests.jsonl

python3 tools/check_provider_poc.py \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --results reports/provider-poc/provider-results.json \
  --out reports/provider-poc/acceptance.json \
  --markdown reports/provider-poc/acceptance.md
```

The generated request transcript uses `${session_id}` after `open`, so the SDK
harness can materialize the actual provider session during replay.

## Package and exercise the SDK bundle

```sh
uv run livefire-rag package-bundle \
  --index indexes/prototype-m21-poc \
  --sdk-specs ../livefire-sdk/specs \
  --out dist/livefire-rag-poc

../livefire-sdk/target/debug/livefire-sdk \
  --specs ../livefire-sdk/specs \
  validate-bundle \
  --manifest dist/livefire-rag-poc/plugin.json \
  --root dist/livefire-rag-poc

uv run ../livefire-sdk/target/debug/livefire-sdk \
  --specs ../livefire-sdk/specs \
  invoke \
  --program dist/livefire-rag-poc/bin/livefire-rag-provider \
  --requests reports/provider-poc/provider-requests.jsonl \
  > reports/provider-poc/sdk-wire-output.jsonl

uv run --extra test python tools/verify_provider_replay.py \
  --demo-results reports/provider-poc/provider-results.json \
  --sdk-wire reports/provider-poc/sdk-wire-output.jsonl \
  --out reports/provider-poc/sdk-replay-verification.json \
  --annotate-demo
```

The final command requires exact parsed-JSON equality for call outputs 3 through
13 in frozen Q1-Q9/S1/S2 order. It records both canonical output digests per case
and annotates the demo result only after every output matches.

The bundle inventories its provider wrapper and Python implementation, both tool
descriptors, and the RAG request/result schemas with exact byte lengths and
SHA-256 digests. Its wrapper resolves Python from `PATH`; invoking it under the
project's `uv` environment supplies the locked NumPy runtime.

## Tests

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --extra test python -m unittest discover -s tests -v
uv run --with jsonschema python tools/validate_evidence_fixtures.py \
  --sdk-specs ../livefire-sdk/specs \
  --report reports/fact-evidence-synthetic/report.json
```
