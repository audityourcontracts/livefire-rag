# Native Rust command-search regression

`rag-legacy-regression` is non-production test and benchmark tooling for the
frozen Q1-Q9 command-search cases and S1/S2 command-neighborhood diagnostics.
It has no Python dependency. The authoritative rules remain in
`fixtures/provider-poc/acceptance-suite.v1.json`; the Rust code reads that file
rather than copying its phrases or rank limits into source code.

The checks use case-insensitive literal substring matching over each result's
preview text and command/document ID. They show whether the historical search
behaviors were reproduced. A match does not establish that a candidate is
evidence, malicious, causally related, chronologically ordered, or sufficient
for an aggregate answer. Q1-Q9 are the nine-case primary denominator. S1/S2
are required diagnostics but do not enlarge that denominator. Q9 passes only
when its declared single-query boundary is reproduced: access-control changes
are present while upload commands are absent.

## Check an existing result without a model

The checker accepts the historical provider result envelope, JSON Lines with
one case per row, or current `rag query` and `rag similar` JSON nested under a
row's `output` field:

```sh
cargo run -p rag-testkit --bin rag-legacy-regression -- check \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --results fixtures/provider-poc/synthetic-provider-results.pass.json \
  --out reports/legacy-regression/check.json
```

The report records the exact byte count and SHA-256 digest of both inputs. It
refuses to overwrite an existing report.

## Run against a command-focused index

The run command uses the same Rust index operations as `rag query` and
`rag similar`. It performs dense search for Q1-Q9 in fixture order. Each query
uses one bounded local embedding request. S1/S2 read stored vectors from the
index and make no model request. Their descriptive historical seed labels are
not document IDs, so each must be bound explicitly to an exact indexed seed:

```sh
cargo run -p rag-testkit --bin rag-legacy-regression -- run \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --index indexes/commands/index \
  --embedding-endpoint http://127.0.0.1:1234 \
  --similar-seed S1=sha256:EXACT_LOGGING_BYPASS_DOCUMENT \
  --similar-seed S2=sha256:EXACT_DENIED_ACCESS_KEY_DOCUMENT \
  --out reports/legacy-regression/command-index-run
```

The output directory must not exist. It contains:

- `requests.jsonl`: exact ordered requests, including the S1/S2 seed bindings;
- `results.jsonl`: complete Rust search hits, scores, previews, and occurrence
  references without post-processing;
- `acceptance.json`: all literal checks and matched ranks;
- `manifest.json`: index identity, execution settings, returned model
  identities, and SHA-256 receipts for the other three files.

The fixed limits are at most 32 cases and 100 results per case. The historical
fixture requests 10. Queries run sequentially, similarity excludes its exact
seed, filters are empty, and stable index tie-breaking is preserved. The tool
does not silently select favorable cases, substitute a seed, change retrieval
mode, or overwrite a prior run.
