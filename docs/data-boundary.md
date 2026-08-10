# Data and trust boundary

## Inputs

The builder reads an admitted normalized OCSF snapshot through a constrained SDK
reader. It may read only declared semantic relations needed for event documents.
It must not read native records, evaluator fixtures, authority-only field
provenance, run traces, prior findings, credentials, or a developer home folder.

The full BOTS m15 snapshot is large and dominated by classes for which general
text embeddings are a poor first index. The initial production policy should
exclude system metrics and configuration snapshots, report excluded counts and
class/time coverage, and fail if input accounting does not close. Metrics belong
in numeric anomaly indexes; configuration history belongs in template/delta
indexes.

## Derived data

Documents, embeddings, lexical terms, previews, and indexes inherit the source
snapshot's tenant, confidentiality, encryption, retention, revocation, and
deletion policy. Embeddings are sensitive derived telemetry, not harmless model
metadata.

Exact timestamps and opaque identifiers remain filter/lexical metadata. Dense
text describes their typed role rather than expecting an embedding to understand
an IP address, hash, session ID, or credential-like value.

## Builder sandbox

- Source snapshot: read-only.
- Staging output: write-only until host verification.
- Network and secrets: denied by default.
- Model/tokenizer/runtime: pre-admitted, content-addressed artifacts.
- Build report: a claim, never self-authorizing.

The host independently checks paths, symlinks, source lineage, policy/model
identity, object digests, coverage, and conformance before issuing an admission
receipt.

## Provider sandbox

The provider reads one admitted index and no ambient filesystem. It receives an
exact dataset/index binding and enforced request/result/deadline limits at
session open. Network, credentials, and source-snapshot access are denied by
default. Logs are bounded diagnostics on stderr; stdout is protocol-only.

