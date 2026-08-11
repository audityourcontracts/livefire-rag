# Data and trust boundary

## Source adapters

Vendor adapters run only while creating a source snapshot. They may receive an
explicit credential handle and network allow-list for the selected source. They
must not write credentials, bearer tokens, session IDs, or mutable API URLs into
records, manifests, diagnostics, or source pointers.

An adapter writes a canonical command snapshot to staging and exits. The host
then validates object digests, schema conformance, pointer completeness, row
counts, time coverage, and path safety before sealing it. A remote source may
change after export; reproducibility begins at the sealed snapshot, not by
assuming a future vendor export will produce the same bytes.

## Index builder

The builder receives sealed source snapshots read-only and a new write-only
staging directory. It may not access vendor APIs, vendor credentials, evaluator
fixtures, Livefire traces/findings, ambient home directories, or unrelated
snapshots.

PowerShell decoding and parsing are static. The builder may decode bounded
representations and invoke a parser, but it must never execute a command, script,
macro, decompressed payload, shell expansion, or PowerShell expression.

Model access is one of:

- a pre-admitted local model artifact; or
- an explicitly allowed loopback LM Studio instance serving that artifact.

No remote embedding endpoint is permitted in v1. Build output and intermediate
embeddings inherit the source telemetry's tenant, confidentiality, encryption,
retention, revocation, and deletion policy.

## Runtime provider

The provider receives one admitted index read-only. It has:

- no Splunk or Panther credentials;
- no vendor-network access;
- no source-snapshot mount by default;
- no arbitrary filesystem access;
- optional loopback access only to the exact bound query embedder;
- bounded request/result sizes, candidates, memory, and wall time.

It returns immutable source pointers. Authoritative hydration is performed by a
separately admitted OCSF/Splunk/Panther evidence tool.

## Derived-data handling

Canonical command text, decoded layers, parse features, vectors, score tables,
and comparison excerpts are sensitive derived telemetry. Indexes are never
published in a plugin bundle. Logs must not contain commands or embeddings unless
an explicit diagnostic policy permits bounded redacted samples.
