# Data and trust boundary

## Admitted source

The active RAG builder accepts only a completed `livefire-ocsf` normalized
snapshot and its immutable build receipt. The current source is M45. Direct
OpenBOTS data, the historical M21/M41 outputs, Splunk exports, Panther exports,
and live vendor APIs are outside this boundary.

`livefire-ocsf` is responsible for admitting upstream data and writing the
normalized Parquet objects. `rag` verifies the receipt, object digests, schema,
row counts, pointer fields, and safe paths before projecting any row.
Reproducibility begins at that admitted normalized snapshot. The RAG repository
does not retain an adapter fallback that reads upstream source bytes.

## Index builder

The Rust builder receives the admitted M45 snapshot read-only and a new output
directory. It may not access vendor APIs, vendor credentials, evaluator
fixtures, Livefire traces/findings, ambient home directories, unrelated
snapshots, or historical OpenBOTS compatibility paths.

PowerShell decoding and parsing are static. The builder may decode bounded
representations and invoke a parser, but it must never execute a command, script,
macro, decompressed payload, shell expansion, or PowerShell expression.

Local development embedding permits model access through:

- a pre-admitted local model artifact; or
- an explicitly allowed loopback LM Studio instance serving that artifact.

The RunPod build path may use explicitly authorized remote embedding workers
only after Rust has produced and verified a sealed prepared corpus. That
exception is limited to embedding and does not grant the runtime provider
general network access. The cloud bundle contains only prepared document
shards and exact execution artifacts. Source occurrence rows remain local.
Remote storage and workers use private access, encryption in transit and at
rest, bounded retention, tenant isolation, and credentials supplied through
environment-backed host secret handling rather than manifests or logs.

Build output, prepared semantic text, and intermediate embeddings inherit the
source telemetry's tenant, confidentiality, encryption, retention, revocation,
and deletion policy.

## Runtime provider

The Rust provider receives one admitted index read-only. It has:

- no source-system credentials;
- no vendor-network access;
- no source-snapshot mount by default;
- no arbitrary filesystem access;
- optional loopback access only to the exact bound query embedder, or a sealed
  cloud-profile query-vector set for a frozen request plan;
- bounded request/result sizes, candidates, memory, and wall time.

It returns immutable OCSF event references. The released OCSF query service
must resolve and confirm those references before their fields are used as
facts.

## Derived-data handling

Canonical command text, decoded layers, parse features, vectors, score tables,
and comparison excerpts are sensitive derived telemetry. Indexes are never
published in a plugin bundle. Logs must not contain commands or embeddings unless
an explicit diagnostic policy permits bounded redacted samples.
