# Livefire RAG

`livefire-rag` builds immutable search indexes over normalized OCSF evidence
candidates and serves typed retrieval tools. It returns ranked leads and exact
event references. Livefire still owns hypotheses, evidence selection,
conclusions, and stopping.

## Current production boundary

The active source is the admitted normalized Parquet output from
`livefire-ocsf` M45. The builder does not read OpenBOTS, vendor exports, or the
historical M21/M41 output trees, and it does not query a live security service.

The production implementation is Rust:

- `rag` performs census, projection, preparation, embedding orchestration,
  finalization, index assembly, inspection, query, similarity, catalogue, and
  RunPod control work.
- `rag-runpod-worker` verifies and executes bounded cloud embedding assignments.
- `rag-provider` opens an immutable index read-only and implements the Livefire
  SDK handshake, open, call, health, and close lifecycle.
- Python is limited to tests, benchmarks, statistical analysis, people reviewing
  and marking which search results are relevant, and PCA visualization. It is
  not a production builder, promoter, query tool, provider, packager, or cloud
  worker.

The provider has no source-system credentials and makes no vendor call at query
time. Search returns candidate references, not verified facts. The released
OCSF query service must resolve and confirm returned event references before
their fields are used as evidence. A production host must run the provider with
enforced file, network, memory, and process limits.

## Current M45 corpus

The admitted source is:

```text
$PWD/../livefire-ocsf/data/builds/m45-progressive-disclosure-run-b
```

Its sealed identities are:

- snapshot: `23077f2605cb4d0ca7f1a857dd0c540d990911197c21a80c886fc1099f6e7d10`
- dataset: `ba9e0c1ff5f1154defc0956e1984fc1168d0424d29f8d4d6b02e1d1c93fbbe46`
- mapping: `641e479d5d830edef80c4e57c8048eed9b26710d35a18101e9441065f4337bb7`
- capability receipt: `d9e7e485213c09abb9862f8620cebc410649bc8241688ae21c53721958493e1b`

The Rust census accounted for 13,905,577 normalized events from 2,030,269
source records with no rejected, unsupported, or unresolved rows. M45 keeps the
normalized event bytes from M44 but changes the authoritative snapshot, mapping,
graph, provenance, and capability identities. Every prepared corpus is therefore
rebuilt rather than relabelled. The system-metric relation remains structured-only;
graph, provenance, and subject-alias relations remain exact lookup or hydration
data rather than embedding input.

The M45 relation datasets are prepared and verified locally under
`indexes/livefire-ocsf-m45-v1/prepared`. This is model-independent work: it does
not mean that embeddings or final indexes are complete. Generated
corpora, model weights, reports, credentials, and built indexes are ignored and
are not committed to Git.

See [the M45 prepared dataset report](docs/m45-prepared-dataset-report.md) for
the source qualification, corpus counts, prepared artifact inventory, and the
exact boundary between files copied for embedding and files retained locally.

## Active Rust workflow

Build the release tools with:

```sh
cargo build --release --workspace
```

### Reproduce preprocessing from `livefire-ocsf` Parquet

Start with a completed M45 snapshot produced by the `livefire-ocsf` pipeline.
Its root must contain `build-receipt.json`, `completeness-receipt.json`, the
`normalized/`, `graph/`, `provenance/`, and `capabilities/` directories, and all
objects named by the receipt. Follow the upstream
[`livefire-ocsf` README](https://github.com/audityourcontracts/livefire-ocsf) to
build and verify that snapshot. The commands below use the independently
reproduced M45 run-b output, but they accept the same snapshot at another local
path through `LIVEFIRE_OCSF_SNAPSHOT`.

From the `livefire-rag` checkout, run this complete model-independent workflow.
Every output directory must be new; publication is atomic and refuses to
overwrite an existing result.

```sh
set -eu

RAG_ROOT=$PWD
M45_SNAPSHOT=${LIVEFIRE_OCSF_SNAPSHOT:-$RAG_ROOT/../livefire-ocsf/data/builds/m45-progressive-disclosure-run-b}
M45_OUTPUT=$RAG_ROOT/indexes/livefire-ocsf-m45-v1
M45_REPORTS=$RAG_ROOT/reports/livefire-ocsf-m45

test -f "$M45_SNAPSHOT/build-receipt.json"
test -f "$M45_SNAPSHOT/completeness-receipt.json"
test -f "$M45_SNAPSHOT/capabilities/snapshot-capabilities.v1.json"

cargo build --release --workspace --bins
mkdir -p "$M45_OUTPUT/prepared" "$M45_REPORTS"

target/release/rag census \
  --snapshot "$M45_SNAPSHOT" \
  --out "$M45_REPORTS/full-census.json" \
  --workers 8

M45_SEARCHABLE_RELATIONS='account-change:ocsf_account_change
api-activity:ocsf_api_activity
application-lifecycle:ocsf_application_lifecycle
authentication:ocsf_authentication
cloud-resources-inventory-info:ocsf_cloud_resources_inventory_info
datastore-activity:ocsf_datastore_activity
detection-finding:ocsf_detection_finding
dns-activity:ocsf_dns_activity
email-activity:ocsf_email_activity
entity-management:ocsf_entity_management
event-log-activity:ocsf_event_log_activity
ext-livefire-configuration-snapshot:ocsf_ext_livefire_configuration_snapshot
file-activity:ocsf_file_activity
http-activity:ocsf_http_activity
inventory-info:ocsf_inventory_info
network-activity:ocsf_network_activity
process-activity:ocsf_process_activity
user-inventory:ocsf_user_inventory'

for specification in $M45_SEARCHABLE_RELATIONS; do
  dataset=${specification%%:*}
  relation=${specification#*:}
  prepared=$M45_OUTPUT/prepared/$dataset

  target/release/rag prepare \
    --snapshot "$M45_SNAPSHOT" \
    --dataset-id "livefire-ocsf-m45-$dataset" \
    --dataset-version 1 \
    --relation "$relation" \
    --out "$prepared" \
    --workers 8

  target/release/rag verify-prepared --prepared "$prepared"
done

# This separate projection retains normalized command, script, and API-operation
# shapes. It does not read OpenBOTS or another raw source directory.
target/release/rag prepare-commands \
  --snapshot "$M45_SNAPSHOT" \
  --out "$M45_OUTPUT/commands-prepared" \
  --workers 8

target/release/rag verify-prepared \
  --prepared "$M45_OUTPUT/commands-prepared"

# Freeze nested 512-, 2,000-, and 10,000-document selections so LM Studio and
# RunPod can be compared using exactly the same formatted document inputs.
set --
for specification in $M45_SEARCHABLE_RELATIONS; do
  set -- "$@" --relation "${specification#*:}"
done

target/release/rag prepare-benchmark \
  --snapshot "$M45_SNAPSHOT" \
  --dataset-id livefire-ocsf-m45-backend-benchmark \
  --dataset-version 1 \
  --selection-seed local-scale-benchmark-v1 \
  --workers 8 \
  "$@" \
  --out "$M45_OUTPUT/benchmark-v1"

for size in 00512 02000 10000; do
  target/release/rag verify-prepared \
    --prepared "$M45_OUTPUT/benchmark-v1/prepared-$size"
done
```

`rag` verifies the exact M45 receipt, normalized Parquet objects, relation
inventory, mapping, relation contract, capability sidecar, and row counts before
publishing. It groups equal formatted text into stable document IDs while
retaining every contributing event reference in occurrence shards. The
system-metric relation is accounted as structured-only and is not embedded.

To move a prepared dataset to an embedding machine, transfer only its
`manifest.json` and every relative path in the manifest's `documents` array.
For example, this creates a backend-neutral archive for one relation:

```sh
PREPARED=$M45_OUTPUT/prepared/process-activity
ARCHIVE=$RAG_ROOT/process-activity-prepared.tar

(
  cd "$PREPARED"
  {
    printf '%s\n' manifest.json
    jq -r '.documents[].path' manifest.json
  } | tar -cf "$ARCHIVE" -T -
)
```

Keep every path in the manifest's `occurrences` array and `accounting.json`
locally. They are not model inputs; they are required later to assemble the
search index and preserve exact M45 event references. The RunPod bundle command
automates the same allowlisted transfer after a cloud profile and token plan
have been sealed.

Stop here when selecting between LM Studio and RunPod. Embedding plans bind the
exact tokenizer, model profile, and runtime conformance evidence, so they are
created only after the backend is chosen. Prepared document shards can be copied
to RunPod; occurrence shards stay local for final index assembly.

For a local LM Studio development profile, freeze exact token-balanced work,
embed it, finalize all task outputs, and assemble one SQLite index:

```sh
target/release/rag plan-embeddings \
  --prepared PREPARED --embedding-profile LOCAL_PROFILE \
  --tokenizer-json TOKENIZER_JSON --tokenizer-ref TOKENIZER_REF \
  --maximum-task-tokens 262144 --maximum-task-documents 2048 --out PLAN

target/release/rag embed \
  --prepared PREPARED --plan PLAN --embedding-profile LOCAL_PROFILE \
  --embedding-endpoint http://127.0.0.1:1234 --out EMBEDDINGS

target/release/rag finalize-embeddings \
  --prepared PREPARED --plan PLAN --embedding-profile LOCAL_PROFILE \
  --embeddings EMBEDDINGS

target/release/rag assemble \
  --prepared PREPARED --plan PLAN --embeddings EMBEDDINGS \
  --embedding-profile LOCAL_PROFILE --index-format sqlite-v3 --out INDEX
```

Query and stored-document similarity are also Rust operations:

```sh
target/release/rag query \
  --index INDEX --query 'encoded PowerShell download' --mode fused --top-n 20 \
  --embedding-endpoint http://127.0.0.1:1234

# This reads the seed vector from the index and makes no model request.
target/release/rag similar \
  --index INDEX --document-id DOCUMENT_ID --top-n 20
```

Several embedding processes may execute non-overlapping task ranges against the
same output directory. `finalize-embeddings` refuses missing, duplicate, or
extra output and publishes the result manifest only when the plan is complete.
Preparation and assembly verify the complete document and occurrence chain.

See [the Rust CLI guide](crates/rag-builder/README.md) for catalogue search,
sealed query-vector sets, recovery, local measurements, and analysis handoff.
See [the provider guide](crates/rag-provider/README.md) for the native provider
and local SDK lifecycle checks.

## RunPod phase

The current cloud phase reuses the exact M45 prepared documents. It uses the
pinned upstream `Qwen/Qwen3-Embedding-8B` revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` and keeps its FP16 cloud vectors
separate from local LM Studio Q4 vectors.

The Rust contracts, worker, S3 transfer, Pod control, conformance, and sealed
query-vector paths are implemented and covered by offline contract tests. The
model artifacts and custom executor image have been built and verified locally.
No paid Pod has been launched, so
there is not yet a measured RunPod throughput, cost, or reproducibility result.
A launch will require explicit credentials and price limits after the offline
checks pass.

The cloud workflow never uploads occurrence shards because embedding needs only
prepared documents. It launches a digest-pinned custom executor image, serves
the model on loopback, writes attempt-scoped outputs, fetches only exact declared
keys, and terminates the Pod under time and cost guards. Cloud-profile indexes
use sealed query vectors created by the same profile; local Q4 query vectors
cannot be mixed with them.

The implemented command family is `rag runpod`. Its exact staged checks and
current limitations are in [the RunPod embedding guide](docs/runpod-embedding.md).

## Python analysis boundary

The installed project has no `livefire-rag` Python console entry point. The
supported Python surface is `livefire_rag_analysis` plus reviewer and benchmark
scripts. For example:

```sh
uv run --extra analysis python -m livefire_rag_analysis inspect --index INDEX
uv run --extra analysis python -m livefire_rag_analysis pca \
  --index INDEX --out REPORT_DIR
uv run --extra analysis python -m livefire_rag_analysis evaluate \
  --run RUN.jsonl --qrels QRELS.jsonl --out REPORT.json \
  --planned-query-id q-1 --planned-query-id q-2
```

The `src/livefire_rag` modules and old Python command implementations remain in
the source tree only as frozen test and comparison oracles. They must not be
used to build or serve the M45 indexes.

## Historical artifacts

Older M21/OpenBOTS provider demonstrations, Python promotion code, the M41
local scale run, and their sealed reports are retained so earlier measurements
and regression fixtures remain auditable. They are not accepted inputs or
compatibility targets for the active M45 path.

Key historical documents are explicitly marked at their source:

- [M41 local-first scale plan](docs/local-first-embedding-scale-plan.md)
- [M41 corpus census](docs/m41-corpus-census.md)
- [M21 full projection build report](docs/generic-evidence-m21-v1-build-report.md)
- [Python M21/OpenBOTS provider proof of concept](docs/standalone-provider-poc.md)
- [early Rust vertical-slice specification](docs/rust-experimental-rag-spec.md)

The active data and trust rules are in [the data-boundary guide](docs/data-boundary.md),
the portable file contracts are in
[the embedding-pipeline specification](docs/portable-embedding-pipeline.md), and
the active cloud execution plan is in
[the RunPod embedding guide](docs/runpod-embedding.md).

## Repository status

This is a private specification and implementation repository. Its GitHub
remote is private. No model weights, credentials, source telemetry, or built
indexes are tracked.
