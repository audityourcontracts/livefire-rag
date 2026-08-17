# RunPod embedding build

Status: implementation in progress; no paid Pod has been launched

## Outcome

Build the search datasets from the normalized Parquet files produced by the
current `livefire-ocsf` M45 build, then create a separate set of Qwen3
Embedding 8B vectors with a pinned RunPod GPU runtime. The local LM Studio Q4
vectors and the RunPod FP16 vectors remain different profiles and must never be
mixed in one result set or index.

The source boundary is the M45 build receipt and its `normalized/` objects.
Upstream raw event files are not an input to this path. The M45 identities
are:

- snapshot `23077f2605cb4d0ca7f1a857dd0c540d990911197c21a80c886fc1099f6e7d10`;
- dataset `ba9e0c1ff5f1154defc0956e1984fc1168d0424d29f8d4d6b02e1d1c93fbbe46`;
- mapping `641e479d5d830edef80c4e57c8048eed9b26710d35a18101e9441065f4337bb7`;
- capability receipt `d9e7e485213c09abb9862f8620cebc410649bc8241688ae21c53721958493e1b`;
- 13,905,577 normalized events from 2,030,269 source records, with no rejected,
  unsupported, or unresolved records in the build receipt;
- 560,842 searchable documents and 6,367,276 retained event references across
  all 18 searchable relations. Network contributes 138,276 documents and
  1,042,076 references; the remaining 17 relations contribute 422,566 and
  5,325,200 respectively. The system-metric relation remains structured-only.

The current Rust census is written to
`reports/livefire-ocsf-m45/full-census.json`. The report is local generated
evidence and is not committed to Git. M45 keeps the normalized Parquet bytes
from M44, but its new snapshot, mapping, graph, provenance, and capability
identities require new prepared manifests and document identifiers.

## Execution design

The first cloud implementation uses one dedicated RunPod Secure Cloud Pod and
one persistent network volume. A custom digest-pinned executor image derives
from a separately pinned Text Embeddings Inference (TEI) base image and adds
the fixed Rust worker at `/usr/local/bin/rag-runpod-worker`. TEI serves Qwen3
Embedding 8B on loopback. The Rust worker runs in the same container, verifies
every input, executes immutable task ranges, and
writes one vector shard, receipt, and sanitized task report per task.

The custom image also has a sealed build receipt. It binds the exact
`linux/amd64` image digest to the official TEI base digest, Dockerfile bytes,
and worker-binary bytes. Conformance stages and verifies the Dockerfile and
receipt; the measured policy and every execution bundle repeat the receipt's
component identity.

No inference port is public. The host uses RunPod's GraphQL create mutation so
the provider receives an absolute `terminateAfter` deletion deadline, then
uses the REST API to admit the returned Pod and manage its lifecycle. The
GraphQL API does not expose that deadline for read-back, so receipts label it
`requested_unobservable`; an in-process wall-clock and cost watchdog remains
mandatory. A separate S3-compatible credential stages and fetches exact named
objects from the network volume. Secrets are read from environment variables
at runtime and are absent from manifests, reports, command arguments, and
logs.

Prepared occurrence shards remain local because embedding needs only prepared
documents. A cloud bundle contains:

- the prepared manifest and every referenced document shard;
- a token-balanced embedding plan and exact token-count object;
- the executable tokenizer, embedding profile, and every model object;
- the pinned Rust worker binary and TEI container identity;
- non-overlapping worker assignments and the exact expected output keys.

The model-independent preparation phase intentionally stops before creating an
embedding plan. A plan binds either the local LM Studio profile or the measured
RunPod profile and must be created only after that backend is selected.

The first version uses static half-open task ranges. Each task owns unique
paths, so a completed receipt is the restart authority. Multiple Pods are not
introduced until one-Pod correctness and the 10,000-document timing test pass.

## Model identity

The RunPod profile uses the upstream `Qwen/Qwen3-Embedding-8B` revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, last-token pooling, inputs capped
at 8,192 tokens, 4,096-dimensional L2-normalized vectors, FP16 model compute,
and float32 API and stored-vector values. The TEI image, model files, tokenizer,
worker binary, formatting rules, served model name, and conformance output all
have separate content identities. Both the official TEI base and the custom
executor image are bound and repeated through conformance; Pod creation always
launches the custom executor image.

An executable profile is sealed only after the same GPU architecture and
pinned TEI image produce the exact conformance vector digest twice, including
one fresh-Pod replay. Until then, the profile is a template and cannot create
plans or execute tasks.

## Staged checks

1. Run Rust fake-server tests for bounded responses, model identity, response
   ordering, retries, timeouts, interruption, and secret redaction.
2. Build the worker container locally and run it against a deterministic fake
   backend. Prove restart, exact output coverage, and failure cleanup.
3. Create a network volume without GPU compute. Upload and download the exact
   bundle object list and verify every byte count and SHA-256 digest.
4. Launch the exact digest-pinned executor image with the network volume and
   no model load. After RunPod reports the Pod as running, upload a fresh
   32-byte challenge through S3. The worker must read and hash it as UID/GID
   1000, publish the exact image-and-challenge-bound response with an immutable
   hard link, and the host must download and verify that response. Retain the
   scheduler request, admitted Pod and returned price, runtime and total-cost
   watchdog, deletion outcome, and exact object identities.
5. Launch one fixed GPU type with explicit hourly, runtime, and total-compute
   price caps. Run only model conformance, terminate the Pod promptly, and
   repeat on a fresh Pod.
6. Embed the small M45 command-focused dataset and run the native Rust search
   and stored-document similarity regression. Missing normalized source fields
   are reported as source limitations; the worker does not reach outside the
   admitted normalized snapshot to fill them in.
7. Embed the frozen 10,000-document sample. Measure model load time, active
   execution time, documents and tokens per second, request latency, retries,
   GPU type, exact bytes transferred, and the hourly price returned by RunPod.
8. Forecast the complete build from the lower measured token-throughput bound:

   `hours = planned_tokens / conservative_tokens_per_second / 3600`

   `compute_cost = returned_hourly_price * hours * 1.25`

   Stop if the forecast exceeds the configured time or cost limit.
9. Re-plan every intended M45 dataset with the sealed cloud profile, execute all
   assignments, fetch only expected objects, finalize locally, assemble
   SQLite-v3 indexes, validate the catalogue, and run Rust search, similarity,
   and provider lifecycle tests.

## Operational rules

- A Pod launch always requires an explicit maximum hourly price. If the created
  Pod reports a different GPU, image, volume, cloud type, or a price above the
  cap, the controller immediately terminates it and reports whether cleanup
  succeeded.
- The network volume is never deleted automatically. It remains available
  until all downloaded results pass local verification.
- One TEI server has one Rust client process. TEI performs its own dynamic
  batching; competing clients on the same GPU are not used as a concurrency
  shortcut.
- Query vectors used to test a cloud index must come from the exact cloud
  profile. Local LM Studio Q4 query vectors cannot search a RunPod FP16 index.
- Frozen cloud query workloads are published as
  `livefire.rag.query-vector-set/1`: the exact JSONL plan, packed float32
  vectors, raw and composed query hashes, vector/order digests, and exact
  profile, policy, execution, accelerator, and executor-image build
  identities. Direct and catalogue search select only a known query ID from
  this set; they never accept raw vectors from a request. Arbitrary dense or
  fused queries still require a private exact-profile endpoint.
- Python is not part of preparation, embedding, finalization, assembly, query,
  similarity, packaging, or provider execution. It remains available only for
  test analysis and visualization.

## Host command boundary

The host uses `rag runpod` commands. `bundle build` creates a new local tree
and verifies the prepared corpus, v2 plan and token counts, measured policy,
conformance fixture, frozen `--query-plan` JSONL, tokenizer, complete model
tree, worker binary, execution identity, and token-balanced assignments.
Worker 0000 embeds each unique dense or fused query after conformance while
the exact TEI model is still loaded. It seals the plan copy and packed vectors
before publishing its completion marker. `bundle validate` re-opens the same
chain without contacting a service. `stage` uploads only declared keys and can
resume by skipping an already-present object only when its byte count and
SHA-256 digest are identical.

`pod dry-run` renders the GraphQL create request, provider deletion deadline,
and redacted credential source without reading credentials. `volume
create|status|terminate` and `pod create|status|terminate` are the only control
operations. Creation requires three different no-overwrite paths: `--create-out`
records the scheduler-returned Pod ID and requested deletion deadline before
admission polling, `--launch-out` records the complete REST-admitted Pod, and
`--out` records the final supervised result. The controller tolerates bounded
transient `404` and `CREATED` states, then requires a running Secure Cloud Pod
with the exact image, GPU display/type, volume, data center, private network
surface, and returned hourly price. It writes that returned Pod and machine
identity to an attempt-scoped worker-observation key; if upload fails, it
terminates the new Pod. It polls only the deterministic completion key, stops
at the runtime or total-compute-USD limit, and always requests Pod deletion.
Termination requires the same exact resource ID in both `--id` and
`--confirm-terminate`.

`storage-challenge dry-run|create` is the mounted-volume gate before model
conformance. It takes the sealed executor-image build receipt, the exact
digest-pinned image, a saved network-volume identity, and explicit hourly,
runtime, and total-compute-USD caps. The host first uploads only a small
bootstrap file so the run directory exists. It launches and admits the Pod,
then uploads a fresh challenge; the challenge is never present before the Pod
is running. The worker permanently leaves root through the normal startup
path, waits for that exact path, byte count, and SHA-256 digest, and publishes
the canonical response through the production no-overwrite hard-link path.
The host requests Pod deletion on every supervised outcome, downloads the
response without listing the bucket, and verifies its exact expected bytes.
Three no-overwrite outputs record the scheduler request, the admitted Pod and
returned hourly price, and the final content-bound watchdog and deletion
receipt. Environment-variable names may appear in redacted request previews;
their secret values never enter command arguments or evidence files.

`fetch` never lists the bucket. It requests the deterministic
`attempts/<worker>/completed.json` key for every assignment, validates each
marker against the sealed bundle, and then downloads only the result, receipt,
task-report, and worker-0000 query-vector objects named by those markers.
Finalizer inputs are published under `embeddings/`; the sealed cloud query
vectors are published under `query-vectors/`; completion markers, the locally
built `run-report.json`, and an exact transfer-byte receipt are isolated under
`evidence/`. `verify` re-hashes all three trees, re-opens the query-vector set
against the staged policy and byte-exact input plan, and proves exact task and
worker coverage again.

The initial measurement does not pretend that an unmeasured policy is already
valid. `rag runpod conformance build` seals a strict candidate template and
copies every candidate-declared input. `conformance validate` re-hashes it and
`conformance stage` uploads only those names. `conformance pod-dry-run` and
`conformance pod-create` use the worker's separate conformance mode; each run
uses a distinct run prefix and run ID. `conformance fetch` reads only
`conformance/results/<run-id>.json`, validates it against the candidate, then
downloads the exact normalized-vector object named by that result.

After two different Pods return identical normalized output,
`conformance seal` combines them with a strict policy draft. The draft must
contain every policy/3 field except `conformance`, must use schema name
`livefire.rag.embedding-policy-draft/3`, and rejects missing or unknown fields.
The command accepts each fetched result tree (or its canonical result path),
re-hashes the normalized-vector object declared by each result, requires exact
byte equality across the two Pods, inserts only measured conformance fields,
parses the result through the normal strict policy/3 parser, and proves that all
candidate execution identities still match. That comparison includes TEI's
maximum client batch size, maximum batch-token capacity, maximum concurrent
requests, and the client's request timeout and response-byte limit. The
batch-token capacity must cover Qwen3-Embedding-8B's 40,960-position window;
it is never inferred from the number of inputs in the conformance fixture.
This is the only path from an unmeasured candidate to an executable cloud
policy.

RunPod and TEI references:

- <https://docs.runpod.io/api-reference/overview>
- <https://docs.runpod.io/api-reference/pods/POST/pods>
- <https://docs.runpod.io/storage/network-volumes>
- <https://docs.runpod.io/storage/s3-api>
- <https://huggingface.co/docs/text-embeddings-inference/en/supported_models>
- <https://huggingface.co/docs/text-embeddings-inference/en/cli_arguments>
- <https://huggingface.co/Qwen/Qwen3-Embedding-8B>
