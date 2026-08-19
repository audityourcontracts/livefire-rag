# RunPod embedding build

Status: full M45 embedding, local verification, and index assembly completed

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
It also publishes immutable phase records as it verifies control objects, the
model tree, the host observation, the GPU, TEI health, conformance, query
vectors, and the first task. If an attempt fails, it writes a sealed failure
record naming the last phase and a bounded public error code. These are the
durable operational logs for a Pod that deliberately has no SSH or public
port; container standard error is not treated as evidence.
One Pod may process a caller-bounded consecutive assignment range. It verifies
and loads the model once, then publishes the normal per-assignment completion
markers in order. A failed Pod never invalidates earlier markers, so recovery
starts at the first unfinished assignment rather than repeating sealed work.

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

SSH is deliberately not used. RunPod's S3-compatible API carries prepared
documents, control objects, and completed embedding artifacts. The first
conformance worker downloads any missing model files directly from the exact
public Hugging Face repository and revision declared by the candidate. It uses
bounded HTTPS redirects, resumable range requests, and immutable publication,
then checks every declared byte count and SHA-256 digest before TEI starts in
offline mode. The second fresh conformance Pod and the full embedding run use
the same network-volume run prefix and reuse that verified model tree.

## Measured M45 result

The completed cloud build embedded all 560,842 searchable M45 documents and
128,329,292 exact tokens. The sealed plan contains 2,191 tasks split across 16
token-balanced assignments. Two fresh RTX 5090 Pods first produced the same
4,096-dimensional normalized conformance bytes. The selected production
setting used document batches of four and one client request in flight.

The final production Pod ran for 12,159.466 seconds and cost USD 3.34385315 at
the provider-returned USD 0.99/hour price. The finalized task reports contain
165,886 model requests, 523 retries, a median request latency of 83.735 ms, and
a 95th-percentile request latency of 110.823 ms. They aggregate 43.224 documents
and 9,890.306 exact tokens per active second. Valid results from guarded pilots
were deliberately reused, so the aggregate preserves their retries and its
reported peak of two requests in flight even though the selected final setting
used one.

The host fetched and verified 9,248,890,270 bytes of declared output. Local
finalization produced 9,188,835,328 bytes of vector payload. The SQLite-v3
index contains 560,842 documents and 6,367,276 event references; independent
inspection passed. A one-entry catalogue over that global index passed 45
frozen lexical, dense, and fused requests using sealed cloud-profile query
vectors and zero new model calls. Stored-document similarity also passed and
made no model call.

The generated ledger at `reports/runpod-m45-full-20260817/ledger.md` records
each pilot, rejected setting, live charge, artifact identity, transfer, local
assembly timing, and cleanup. It is ignored by Git because it is run evidence,
not source.

## Diagnostic escalation

Normal operation should first inspect the immutable worker evidence on the
network volume: stage events show the last completed phase; task reports show
latency, retries, and throughput; assignment markers prove closed ranges; and
failure records provide a bounded error code. These files remain available
even when the host process is quiet or a Pod terminates.

RunPod's web console provides two additional views for a live Pod: container
logs contain application standard output, and system logs contain lifecycle
events such as startup, shutdown, and errors. The production image deliberately
publishes no inference port, SSH daemon, or shell service, so the normal run
cannot be entered interactively.

If the sealed records and console logs cannot explain a failure, launch a
separate diagnostic image under a small time and total-cost cap. That image may
run an SSH daemon and expose only its SSH port. Use the dedicated public key
already registered with RunPod; its Ed25519 private key stays in 1Password and
is offered by the Touch ID-backed 1Password SSH agent. Copy the exact SSH
command from the Pod's Connect panel or `runpodctl ssh info POD_ID`. Do not
export the private key to a repository file, `.env`, image, network volume, or
RunPod secret. Terminate the diagnostic Pod immediately after collecting
redacted logs and hardware/runtime state.

Prepared occurrence shards remain local because embedding needs only prepared
documents. A cloud bundle contains:

- the prepared manifest and every referenced document shard;
- a token-balanced embedding plan and exact token-count object;
- the executable tokenizer, embedding profile, and exact model-object manifest;
- the pinned Rust worker binary and TEI container identity;
- non-overlapping worker assignments and the exact expected output keys.

The bundle still binds every model object when the host uses
`--skip-model-objects`. That option changes only the transport: the worker must
download each missing object from the manifest-bound revision and prove the
same bytes before inference. It does not weaken or replace the model identity.

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
   small control-object list and verify every byte count and SHA-256 digest.
4. Launch the exact digest-pinned executor image with the network volume and
   no model load. After RunPod reports the Pod as running, upload a fresh
   32-byte challenge through S3. The worker must read and hash it as UID/GID
   1000, publish the exact image-and-challenge-bound response with an immutable
   hard link, and the host must download and verify that response. Retain the
   scheduler request, admitted Pod and returned price, runtime and total-cost
   watchdog, deletion outcome, and exact object identities.
5. Launch one fixed GPU type with explicit hourly, runtime, and total-compute
   price caps. On the first Pod, download and verify the exact pinned model
   files into the volume, run only model conformance, and terminate promptly.
   Repeat conformance on a fresh Pod using the verified model tree.
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
terminates the new Pod. It polls the deterministic completion key and the one
attempt-scoped failure key, stops immediately when either appears, otherwise
stops at the runtime or total-compute-USD limit, and always requests Pod
deletion.
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

For the no-SSH transport, use one safe run prefix for conformance and the later
full bundle. Stage the candidate without copying model weights from the host:

```sh
rag runpod conformance stage \
  --candidate CONFORMANCE_CANDIDATE \
  --run-prefix RUN_PREFIX \
  --network-volume-id VOLUME_ID --datacenter-id DATACENTER_ID \
  --skip-model-objects

rag runpod stage \
  --bundle BUNDLE --run-prefix RUN_PREFIX \
  --network-volume-id VOLUME_ID --datacenter-id DATACENTER_ID \
  --skip-model-objects
```

The first command uploads only the candidate and its declared control files.
The conformance worker downloads missing model files from the exact public
revision. The second command later uploads the prepared documents, plan, and
remaining bundle inputs while leaving that verified model tree in place.

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
- <https://docs.runpod.io/pods/manage-pods>
- <https://docs.runpod.io/pods/configuration/use-ssh>
- <https://huggingface.co/docs/text-embeddings-inference/en/supported_models>
- <https://huggingface.co/docs/text-embeddings-inference/en/cli_arguments>
- <https://huggingface.co/Qwen/Qwen3-Embedding-8B>
