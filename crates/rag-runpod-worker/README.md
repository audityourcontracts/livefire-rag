# RunPod embedding worker

This Rust binary executes exactly one assignment from a sealed RunPod bundle.
It reads and writes only beneath one mounted run directory, starts the pinned
Text Embeddings Inference server on loopback, confirms the exact model response
with the profile's frozen fixture, and publishes a deterministic completion
marker only after every assigned vector, receipt, and report is valid.

The host launches one Pod for one assignment:

```text
/usr/local/bin/rag-runpod-worker run \
  --root /workspace/<run-prefix> \
  --bundle bundle.json \
  --worker-id worker-0000 \
  --attempt-id attempt-0001 \
  --attempt-number 1 \
  --observation runtime/worker-0000/attempts/attempt-0001/observation.json
```

The container starts as root only to take ownership of that exact run
directory. It immediately clears supplementary groups and permanently changes
to user and group 1000 before reading model data or starting the inference
server. Every normal run performs an atomic create, sync, rename, read, and
delete probe after the privilege change. The same check can be run without a
GPU before launch:

```text
/usr/local/bin/rag-runpod-worker storage-probe \
  --root /workspace/<run-prefix> \
  --required-object candidate.json <bytes> <sha256> \
  --required-object model/config.json <bytes> <sha256>
```

Required object paths must be sorted, unique, relative paths. The probe checks
each exact byte count and SHA-256 after dropping privileges. It also exercises the same
hard-link publication operation used for immutable worker receipts, along with
atomic rename publication, and removes its temporary files before returning.

Before an executable embedding policy exists, the host launches two fresh Pods
with the same sealed candidate and different run IDs:

```text
/usr/local/bin/rag-runpod-worker conformance \
  --root /workspace/<run-prefix> \
  --candidate candidate.json \
  --run-id run-0001 \
  --observation runtime/conformance/run-0001/observation.json
```

Candidate-declared object paths are relative to this root. In particular, the
model artifact paths remain their exact upstream paths, so the run root is the
model directory. The normalized vector artifact uses the candidate's declared
key; the sealed result is written to
`conformance/results/<run-id>.json`.

The host writes the observation file after the Pod API returns its Pod and
machine identities. The worker waits for it for up to five minutes by default.
No storage or provider credential is accepted on the command line or written
to an artifact. The child inference process receives only an explicit CUDA and
process-path environment allowlist, so provider credentials are not inherited.

Build the `worker-artifact` target and export `/rag-runpod-worker` when creating
the bundle. The final image copies that same build-stage file, which lets the
worker prove that its running executable is byte-for-byte the binary sealed in
the bundle.

Before launching a paid worker, seal an executor-image build receipt with the
custom image digest, official TEI base digest, `linux/amd64` platform,
Dockerfile object, and exported worker object. Conformance verifies the receipt
and Dockerfile bytes; normal runs verify the same receipt identity from the
sealed embedding policy and bundle.

Create that receipt with the Rust host CLI after exporting the
`worker-artifact` target and obtaining the final image's manifest digest:

```text
rag runpod executor-image seal \
  --executor-image ghcr.io/<owner>/livefire-rag-worker@sha256:<manifest> \
  --executor-version <build-version> \
  --tei-base-image ghcr.io/huggingface/text-embeddings-inference@sha256:39a82da7c30641fbc5f7776f512db0878cf3573e1ea8d60a886391fb4bdcba66 \
  --tei-base-version 1.9.3 \
  --dockerfile crates/rag-runpod-worker/Dockerfile \
  --worker-binary <export-directory>/rag-runpod-worker \
  --out <new-receipt.json>
```

`rag runpod executor-image validate` rehashes both local files and refuses a
receipt whose canonical self-digest, image identities, platform, Dockerfile,
or worker bytes have changed.
