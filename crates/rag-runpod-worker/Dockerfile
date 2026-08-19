# syntax=docker/dockerfile:1.7
# The multi-platform Rust index is pinned so an ARM64 development machine can
# build the same Linux/AMD64 worker without executing rustc through QEMU.
FROM --platform=$BUILDPLATFORM docker.io/library/rust:1.97.0-bookworm@sha256:8fa55b2f3ddf97471ab6a767bfa3f37e6bad0986ba823e75fea57e2a2a5c3073 AS build
WORKDIR /source
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc-x86-64-linux-gnu libc6-dev-amd64-cross \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add x86_64-unknown-linux-gnu
COPY . .
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc
RUN cargo build --locked --release --target x86_64-unknown-linux-gnu -p rag-runpod-worker

FROM scratch AS worker-artifact
COPY --from=build /source/target/x86_64-unknown-linux-gnu/release/rag-runpod-worker /rag-runpod-worker

FROM --platform=linux/amd64 ghcr.io/huggingface/text-embeddings-inference:120-1.9@sha256:144aaa80ddcb520d49df83f915dc188ddd7cc6b1b3b9684a829c21dd39cbe3c5
# The CUDA-linked TEI router cannot execute on a CPU-only build host, even for
# `--help`. The Rust worker fixes `--auto-truncate=false`; the fresh-GPU
# conformance run proves that exact argument against this digest-pinned base.
LABEL org.opencontainers.image.title="Livefire RAG RunPod worker" \
      org.opencontainers.image.base.name="ghcr.io/huggingface/text-embeddings-inference:120-1.9" \
      org.opencontainers.image.base.digest="sha256:144aaa80ddcb520d49df83f915dc188ddd7cc6b1b3b9684a829c21dd39cbe3c5"
COPY --from=build /source/target/x86_64-unknown-linux-gnu/release/rag-runpod-worker /usr/local/bin/rag-runpod-worker
# The worker starts as root only long enough to take ownership of its exact
# /workspace/<run-prefix> directory. It then permanently drops to 1000:1000
# before reading model data, starting TEI, or writing any result.
USER 0:0
ENTRYPOINT ["/usr/local/bin/rag-runpod-worker"]
