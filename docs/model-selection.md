# Local embedding model selection

Local experiment record. It documents the LM Studio profile and the upstream
model choice, but cloud execution identity and measured admission are governed
by [`runpod-embedding.md`](runpod-embedding.md).

## Quality-first candidate

The v1 quality-first candidate is `Qwen/Qwen3-Embedding-8B` from Hugging Face under
Apache-2.0. It was selected as the quality-first candidate because its model card
reports the strongest Qwen3 embedding result, supports text and code retrieval,
task instructions, 100+ natural/programming languages, a 32K context, and
Matryoshka output dimensions up to 4,096.

The Qwen quality-reference revision selected for the first experiment is
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. It is never resolved from mutable
`main` during a build.

The development machine is an Apple M5 Max with 128 GB unified memory. The
official BF16 model is therefore a viable local quality reference. A smaller
model is not selected merely to satisfy memory constraints.

## Runtime profiles

Two local profiles are evaluated:

1. **Reference:** official Hugging Face BF16 safetensors and tokenizer at an
   exact commit, Sentence Transformers/Transformers at pinned versions, PyTorch
   MPS, deterministic batching policy.
2. **LM Studio:** an exact locally stored embedding artifact served through
   LM Studio's loopback `/v1/embeddings` interface. The model file digest,
   conversion source, quantization, LM Studio/engine version, load settings, and
   output dimension are pinned.

The string sent in the API's `model` field is not sufficient identity. The
builder performs a conformance challenge and records the digest of normalized
outputs for fixed test texts before admitting a build.

## Prompts

Documents use the model's document form without a query instruction. Search
queries use this versioned instruction:

```text
Retrieve security command-line activity relevant to this investigation query.
Preserve executable or API action, target, arguments, process context, and intent.
```

Anomaly component projections have separate versioned document templates for
action, target, structure, and obfuscation. Prompt or template changes require a
new index.

## Required local bake-off

Evaluate:

- Qwen3-Embedding-8B at 1,024, 2,048, and 4,096 dimensions;
- official BF16 versus the best LM Studio-compatible F16/Q8 artifact available;
- Qwen3-Embedding-4B and Qwen3-Embedding-0.6B as latency/cost ablations;
- EmbeddingGemma 300M as a small code-retrieval baseline;
- BAAI/BGE-Code-v1 as an Apache-2.0 code/command retrieval challenger;
- Perplexity `pplx-embed-v1-0.6b` or 4B as a current MIT-licensed challenger;
- BM25 and exact token/feature baselines with no embedding model.

Use held-out security queries and labelled command pairs covering benign admin,
encoded PowerShell, living-off-the-land commands, cloud CLI/API activity,
sensitive-resource access, process-tree anomalies, and close benign decoys.

Primary quality gates:

- highest held-out Recall@20 and worst-paraphrase Recall@20;
- no regression in principal/population outlier ranking nDCG@20;
- 100% filter and pointer correctness.

Operational measures are build commands/second, query p50/p95, peak unified
memory, vector bytes, exact-scan latency, and cold model-load time. Select 8B BF16
unless a smaller dimension or quantized/local profile stays within one percentage
point of its domain Recall@20 and materially improves build/query cost.

NVIDIA `llama-embed-nemotron-8b` ranks above Qwen3 by Borda votes on the model
card's October 2025 multilingual MTEB table, but its published weights are
research/non-commercial. It may be measured in research evaluation but is not a
deployable default. Jina v5 is similarly excluded from the default because its
weights are CC BY-NC. General leaderboards are shortlist evidence, not a proxy
for security command retrieval.

The selected result becomes an immutable embedding-policy component. A later
fine-tuned security model is a new component and requires a new snapshot; it does
not overwrite the v1 model.

## Current local development profile

The development LM Studio instance currently exposes:

```text
model key:      text-embedding-qwen3-embedding-8b
source repo:    Qwen/Qwen3-Embedding-8B-GGUF
repo revision:  69d0e58a13e463cd99a9b83e3f5fee7c10265fab
artifact:       Qwen3-Embedding-8B-Q4_K_M.gguf
artifact sha256:3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85
artifact-set lock:9e5f2156b767ea1d403e5f1c217455d48095095c1603bf9e79feab19aac9561f
quantization:   Q4_K_M
loaded context: 8192
dimensions:     4096
```

Two repeated synthetic conformance batches returned byte-identical L2-normalized
JSON vectors with digest
`f9f0200562118a137fa352ecde5a786490fe56a94a1f820909b461536ab98518`.
The fixture covers one document input and one exactly composed instructed query;
the server output is already L2-normalized, so the client performs no
renormalization.
This proves repeatability for that probe, not equivalence with BF16. The Q4 model
must still pass the domain-quality comparison before becoming an admitted
production profile.

The exact development policy is
`profiles/qwen3-embedding-8b-lmstudio-q4.dev.json`; its fixed request corpus and
normalization procedure are in `fixtures/embedding-conformance.v1.json`. It is a
development profile, not an admission receipt: the production profile must also
pin the underlying inference engine/load configuration (not only the LM Studio
application executable) and pass the security-domain bake-off.
