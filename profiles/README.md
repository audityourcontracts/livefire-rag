# Local embedding profiles

The local Qwen profile and executable tokenizer are separate on purpose:

- `qwen3-embedding-8b-generic-evidence-lmstudio-q4.dev.json` binds the exact
  GGUF model, LM Studio runtime, vector shape, prompts, and conformance output.
- `qwen3-embedding-8b-gguf-q4-k-m-tokenizer.ref.json` binds the executable
  tokenizer used to count and balance local LM Studio inputs.
- `qwen3-embedding-8b-upstream-tokenizer.ref.json` retains the original
  Hugging Face tokenizer identity for the later Hugging Face/Runpod profile.

Download the tokenizer bytes from the pinned official Qwen revision and verify
them before planning:

```sh
mkdir -p indexes/tokenizers/qwen3-embedding-8b-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
curl -fL \
  https://huggingface.co/Qwen/Qwen3-Embedding-8B/resolve/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json \
  -o indexes/tokenizers/qwen3-embedding-8b-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json
shasum -a 256 indexes/tokenizers/qwen3-embedding-8b-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json
```

The expected SHA-256 is
`83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d`.
The 11 MB tokenizer file is a downloaded model artifact under ignored
`indexes/`; only its small identity record is committed.

The upstream file applies NFC Unicode normalization. The tokenizer embedded in
the pinned GGUF does not, so that upstream file is for the later Hugging Face
backend and must not be used to plan LM Studio work.

For the local GGUF profile, derive the executable tokenizer by removing only
the upstream NFC normalizer and writing canonical compact JSON:

```sh
mkdir -p indexes/tokenizers/qwen3-embedding-8b-gguf-q4-k-m-69d0e58a13e463cd99a9b83e3f5fee7c10265fab
jq -c '.normalizer=null' \
  indexes/tokenizers/qwen3-embedding-8b-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json \
  > indexes/tokenizers/qwen3-embedding-8b-gguf-q4-k-m-69d0e58a13e463cd99a9b83e3f5fee7c10265fab/tokenizer.json
shasum -a 256 indexes/tokenizers/qwen3-embedding-8b-gguf-q4-k-m-69d0e58a13e463cd99a9b83e3f5fee7c10265fab/tokenizer.json
```

The expected derived SHA-256 is
`c939bedf6c07a8d8b0872069748b04194591c77e566d80201e46a07702a5f40c`.
Its tracked reference is
`qwen3-embedding-8b-gguf-q4-k-m-tokenizer.ref.json`.

The reference enables special tokens. The pinned GGUF's llama.cpp tokenizer
adds the final `<|endoftext|>` token for embedding input: for `hello world`,
the plain token IDs are `14990, 1879` and the embedding input is
`14990, 1879, 151643`. The fixed parity fixture covers whitespace, composed
and decomposed Unicode, CJK, Arabic, emoji joiners, security text and the
16,384-byte planning boundary.
