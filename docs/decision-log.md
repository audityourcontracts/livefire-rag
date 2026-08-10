# Decision log

## Accepted for the first slice

- Two repositories only: `livefire-sdk` and `livefire-rag`.
- Embedder, projector, index builder, and provider remain in `livefire-rag` until
  a second independent consumer justifies `livefire-embeddings`.
- JSONL/stdio provider boundary; no compile-time dependency on Livefire.
- Retrieval only: `rag.search` and `rag.more_like_event`; no hidden generation.
- Natural-language query plus closed typed filters; no arbitrary backend query.
- Candidate event pointers are not evidence and must hydrate through exact OCSF.
- No pagination cursor in v1; bounded top-k and stable tie-breaking only.
- A deterministic fake embedder and exact scorer are the conformance oracle.
- Telemetry, hunt knowledge, and prior-case memory are separate governed indexes.

## Experiment-gated decisions

### Projection granularity

Start fixtures with atomic event documents because pointer integrity and relevance
judgements are simple. Before the full corpus, compare them with documents grouped
by exact source `support_ref`. Select using duplicate rate, token distribution,
retrieval quality, index size, and hydration accuracy. Either choice is a new
projection-policy identity.

### Retrieval backend

Compare exact typed OCSF, BM25, dense vectors, and deterministic hybrid rank
fusion over identical documents and filters. Add an approximate vector index only
after measuring its recall against the exact vector scorer.

### Embedding model

Do not bake a mutable model name into the protocol. Benchmark local candidates,
then pin the chosen weights, tokenizer, runtime, target, dimensions, and digests.
CI uses a clearly non-production deterministic test embedder.

### Generation

Generation remains outside this plugin until a separate use case demonstrates
value. Any future generator is a separate tool and traced model effect; it cannot
upgrade pointers into evidence.

