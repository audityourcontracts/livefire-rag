# Semantic retrieval architecture

## First vertical slice

```text
immutable OCSF snapshot
  -> deterministic ocsf_event_document.v1 projection
  -> pinned embedding policy
  -> document table + vectors + searchable index + manifest
  -> rag.search / rag.more_like_event
  -> answer/pointer/miss envelope (normally pointer)
  -> exact OCSF hydration and evidence verification by Livefire
```

The builder and provider live together initially because the projection,
embedding model, distance function, normalization, and index format form one
compatibility unit. They remain separate executables and SDK interfaces, so a
deployment may build offline and run a read-only provider elsewhere.

## Artifact layout

```text
index/
  manifest.json
  documents.jsonl
  vectors.bin
  search-index/
```

`manifest.json` binds the source snapshot and mapping, builder artifact,
projection policy, embedding model and revision, chunking policy, vector
dimensions, distance metric, backend format, document counts, and digests of all
files. The provider refuses a mismatched or corrupt artifact before serving.

## Document projection

The conformance slice emits one document per normalized event envelope. Its text is a
canonical rendering of event class, activity, status, time, typed operation and
material facets, actor/resource stable identities and display values,
observables, source family, and exact event-bound relationships.

Before a full-corpus build, benchmark grouping events that share an exact
`support_ref` into one token-bounded document. That can avoid duplicate embeddings
when one source record normalizes into several events, but it must preserve all
event IDs and split only at event boundaries. The measured result, not an
assumption, selects the production projection policy.

The projection excludes raw/native payloads, expected findings, evaluator labels,
model conclusions, credentials, and arbitrary source fields. Event IDs and
support references are stored as metadata rather than treated as semantic text.

Later projections may add exact entity activity windows and exact session
episodes. Those must be built only from stable typed identities/relationships;
semantic similarity may not invent continuity.

## Search

`rag.search` accepts natural-language semantic content plus closed typed filters.
`rag.more_like_event` accepts one exact indexed event ID. Both return ranked event
pointers whose `source_ref` is an exact event ID, plus projection digests, score,
rank, non-definitive coverage, and index identity.

V1 intentionally has no cursor: pagination over approximate rankings is difficult
to define honestly. Calls return a bounded top-k from one immutable index, with
stable tie-breaking.

The initial reference backend performs exact cosine search for deterministic
conformance. A production approximate-nearest-neighbour backend is allowed only
after recall is measured against the exact scorer. BM25 and hybrid rank fusion
are first-class baselines; vector retrieval must demonstrate value over them.

## Generation

There is no `rag.answer` in v1. If generation is later added, it must be a
separate declared model effect whose prompt, inputs, outputs, model identity, and
citations are retained in the trace. Generated prose cannot replace exact OCSF
hydration or verifier-bound evidence.

## Deployment boundary

The index build is an offline job with read access to the source snapshot and
write access only to its output directory. The runtime provider is read-only,
loads one pinned index, and needs no network or model credentials for local
embeddings. Remote embedding services are a separate policy profile with explicit
network and secret effects.
