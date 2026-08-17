# Evidence pilot evaluation

Historical Python pilot record. It remains useful for evaluator regression and
does not describe the active Rust M45 build.

This stage evaluates a sealed, explicitly non-admitted pilot index. It never
changes projection, sampling, semantic text, embeddings, or the frozen query
set. Results describe only the selected sample and cannot establish corpus-wide
recall or absence.

## Frozen retrieval run

`fixtures/generic-evidence-pilot-queries.v1.json` was authored before pilot
rankings were inspected. Its exact SHA-256
`3da177d46ffd87c5b284db1983828ce64d8d6c76999cf9729cef6d054706f456` is
pinned by the evaluator. `evaluate-evidence-pilot` rejects any other bytes or a
fixture that does not retain its predeclared, answer-neutral, sample-scoped
status. Before the first call it writes and hashes the complete Cartesian plan:

- every fixture query;
- lexical, dense, and reciprocal-rank fused retrieval;
- one fixed top-N depth for every run; and
- three fixed comparisons per query: dense/lexical, fused/lexical, and
  fused/dense.

The runner uses `EvidenceService`, validates every public output schema, requires
`coverage.status=partial` and `pilot_sample_not_corpus_coverage`, and checks each
returned source pointer against its sealed occurrence row. It writes:

- `query-fixture.json`: exact copied fixture bytes;
- `execution-plan.json`: the pre-execution plan and request bodies;
- `rankings.jsonl`: all returned rankings and scores;
- `comparisons.jsonl`: overlap-at-1/5/10/20, shared ranks, rank deltas, and
  left/right-only document IDs;
- `report.json`: counts, bindings, closure result, and explicit no-qrel status;
- `objects.lock.json` and `manifest.json`: byte and component identities.

Expected relation families never filter, boost, label, or stop a run. They are
copied into each ranking only to show which relation families occur among the
returned pointers. That is a diagnostic, not relevance and not a substitute for
qrels. The negative control is executed exactly like every other query; the
runner does not require it to miss.

## Separate PCA/kNN geometry report

Corpus geometry is an index-only operation with no query fixture input. It is
implemented as a separately sealed artifact whose input dependency graph may
contain only the promoted index, embedding profile, the sealed pilot-selection
metadata, and a fixed
geometry-policy component. It must reject query,
qrel, expected-family, scenario, anchor-ID, or analyst-label inputs.

The fixed policy is:

1. Read every selected searchable `document_id`, its single relation, and L2
   vector in ascending document-ID order; verify vector/document/selection
   closure and finite unit norms.
2. Center vectors with a float64 mean. Compute a deterministic randomized PCA
   with a seed derived from the sealed index, embedding object, pilot selection,
   geometry policy, and caller seed. Resolve component sign by making the
   largest-absolute loading positive. PCA is visualization-only; emit PC1/PC2
   and PC1/PC3 plots.
3. Compute exact cosine neighbors in the original L2 embedding space in bounded
   blocks, excluding self and breaking equal distances by document ID. Use fixed
   same-relation k values 10, 25, and 50, reducing k only when a relation is
   smaller.
4. For each document report its neighbor IDs and distances, mean and kth
   distance, reciprocal-neighbor rate, global and cross-relation nearest
   neighbors, and a median/MAD robust within-relation isolation score. Zero-MAD
   relations receive an unavailable score rather than an invented value.
5. Aggregate by relation and report cross-relation nearest-neighbor confusion.
   Compare document, occurrence, inverse-inclusion, and combined
   occurrence/inclusion-weighted means. Weights never fit PCA, choose neighbors,
   or change a per-document isolation score.

Required artifacts are `geometry-policy.json`, `coordinates.parquet`,
`neighbors.parquet`, `relation-summary.json`, `pca-pc1-pc2.png`,
`pca-pc1-pc3.png`, `report.json`, `objects.lock.json`, and `manifest.json`. The
report must say isolation is geometric—not maliciousness or retrieval
quality—bind the exact embeddings and selection objects, record counts, and remain
`local_evaluation_only_not_sdk_admitted`. Rebuilding from identical index and
policy bytes and the same caller seed must produce identical row order,
coordinates within a frozen numeric tolerance, neighbor identities, and report
digest material.

PCA separation and original-space neighbor isolation help explain corpus/model
geometry. They do not prove anomaly, maliciousness, retrieval relevance, or
Livefire evidence. Those claims require independently authored labels/qrels and
separate evaluation.
