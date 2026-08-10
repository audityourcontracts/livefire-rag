# Standalone evaluation plan

Evaluation runs through the CLI and provider protocol without the Livefire
runner. Expected findings are stored only in the evaluation harness, never in
projection templates or index metadata.

## Correctness gates

- Every pointer exists in the bound snapshot and hydrates successfully.
- Time, class, activity, status, and source-family filters have zero violations.
- Snapshot, policy, model, and artifact digest mismatches fail before search.
- Empty, truncated, partial, and unavailable coverage are reported honestly.
- Repeated requests have stable ranking, tie-breaking, cursor, and result digest.
- Forbidden raw/native sentinel fields never appear in documents, excerpts,
  diagnostics, or errors.

## Quality measures

- Recall@1/5/10/20, MRR, and nDCG@k.
- Worst-paraphrase Recall@k and rank variance.
- Relevant-versus-benign score margin and false-candidate rate.
- Recall broken down by source family and investigation scenario.
- Exact-search, BM25, dense-vector, and hybrid ablations on the same projections.

## Operational measures

- Build duration, documents/second, output size, and peak memory.
- Query p50/p95 latency, queries/second, provider startup, and resident memory.
- Approximate-index recall against the exact vector scorer.

## Anti-overfit suite

Hold out whole scenarios. Rename entities, shift timestamps, add benign decoys,
change background volume, paraphrase queries, and delete relevant records. Include
public-storage exposure, credential misuse, a compound cloud incident, benign
near-misses, and relevant records beyond the first result page.

