# Standalone provider POC acceptance

This is a small, honest acceptance layer for the standalone RAG provider. It
freezes all nine analyst queries and both command-neighborhood diagnostics from
the first real OpenBOTS/OCSF prototype. A run must report every case; selecting
only favorable examples is a failure.

The machine-readable contract is
`fixtures/provider-poc/acceptance-suite.v1.json`. It records each frozen query,
qualitative rank bounds, mandatory observed hard negatives, and investigation
boundaries. Matchers use case-insensitive literal substrings over provider
candidate `preview` and `command_id`. They are smoke-test predicates, not qrels
or a model-quality benchmark. The frozen prototype requested ten results per
case; changing that denominator requires a new suite version.

## Required outcomes

| Cases | Required demonstration |
|---|---|
| Q1-Q5 | Recover the declared command/script behaviors at the frozen rank bounds. |
| Q4 | Preserve and label related logging-bypass activity as a hard negative for the scan/firewall question. |
| Q6 | Retrieve failed `RunInstances` candidates, then require exact structured expansion for totals and regional distribution. |
| Q7 | Rank denied create operations first while retaining the successful IAM read as a wrong-polarity hard negative. |
| Q8 | Retrieve both ACL states, then hydrate and order source records; rank order is not chronology. |
| Q9 | Reproduce and explicitly report the flat dense query's facet-collapse: ACL candidates are present and archive-upload candidates are absent. This is an expected boundary failure, not successful retrieval. The investigation must decompose, hydrate, and correlate two searches. |
| S1/S2 | Report neighborhood behavior diagnostically. Similarity is not equivalence, evidence, or a verdict. |

Q1-Q9 are the non-cherry-picked primary denominator. S1/S2 are mandatory
diagnostics but do not enlarge that denominator. The checker fails on a missing
case, an unknown case, duplicate/non-contiguous ranks, a missed expected
behavior, an absent mandatory hard negative, or a Q9 result that is presented
as though the frozen boundary failure occurred when both facets were returned.

## Provider result envelope

Capture each provider response without rewriting it and add only its case ID:

```json
{
  "schema_version": "livefire.rag.provider-poc-results/1",
  "run_id": "local-run-id",
  "calls": [
    {"case_id": "Q1", "response": {"tool": "cli.search", "rankings": []}}
  ]
}
```

`response` may be the provider's `livefire.rag.semantic-result/1` object, whose
ranked candidates are in `pointers`, or an unmodified JSONL protocol response,
where that object is nested under `result.output`. The checker also accepts the
prototype's legacy `semantic_search`/`similar_command` ranking envelope so old
artifacts remain replayable; it otherwise leaves responses untouched.

Run the dependency-free check and produce auditable JSON and Markdown:

```sh
python3 tools/check_provider_poc.py \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --results path/to/provider-results.json \
  --out reports/provider-poc/acceptance.json \
  --markdown reports/provider-poc/acceptance.md
```

The checked-in synthetic file exercises the harness only:

```sh
python3 tools/check_provider_poc.py \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --results fixtures/provider-poc/synthetic-provider-results.pass.json \
  --out /tmp/provider-poc-acceptance.json
```

Passing this smoke suite means the provider can reproduce the declared useful
and failure behaviors. It does not prove that RAG improves investigation
quality. That claim still requires a sealed occurrence-preserving candidate
universe, blinded qrels, same-universe exact/BM25 baselines, pointer hydration,
and the metrics in `docs/fact-evidence-benchmark.md`.

## Same-corpus lexical comparison

`tools/run_lexical_provider_poc.py` provides a deterministic dependency-free
baseline over the exact sealed `documents.jsonl` object used by the dense
provider. Its frozen scorer is standard single-field BM25 with `k1=1.2` and
`b=0.75`. The only field is `semantic_text`, exactly the document text embedded
by the dense system; `preview` is excluded so principal, host, and display
metadata do not confound the representation comparison. Tokenization
splits lower-camel boundaries, extracts ASCII alphanumeric/underscore terms,
lowercases them, and performs no stemming or stop-word removal. Results sort by
score descending and then `command_id` ascending. Each result retains the
original candidate source pointer.

Run it without changing the suite queries or predicates:

```sh
python3 tools/run_lexical_provider_poc.py \
  --suite fixtures/provider-poc/acceptance-suite.v1.json \
  --index indexes/prototype-m21-poc \
  --dense-results reports/provider-poc/provider-results.json \
  --out-results reports/provider-poc/lexical-results.json \
  --out-report reports/provider-poc/effectiveness-comparison.json \
  --markdown reports/provider-poc/effectiveness-comparison.md
```

The comparison runs the same checker in the fixed Q1-Q9 acceptance scope. It
reports positive behavior checks separately from hard-negative exposure and
the Q9 boundary diagnostic. Q4/Q7 hard negatives are mandatory in the smoke
checker only because it reproduces known dense behavior; in an effectiveness
comparison their appearance is a weakness and lower exposure is better.

The frozen real-data run on the 3,806-document prototype produced 15/15 dense
positive behavior checks and 5/15 BM25 checks. Dense completed Q1, Q3, Q4, and
Q6 where BM25 returned no declared positive behavior, and completed all checks
for Q2, Q5, and Q8 where BM25 was partial. Both systems recovered Q7's two
positive behaviors, while BM25 avoided the two declared dense hard-negative
exposures. Q9 reproduced the expected single-query facet-collapse boundary.

This is qualitative POC evidence: there are no blinded qrels, so checker passes
are not nDCG, Recall, precision, or evidence of general model superiority.
