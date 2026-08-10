# Investigation use cases

## Strong fits

1. Translate a natural-language hypothesis into candidate OCSF events before the
   runner knows the exact class, activity, or facet vocabulary.
2. Rank material observations from a large bounded result so the runner can test
   the most promising lead first.
3. Find semantically similar activity across source families whose native terms
   differ but whose normalized actor/action/resource pattern is alike.
4. Find events similar to an exact seed event while excluding the seed and
   applying time, class, status, and source-family filters.
5. Retrieve candidates for both confirming and benign explanations of a
   hypothesis. The runner, not retrieval rank, decides which survives testing.
6. Triage queued and ancillary leads without merging them into the active
   investigative component.

## Separate corpus

Playbooks, OCSF documentation, ATT&CK material, and organizational runbooks can
use the same SDK but must be a separately built `knowledge` index and separately
named tool. Knowledge helps formulate questions; it is not telemetry evidence.

## Bad fits

Use exact OCSF, SQL, baseline, or authority tools for:

- event IDs, IPs, hashes, accounts, resources, time ranges, and exact statuses;
- counts, aggregation, rarity, prevalence, ordering, and pagination;
- exhaustive absence or negative-evidence claims;
- identity continuity, typed relationships, and causal conclusions;
- evidence verification or finding submission.

A high similarity score is a lead, not proof. A zero-result vector search is not
proof of absence unless the declared index coverage and supported query class
make that claim possible—which v1 does not.

## Initial experiment

Index atomic OCSF event documents and compare four methods over identical queries
and typed filters: exact OCSF search, BM25, dense vectors, and hybrid rank fusion.
Promote the provider only if vector or hybrid search improves held-out Recall@20
and paraphrase robustness without returning out-of-filter or unhydratable IDs.

