# Fact-to-evidence retrieval benchmark

Status: proposed hidden-test contract.

This benchmark asks one primary question:

> Given an answer-neutral investigation query, does the standalone RAG provider
> retrieve the authoritative source records that support the corresponding
> evaluator fact?

It evaluates evidence discovery, not whether a language model already knows the
answer and not whether a tool can compute a final count, set, ordering, join, or
incident conclusion. It runs outside the Livefire runner and keeps evaluator
facts and qrels out of the index builder and runtime provider.

## Declared fact inventory and unit of analysis

The sealed evaluator inventory contains 76 fact atoms:

| Cohort | Declared atoms | Notes |
|---|---:|---|
| Cloud | 23 | Facts whose support is expected in cloud/API activity or its authoritative source snapshot. |
| BOTS | 53 | Facts associated with the BOTS investigation corpus. |
| **Total** | **76** | Every atom must have exactly one terminal eligibility disposition. |

Ten of the 53 BOTS atoms are external-enrichment atoms. They require an
authoritative source outside the native BOTS/OCSF telemetry. The suite manifest
must assert and reconcile this membership before sealing:

```text
23 cloud + 43 BOTS-native + 10 BOTS-external-enrichment = 76
```

If source reconciliation shows that the ten atoms are not members of the 53,
the suite must not run under this version; the inventory and denominators must be
versioned instead of silently changing the total.

A fact atom is the scoring unit. Facts that resolve to the same source event or
incident chain remain distinct questions but share one `resampling_cluster_id`,
so bootstrap samples preserve their dependence. Duplicate source records for a
single query share one qrel `evidence_group_id` and receive retrieval credit only
once. Paraphrases of one atom do not create extra statistical weight.

## Eligibility ledger and denominators

Eligibility is determined from source truth and reviewed fact meaning before any
candidate ranking is inspected. Each atom receives exactly one disposition:

- `eligible_native`: at least one supporting record is expected in the bound
  cloud or BOTS-native source snapshot;
- `eligible_external`: at least one supporting record is expected in a separately
  bound and admitted external-enrichment snapshot;
- `external_source_unbound`: one of the ten external-enrichment atoms, but no
  eligible enrichment snapshot is part of this run;
- `outside_index_domain`: authoritative record-level support exists, but only in
  a source family the bound command/script/cloud-action index does not declare;
- `not_retrieval_testable`: the fact has no record-level support and can only be
  established by an aggregate, absence proof, or computation;
- `unresolvable_fact`: reviewers cannot map the atom to authoritative support;
  this prevents suite sealing rather than becoming a convenient exclusion.

An atom is eligible when the sealed **source snapshot contains supporting
evidence inside the index's declared command, script-block, or cloud-action
domain**. Facts whose only support is network, inventory, email, endpoint-
protection, or other deliberately unindexed telemetry are
`outside_index_domain`; they remain visible in coverage but do not become
artificial zeroes against a command RAG. Once a fact is admitted as in-domain,
an adapter or index omission cannot remove it from the denominator: it receives
zero retrieval credit and creates a source-fidelity violation.

Every report publishes this reconciliation separately for cloud, BOTS-native,
and BOTS-external-enrichment:

```text
declared
  = eligible_native
  + eligible_external
  + external_source_unbound
  + outside_index_domain
  + not_retrieval_testable
  + unresolvable_fact
```

The primary retrieval denominator is the set of index-domain
`eligible_native + eligible_external` fact atoms bound to the run. The 66
non-external atoms are an inventory to classify, not a claim that all 66 belong
in a command-index retrieval denominator. The ten enrichment atoms are reported
separately and enter retrieval scoring only when an admitted external snapshot
and provider are explicitly in scope. A run must never report its score as if
all 76 had been searched. Model comparisons use the identical frozen eligible
set.

The answer-free coverage-plan fixture has a preliminary, orthogonal
`preliminary_eligibility` axis: `direct_single`, `direct_multi`,
`retrieve_then_compute`, `exact_metadata`, `external_enrichment`,
`outside_current_index`, or `needs_review`. This records *how* the current index
could assist before adjudication. Sealing translates that planning class into
the normative source/index eligibility disposition; it does not let the model
choose its denominator after rankings are visible.

The current answer-free planning review is closed at 34 `eligible_native`
atoms, 10 `external_source_unbound` atoms, and 32 `outside_index_domain`
atoms. The tracked real-suite eligibility ledger is the normative denominator
input; the preliminary plan is retained as review provenance. The companion
query-authoring worklist requests 99 surfaces for the 33 eligible atoms but
contains no query text, answer values, source pointers, qrels, or rankings.
Those surfaces must be authored by a reviewer who has not seen the answer
material, then pass a literal-leakage audit before they are compiled into active
query rows.

Also report:

- `N_declared`, `N_resampling_clusters`, `N_eligible`, `N_queryable`, and
  `N_scored`;
- mapping, query-construction, execution, and qrel coverage rates; and
- every excluded or failed atom by ID and reason.

A runtime error is not an eligibility exclusion. It scores zero and is reported
as an execution failure.

## Query and qrel construction

Each eligible fact has one intent group with at least three independently
reviewed query surfaces:

1. natural analyst language;
2. terse SOC language; and
3. a vocabulary-changing, entity-light paraphrase.

The query author sees the pre-answer case framing or evaluator question, but not
the fact answer, qrels, source pointers, or retrieved candidates. Qrel reviewers
see the fact and source records only after the query surfaces are locked. Queries
describe the investigative need without stating the evaluator answer.
They must not copy answer-only literals such as an event ID, exact timestamp,
principal, host, resource identifier, hash, command fragment, or conclusion from
the fact. A literal that an analyst is explicitly assumed to know may be used
only in a separately tagged `seeded_query` diagnostic; seeded-query results do
not enter the primary answer-neutral score.

Qrels point to immutable source records or semantic evidence groups and use:

- `3`: directly supports the fact;
- `2`: substantial supporting evidence;
- `1`: useful context but not sufficient support; and
- `0`: irrelevant, contradictory, or a reviewed hard negative.

Grades 2 and 3 are relevant for Recall. Exact semantic duplicates collapse to
one evidence group for metrics, while all event-level source pointers remain
available for audit and hydration. Two reviewers grade independently, adjudicate
relevance disagreements and grade gaps greater than one, and seal reviewer
agreement and the adjudication receipt. Queries, qrels, and hard negatives are
content-bound objects in the suite manifest.

Every candidate appearing in a prespecified system's top 20 must receive a grade
before metrics are finalized. An unjudged candidate is not silently treated as
irrelevant; it returns the suite to blinded adjudication and produces a new
qrel-object digest before metrics are calculated or released.

Each active query also binds `expected_top_k_cardinality = min(20, eligible
corpus cardinality after its sealed filters)`, computed independently of the
retriever. A ranking file must contain exactly that many contiguous rows. This
prevents a system from returning only an easy positive and suppressing difficult
or still-unjudged candidates.

## Primary metric and gates

### Primary: macro nDCG@20

For query surface `s`, use the standard graded gain and logarithmic discount:

```text
DCG@20(s) = sum from rank r=1..20 of (2^grade_r - 1) / log2(r + 1)
nDCG@20(s) = DCG@20(s) / IDCG@20(s)
```

Average surfaces within each fact atom, then average eligible fact atoms:

```text
atom_nDCG@20 = mean_surface nDCG@20
macro_nDCG@20 = mean_eligible_atom atom_nDCG@20
```

This prevents an atom with more paraphrases from receiving more weight. Publish
the overall eligible macro score and cloud, BOTS-native, and, when bound,
external-enrichment cohort scores. Also publish worst-surface nDCG@20 per atom.

### Required promotion gates

The development pilot freezes exact numerical gates before validation and hidden
test are opened. The initial proposed floors, consistent with the broader test
program, are:

- macro nDCG@20 is at least `0.70`;
- macro Recall@20 for grade >= 2 is at least `0.85`;
- evidence-group cohort coverage at 20 is at least `0.90` overall and at least
  `0.80` in each eligible cohort;
- every active query surface has at least one adjudicated matched hard-negative
  pair (`1.0` declaration coverage);
- hard-negative triplet accuracy is at least `0.90`; and
- the cluster-bootstrap 95% lower confidence bound for the median hard-negative
  triplet margin is greater than zero.

`Recall@20` measures the fraction of all relevant evidence groups retrieved for
an atom. `CohortCoverage@20` measures the fraction of eligible atoms
for which every required query surface retrieves at least one grade-2-or-3
evidence group in the top 20. A system can therefore have good coverage while
missing important corroborating records; both are required.

For a dense system with cosine distance, each reviewed triplet has query `q`, a
supporting evidence group `p`, and a matched hard negative `n`:

```text
triplet_margin = cosine_distance(q, n) - cosine_distance(q, p)
triplet_win = triplet_margin > 0
```

Ties are failures. For sparse or reranked systems, use the system's native score
with direction normalized so positive means the support outranks the negative.
Raw margin magnitudes are comparable only within the same scoring family;
triplet win rate and paired rank deltas are used across families.

Macro nDCG@20 remains the selection objective. Recall, coverage, and hard-negative
results are gates so a model cannot win by ranking one easy positive highly while
missing cohorts, corroboration, or close decoys. MRR@20, MAP@20, Precision@5,
Recall@1/5/10, and worst-paraphrase Recall@20 are secondary diagnostics.

## Matched hard negatives

Hard negatives are reviewed non-supporting records, not presumed-benign events.
They should resemble the positive on as many non-answer facets as possible while
contradicting one required facet. Match, as applicable, on:

- source and OCSF class/activity;
- executable, interpreter, or API action family;
- target/resource kind and command length;
- shell, decode state, AST availability, and ancestry availability;
- success/denial status, unless polarity is the tested facet;
- principal-history, host-activity, and event-time buckets; and
- frequency/duplicate-count bucket.

Required negative families include same action/wrong target, same target/wrong
action, benign read versus sensitive change, denied versus successful action,
same encoded wrapper/different decoded payload, quoted or commented text instead
of execution, reversed/unrelated ancestry, cloud list/get versus mutation, and
path/resource substring collisions.

Matching uses source fields and reviewed semantics, never evaluator labels in a
model feature and never the candidate system's distance. Fixed match tiers,
calipers, exclusion windows, and deterministic tie-breaking are sealed before
the hidden rankings are produced. A missing match is reported explicitly;
criteria are not relaxed after observing a result. Candidate pooling may combine
outputs from all prespecified systems for reviewer coverage, but the pooled
candidates and qrels are sealed before system comparison.

## Leakage controls

The fact bundle is a hidden overlay. Before it is revealed, freeze and digest:

- source and command snapshots;
- every document projection, redaction and static-decoding policy;
- exact model revision, weight objects, tokenizer, runtime, quantization,
  dimension, prompt, normalization, and batching policy;
- deduplication, filtering, distance, tie-breaking, and top-N behavior;
- the model-by-projection matrix, metric implementation, gates, matching policy,
  statistical plan, and report templates; and
- development/validation/hidden-test partitions.

The builder must not read evaluator prose, qrels, answer keys, scenario labels,
fact-derived tags, or query text, and none may be injected into document text,
metadata, prompts, model training pairs, thresholds, filenames, or index objects.
Authentic source values are not leakage merely because they support or equal an
evaluator answer; their provenance must resolve to the independently frozen
source snapshot and projection policy. Audit provenance paths for evaluator-
derived names and features. Secret redaction happens before model input
construction.

Do not use the 76 atoms to tune a model, projection, instruction, dimension,
decoder, score, hard-negative threshold, or stopping rule. Development and
candidate selection use the separate authored intent suite described in
`docs/test-program.md`. Before opening this benchmark, freeze exactly one
candidate, one reference, and their promotion policy. Opening the hidden 76-atom
evaluation is a one-way recorded operation; it may accept or reject that locked
candidate but must not select among a matrix. Any post-open change creates a new
benchmark version and leaves the original result intact.

Hold all surfaces and evidence from one atom, duplicate group, incident chain,
principal, and host in one split. Apply identifier aliasing and timestamp shifts
consistently across query, source metadata, and qrels. A raw-versus-typed
identifier ablation detects retrieval driven by unique strings rather than
security semantics.

## Development model-by-projection experiment matrix

Run every admitted cell against the same development corpus, intent queries,
qrels, exact top-20 depth, and source filters from `docs/test-program.md`. Do not
run this selection matrix against the hidden 76-atom overlay. At minimum,
compare:

| System/model | Required identity |
|---|---|
| Exact typed-field/token and BM25 | implementation, tokenizer, fields, boosts, and tie-breaking |
| Qwen3-Embedding-8B BF16 | exact Hugging Face revision and runtime |
| Qwen3-Embedding-8B Q4_K_M | exact GGUF digest, engine, and load profile |
| Qwen3-Embedding-4B and 0.6B | exact revision, dimension, and runtime |
| EmbeddingGemma 300M | exact revision and runtime |
| BAAI/BGE-Code-v1 | exact revision and runtime |

Use one-factor cumulative projections:

| Projection | Content |
|---|---|
| P0 `raw-redacted` | Original command/script/API text after mandatory secret redaction. |
| P1 `canonical` | P0 plus typed action, target, shell/interpreter, status, and OCSF semantics. |
| P2 `canonical-decoded` | P1 plus bounded, static decoded content; no execution. |
| P3 `canonical-structure` | P2 plus versioned AST/argument structure. |
| P4 `canonical-context` | P3 plus bounded ancestry or explicitly declared relational context. |
| P5 `typed-identifiers` | P4 with unstable identifiers replaced by typed placeholders. |

P5 is the privacy-preserving semantic candidate; the P4-to-P5 delta tests
identifier dependence. If source policy forbids identifier-preserving P4, run it
only on an approved synthetic/aliased corpus and mark the production cell
`not_permitted`, not missing.

For each development cell, record exact document and query projection digests.
Compare models within projection and projections within model with paired
statistics. Freeze the winning candidate and one reference before the hidden
fact benchmark is opened. Evaluate
1,024/2,048/4,096 dimensions and BF16/Q4 as separately named cells rather than
changing two factors at once. Dense+BM25 fusion, ANN, LLM reranking, or fine-tuned
security embeddings are later, separately versioned matrices; they must not be
introduced while selecting the base dense cell.

## Statistics and uncertainty

Use 10,000 bootstrap resamples over sealed resampling clusters. Keep an
atom's surfaces, duplicate records, triplets, and incident-chain members in the
same resample. Publish percentile 95% confidence intervals for macro nDCG@20,
Recall@20, cohort coverage, triplet accuracy, and median margin.

System comparisons are paired on the same atoms. Report the candidate-minus-
reference mean/median delta, its bootstrap interval, the number of improved/tied/
regressed atoms, and a 10,000-draw paired permutation test. Apply Holm correction
to the prespecified family of model/projection comparisons. Do not promote based
on a p-value alone; the candidate must meet absolute gates and any declared
non-inferiority margin.

The initial policy requires both a paired macro-nDCG lower confidence bound no
worse than `-0.01` and a non-negative point delta. Recall@20 and hard-negative
triplet accuracy may regress by at most `0.01`, while cohort coverage may not
regress. These are versioned selection-policy choices, not universal literature
thresholds.

`tools/evaluate_fact_evidence.py` is the deterministic local metric and gate
calculator. It validates inventory reconciliation, complete top-20 judgments,
score-family direction, pointer/filter results, and the paired cluster bootstrap.
A formal promotion receipt additionally requires the sealed run manifest,
leakage and control audits, repeatability results, the predeclared paired
permutation tests, and Holm correction defined by
`evidence-benchmark-run.v1`.

Report cohort and diagnostic slices only when their denominator is shown:
cloud/BOTS/external, command/API/script, PowerShell/POSIX/Windows, common/rare,
decoded/not-decoded, single-record/multi-record, and answerable/aggregate-backed.
Small slices are descriptive and are not used for unplanned threshold tuning.

## Retrieval is not answer computation

The benchmark credits retrieval of support records. It does not credit a final
answer merely because the expected text occurs in a generated response.

Examples:

- Retrieving `RunInstances` records is retrieval; counting launches, grouping by
  region, and calculating failure rates are structured aggregate operations.
- Retrieving both S3 ACL changes is retrieval; proving their order and deriving
  the state transition requires timestamp-aware tooling.
- Retrieving an archive upload and a later cloud event is retrieval; joining the
  two sources into an incident claim is correlation performed by Livefire or a
  separately evaluated tool.
- An absence claim cannot be established by top-N retrieval. It requires a
  complete, scoped predicate and coverage proof.
- An external-enrichment conclusion requires its admitted enrichment adapter and
  source pointer; model prior knowledge is not evidence.

A fact that needs aggregation may still define retrieval qrels for its supporting
rows, but the report says only that evidence candidates were found. A separate
tool-answer benchmark must evaluate exact counts, sets, grouping, ordering,
joins, and absence predicates, with exact source-pointer provenance and coverage
checks. Retrieval nDCG and final-answer accuracy must never be merged into one
score.

## Required artifacts and report

The content-addressed suite adds:

```text
fact-evidence-suite/
  eligibility-ledger.json
  evidence-groups.parquet
  queries.jsonl
  qrels.parquet
  hard-negative-triplets.parquet
  candidate-universe-receipts.parquet
  split-lock.json
  adjudication-receipt.json
  leakage-audit.json
  suite-manifest.json
  objects.lock.json

run/
  locked-systems.json
  rankings.parquet
  per-surface-metrics.parquet
  per-atom-metrics.parquet
  triplet-results.parquet
  cohort-metrics.parquet
  paired-comparisons.parquet
  failures.jsonl
  report.json
  report.md
  result-receipt.json
```

Every retrieved item retains an exact local source pointer and evidence-group ID.
Governed raw commands, evaluator answers, qrels, and external enrichment remain
in access-controlled objects; the public report uses opaque IDs and redacted
previews.

`report.md` contains:

1. identities, hidden-test opening receipt, and executive disposition;
2. the 23/53/10 denominator waterfall and all eligibility dispositions;
3. primary/gate metrics with confidence intervals by cohort;
4. a model-by-projection heatmap for macro nDCG@20 and gate failures;
5. Recall@20 and coverage dot plots with bootstrap intervals;
6. hard-negative margin ECDFs and triplet failure classes;
7. paired per-atom candidate-versus-reference delta plots;
8. rank distributions and worst-surface regressions;
9. a complete searchable per-atom table with source pointers and failure reason;
10. a separate retrieval-versus-tool-computation accounting table; and
11. leakage audit, limitations, violations, and waivers.

No UMAP or PCA plot is evidence of retrieval quality. If included as an
exploratory appendix, it is fitted without evaluator labels, uses frozen
parameters, and is explicitly excluded from selection and gates. Example cases
are selected deterministically: fixed predeclared IDs, the median atom, the
largest regressions, and every gate failure. Hand-picked successful queries are
not the report.

## Research basis

The design follows results from recent primary work:

- [CmdCaliper (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1126/)
  shows that command-specific contrastive training can materially improve
  command retrieval, motivating domain qrels and hard negatives rather than a
  general embedding leaderboard.
- [STaRK (NeurIPS Datasets and Benchmarks 2024)](https://arxiv.org/abs/2404.13207)
  evaluates retrieval requiring both textual and relational evidence, supporting
  separate structured-context projections and retrieval/tool boundaries.
- [RelBench (NeurIPS Datasets and Benchmarks 2024)](https://arxiv.org/abs/2407.20060)
  treats relational databases as temporal heterogeneous graphs, reinforcing that
  source relationships should remain available instead of being flattened away.
- [MAGIC (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/jia-zian)
  learns provenance context with self-supervised masked graph representation and
  performs multi-granularity outlier detection.
- [KAIROS (IEEE Symposium on Security and Privacy 2024)](https://arxiv.org/abs/2308.05034)
  uses temporal provenance-graph evolution and attack reconstruction, showing
  why flat semantic retrieval is not a complete investigation detector.
- [ORTHRUS (USENIX Security 2025)](https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang)
  emphasizes quality of attribution and analyst inspection burden even when
  detector headline metrics are high.
- [LogSD (FSE 2024)](https://doi.org/10.1145/3660800) demonstrates frequency
  bias in learned normal log patterns, motivating frequency-matched negatives
  and separate rarity versus relevance reporting.
- [ADGym (NeurIPS 2023)](https://arxiv.org/abs/2309.15376) shows that
  preprocessing and other design choices materially alter anomaly results and no
  single choice wins universally, motivating a frozen one-factor matrix.
- [The Need for Unsupervised Outlier Model Selection (SIGKDD Explorations
  2023)](https://doi.org/10.1145/3606274.3606277) documents the unreliability of
  label-free outlier model selection, supporting the rule that evaluator facts
  are an overlay, not a tuning set.

These papers motivate the protocol; none supplies the proposed numerical gates.
The gates are Livefire-RAG policy and must be frozen by the development pilot
before validation or hidden-test results are opened.
