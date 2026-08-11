# Command RAG test, evaluation, and reporting program

Status: implementation plan and draft promotion contract.

This program evaluates six different things without conflating them:

1. source-to-command fidelity;
2. deterministic projection, decoding, scoring, and protocol conformance;
3. embedding and distance correctness;
4. semantic retrieval quality;
5. principal/population anomaly-ranking quality; and
6. operational qualification and standalone-provider isolation.

A model cannot compensate for a missing command, an invalid pointer, temporal
leakage, or an incorrect cosine calculation. Quality scores are therefore
reported separately from conformance and performance.

## Dataset decision

The production-equivalent lineage is:

```text
OpenBOTS native Parquet (source truth)
  -> admitted Livefire OCSF normalized snapshot (typed source boundary)
  -> livefire-ocsf command snapshot (adapter-neutral command records)
  -> livefire-rag command projection (model input)
  -> embeddings and immutable RAG index
```

The embedding model does not consume the entire raw OpenBOTS row and does not
consume a serialized OCSF event. It consumes `command_document.semantic_text`
and the separate action/target projections produced by the versioned projection
policy. Those projections contain the original or statically decoded command,
typed action and target, shell/interpreter, bounded process context, and stable
semantic features. Principal, host, timestamps, event IDs, source names, and
unstable identifiers remain metadata unless a particular context feature is
explicitly included by policy.

OpenBOTS native data remains essential for a source-fidelity oracle. The OCSF
snapshot remains essential for portable identity, time, actor, device, class,
relationship, and provenance fields. The command snapshot is the stable boundary
that later Splunk and Panther adapters must also emit.

### Available local corpus

The pinned OpenBOTS catalogue identifies:

- dataset digest `ba9e0c1ff5f1154defc0956e1984fc1168d0424d29f8d4d6b02e1d1c93fbbe46`;
- 2,030,269 native rows in 42 Parquet objects;
- 107 sourcetypes, 1,007 sources, and 1,104 source/sourcetype pairs.

The lossless authority is `raw_bytes`, not the convenience `raw` string: the
OpenBOTS validation manifest records 109,332 rows containing invalid UTF-8. The
source adapter must bind/extract exact bytes and record any decoding policy rather
than silently accepting replacement characters.

The latest complete local normalized snapshot inspected for this plan is
`outputs/ocsf/botsv3-m21-v1`; M22 currently exists only as a staging directory
and is not an admissible test corpus. M21 contains, among other relations:

- 234,858 normalized process-activity rows;
- 407,729 normalized event-log-activity rows;
- 8,592 normalized API-activity rows;
- 142 distinct API operations and 37 actors in the API relation;
- 13,905,577 normalized event rows after one-to-many normalization.

The M21 authority relation contains 127,490,228 field-provenance rows. It is
available to the source adapter to resolve support references to exact native
object/row/raw-byte identities, but it is not exposed to the RAG builder after
the command snapshot is sealed.

There is a known command-fidelity issue to test rather than conceal: 214,922 of
the M21 process rows carry `cmdline` under the preserved native/unmapped payload,
while none of the inspected rows promote it to an OCSF `process.cmd_line` field.
PowerShell-bearing native rows also occur across Sysmon, PowerShell Operational,
Security, WinHostMon, and other sourcetypes. The first adapter milestone must
therefore prove native-to-command recall across multiple normalized classes and
must not index only `ocsf_process_activity` or silently discard preserved command
text.

M21 is the first reproducible full-corpus baseline. A sealed M22 becomes a new
suite/corpus version and is compared as a migration; it never overwrites M21
results.

## Three test programs

### Program A: deterministic conformance

This uses small, hand-audited fixtures and an independent oracle. The fake
embedder returns declared vectors so expected distances and ranks are exact.
It gates:

- snapshot and pointer integrity;
- command extraction and exclusion accounting;
- PowerShell static decoding/AST safety;
- strict-prior 30-day history selection;
- all four anomaly components and calibration;
- distance encoding and stable ordering;
- JSON schemas, tool semantics, and provider lifecycle;
- deterministic rebuild and replay; and
- sandbox, mount, credential, and network isolation.

All required Program A tests must pass. A high retrieval score cannot waive one.

### Program B: held-out quality evaluation

This uses the full corpus or a content-addressed representative corpus plus a
sealed, reviewed query/qrel bundle. It measures embedding retrieval, command
similarity, and anomaly ranking. It supports model, projection, dimension,
quantization, and component ablations. Development, validation, and hidden test
intent groups are disjoint.

### Program C: operational qualification

This measures full build cost and provider behavior on the declared hardware and
workloads. It retains raw samples rather than only percentiles. Performance gates
are versioned by environment/budget and do not alter correctness or quality.

## Fixture ladder

### L0: mathematical micro-fixtures

Approximately 20 commands with artificial two- or four-dimensional vectors.
Use known vectors for distance boundaries:

```text
A=[1,0], B=[1,0]  -> cosine distance 0
A=[1,0], C=[0,1]  -> cosine distance 1
A=[1,0], D=[-1,0] -> cosine distance 2
```

Assert float64 accumulation, half-even millionths encoding, ascending semantic
distance order, descending anomaly-score order, and command-ID tie-breaking.

### L1: static-analysis and history fixtures

Approximately 100 authored records covering shells, PowerShell encodings,
process trees, cloud actions, identities, timestamps, cold start, and missing
fields. Expected history IDs, scores, comparisons, AST shapes, and pointers are
hand-reviewed and stored as oracle artifacts.

### L2: native-to-OCSF fidelity sample

Use a deterministic, stratified sample selected before extraction:

- every native row from rare command-bearing sourcetypes;
- all PowerShell Operational rows;
- stratified Sysmon/Security/WinHostMon process rows;
- stratified osquery process rows by host and principal;
- all known incident command/API rows;
- API actions stratified by actor, operation, status, and resource presence; and
- random non-command rows to measure false extraction.

For every native row, record whether zero, one, or several canonical command
records should be emitted. Review missing text, principal, host, parent, event
time, action, target, and source-pointer fidelity separately.

Initial gates:

- 100% recall on reviewed command-bearing rows;
- 0 false command records from reviewed negative rows;
- 100% exact native-pointer resolution;
- 100% event-time agreement;
- every absent principal/parent/target represented as unavailable, never guessed;
- no dependence on a mutable native file path after snapshot admission.

### L3: pilot corpus

Use 500 to 2,000 deduplicated command groups with deliberate representation of
PowerShell, POSIX, Windows, cloud APIs, rare behavior, common background commands,
and benign hard negatives. Use this for rapid builder/provider iteration.

### L4: full M21 command corpus

Extract all qualifying command/script/API observations from the complete M21
snapshot. Report extraction counts by normalized class, native sourcetype,
shell, host, principal status, observation kind, and exclusion reason. Never
declare the number of commands before this accounting report is produced.

### L5: portability and source-adapter corpora

Apply the same command-snapshot contract to a later OCSF edition, synthetic
Splunk export, and synthetic Panther export. Equivalent input command records
must produce identical projections and index semantics regardless of adapter.

## Test catalogue

### `LF-SOURCE-*`: source and command fidelity

- Verify source catalogue, OCSF snapshot, receipts, objects, and digests.
- Match M21 lineage to the pinned OpenBOTS dataset digest.
- Enumerate command-bearing native and normalized classes.
- Reconcile native-positive, normalized-positive, emitted, and excluded counts.
- Verify original command bytes/digest, timestamps, principal namespace/ID,
  device, image, parent, ancestry, API service/action/resources, and status.
- Detect commands present only in preserved native/unmapped fields.
- Detect PowerShell/script blocks outside the process relation.
- Reject duplicate canonical command IDs and unresolved support references.
- Publish a field-level loss matrix and every excluded-row reason.

### `LF-EMBED-*`: embedding conformance

For document, action, target, and query inputs assert:

- exact profile/model/tokenizer/runtime/prompt identity;
- dimension, finite float32 values, and L2 norm tolerance;
- input order preservation and overlength rejection;
- exact document and instructed-query composition;
- repeatability over five cold model loads and three provider processes;
- vector and ranking digests; and
- explicit `exact`, `within_declared_tolerance`, or `mismatch` disposition.

Compare current Qwen3 8B Q4_K_M with the official 8B BF16 quality reference.
Cross-profile vectors need not be byte-identical; compare vector alignment,
top-20/top-100 overlap, distance deltas, and relevance margins.

### `LF-DIST-*`: distance and engine oracle

An independent implementation consumes stored float32 L2 vectors, accumulates
the dot product in float64, computes cosine distance, and applies the manifest's
half-even millionths rule. Compare it with DuckDB/provider results for:

- all returned pairs;
- the mathematical micro-fixtures;
- 10,000 stratified random query/document pairs;
- equal-distance ties;
- distances near 0, 1, and 2; and
- corrupted, wrong-dimension, non-finite, or non-normalized vectors.

The exact engine must have zero encoded-distance or rank disagreements. An ANN
cache is a later independent experiment and must achieve Recall@20 >= 0.98
against exact search.

### `LF-PS-*`: PowerShell safety and structure

Cover UTF-16LE/UTF-8 Base64, URL/escape encoding, recognized compression,
constant concatenation, backticks, mixed case, depths one through three,
malformed input, a fourth layer, expansion bombs, boundary-plus-one inputs,
recoverable parse errors, dynamic expressions, and sentinel payloads.

Assert decode order/status, limits, AST node/parent ordering, feature signals,
stable digests, and zero command/process/network/DNS execution. The sandbox audit
trace is a required report artifact.

### `LF-HIST-*`: principal and population histories

Construct histories where:

- A repeatedly uses X and B repeatedly uses Y;
- A/Y is principal-novel but population-familiar;
- A/X is principal-familiar;
- A/Z is novel to both;
- a new principal has population but no principal history;
- two namespaces share the same native ID;
- identity is ambiguous or unavailable; and
- tenant/scope membership differs.

Assert exact eligible history IDs, independent counts/calibration/scores,
structured principal-key isolation, and comparison-universe membership.

### `LF-TIME-*`: future-leakage gate

Test the instant before, equal timestamp, 30-day boundary, just-expired, and
future records. Adding arbitrary future records must not alter an existing
candidate's projection-independent score, calibration population, or comparisons.
Input permutation at equal timestamps must not alter output. Every unexpected
equal/future/expired ID is retained in a leakage ledger; all lists must be empty.

### `LF-COMP-*`: four anomaly components

Change one factor at a time for action, target, structure, and obfuscation.
Assert raw distance, scorer identity, calibration midrank/ties, millionths,
availability/null semantics, minimum-history boundary, retained comparison cap,
weight sum, available-component weight renormalization, combined score, and
status. Run each component alone, leave-one-out, and the full policy, separately
for principal and population rankings.

### `LF-TOOL-*`: tool behavior

Run every case through both the standalone CLI and provider protocol.

Common gates include schema rejection, closed filters, exact index/source refs,
no duplicate IDs, consecutive ranks, cardinality/accounting, stable ties,
locally resolvable pointers, byte/candidate/deadline limits, and absence of
filesystem paths, vectors, vendor locators, and credentials.

- `cli.outliers`: principal, population, both, cold start, top N, score ordering.
- `cli.search`: instructed query, exact filters, cosine ordering, no-answer query.
- `cli.similar`: seed existence, exclude/include seed, profile match, filters.
- `cli.explain`: both scopes, baseline, four components, policies, comparisons,
  and agreement with materialized rows.

### `LF-PROVIDER-*`: lifecycle and isolation

Test handshake/open/call/health/close, ID correlation, invalid order/session,
partial JSONL writes, invalid UTF-8/JSON, stdout contamination, incompatible or
corrupt bindings, restart/reopen, concurrent isolation, read-only index bytes,
blocked source/vendor access, exact loopback-only embedder permission, arbitrary
filesystem denial, limits, crash handling, and clean termination.

The provider must pass without a Livefire checkout, source mount, Splunk/Panther
credential, or vendor endpoint.

## Semantic retrieval benchmark

Create 180 intent groups, initially 30 in each family:

1. executable/action;
2. action plus target/resource;
3. decoded PowerShell intent;
4. argument/AST/process structure;
5. cloud API action/resource semantics; and
6. investigation/threat-language vocabulary mismatch.

Each intent has four reviewed surfaces: natural analyst language, terse SOC
language, a vocabulary-changing paraphrase, and an entity-light/role-oriented
query. Keep every surface and synthetic variant of one intent in the same split.
Use 60 intent groups for development, 60 for validation, and 60 for a hidden test.
Add at least 120 command-to-command similarity seeds and 20-30 no-answer controls.

### Ground truth

Use predicate-complete qrels where canonical fields define all relevant records.
For semantic intent, pool the deduplicated top 100 from BM25, exact field/token
search, Qwen 8B BF16, Qwen 8B Q4, smaller Qwen variants, BGE-Code, projection
ablations, deterministic predicates, and authored hard negatives.

Two reviewers independently grade semantic command groups:

- 3: directly satisfies the required action/target/intent;
- 2: strongly relevant and useful;
- 1: contextual but not an answer; and
- 0: irrelevant or contradicts a required facet.

Adjudicate relevant/non-relevant disagreements or grade gaps greater than one.
Report weighted kappa and adjudication rate; target kappa >= 0.75 before sealing.
Event-level pointers remain in the results, but relevance metrics collapse exact
semantic duplicates so repeated background events cannot inflate quality.

### Mandatory hard negatives

Every intent includes reviewed examples of applicable classes:

- same executable, different action;
- same action, wrong target;
- same target, benign read instead of sensitive write/change;
- token-overlapping quoted, echoed, commented, negated, or failed command;
- same `powershell -enc` wrapper, different decoded payload;
- benign encoded administration versus suspicious decoded behavior;
- same process binaries with reversed/unrelated ancestry;
- cloud list/get/describe versus create/update/expose/delete;
- path/resource substring collision; and
- frequent near-duplicate background commands.

### Primary metrics

Macro-average by intent group, not query string:

- Recall@1/5/10/20 for grade >= 2;
- nDCG@10/20 with grades 0-3;
- MRR@20, MAP@20, and Precision@5;
- worst-paraphrase and bottom-decile Recall@20;
- family/common/rare/source/shell slices;
- hard-negative triplet accuracy and distance margin;
- positive, hard-negative, easy-negative distance distributions; and
- `cli.similar` Recall/nDCG.

Use 10,000 paired bootstrap resamples over intent groups for 95% confidence
intervals. Report candidate-minus-reference deltas with the declared multiple
comparison correction.

## Anomaly-quality benchmark

Labels identify commands unusual for the principal, population, both, or neither
at that point in time. Include encoded PowerShell, unusual ancestry, new action,
new target, new action-target combination, sensitive API use, sparse principals,
cold start, and benign administrative decoys.

Report principal and population independently:

- Recall and nDCG at 5/10/20;
- MRR and rank of each labelled candidate;
- benign-admin false-candidate rate;
- cold-start/unscored rate;
- per-principal/source/host/history-density slices; and
- component-only, leave-one-component-out, and full-policy results.

The benchmark is ranking-based. It does not turn top-N retrieval into an alert
threshold or claim that an outlier is malicious.

## Predeclared experiment matrix

Run paired, one-factor comparisons on identical corpus/splits:

- Qwen3 8B BF16 reference versus current LM Studio Q4_K_M;
- Q8/F16 if available, Qwen3 4B/0.6B, BGE-Code, and EmbeddingGemma;
- BM25 and exact typed-field/token baselines;
- 1,024, 2,048, and 4,096 dimensions;
- raw command only versus canonical semantic projection;
- original-only versus bounded decoded content;
- AST and ancestry on/off;
- action/target/structure context on/off;
- raw identifiers versus typed/redacted identifiers;
- query instruction versions/on/off; and
- dense-only versus a separately reported dense+BM25 hybrid.

Do not combine model selection with ANN or LLM reranking. Those are separately
versioned experiments after exact dense retrieval is selected.

## Initial promotion gates

Correctness requires zero:

- unresolved pointers or wrong filters;
- temporal-leakage records;
- schema/profile/manifest/vector-shape violations;
- exact-engine distance/ranking disagreements;
- PowerShell execution attempts;
- required skipped tests; and
- unaccounted source/index records.

Provisional quality floors, frozen after the development pilot but before
validation/test, are overall Recall@20 >= 0.85, nDCG@20 >= 0.70, every family
Recall@20 >= 0.70, and hard-negative triplet accuracy >= 0.90.

Promote Q4 instead of BF16 only if:

- macro Recall@20 and nDCG@20 are each within 1 percentage point;
- no family Recall@20 loses more than 3 points;
- worst-paraphrase Recall@20 loses no more than 2 points;
- hard-negative triplet accuracy and principal/population anomaly nDCG@20 each
  lose no more than 1 point; and
- Q4 materially improves a predeclared operational measure, provisionally at
  least 25% lower peak memory or 25% higher throughput.

Confidence intervals must support the declared non-inferiority margins. If Q4
does not provide a material operational benefit, retain BF16 even if it ties.

## Reproducible suite and run artifacts

```text
eval-suite/
  manifest.json
  corpus-binding.json
  queries.jsonl
  predicates.json
  qrels.parquet
  hard-negatives.parquet
  similarity-seeds.jsonl
  anomaly-labels.parquet
  splits.json
  adjudication-receipt.json
  objects.lock.json

run/
  run-manifest.json
  build-transcript.jsonl
  provider-transcript.jsonl
  rankings.parquet
  distances.parquet
  per-query-metrics.parquet
  anomaly-metrics.parquet
  correctness.json
  repeatability.json
  resources.json
  failures.jsonl
  report.json
  report.md
  result-receipt.json
```

The suite manifest and report validate against the repository schemas. Large or
sensitive commands, vectors, labels, and transcripts remain in governed
content-addressed artifacts; the report references them by digest.

## Execution order and review checkpoints

| Phase | Work | Reviewable output | Stop condition |
|---|---|---|---|
| 0 | Seal M21/native lineage and sample plan | corpus binding and sample lock | any digest/receipt mismatch |
| 1 | Native-to-command fidelity audit | loss matrix, exclusions, pointer ledger | missed reviewed command or false extraction |
| 2 | L0/L1 fake-embedding conformance | exact distance/history/tool report | any required conformance failure |
| 3 | L3 pilot projection and Q4 index | projection audit, pilot rankings, resource estimate | pointer/filter/distance mismatch |
| 4 | Author/pool/adjudicate evaluation suite | sealed queries, qrels, kappa receipt | leakage or insufficient agreement |
| 5 | Full M21 command snapshot and exact index | closure report and index receipt | unaccounted command candidate |
| 6 | BF16/Q4/model/projection matrix | paired quality and cost report | hidden split remains sealed until selection |
| 7 | Principal/population anomaly evaluation | temporal ledger and component ablations | any future leak or score-oracle mismatch |
| 8 | Provider isolation and performance | protocol, denial, replay, raw sample reports | vendor/source dependency or budget failure |
| 9 | Promotion review | signed combined report and receipt | any unresolved required violation |

After Phase 1, calculate the actual command count and projection-deduplication
rate before estimating full embedding time/storage. The 13.9 million normalized
events are not automatically embedded: only admitted command/script/API command
records are. Repeated semantic/action/target projections may reuse vectors by
their projection digest, while each event retains its own pointer and
chronological score row. The report must show physical vectors, reused references,
and savings so deduplication cannot alter ranking silently.

## Full report

One report contains three independently visible dispositions:

```text
conformance: pass | fail | incomplete
quality:     promoted | rejected | informational | incomplete
operations:  qualified | unqualified | informational | incomplete
```

It binds exact source/index/provider/model/policy/schema/suite/harness identities,
environment, every suite result, per-family/slice metrics, confidence intervals,
distance diagnostics, regressions, repeatability, raw performance samples,
violations, waivers, audit artifacts, and the final signed receipt.

Required audit artifacts include source/build/index admission records, object
locks, model/profile/conformance transcripts, canonical requests/results,
pointer-resolution and temporal-leakage ledgers, no-execution sandbox trace,
network/vendor-denial trace, repeat-build diff, repeat-query digest matrix,
quality labels/adjudication receipt, raw performance samples, and harness/SBOM
identity.

`report.md` presents:

1. executive disposition and failed gates;
2. corpus lineage and extraction/loss accounting;
3. embedding profile and repeatability;
4. overall/family/slice retrieval results;
5. distance distributions and hard-negative failures;
6. principal/population anomaly results and ablations;
7. tool/provider conformance and isolation;
8. build/query resource measurements;
9. per-query and per-candidate regressions; and
10. limitations, waivers, and next experiment.

## Planned standalone commands

```text
livefire-rag evaluate audit-source --native OPENBOTS --ocsf M21 --out AUDIT
livefire-rag evaluate prepare --snapshot COMMAND_SNAPSHOT --draft DRAFT --out STAGING
livefire-rag evaluate pool --suite STAGING --matrix MATRIX --out POOL
livefire-rag evaluate adjudicate-import --pool POOL --judgments JUDGMENTS --out SEALED
livefire-rag evaluate run --index INDEX --suite SEALED --split validation --out RUN
livefire-rag evaluate compare --reference BF16_RUN --candidate Q4_RUN --gates GATES --out REPORT
livefire-rag evaluate verify --suite SEALED --run RUN
livefire-sdk test-provider ./bin/livefire-rag-provider --index INDEX --suite CONFORMANCE
```

These are target interfaces for implementation; the current repository contains
the specifications, not these executables.
