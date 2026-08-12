# Generic evidence index v1

## Purpose and boundary

The generic evidence index is a faithful, immutable retrieval projection of
complete admitted source snapshots. It is not an investigation, a finding, a
scenario model, or an evidence-admission engine. It must be built without query
sets, expected answers, evaluation identifiers, incident labels, or
scenario-specific indicators as inputs.

The index makes heterogeneous OCSF-normalized observations discoverable through
one typed search surface. It retains exact source identity so a structured tool
can hydrate a candidate and establish facts using authoritative fields. A ranked
candidate is never itself evidence.

## Normative invariants

1. **Immutable inputs.** Every v1 projection build binds exactly one admitted source snapshot
   component references. A change to any source byte, schema, projection policy,
   derivation policy, model, or physical object creates a new index identity.
2. **Complete closure.** Every source record has exactly one occurrence row and
   exactly one terminal disposition. The coverage report must declare zero
   unaccounted records and zero multiply dispositioned records. Records are never
   silently dropped.
3. **Immutable pointers.** Every occurrence carries an SDK
   `source-record-pointer.v1`. Admission resolves every pointer locally against
   its bound snapshot and rejects unresolved or digest-mismatched pointers.
4. **Generic identity.** Relation identity uses a source namespace and relation
   name, with optional OCSF category, class, and activity identifiers. No source
   is forced to invent OCSF values it does not possess.
5. **Documents and occurrences are separate.** A semantic document describes a
   retrievable meaning. Occurrence rows preserve every source observation's
   membership, event time when parsed, bounded indexed filter attributes,
   relation identity, and exact pointer. Deduplication may reduce vector count
   but may not erase occurrence membership.
6. **Deterministic derivation.** Projections and derived documents bind immutable
   policy component references. Window boundaries, grouping keys, state ordering,
   aggregation rules, text rendering, redaction, and canonicalization are policy,
   not runtime discretion.
7. **Closed document kinds.** V1 admits `activity`, `state`, `state_transition`,
   `metric_window`, `network_window`, `entity`, `detection`, and
   `structured_only`. Adding a kind requires a versioned schema and policy
   change.
8. **No filter borrowing.** Time, source, relation, entity, and declared indexed
   attribute filters apply to occurrence rows before candidate occurrences are
   returned. A semantic group cannot borrow matching metadata from a different
   occurrence. Fields outside the bounded projection require authoritative
   pointer hydration and are not falsely advertised as local filters.
9. **Portable authority.** Canonical Parquet objects and their object lock are
   authoritative. DuckDB files, full-text indexes, ANN graphs, and engine caches
   are derived and rebuildable.
10. **Candidate-only output.** Search returns deterministic ranked document IDs
    and immutable occurrence pointers. It does not assert chronology, counts,
    causality, maliciousness, or evidentiary sufficiency.

## Terminal dispositions

Every occurrence ends in one of these states:

| Disposition | Meaning |
|---|---|
| `direct_semantic_document` | The record produces its own searchable document. |
| `semantic_group_occurrence` | The record joins a meaning-equivalent document while retaining a distinct occurrence row. |
| `derived_document_input` | The record is an input to one or more deterministic derived documents. |
| `structured_only_occurrence` | The record remains exactly pointer-addressable, with any declared indexed attributes, but intentionally has no semantic document. |
| `rejected` | The record cannot be admitted; a stable reason code is mandatory. |

`structured_only_occurrence` and `rejected` are explicit closure outcomes, not
omissions. A build can therefore report complete input accounting even when a
record is unsuitable for semantic retrieval.

## Exact-attribute boundary

`exact_attributes` is a bounded local-filter subset, not a second copy of the
typed event. Every published value is copied unchanged from one parsed
typed-JSON scalar at the declared RFC 6901 JSON Pointer. The builder independently
resolves each pointer against the typed event and rejects a type or value mismatch.
It MUST NOT publish semantic
redactions, hashes, normalized whitespace, stringified objects, truncated
strings, normalized timestamps, or coerced large integers as exact values.

The generic v1 policy includes booleans, JCS-safe integers, finite numbers, and
safe strings of at most 1,024 UTF-8 bytes. It omits nulls, explicit secret
fields, strings containing credential material, free-text fields, oversized
strings, non-finite numbers, non-JCS-safe integers, and values beyond the
attribute or scan bounds. Command lines, request bodies, messages, scripts, and
similar free text can still contribute to the separately bounded and redacted
semantic projection; they are never mislabelled as exact local filters.

Every occurrence carries `exact_attribute_projection` accounting: selected and
known-omitted scalar counts, omission reasons, scan-bound state, and whether
authoritative source hydration is required. An incomplete subset also adds the
stable reason code `exact_attribute_subset_requires_source_hydration`. A scan
bound reports omitted subtree roots because the exact number of unseen scalars
is intentionally unknown. Consumers MUST hydrate the immutable source pointer
for any field not present in `exact_attributes`; absence is never a negative
fact about the source event.

## Document families

### Activity

Projects discrete actions such as process, API, authentication, datastore,
file, HTTP, DNS, network, email, entity-management, and application-lifecycle
activity. Common semantic facets are actor, action, target, outcome, and context.

### State

Projects observations of configuration, inventory, resource, identity, or
service state. Selected identifiers and typed values remain occurrence
attributes; the semantic text describes the state generically.

### State transition

Represents a policy-defined before/after change for the same stable subject.
Ordering key, equality key, compared fields, missing-state behavior, and time
semantics are fixed by a derivation policy.

### Metric window

Summarizes a fixed window over a declared subject, metric, and grouping key.
Every sample remains an occurrence. Window width, origin, coverage requirements,
statistics, and prior-window comparison rules are immutable policy inputs.

### Network window

Summarizes fixed-policy network, DNS, or HTTP populations while preserving all
flows and requests as occurrences. Grouping, cardinality, duration, byte, and
status calculations are policy-defined.

### Entity

Projects stable user, device, process, application, identity, or resource
descriptions. Search aids discovery; exact identity resolution remains a
structured relationship operation.

### Detection

Projects findings and product actions with separate behavior and outcome facets.
An embedding similarity does not establish whether an action was blocked,
allowed, successful, or failed; those remain exact fields.

### Structured only

Retains records for local resolution and structured querying when a faithful,
useful semantic projection is unavailable or intentionally disabled by policy.
These documents are not embedded or returned by semantic search.

## Two-stage artifact boundary

Projection and search-index construction are separate immutable stages.

The builder first emits an `evidence-projection-pack.v1`. It contains canonical
JSONL documents, occurrences, a coverage report, and an object lock. It has no
embeddings, makes no searchability claim, and is not an SDK evidence index. This
stage is useful for schema validation, closure reconciliation, source-pointer
admission, deterministic projection review, and repeatability tests without an
embedding service.

```text
evidence-projection-pack/
  manifest.json
  documents.jsonl
  occurrences.jsonl
  coverage-report.json
  objects.lock.json
```

The projection pack still requires exact SDK source pointers. A
`record_id_only` locator is valid only when the admitted source snapshot provides
the required local pointer table. An opaque source reference, event identifier,
or vendor key is not promoted into an SDK pointer by assertion. If exact local
resolution is unavailable, the builder must stop before emitting a conformant
pack; it must not fabricate pointer digests or claim pointer closure.

Both the projection-pack manifest and the final evidence-index manifest carry a
`component` self-reference. Its SHA-256 is the RFC 8785 JCS digest of the entire
manifest with `/component/sha256` omitted, following the SDK self-digest rule.
The component ID and version are build inputs; URI is optional and, when
present, identity-bearing.

After projection-pack admission, a separate promotion step embeds all searchable
documents, materializes the portable Parquet representation, and emits an
`evidence-index-manifest.v1`. Only this second artifact is a searchable evidence
index.

The current projection implementation emits generic `activity`, `state`, and
`detection` semantic groups for every eligible typed observation. Raw system
metric samples receive `structured_only_occurrence` with the stable reason
`awaits_deterministic_window_derivation`; they remain exactly pointer-addressable
and are not embedded as millions of context-free scalar samples. Metric windows,
state transitions, network windows, and entity consolidations are distinct
policy-bound derivation stages. Their absence must remain visible in the
coverage report and cannot be represented as silent omission.

## Standalone projection commands

The builder consumes a completed normalized-snapshot receipt rather than a
manually selected list of records. It verifies the exact digest and row count of
every typed relation before projection.

```sh
livefire-rag build-evidence-projection \
  --snapshot-root NORMALIZED_SNAPSHOT \
  --source-build-receipt NORMALIZED_SNAPSHOT/build-receipt.json \
  --snapshot-id SOURCE_COMPONENT_ID \
  --snapshot-version SOURCE_COMPONENT_VERSION \
  --index-id PROJECTION_PACK_COMPONENT_ID \
  --index-uri urn:example:projection-pack:1 \
  --index-version 1 \
  --out PROJECTION_PACK

livefire-rag verify-evidence-projection \
  --pack PROJECTION_PACK \
  --snapshot-root NORMALIZED_SNAPSHOT \
  --source-build-receipt NORMALIZED_SNAPSHOT/build-receipt.json \
  --sdk-specs ../livefire-sdk/specs

# Override only when validating against an explicit RAG schema checkout:
#   --rag-specs /path/to/livefire-rag/specs

livefire-rag inspect-evidence-projection \
  --pack PROJECTION_PACK \
  --snapshot-root NORMALIZED_SNAPSHOT \
  --sdk-specs ../livefire-sdk/specs
```

The wheel includes the generic evidence schemas, projection policy, and typed
Parquet record profile. Verification discovers those packaged RAG contracts by
default and deliberately excludes scenario/benchmark schemas from its offline
registry. The SDK schema directory remains an explicit host input so the pack
is checked against the SDK contract set selected by the caller.

SDK component references may include an optional non-empty `uri`. The URI is
preserved on admitted source snapshots, projection policies, and pack
components. It participates in canonical manifest and row identities; it is
not display-only metadata.

Discovery is driven only by the closed typed-relation contract. The builder
does not accept queries, labels, expected facts, scenario IDs, include patterns,
or event-value predicates.

The verify command validates every manifest, coverage object, document row, and
occurrence row against the offline JSON Schema registry. It then mounts every
receipt-fenced typed relation and resolves every pointer against the exact
object digest, row group, row ordinal, record ID, record digest, and support
reference. Finally, it replays the bound deterministic projection for every
source row and requires the occurrence and semantic document to match the
pack exactly. Structural verification without the source snapshot and bound
projection policy is not admission and cannot satisfy
`all_pointers_resolved`.

## Canonical search-index artifact set

```text
evidence-index/
  manifest.json
  documents.parquet
  occurrences.parquet
  embeddings.parquet
  coverage-report.json
  objects.lock.json
  relations.parquet       # optional exact relationship projection
  lexical-inputs.parquet  # optional canonical input for a lexical cache
```

`documents.parquet` validates logically against `evidence-document.v1` and is
ordered by `document_id`. `occurrences.parquet` validates against
`evidence-occurrence-row.v1` and is ordered by `(occurrence_id)`. Duplicate IDs
are admission failures.

`embeddings.parquet` contains only searchable document IDs and vectors bound to
the manifest's embedding profiles. It must not contain a vector for a
`structured_only` document. Exact vector search is the correctness oracle;
approximate indexes are disposable caches.

`coverage-report.json` validates against `evidence-coverage-report.v1`. Its
global, per-relation, per-disposition, per-kind, rejection-reason, and pointer
counts must reconcile during admission. JSON Schema fixes the shape and required
zero-failure claims; the builder/admission harness verifies the arithmetic.

All artifacts are exact-byte `artifact-ref.v1` members of the SDK object lock.
The top-level `evidence-index-manifest.v1` binds the SDK base index manifest,
schemas, physical profile, snapshots, policies, model profiles, artifacts,
closure summary, and query contract.

## Projection pipeline

```text
admitted source snapshots
        |
        v
enumerate every relation and record
        |
        v
canonical typed evidence record
        |
        +--> direct document
        +--> semantic group + occurrence
        +--> deterministic derived document input
        +--> structured-only occurrence
        `--> rejected occurrence + reason
        |
        v
closure and pointer reconciliation
        |
        v
semantic text/facets + embeddings + lexical inputs
        |
        v
sealed evidence index
```

Projection policies operate on schema fields and typed values. They may not
inspect evaluation queries, labels, expected results, analyst annotations, or
future scenario content. Human-readable text is a deterministic rendering of the
bound fields. The bounded indexed-attribute set is useful for declared local
filters but is not a source replica: omitted list members, oversized text, and
non-indexed fields remain available only through the immutable source pointer.

Derivation policies must declare at least:

- accepted relation and schema identities;
- stable grouping and subject keys;
- ordering fields and null handling;
- fixed window origin and duration where applicable;
- aggregation functions and numeric types;
- minimum coverage and incomplete-window behavior;
- semantic rendering version; and
- policy component identity.

## Search contract

`evidence.search` accepts natural-language text, a bounded `top_n`, explicit
retrieval methods, and closed typed filters. V1 supports dense, lexical, or
deterministically fused retrieval. It does not accept SQL, arbitrary expressions,
model-authored field paths, or source credentials.

The provider performs these logical steps:

1. resolve the immutable index binding;
2. select eligible occurrence rows using the closed filters;
3. derive eligible semantic document IDs without losing occurrence membership;
4. run dense and/or lexical retrieval over that closed universe;
5. fuse rankings according to the bound policy;
6. apply the manifest tie break; and
7. return candidate documents with matching immutable occurrence pointers.

Score fields use deterministic integer encodings. Missing retrieval channels are
represented as `null`, not fabricated zero scores. `matched_facets` are copied
from deterministic projected fields and are not generated explanations.

Search output is deliberately insufficient for evidence admission. Hydration,
exact aggregation, entity joins, state ordering, chronology, and factual claims
remain responsibilities of structured data tools and the Livefire investigative
brain.

## SDK and runner boundary

The projection pack is a builder artifact, not a tool-provider index binding.
It is therefore verified with `verify-evidence-projection` and must not be
opened by the Livefire runner. The promotion step will produce the SDK base
index manifest, specialized `evidence-index-manifest.v1`, admission receipt,
object lock, format descriptor, and `evidence.search` tool descriptor required
by a provider binding.

The eventual provider follows the existing SDK JSONL lifecycle:
`handshake -> open(binding lock + admitted index mount) -> call -> health ->
close`. It opens the final index read-only, returns only
`source-record-pointer.v1` candidates, and has no normalized-source mount,
vendor credentials, or Livefire repository dependency. This contract does not
require a new Livefire brain interface and does not move hypothesis, evidence,
or finding logic into RAG.

The current SDK component/artifact/source-pointer types are sufficient for the
projection stage. A real tool bundle is intentionally deferred until there is a
searchable, admitted evidence index; packaging the pre-embedding pack as a
provider index would be a false capability claim.

## Admission checks

An index is admissible only when the harness verifies:

- every declared artifact digest and byte length;
- schema validity for all logical rows;
- unique document and occurrence IDs;
- exactly one occurrence/disposition per input source record;
- source, disposition, relation, kind, and reason-count reconciliation;
- zero unaccounted and multiply dispositioned records;
- resolution and record-digest validation for every source pointer;
- document occurrence counts against the occurrence join;
- semantic group membership without filter leakage;
- derivation-policy references for every derived document;
- no embedding for structured-only documents and one conformant embedding per
  searchable document/profile;
- deterministic row ordering and canonical record chains; and
- absence of evaluation and scenario artifacts from the build dependency graph.

These checks measure indexing fidelity. Retrieval evaluation may be performed on
a sealed index, but evaluation inputs cannot influence projection, grouping,
derivation, or admission.
