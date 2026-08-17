# Generic evidence derivation overlay

Historical Python design record. This overlay is not part of the active M45
preparation, embedding, index, or provider path.

The derivation overlay adds deterministic, searchable summaries to a sealed
generic evidence projection pack. It does not rewrite source occurrences or
change their terminal dispositions. A separate membership relation allows one
source occurrence to support multiple derived documents while retaining its
exact Parquet pointer in the base pack.

The overlay emits four document kinds:

- `metric_window`: observed metric samples in fixed, UTC epoch-aligned
  five-minute windows;
- `network_window`: network, DNS, and HTTP observations in the same fixed
  windows;
- `state_transition`: changes between adjacent, strictly time-ordered states
  in a canonical entity scope;
- `entity`: receipt-bound entity summaries over typed-relation participation
  and relationship structure.

All exact grouping scope comes from the normalized snapshot's `participants`
and `entities` relations. The canonical scope is the sorted unique tuple of
participant role and entity ID. Exact identifiers affect document identity and
membership but never appear in semantic text. Missing scope is an explicit
ineligibility reason; unrelated anonymous records are never merged.

## Immutable boundary

An overlay manifest binds all of the following by digest:

- the normalized source-snapshot component;
- the sealed base projection-pack component;
- the built-in derivation policy;
- the receipt-bound entity, participant, and relationship objects;
- derived documents, memberships, coverage, and the object lock.

Document identity includes the source and base components, policy, exact group
and aggregate material, and a digest of the complete sorted input membership
set. Membership rows resolve a derived document ID to a base occurrence ID and
an input role. The base occurrence file is not copied.

Window closure is `snapshot_sealed_observed`: all observations present in the
immutable snapshot were consumed. This does not claim that upstream telemetry
was complete or that an expected sampling cadence was satisfied. Empty windows
are not synthesized, missing numeric values are counted rather than replaced
with zero, and boundary windows are retained.

State transitions compare hydrated typed state, not the bounded exact-attribute
subset. The first observation has no predecessor, equal adjacent states are
unchanged, and distinct states at the same timestamp are ambiguous rather than
ordered using an arbitrary event ID.

Entity documents are emitted only for entities that resolve to at least one
admitted typed occurrence through `participants`. Orphan entities remain in
coverage with an explicit reason. Dangling participant or relationship keys are
integrity failures.

## Scenario-blindness

The public builder accepts only the snapshot root, its receipt, the base pack,
an output location, and component identity. It has no query, label, expected
evidence, qrel, benchmark, include-list, or per-relation selector. Relations are
selected solely by the fixed typed contract, and window sizes, structural field
adapters, null behavior, aggregate functions, and volatility exclusions are all
bound in the versioned policy before execution.

The derivation overlay is a pre-embedding artifact. Promotion may union its
documents with the base documents, embed all searchable rows under bound model
profiles, and expose membership-backed hydration without altering either
component.
