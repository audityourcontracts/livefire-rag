# Immutable command source snapshots

## Purpose

A source adapter converts one bounded data export into canonical command records.
It is not part of the RAG query path. The adapter may speak OCSF/Parquet, Splunk
SPL and REST, Panther GraphQL/SQL, or another vendor protocol; the builder sees
the same sealed snapshot contract.

```text
vendor or file source
  -> adapter establishes a bounded read fence
  -> adapter writes canonical records to staging
  -> host verifies and seals snapshot
  -> credentials and vendor access are removed
  -> RAG builder consumes snapshot read-only
```

## Canonical command record

Every record has:

- a stable command ID within the snapshot;
- event time in UTC;
- canonical principal identity and its confidence/availability state;
- host, process, parent, and optional ancestry metadata;
- original command/script text, or a deterministic canonical action string for
  a cloud API event;
- source-declared shell/interpreter;
- cloud service/API action fields when the event represents CLI or API activity;
- an exact local source-record pointer and content digest;
- adapter and record-schema identities.

The adapter does not decode PowerShell or calculate anomaly scores. Those are
versioned index-builder operations so OCSF, Splunk, and Panther records receive
identical treatment.

## Read fences and checkpoints

The snapshot request uses a half-open event-time interval `[start, end)` and a
closed extraction policy. The adapter must complete or freeze its vendor-side
result before it writes the final snapshot:

- OCSF/file input binds exact input manifests and object digests.
- Splunk binds one completed bounded search/export job and its result count.
- Panther binds one completed data-lake query and all result pages/objects.

Resume checkpoints are append-only and digest chained. Expired vendor jobs or
page tokens fail the export; the adapter may not silently rerun a query and append
different results to the same snapshot.

After staging completes, the host verifies record schema, unique IDs, pointers,
counts, time bounds, object digests, safe paths, and coverage before issuing a
source-admission receipt. Operational timestamps belong in that receipt, not in
the canonical snapshot identity.

## Source pointers

The RAG index retains a structured pointer that resolves within the immutable
source snapshot:

```json
{
  "schema_version": "livefire.source-record-pointer/1",
  "snapshot": {"id": "commands-openbots-v1", "version": "1.0.0", "sha256": "..."},
  "snapshot_profile": {"id": "command-source-snapshot", "version": "1.0.0", "sha256": "..."},
  "record_id": "cmd_...",
  "record_sha256": "...",
  "locator": {
    "kind": "parquet_row",
    "object_sha256": "...",
    "row_group": 4,
    "row_ordinal": 123
  },
  "native_locator_sha256": "..."
}
```

The local record pointer is the reproducibility boundary. A safe vendor locator
may be retained by the vendor adapter for later hydration, but credentials,
URLs, Splunk SIDs, Panther page tokens, and mutable cursors are forbidden.

## Coverage

A snapshot reports requested and observed time ranges, exported/rejected record
counts, command-text availability, principal availability, process-parent
availability, cloud-action availability, and exclusions by stable reason code.
The index builder carries these coverage facts forward rather than treating
missing fields as negative observations.
