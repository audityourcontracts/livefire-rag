# Generic evidence M21 v1 build report

## Result

The scenario-blind generic evidence projector completed a full build from the
admitted BOTS v3 M21 normalized OCSF snapshot. The builder enumerated all 18
typed relations and assigned exactly one terminal disposition and one immutable
source pointer to every one of the 13,905,577 source records.

This artifact is a sealed, source-replayed **pre-embedding projection pack**.
It is not yet a searchable RAG index and must not be opened by a Livefire tool
binding. Independent offline schema validation is recorded separately from the
builder's source replay so those two guarantees are not conflated.

| Identity | Value |
|---|---|
| Projection pack | `livefire.rag.evidence.botsv3-m21-v1@1` |
| Pack SHA-256 | `67040490774e09b0ab598e09921fc4c4ac2fc7b32e234fccd362c1f4f5525ed6` |
| Source snapshot | `botsv3-ocsf-normalized-snapshot@21` |
| Source SHA-256 | `1fda84fc2ab33c8d2dac4f72d44cc27c5a3e19d5c72a796bcb2e342012970dd0` |
| Source build receipt SHA-256 | `b1cb0af18856c91e6aedd269744d92f19197dc1b336c90aa1bb70ea50a1b9de8` |
| Projection policy SHA-256 | `0fa789e49c388f4bfda9239aca0edf5d1f63c6fcec3b6775ffdb4622f335bdf6` |

## Closure and document materialization

| Measure | Count |
|---|---:|
| Typed relations | 18 |
| Source records | 13,905,577 |
| Terminal dispositions | 13,905,577 |
| Resolved immutable pointers | 13,905,577 |
| Unresolved pointers | 0 |
| Unaccounted records | 0 |
| Multiply dispositioned records | 0 |
| Semantic occurrences | 6,367,276 |
| Structured-only occurrences | 7,538,301 |
| Searchable documents | 1,319,974 |
| Activity documents | 1,168,134 |
| State documents | 151,787 |
| Detection documents | 53 |

Semantic grouping reduced 6,367,276 semantic occurrences to 1,319,974
documents, a 4.823789-to-1 reduction. This does not delete or merge evidence:
the occurrence ledger retains every event, exact source pointer, filterable
attribute subset, omission state, and document membership.

System metrics account for 7,538,301 records, or 54.21% of the source corpus.
They are deliberately retained as structured-only occurrences. Embedding a
context-free scalar sample would create misleading semantics, so searchable
metric documents require a separately versioned, fixed-window derivation
policy. No metric row was dropped.

## Relation closure

| Typed relation | Source records | Semantic | Structured only |
|---|---:|---:|---:|
| `ocsf_api_activity` | 8,592 | 8,592 | 0 |
| `ocsf_application_lifecycle` | 1,260 | 1,260 | 0 |
| `ocsf_authentication` | 1,125 | 1,125 | 0 |
| `ocsf_cloud_resources_inventory_info` | 522 | 522 | 0 |
| `ocsf_datastore_activity` | 36,494 | 36,494 | 0 |
| `ocsf_detection_finding` | 2,240 | 2,240 | 0 |
| `ocsf_dns_activity` | 115,145 | 115,145 | 0 |
| `ocsf_email_activity` | 927 | 927 | 0 |
| `ocsf_entity_management` | 60 | 60 | 0 |
| `ocsf_event_log_activity` | 407,729 | 407,729 | 0 |
| `ocsf_ext_livefire_configuration_snapshot` | 4,480,933 | 4,480,933 | 0 |
| `ocsf_ext_livefire_system_metric` | 7,538,301 | 0 | 7,538,301 |
| `ocsf_file_activity` | 330 | 330 | 0 |
| `ocsf_http_activity` | 25,114 | 25,114 | 0 |
| `ocsf_inventory_info` | 9,643 | 9,643 | 0 |
| `ocsf_network_activity` | 1,042,076 | 1,042,076 | 0 |
| `ocsf_process_activity` | 234,858 | 234,858 | 0 |
| `ocsf_user_inventory` | 228 | 228 | 0 |

## Physical artifacts

| Object | Bytes | SHA-256 |
|---|---:|---|
| `documents.jsonl` | 4,540,350,179 | `41c5f385a4dc3413850f74ee87d0d93dc722ec47c52af482eedff5e7a1413f0e` |
| `occurrences.jsonl` | 49,079,822,081 | `527a2b676c02ff8e9a83cf38ac69e5dfbd43eda1ab34e764c5d188462a511736` |
| `coverage-report.json` | 7,115 | `bc02a331c53d57a8d5f1c0fe0a8990a0f0063d0213f145e1c8a27830d5529f37` |
| `objects.lock.json` | 528 | `285f7a113481aeeb8417be7bb8e30a43711cc9bf3b31c7566ef92278d4fc9f27` |

The canonical JSONL objects total about 49.94 GiB. The occurrence ledger is a
fidelity artifact; embedding promotion operates on the smaller semantic
document layer while preserving the occurrence join for filters, aggregation,
and exact hydration.

## Verification

The builder independently replayed every receipt-fenced Parquet row against the
sealed occurrence and document material. It checked the source object digest,
row group and row ordinal, record ID and digest, support reference, exact typed
attributes, terminal disposition, deterministic projection, semantic document,
occurrence count, relation count, reason count, and global closure. The build
would not atomically publish the pack if any replay differed.

A separate offline Draft 2020-12 pass then validated the sealed manifest,
coverage report, all 1,319,974 document rows, and all 13,905,577 occurrence rows
against the packaged generic evidence schemas and the adjacent SDK pointer
contracts. The observed pass took 11,087.19 seconds (3 hours, 4 minutes,
47.19 seconds) with maximum resident set size 30,670,848 bytes. These timing
figures are operational observations, not a sealed validation receipt.

The observed `/usr/bin/time` result for the full build and source replay was
40,297.68 seconds (11 hours, 11 minutes, 37.68 seconds), with maximum resident
set size 346,226,688 bytes. These operational observations are not fields in a
sealed run receipt. Filesystem inspection also observed about 70 GiB of
temporary replay state. The large runtime is an implementation limitation, not
a corpus or model limitation: the current verifier performs tens of millions
of row-at-a-time SQLite inserts, updates, lookups, and deletes.

The next verifier revision should preserve all guarantees while replacing full
payload storage and per-row mutations with compact occurrence/document digests,
batched writes, and bulk anti-joins. It should then remove duplicate JSON parses
and parallelize replay by receipt-fenced Parquet row group. Sampling, skipping
structured-only rows, or trusting an un-replayed builder receipt are not
acceptable optimizations.

## Scenario-blindness and limitations

The generic build dependency graph contains only the admitted typed snapshot,
the generic typed-field projection policy, generic evidence schemas, and SDK
pointer contracts. The executable generic builder/projector/source modules and
their bound policy/contracts contain no cloud-hunt or BOTS answer literals,
fact IDs, actor names, expected API operations, or benchmark query artifacts.
The wider repository intentionally contains separate evaluator fixtures and
prototype reports, but they are not loaded by the generic build or verifier.
Known evidence was not used to select relations, documents, fields, thresholds,
groups, or dispositions.

Exact attributes are an explicitly bounded typed subset for local predicates,
not a source replica. Any omitted, unsafe, oversized, or non-portable field is
accounted for and requires source hydration. The immutable pointer remains the
authority.

The next phase is to add versioned metric/state/network derivations, canonical
Parquet materialization, embeddings for every searchable document, lexical
inputs, the final evidence-index manifest, and an `evidence.search` provider
bundle tested through `livefire-sdk`. Retrieval quality and investigation
effectiveness are evaluated only after that index is sealed; they cannot alter
this projection pack.
