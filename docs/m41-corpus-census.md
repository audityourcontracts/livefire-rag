# M41 corpus census

Historical measured evidence. This report is preserved unchanged in meaning so
the completed local M41 run remains auditable. M41 is not accepted by the
active builder path; current corpus identities and status are in
[`runpod-embedding.md`](runpod-embedding.md).

This report fixes the local embedding target used by the scale work. It was
produced by the read-only Rust `rag census` command from the accepted M41 OCSF
snapshot. It counts projected document groups; it does not create embeddings or
claim search quality.

## Bound inputs

| Input | Identity |
|---|---|
| OCSF snapshot | `botsv3-ocsf-normalized-snapshot@41` / `5a93d30760a51a866a9411b53d985fc7b39eaf7e5248a2dde7ae184dc824d49c` |
| Mapping | `botsv3-ocsf-m41@1` / `caaf0e47d07bdfe022757cd0bab64e2460122507d36b5228a85874d2e473512f` |
| Projection policy | `livefire.rag.generic-evidence-projection-policy@2` / `426a3543df1c11990bfdf32f269da25808fda65586ecd855d07192a60f375acd` |
| Census report | `5237465413ad56e382639a4e2746364c4e594c95c7392328f10e84bbe15c1deb` |
| Document order | `660feeaba83d9b2e26a5b69ba46cb0375b70fb17e1d1db1db5963acce36fdc39` |
| Report file SHA-256 | `3be9ccda6a0ec56339104a8ce10788cbba46eee4dd221497fb3bb584cd1ca95c` |

## Totals

| Measure | Count |
|---|---:|
| Source rows | 13,905,577 |
| Searchable event references | 6,367,276 |
| Structured-only metric rows | 7,538,301 |
| Searchable document groups | 560,842 |
| Raw-network document groups | 138,276 |
| Non-network document groups selected for embedding | 422,566 |
| Non-network event references | 5,325,200 |

The document kinds are 409,914 activity groups, 150,875 state groups, and 53
detection groups.

## Per-relation counts

| Relation | Source rows | Searchable references | Structured only | Documents |
|---|---:|---:|---:|---:|
| `ocsf_api_activity` | 8,592 | 8,592 | 0 | 6,531 |
| `ocsf_application_lifecycle` | 1,260 | 1,260 | 0 | 235 |
| `ocsf_authentication` | 1,125 | 1,125 | 0 | 695 |
| `ocsf_cloud_resources_inventory_info` | 522 | 522 | 0 | 256 |
| `ocsf_datastore_activity` | 36,494 | 36,494 | 0 | 29,600 |
| `ocsf_detection_finding` | 2,240 | 2,240 | 0 | 53 |
| `ocsf_dns_activity` | 115,145 | 115,145 | 0 | 76 |
| `ocsf_email_activity` | 927 | 927 | 0 | 927 |
| `ocsf_entity_management` | 60 | 60 | 0 | 58 |
| `ocsf_event_log_activity` | 407,729 | 407,729 | 0 | 199,749 |
| `ocsf_ext_livefire_configuration_snapshot` | 4,448,673 | 4,448,673 | 0 | 148,110 |
| `ocsf_ext_livefire_system_metric` | 7,538,301 | 0 | 7,538,301 | 0 |
| `ocsf_file_activity` | 330 | 330 | 0 | 251 |
| `ocsf_http_activity` | 25,114 | 25,114 | 0 | 12,045 |
| `ocsf_inventory_info` | 9,643 | 9,643 | 0 | 2,468 |
| `ocsf_network_activity` | 1,042,076 | 1,042,076 | 0 | 138,276 |
| `ocsf_process_activity` | 267,118 | 267,118 | 0 | 21,471 |
| `ocsf_user_inventory` | 228 | 228 | 0 | 41 |

## Why the earlier Rust count was wrong

The earlier Rust estimate was 638,216 documents in total and 505,835 after
excluding network data. The projection normalized camel-case keys with the
replacement `$1_$2`. Rust interpreted `$1_` as a capture name, dropping the
character before the capital letter. Important keys such as `hostIdentifier`
and `calendarTime` therefore stopped acting as identity and time fields, which
split equivalent events into too many document groups. The replacement is now
`${1}_${2}`, with shared Python/Rust golden tests.

The useful historical comparison is the Python M21 projection-policy-v2 build:
560,574 documents in total and 422,550 without network data. Focused corrected
Rust censuses reproduce its 695 authentication, 12,045 HTTP, and 138,024
network document groups exactly; the process-activity count also matches, and
the same-row test below covers every searchable relation. A fresh full Rust
M21 census was not run, so the historical all-relation totals remain attributed
to the Python policy-v2 artifact. Against that artifact, the corrected M41
snapshot contains 16 more non-network groups and 268 more groups overall. The
old 1,319,974-document Python policy-v1 build is not a valid cost baseline for
the current policy.

## Reproduction status

The accepted census is bound by the component, order, and file hashes above.
Earlier serial and eight-worker runs established deterministic report ordering;
their old pre-boundary component identities are not reused for this corrected
report. A separate clean fixture supplies the concurrency measurement below.

The final implementation also divides each decoded Arrow block into ordered
row ranges when a file has too few row groups to occupy the worker pool. A
clean 24,593-row, one-row-group fixture produced exact output bytes at every
worker count. Median projection time improved from 1,792,597 microseconds with
one worker to 836,900 with four workers and 705,621 with eight workers: 2.14×
and 2.54× faster respectively. The real benchmark preparation uses this path;
the synthetic ratio is not presented as a forecast for the whole M41 scan.

The final same-row comparison covered 4,128 rows across all 17 searchable
relations. Python and Rust matched on searchability, document kind, semantic
text, facets, grouping identity, and document ID for all 4,128 rows. Focused
M21 count comparisons for the relations affected by the fixes also match the
Python policy-v2 artifact.
