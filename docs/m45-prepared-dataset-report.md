# M45 prepared dataset report

Status: model-independent preparation complete on 2026-08-17. No embedding
backend was contacted and no embedding plan was created.

## Source accepted for this build

The source is the clean `livefire-ocsf` M45 run-b snapshot at:

```text
$PWD/../livefire-ocsf/data/builds/m45-progressive-disclosure-run-b
```

The source checkout was clean at commit
`0f3b6f34dce387306b5d788d83212196a7476f4a`. The prepared datasets bind:

- snapshot `23077f2605cb4d0ca7f1a857dd0c540d990911197c21a80c886fc1099f6e7d10`;
- dataset `ba9e0c1ff5f1154defc0956e1984fc1168d0424d29f8d4d6b02e1d1c93fbbe46`;
- mapping `641e479d5d830edef80c4e57c8048eed9b26710d35a18101e9441065f4337bb7`;
- relation-contract content
  `a40656d2b8e233326157a40c08a257bffe8ef2b97ca76ff62740fbef43eca549`;
- snapshot capability file
  `d9e7e485213c09abb9862f8620cebc410649bc8241688ae21c53721958493e1b`.

The upstream M45 qualification reports two byte-identical clean builds, all 28
Parquet files matching the version-4 relation contract, and complete support
closure. Its support check found zero missing references, unwitnessed declared
references, unreachable objects, or malformed edge supports.

M45 keeps all 19 normalized OCSF relation files byte-identical to M44, but it
changes the authoritative snapshot, mapping, graph, provenance, and capability
identities. The RAG preparation was therefore rebuilt instead of relabelling
the earlier M44 artifacts.

## Rust census

The source-admission-bound census is
`reports/livefire-ocsf-m45/full-census.json`, component
`8cafa5007bb4100e17c9fb41f7c9f31906f585c5e1d2653b4903ecd6e6e421be`.
Its file SHA-256 is
`c51d978a4779e23eb2eb56a6b04444171ac5f7454271b57b74b0a828981c3bfc`.
It accounted for:

| Measure | Count |
| --- | ---: |
| Normalized source events | 13,905,577 |
| Searchable documents | 560,842 |
| Retained searchable event references | 6,367,276 |
| Structured-only system-metric rows | 7,538,301 |
| Searchable relations | 18 |
| Network documents | 138,276 |
| Network event references | 1,042,076 |
| Non-network documents | 422,566 |
| Non-network event references | 5,325,200 |

System metrics are deliberately not embedded. Graph, provenance, capability,
and subject-alias objects are also not embedding inputs; they remain source
admission, exact lookup, or event-confirmation data.

## Prepared outputs

The generic one-relation datasets are under:

```text
indexes/livefire-ocsf-m45-v1/prepared/
```

Their 18 manifests close exactly to the census: 560,842 documents and
6,367,276 event references. The embedding-copy set is 18 manifests plus 287
document Parquet files: 99,939,871 bytes in total, of which 99,441,099 bytes
are document objects. The 792 occurrence Parquet files retained locally total
1,380,893,434 bytes. Accounting files add 66,173 local bytes.

| Relation | Documents | Event references | Prepared component |
| --- | ---: | ---: | --- |
| Account change | 1 | 1 | `066f157b…656a2` |
| API activity | 6,526 | 8,587 | `13c7f4aa…9a724` |
| Application lifecycle | 235 | 1,260 | `e3fee0d5…86320` |
| Authentication | 699 | 1,129 | `13ff5108…69378` |
| Cloud resource inventory | 256 | 522 | `37ab7348…351f` |
| Datastore activity | 29,600 | 36,494 | `46fdd51a…c879` |
| Detection finding | 53 | 2,240 | `43223daa…e7a6` |
| DNS activity | 76 | 115,145 | `28fbaa7b…842` |
| Email activity | 927 | 927 | `0c1d6e75…bbbe` |
| Entity management | 58 | 60 | `f32fe878…1593` |
| Event-log activity | 199,749 | 407,729 | `e25987a1…653d` |
| Configuration snapshot | 148,110 | 4,448,673 | `1b756d67…b6bcf` |
| File activity | 251 | 330 | `29f1d296…009b4` |
| HTTP activity | 12,045 | 25,114 | `7a985b5c…f13` |
| Inventory information | 2,468 | 9,643 | `0ea90928…bb6a7` |
| Network activity | 138,276 | 1,042,076 | `d6decb99…873ab` |
| Process activity | 21,471 | 267,118 | `7b6e753d…a8132` |
| User inventory | 41 | 228 | `56fc686e…8590` |

The separate command-focused projection is verified under:

```text
indexes/livefire-ocsf-m45-v1/commands-prepared/
```

It contains 2,226 documents and 227,218 retained event references. Its prepared
component is
`07e0dc40f05f574d1f50b9990df01656c065e04845243915f5f8e7822b00ba13`.
It reads only normalized M45 API, event-log, and process relations. It does not
read OpenBOTS or another raw source tree.

Its embedding-copy set is one manifest plus two document Parquet files,
414,331 bytes in total. Its 30 occurrence files total 28,566,110 bytes and
remain local.

The fixed nested backend-comparison datasets are under:

```text
indexes/livefire-ocsf-m45-v1/benchmark-v1/
```

They contain the same deterministic 512-, 2,000-, and 10,000-document selections
for later LM Studio and RunPod measurements. Their preparation does not choose
a tokenizer, model, quantization, GPU, or embedding server.

The selector admitted all 560,842 candidates and sealed 10,000 ordered
selections. Selection component:
`d0bb273af30a7a97005f4e30de4c2676ec9de09544728b4b0454d1175ae4bd18`.
The full scan took 2,735.29 seconds and reached 4,182,769,664 bytes maximum
resident memory.

| Prepared comparison set | Documents | Event references | Copy bytes: manifest + documents | Prepared component |
| --- | ---: | ---: | ---: | --- |
| 512 | 512 | 60,872 | 161,801 | `946d69b7…6c652` |
| 2,000 | 2,000 | 141,238 | 497,290 | `b6911e14…daf1` |
| 10,000 | 10,000 | 394,400 | 2,447,724 | `1c13c725…248e` |

All three prepared comparison sets passed full offline verification. The
4,954,936-byte selection manifest remains local comparison evidence; an
embedding worker needs only the chosen prepared manifest and its listed
document files.

An earlier M45 draft that did not retain the relation-contract and capability
components is preserved for audit at:

```text
indexes/livefire-ocsf-m45-v1/superseded-pre-source-admission-20260817/
```

It must not be planned, embedded, copied, or indexed.

## What to copy after choosing a backend

For every selected prepared dataset, copy its `manifest.json` and every file
listed in the manifest's `documents` array. Those are the only prepared inputs
the embedding worker needs.

Keep the files listed in `occurrences` local. They bind each grouped search
document back to all contributing M45 event references and are needed when the
final SQLite index is assembled. `accounting.json` also stays local.

Do not create an embedding plan until the backend is selected. The plan binds
the exact tokenizer and embedding profile. LM Studio Q4 vectors and RunPod FP16
vectors are separate artifact families and cannot be mixed.

## Completed preparation checks

The final preparation run:

1. verified every one-relation dataset, the command dataset, and all three nested
   comparison datasets with `rag verify-prepared`;
2. reconciled document and occurrence totals exactly with the census;
3. recorded exact document bytes to copy and occurrence bytes retained locally;
4. confirmed every current manifest carries the same M45 snapshot, mapping,
   relation-contract, and capability identities; and
5. did not create a tokenizer plan, contact an embedding service, or create an
   embedding vector.

Repository tests are reported separately because they validate the Rust and
analysis tooling rather than the generated corpus bytes.

## Tool and test result

The final release `rag` binary used for verification is 38,816,656 bytes with
SHA-256
`3391c57bd7c4e8563f56fbe439adf08cd28f5bc025a527c69f1c8a40914c1006`.

The final checks passed:

- `cargo fmt --all -- --check`;
- the complete Rust workspace test suite, with only explicitly ignored manual
  scale or local-snapshot tests skipped;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo build --release --workspace --bins`;
- 169 Python analysis, schema, packaging, comparison, and visualization tests;
  and
- Git whitespace checks and repeated full verification of all three comparison
  corpora.

Generated `indexes/` and `reports/` paths remain ignored by Git. No generated
dataset, model, vector, credential, or report was staged or committed.
