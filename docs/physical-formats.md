# Physical snapshot and index profiles (normative draft v1)

Historical command-snapshot draft. The current prepared, embedding, and fast
index formats are implemented by Rust and documented in
[`portable-embedding-pipeline.md`](portable-embedding-pipeline.md).

## Command source snapshot profile

`livefire.rag.command-snapshot-profile/1` contains one or more Parquet objects
whose logical rows validate against `command-record.v1`. Objects are ordered by
artifact path. Rows are ordered by `(event_time, command_id)`; duplicate command
IDs fail admission. The snapshot profile binds the record schema component, the
Parquet writer component, and this physical profile component.

Required physical mappings are:

- timestamps: Parquet/Arrow `timestamp[us, tz=UTC]`;
- IDs, hashes, enum values, text, and JSON subdocuments: non-dictionary UTF-8;
- counts/ordinals: signed little-endian 64-bit integer;
- nullable values: Arrow validity bitmap, never a sentinel string or number;
- repeated scalar fields: standard three-level Parquet LIST;
- structured objects: named Parquet STRUCT fields in schema order;
- compression: Zstandard level 3;
- Parquet format version: 2.6, data page version 2.0.

Each artifact is hashed as exact bytes. Different admitted writer components may
produce different artifact/index identities while remaining logically compatible;
they do not pretend to be byte-identical. Cross-language conformance compares
the typed logical rows and canonical record chain, then verifies the exact bytes
declared by that build.

## Command index profile

Every object uses the same Parquet rules and these row schemas/orderings:

| Object | Row schema | Order |
|---|---|---|
| `commands.parquet` | `command-document.v1` | `(event_time, command_id)` |
| `powershell-asts.parquet` | `powershell-ast-document.v1` | `(source_sha256)` |
| `embeddings.parquet` | `embedding-row.v1` | `(purpose, command_id)` |
| `outlier-scores.parquet` | `outlier-score-row.v1` | `(comparison, command_id)` |
| `comparisons.parquet` | `comparison-row.v1` | `(comparison, component, command_id, rank)` |

Embedding vectors use Arrow fixed-size list of IEEE-754 little-endian float32 at
the policy dimension. Values must be finite, the list length must equal the row
and policy dimension, and L2 norm must be within the bound tolerance. Providers
accumulate cosine dot products in float64 as required by the manifest.

The base SDK index manifest and command manifest are a non-circular pair: the
command manifest is the top-level index selected by a tool binding and references
the admitted base manifest. The base manifest never references the command
manifest. Admission resolves both and requires their format, source-snapshot,
object-lock, and pointer-table claims to agree.

Source pointers are local. A Parquet `row_ordinal` is zero-based within its row
group. A `record_id_only` locator must resolve through the base manifest's
required `source_pointer_table`; it never invokes an adapter or vendor API.
