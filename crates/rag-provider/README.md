# rag-provider

Native JSONL provider for the fast evidence index. It implements the Livefire
SDK handshake/open/call/health/close lifecycle and exposes two separately
granted tools:

- `evidence.search` embeds natural-language text through the exact bound local
  LM Studio model, then returns ranked OCSF event references.
- `evidence.similar` reads one indexed document's stored vector, makes no model
  or network call, excludes the seed by default, and returns nearby event
  references.

Each SDK session grants exactly one tool. The binding lock must name that
tool's descriptor, input and output schemas, and retrieval policy. Component
admission and production packaging remain host responsibilities.

Create local SDK loadouts with `rag-prepare-local-tool --tool search` or
`--tool similar`. Similarity additionally requires `--document-id` for its
seed. The preparer writes separate index-admission and tool-binding contracts;
it never modifies the assembled physical index.

This executable is an experimental lifecycle and retrieval harness, not an
admission or sandbox boundary. The host must mount the entire index directory
immutably for the entire session. The provider retains an open handle for the
vector file, but documents and the occurrence lookup are opened lazily by path.
Local-development digest and association checks do not prevent path replacement
or in-place mutation by another process with write access. Do not use a
caller-writable index directory for a provider session; production hosts must
enforce a read-only/immutable mount externally.

`limits.memory_bytes` is accepted and validated for SDK binding-lock
compatibility but is enforced by the host process sandbox, not by this direct
executable. Request, result, wall-time and candidate limits are enforced here.
