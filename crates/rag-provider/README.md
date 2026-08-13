# rag-provider

Native JSONL provider for the fast experimental evidence index. It implements
the Livefire SDK handshake/open/call/health/close lifecycle and returns
snapshot-scoped OCSF event references for authoritative hydration. Component
admission and production packaging remain host responsibilities.

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
