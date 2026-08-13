# rag-provider

Native JSONL provider for the fast experimental evidence index. It implements
the Livefire SDK handshake/open/call/health/close lifecycle and returns
snapshot-scoped OCSF event references for authoritative hydration. Component
admission and production packaging remain host responsibilities.

This executable is an experimental lifecycle and retrieval harness, not an
admission or sandbox boundary. The host must mount index bytes immutably for the
entire session. The provider pins open file handles and validates source and
cross-artifact associations, but it cannot prevent another process with write
access from modifying the same inode after `open`. Do not use a caller-writable
index directory for a provider session.

`limits.memory_bytes` is accepted and validated for SDK binding-lock
compatibility but is enforced by the host process sandbox, not by this direct
executable. Request, result, wall-time and candidate limits are enforced here.
