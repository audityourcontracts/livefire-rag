# Investigation use cases

## Strong fits

- Rank a user's commands against that user's prior 30-day history.
- Rank commands against prior behavior across all users.
- Surface encoded or obfuscated PowerShell without executing it.
- Find unusual action/target combinations such as a familiar CLI accessing a new
  sensitive resource.
- Surface abnormal invocation shapes and process ancestry.
- Find principals newly invoking sensitive AWS or other cloud actions.
- Search command history with analyst language when exact syntax is unknown.
- Find behaviorally similar commands across shells, hosts, and source products.
- Explain why a command ranked highly using components and prior examples.

## Output interpretation

The tools answer “what is most different or most semantically relevant within
this snapshot?” They do not answer “is this malicious?” A high score is relative
to declared history and may reflect a legitimate new task. Sparse or left-censored
history reduces confidence and is always returned as coverage.

The provider returns the requested top N rather than applying an alert threshold.
Livefire decides whether to hydrate a result, compare it with other telemetry,
form a hypothesis, or dismiss it as benign.

## Bad fits

Use exact source/OCSF/SIEM tools for exhaustive retrieval, counts, authoritative
raw records, causal relationships, and negative-evidence claims. Use dedicated
numeric or delta indexes for system metrics and configuration changes. Similarity
and novelty scores are not evidence and cannot establish identity continuity.

## Future separate indexes

Playbooks, ATT&CK, OCSF documentation, prior cases, configuration deltas, and
numeric metrics have different confidentiality, lineage, deletion, and ranking
semantics. They may reuse the SDK but do not share the command index or tool
binding.
