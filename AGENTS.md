# Working style

Use clear, ordinary language in plans, progress updates, documentation, commit
messages, and handoffs. Prefer a short explanation of what a technical step
does over internal project shorthand.

When a specialist term is necessary, define it on first use. In particular:

- Say "people review the search results and mark which are relevant" before
  using "qrel review".
- Say "confirm returned event references through the released OCSF query
  service" before using a release-gate name such as "E9 hydration".
- Say "run the provider with enforced file, network, memory, and process
  limits" before using "production admission" or "sandboxing".
- Say "connect the browser UI to the server-side tool process" before using
  "browser IPC integration".

Do not use an acronym or milestone label as a substitute for the outcome it
represents. Clearly distinguish what works locally now, what has been measured,
and what still depends on another service or release.
