# Agent guidance

## Communication

Use clear, direct language that a technically informed reader can understand without knowing this project's internal vocabulary.

- Lead with the practical outcome: what works, what does not, and what the user can do next.
- Prefer ordinary words over acronyms, milestone codes, and governance terminology.
- If a precise technical term is necessary, explain it immediately in plain language.
- Do not compress several unfinished items into a dense phrase. State each remaining task and why it matters.
- Distinguish clearly between a working local prototype, something integrated into Livefire, and something ready for production.
- Describe tests in terms of what they proved. Do not imply that passing plumbing tests proves search quality.
- Keep status updates concise, but never make them cryptic.

Examples:

- Instead of "blinded qrel adjudication," say "people still need to review the search results without knowing which search method produced them, then mark which results are relevant."
- Instead of "qualified E9 hydration," say "we still need to test that each returned event reference can be opened through the final OCSF data service."
- Instead of "production admission and sandboxing," say "before production use, the tool still needs formal approval and operating-system limits on its file, network, and memory access."
- Instead of "browser IPC integration," say "the browser interface still needs a safe way to send requests to the local RAG service."

When reporting completion, use a short structure such as:

1. What is ready now.
2. What was tested.
3. What remains, in plain language.
4. Whether the remaining work blocks local use, Livefire use, or production use.
