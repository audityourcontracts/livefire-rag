# Command index and scoring policy

## Canonical projection

Each command, script-block, or cloud-action record becomes a deterministic
`command_document.v1` containing:

```text
interpreter/shell
executable or cmdlet/API action
normalized action tokens
normalized target/resource tokens
argument shape with secrets and unstable IDs typed or redacted
PowerShell AST shape and bounded decoded layers
parent executable and bounded process ancestry
principal/host/source metadata for filtering, not semantic identity
```

Original and decoded command text remain retrievable preview fields, subject to
the telemetry policy. Exact timestamps, principal IDs, event IDs, hashes, and
opaque identifiers are metadata rather than dense semantic content.

## Static decoding

The builder attempts, in order and only when positively identified:

1. Base64 as UTF-8 and UTF-16LE.
2. URL percent encoding and recognized escape sequences.
3. gzip or deflate after header/format validation.
4. Safe constant-string concatenation.
5. PowerShell backtick and case normalization.

It stops after three successful layers or the configured expanded-byte bound.
Each transformation records type, input/output digests, sizes, success/failure,
and truncation. The builder never evaluates a variable, substitution, macro,
script block, command, or decompressed payload.

PowerShell text is parsed natively with Microsoft's parser during indexing. The
index stores only `powershell_ast_document.v1`: stable node kinds, invocation
shapes, literals classified by kind, flags, nesting, and parse diagnostics.
Microsoft object layouts and source-code execution are outside the contract.

## Chronological baselines

For candidate command `c` at time `t`, eligible history is strictly earlier than
`t` and no older than 30 days:

```text
principal history  = commands where (principal namespace, principal ID) == c.principal key
population history = commands from all principals in the bound tenant/scope
```

Equal-timestamp commands are not history for one another. OpenBOTS spans about
one day, so the same rule naturally means all earlier data. Snapshot time bounds
do not change the rule; history missing before the snapshot start is reported as
left-censored coverage.

Scores are materialized in chronological order and never recomputed using events
that arrived later in the snapshot.

The comparison-universe component binds the tenant/scope digest, identity and
population policies, and the exact source-snapshot set. Admission rejects a
command index if its base manifest, universe, and command manifest do not name
the same source snapshots. Principal filters and results always use the
structured `(namespace, id)` key; a bare vendor ID is never a principal key.

## Four score components

Every available component is reported as integer millionths from 0 to 1,000,000,
together with its raw distance, calibration population, and comparison count:

1. **Action novelty** — rarity/distance of executable, cmdlet, operation, or API
   action semantics in the selected history.
2. **Target novelty** — rarity/distance of paths, resources, services, endpoints,
   arguments, and target roles.
3. **Structural novelty** — distance of argument/AST shape, parent/ancestry shape,
   and invocation structure.
4. **Obfuscation novelty** — distance of decode layers, encoding indicators,
   entropy/escape features, dynamic-expression flags, and parse irregularities.

The first experiment policy uses an equal-weight integer mean of available
calibrated components. The protocol permits another explicitly versioned weight
vector whose four parts-per-million values sum to 1,000,000. Missing components
are not zero. A result carries explicit per-component availability and
`insufficient_history` when calibration is not meaningful. Changing feature
definitions, calibration, minimum history, or weights creates a new score-policy
identity and index snapshot.

Scores are relative ranking measures, not probabilities of compromise.

## Materialized explanations

For each candidate and comparison scope, store:

- baseline window and count;
- component and combined scores;
- coverage/cold-start state;
- nearest prior command IDs for each component;
- exact distances and the features responsible for deterministic components;
- parser/decode signals;
- source pointers.

This allows `cli.explain` to be fast, deterministic, and model-free.

## Retrieval

`cli.search` embeds a natural-language investigation query and performs exact
cosine search over command documents after applying closed principal, host,
source, and time filters. `cli.similar` uses one stored command vector as the
query and excludes the seed by default.

`cli.outliers` reads the materialized score table. It returns the requested top N
scored candidates, followed by explicitly unscored/cold-start candidates only if
needed to fulfill N. Stable ties use `(score desc, event_time asc, command_id
asc)`. No minimum score is accepted in v1.
