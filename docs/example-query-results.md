# Prototype query results

Historical M21/OpenBOTS experiment. These results are retained as evidence of
the early prototype and must not be presented as M45 search results.

Status: **ad-hoc prototype; informational only**

Conformance: **incomplete**
Run date: 2026-08-11

This report records the first semantic-retrieval run against real BOTS v3 data. It
is intended to demonstrate the investigations the RAG tool should support and to
expose design defects before implementing the production adapter, immutable
index, and LiveFire tool provider. It is not a model-quality evaluation.

## Run boundary

The prototype combined:

- typed process and API activity from the completed M21 OCSF snapshot;
- exact Sysmon process command lines, PowerShell 4104 script blocks, and Bash
  history hydrated from the authority-linked OpenBOTS source data;
- deterministic semantic projections and semantic deduplication;
- Qwen3-Embedding-8B Q4_K_M served locally by LM Studio;
- 4,096-dimensional float32, L2-normalized vectors;
- an exact cosine scan with float64 accumulation and round-half-even distance
  millionths.

The source scan read 227,278 observations and produced 3,806 semantically unique
documents:

| Document kind | Unique documents | Source observations |
|---|---:|---:|
| M21 process command | 1,279 | 214,922 |
| M21 API activity | 1,891 | 8,592 |
| Source Sysmon process command | 572 | 3,616 |
| Source PowerShell 4104 script block | 29 | 48 |
| Source Bash history | 35 | 100 |

The corpus contained six command variants for which bounded static PowerShell
Base64 decoding succeeded. Corpus embedding took 823.5 seconds (4.62 documents
per second). Embedding all nine queries in one request took 1.11 seconds. Exact
scans of 3,806 vectors took approximately 2.9-8.9 ms per query. Vector norms were
within `0.999999943` and `1.000000061`.

## Results

Every query requested the top 20. The table summarizes the most important ranks;
distance is cosine distance, so lower is closer.

| ID | Investigation question | Observed result |
|---|---|---|
| Q1 | Encoded or obfuscated PowerShell disabling logging or executing registry content | The logging-bypass 4104 script blocks occupied ranks 1-5; rank 1 distance `0.238528`. Case randomization and reflection did not prevent retrieval. |
| Q2 | SYSTEM scheduled-task persistence launching hidden PowerShell from the registry | The `schtasks /Create ... /RU system /TN Updater` command ranked 1 (`0.354978`); its registry-backed PowerShell payload ranked 2. |
| Q3 | PowerShell-spawned local service/VNC account creation | Account creation ranked 1-2 and administrator-group addition ranked 3-4; rank 1 distance `0.270487`. |
| Q4 | PowerShell-spawned scanning or Windows Firewall disablement | Firewall disablement ranked 1 (`0.349727`) and the `hdoor` address-range scan ranked 2 (`0.427442`). Logging-bypass PowerShell at ranks 3-5 was related defense evasion but not a complete match. |
| Q5 | Linux commands uploading the Frothly archive to S3 | Exact Bash upload commands and their M21 process projections occupied ranks 1-4; rank 1 distance `0.273492`. An `scp` staging command ranked 5. |
| Q6 | Large EC2 fleet launches across regions and their failures | `RunInstances` failures by `web_admin` occupied ranks 1-20 across regions. The first result was `UnauthorizedOperation` at `0.395710`. Retrieval found the candidates; aggregation is still required to answer counts and failure distribution. |
| Q7 | Denied IAM user/access-key manipulation | Denied `CreateAccessKey` and `CreateUser` ranked 1-2 (`0.321795`, `0.342531`). A successful `ListAccessKeys` ranked 5, illustrating that action similarity does not enforce result polarity. |
| Q8 | Public S3 ACL followed by access tightening | The public `PutBucketAcl` event ranked 1 (`0.292628`) and the later owner/log-delivery-only ACL ranked 2 (`0.298432`). Retrieval found both states; time ordering is a structured query responsibility. |
| Q9 | Cross-source archive staging followed by an ACL change | The ACL changes ranked 1-2, but no Bash upload appeared in the top 20. Dense retrieval failed to assemble both halves of the multi-step question. |

Two command-to-command searches were also run:

- An obfuscated logging-bypass 4104 script retrieved four differently cased
  variants at distances `0.005972`-`0.015059`, followed by an encoded process
  command at `0.081896`.
- A denied `CreateAccessKey` event retrieved denied `CreateUser`,
  `ListAccessKeys`, and `DeleteAccessKey` activity at distances
  `0.082730`-`0.099289`. This is a useful investigation neighborhood, but it
  includes different actions and statuses and must not be presented as semantic
  equivalence.

## What this demonstrates

Dense retrieval is already useful for analyst-language-to-evidence candidate
generation. It handled vocabulary mismatch, command syntax, PowerShell case
obfuscation, decoded content, and cloud action/resource descriptions well in
these examples.

It is not the complete investigation engine:

- A semantic result is a candidate, not admitted evidence. LiveFire still owns
  hypothesis management, evidence hydration, verification, and conclusions.
- Temporal questions need structured filters, joins, grouping, and ordering after
  retrieval. Q6, Q8, and especially Q9 demonstrate this boundary.
- `top_n` always returns something. Thresholds cannot be chosen from these nine
  examples; they require reviewed positives, hard negatives, and held-out qrels.
- Retrieval distance is not an anomaly score. User-history and population-history
  novelty need strict-prior materialized baselines and the separate four-component
  scoring policy.
- A single dense vector favors one facet of a composite query. Q9 argues for tool
  planning that decomposes a question into multiple retrievals or combines dense
  retrieval with structured/BM25 candidates before evidence correlation.

## Defects found by the run

The first pass allowed a service-account password from a source command into an
embedding projection and preview. The report was sanitized, the generated vector
cache was deleted, and the prototype redactor now handles both common `net user`
argument orders. This run is therefore permanently non-admissible. Production
admission must include secret fixtures and prove redaction occurs **before** model
input construction; display-only redaction is insufficient.

M21 also does not yet promote all exact command/script text into typed semantic
fields. The prototype had to hydrate raw Sysmon, PowerShell, and Bash content.
The production source adapter must perform this work once, account for every
candidate/exclusion, and seal canonical command records. The RAG builder must not
reach back into OpenBOTS or source-specific authority data.

## Reproduction and artifacts

Run the prototype with the loaded LM Studio profile:

```sh
uv run --with duckdb --with numpy --with pytz \
  python tools/prototype_query_demo.py --top-n 20 --batch-size 32
```

The ignored local machine-readable artifact is
`reports/prototype-rag-demo/report.json`. It records all top-20 results, source
locators, exact distances, corpus/model identities, timings, history examples,
and limitations. Because the unsafe embedding cache was removed, the next run
will rebuild all vectors using the corrected pre-embedding redaction.

The next meaningful test is the sealed evaluation suite described in
`docs/evaluation-plan.md`: fixed queries and hard negatives, independently
reviewed qrels, BM25/exact-field baselines, BF16-versus-Q4 comparison, and
Recall/nDCG/MRR with confidence intervals. Until that exists, these results show
capability and failure modes, not measured quality.
