# Fixtures

Fixtures will contain small synthetic OCSF snapshots, projection goldens, query
requests, relevance judgements, benign decoys, and corrupt/mismatched manifests.
No customer telemetry, evaluator secrets, model weights, or prebuilt indexes are
committed here.

`fact-evidence-coverage-plan.v1.json` is an answer-free planning inventory of
the 23 deep-cloud and 53 BOTS scored atoms. It records preliminary eligibility
against the current command/script/cloud-action index without including answer
values or strict-match literals.

`fact-evidence-synthetic/` is a five-query, fully synthetic conformance suite
for the standalone fact-to-evidence metric calculator. It includes graded
qrels, matched hard negatives, a weak baseline, an improved candidate, and
promotion gates. Its sealed eligibility ledger and independently counted
candidate-universe receipts exercise denominator and top-N closure. It is not a
quality benchmark and does not contain BOTS answer material.
