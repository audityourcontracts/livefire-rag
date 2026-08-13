# rag-builder

Native `rag` CLI for direct `livefire-ocsf` snapshot projection, resumable local
embedding, fast-index construction, inspection, and dense/lexical/fused query.
Use `--representative-sample` for the fixed scenario-blind experiment path.
It declares a census for searchable relations with at most 1,000 documents and
a 2,000-document snapshot-bound hash-min cap above that threshold. Consequently
relations with 1,001 through 2,000 documents are also fully retained; only
larger relations are reduced. A second source scan spills every occurrence for
the final selected documents, so high-fanout membership is complete without
being held in memory.
