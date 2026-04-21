# rerank_experiment

Purpose
- Experimental reranker code and notes used to improve section precision and reduce hallucinated citations.

What’s here
- `run_100.py` — example runner for reranking experiments.
- `README.md` — (existing) high-level notes.

Results
- See CSV outputs produced by `run_100.py` and comparison artifacts under `bns_comparison/results/`.

Notes
- Reranker experiments are optional; they typically post-process top-k retrieved chunks to re-order by a learned scorer.

