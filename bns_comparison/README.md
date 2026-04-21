# bns_comparison

Purpose
- Tools and adapters for building FAISS indexes over BNS sections and comparing retrieval baselines.

What’s here
- `build_bns_faiss.py` / `faiss_bns_only` — scripts and indexes for section-level retrieval.
- `adapters/` — adapters for System3 / FullPipeline integration.
- `results/` — comparison CSVs (e.g., `comparison_results.csv`, `comparison_system1.csv`).

How to reproduce
- Build index: `python bns_comparison/build_bns_faiss.py` (reads `bns_comparison/sections.csv`).
- Run comparisons: `python bns_comparison/run_comparison.py` (outputs `results/`).

Notes & results
- Contains both finetuned and non-finetuned index artifacts (see `faiss_bns_only_nonft/`).
- Use these artifacts to assess hit-rate, MRR, and section-level precision.

