# Figures

This file lists the figures to include in the M.Tech thesis, with caption, data source, and a short recipe to generate each figure.

## Figure 1 — Retrieval overlap (cited vs retrieved)
- Caption: Overlap between sections cited in generated answers and sections found in retrieved chunks (first 100 cases).
- Data source: `evaluation/results/first10_cited_found.csv` (diagnostic) and `evaluation/results/system3_raw_results.jsonl` (full).
- Generation recipe (Python/pandas + matplotlib):
  - Load per-case cited_sections and retrieved_sections, normalize section ids, compute per-case intersection size and hit-rate (|intersection|/|cited|), then plot histogram or bar for first N cases and aggregate histogram for all cases.
  - Recommend output: `MTech Thesis/figures/fig1_overlap.png`

## Figure 2 — Groundedness distribution
- Caption: Distribution of groundedness scores assigned during manual evaluation (0.0–1.0).
- Data source: `evaluation/results/gpt4_eval_results_manual_100.csv` (groundedness column).
- Generation recipe:
  - Read CSV, plot histogram and KDE, annotate mean/median. Save as `MTech Thesis/figures/fig2_groundedness.png`.

## Figure 3 — Answer vs Context relevance heatmap
- Caption: Joint distribution heatmap of answer_relevance vs context_relevance across all cases.
- Data source: `evaluation/results/gpt4_eval_results_manual_100.csv`
- Generation recipe:
  - Bin both scores (e.g., 0.0–0.2,...,0.8–1.0), compute 2D histogram, render heatmap with counts and colorbar. Save as `MTech Thesis/figures/fig3_relevance_heatmap.png`.

## Figure 4 — Groundedness vs Avg relevance (scatter)
- Caption: Scatter showing groundedness against avg_relevance with points labelled by failure modes (low grounding).
- Data source: `evaluation/results/gpt4_eval_results_manual_100.csv` and `evaluation/results/gpt4_citation_mismatches.csv` (to highlight mismatches).
- Generation recipe:
  - Scatter plot, highlight points with groundedness<0.5 or cases in mismatches CSV. Save as `MTech Thesis/figures/fig4_grounded_vs_avg.png`.

## Figure 5 — Retrieval latency summary (optional)
- Caption: Generation / retrieval / total latency statistics (median, IQR).
- Data source: `evaluation/results/system3_raw_results.jsonl` (timings fields) or `evaluation/results/system3_results_100_nonft_bge.csv`
- Generation recipe:
  - Extract timing fields (retrieval_sec, generation_sec, total_sec) and plot boxplots.

---
Notes / reproducibility
- Scripts to reproduce all figures should live under `evaluation/figures/` (e.g. `evaluation/figures/plot_overlap.py`).
- Save generated PNGs in `MTech Thesis/figures/` and reference file names in the thesis.

