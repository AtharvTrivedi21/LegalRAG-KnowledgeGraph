# Tables

This file describes the tables to include in the M.Tech thesis, with purpose and source for each.

## Table 1 — Manual evaluation results (full)
- Description: "Final manual evaluation for cases 1–100 with scores and one-line justification."
- Columns: case_id, answer_relevance, context_relevance, groundedness, avg_relevance, justification
- Source file: `evaluation/results/gpt4_eval_results_manual_100.csv`

## Table 2 — Citation mismatches
- Description: "Cases where cited sections did not overlap with retrieved ('gold') sections."
- Columns: case_id, cited_sections, retrieved_sections
- Source file: `evaluation/results/gpt4_citation_mismatches.csv`

## Table 3 — First 10 cited vs found (compact)
- Description: "For initial inspection — cited_sections and retrieved (found) sections for cases 1–10."
- Columns: case_id, cited_sections, found_sections
- Source file: `evaluation/results/first10_cited_found.csv`

## Table 4 — Summary statistics
- Description: "Aggregate metrics (mean/median/std) for relevance and groundedness, counts of low-grounding cases, and hit-rate summary."
- Recommended columns: metric, mean, median, std, notes
- Source: computed from `gpt4_eval_results_manual_100.csv` and retrieval metrics CSVs

---
Notes:
- Export these CSVs to LaTeX or include as CSV snippets in the thesis appendix.
- Save generated table images (if any) under `MTech Thesis/tables/`.
 
## Reproducible commands
- Generate summary statistics (example Python/pandas):
  - python -c "import pandas as pd; df=pd.read_csv('evaluation/results/gpt4_eval_results_manual_100.csv'); print(df.describe())"

- Export Table 1 as LaTeX:
  - python - <<'PY'\nimport pandas as pd\npd.read_csv('evaluation/results/gpt4_eval_results_manual_100.csv').to_latex('MTech Thesis/tables/manual_eval_full.tex', index=False)\nPY

## Additional recommended tables (appendix)
- Per-experiment retrieval metrics (hit_rate, mrr, section_precision/recall) — source: `evaluation/results/system3_results_100_nonft_bge.csv` or exported metrics CSVs.
- Failure-mode examples: small table with case_id, short failure description, why it failed (missing retrieved sections / fabricated claims).

