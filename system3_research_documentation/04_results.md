# Quantitative Results (100-case benchmark)

All figures below are **row-wise means** over `n=100` cases in the exported CSVs (computed with pandas on the same machine as the development logs). Slight floating-point display differences are normal.

## Baseline vs Groq-supervised embedding (primary comparison)

| Metric | Baseline `system3_results_100_baseline.csv` | Groq FT `system3_results_100_groq_ft.csv` | Delta |
|--------|---------------------------------------------|-------------------------------------------|-------|
| `hit_rate` | 0.460 | 0.590 | **+0.130** |
| `mrr` | 0.192785 | 0.345011 | **+0.152226** |
| `section_precision` | 0.130191 | 0.158842 | **+0.028651** |
| `section_recall` | 0.249831 | 0.321996 | **+0.072165** |
| `section_f1` | 0.159828 | 0.203234 | **+0.043406** |
| `grounding_score` | 0.745879 | 0.796440 | **+0.050561** |

**Interpretation (short):**

- **Retrieval ranking improved strongly** (`mrr`, `hit_rate`), consistent with better query–section alignment from Groq-style training pairs.
- **Section extraction improved** across precision, recall, and F1—still modest in absolute terms, which matches the user’s assessment that quality is “not good enough” yet.
- **Grounding increased**, suggesting answers were somewhat more anchored to retrieved material under the new embedding + same generation stack.

## Other CSVs in `evaluation/results/`

| File | Role |
|------|------|
| `system3_results_100.csv` | Often the “current default” run; compare filenames before treating as baseline |
| `system3_results_100_rerank_smoke.csv` | Tiny smoke test (not statistically meaningful) |
| `backups/` | Timestamped copies of important runs—use if primary CSV was overwritten |

## Reranker experiment

A **full 100-case mean table** for `system3_results_100_rerank_exp.csv` should be filled in after the run completes end-to-end. Partial raw logs: `evaluation/results/system3_raw_results_rerank_exp.jsonl`.

**To reproduce aggregation:**

```text
python -c "import pandas as pd; df=pd.read_csv('evaluation/results/system3_results_100_rerank_exp.csv'); print(df.mean(numeric_only=True))"
```

(From repo root, with venv activated.)

## Complaint-based fine-tuning

No stable final CSV was promoted to `evaluation/results/` as the new best; internal eval and/or 100-case checks showed **regression** versus the Groq-synthetic model. Treat complaint runs as **experimental / negative results** unless a cleaned subset is introduced later.
