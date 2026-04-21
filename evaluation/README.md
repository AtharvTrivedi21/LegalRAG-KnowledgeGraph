# evaluation

Purpose
- Contains system outputs, manual evaluation results, and helper scripts for analyzing System3 (BNS) runs.

What’s here
- `results/` — JSONL and CSV outputs from System3 runs and manual evaluations:
  - `system3_raw_results.jsonl` — full per-case outputs (answers, retrieved_chunks, timings).
  - `gpt4_eval_results_manual_100.csv` — final manual evaluation (cases 1–100).
  - `first10_cited_found.csv`, `gpt4_citation_mismatches.csv` — diagnostics produced during analysis.
- `scripts/` — helper scripts used to extract diagnostics and compute overlaps.
- `helpers/` — utilities for processing BNS sections and fixing JSONL.

How to reproduce key artifacts
- Recreate manual-eval CSV: follow `EVAL_AGENT_INSTRUCTIONS.md` and use `system3_raw_results.jsonl` as input.
- Diagnostic scripts:
  - `evaluation/scripts/extract_first10_cited_found.py`
  - `evaluation/scripts/find_unmatched_citations.py`

Notes & results
- Manual evaluation completed and saved to `results/gpt4_eval_results_manual_100.csv`.
- One citation mismatch was identified and saved to `results/gpt4_citation_mismatches.csv`.

