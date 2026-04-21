# Experiments — approaches and results

This document summarises each experiment, methodology, and key results for inclusion in the thesis. Use this as the canonical "detailed content" file.

## 1) System3 — FullPipelineBNS (primary system)

- Goal: Produce legal-domain answers (BNS) using retrieval-augmented generation and measure grounding and relevance.
- Input: `evaluation/results/system3_raw_results.jsonl` (100 cases)
- Retrieval: FAISS index over BNS sections; retrieved_chunks include `source_id` and `text`.
- Generation: FullPipelineBNS adapter (generation model + prompt templates).
- Manual evaluation: 100 cases scored by the selected LLM (manual read) on:
  - answer_relevance (0.0–1.0)
  - context_relevance (0.0–1.0)
  - groundedness (0.0–1.0)
  - avg_relevance (mean of answer/context)
- Outputs:
  - `evaluation/results/gpt4_eval_results_manual_100.csv` — per-case scores and one-sentence justifications
  - `evaluation/results/first10_cited_found.csv` — diagnostic table for cases 1–10
  - `evaluation/results/gpt4_citation_mismatches.csv` — cases with zero overlap between cited and retrieved sections (case 19)
- Key findings (to be expanded in thesis):
  - High prevalence of correct section overlap for most cases.
  - A small number of citation mismatches observed; manual re-check recommended.
  - Retrieve the boilerplate Section "1" appears in many retrieved chunks (metadata noise) — consider ignoring it in overlap metrics.

### Reproducibility / important paths
- Raw JSONL (per-case system outputs): `evaluation/results/system3_raw_results.jsonl`
- Manual evaluation CSV (final): `evaluation/results/gpt4_eval_results_manual_100.csv`
- Diagnostic CSVs: `evaluation/results/first10_cited_found.csv`, `evaluation/results/gpt4_citation_mismatches.csv`
- Scripts used:
  - `evaluation/scripts/find_unmatched_citations.py` — find no-overlap cases
  - `evaluation/scripts/extract_first10_cited_found.py` — extract cited vs found for first 10
  - `evaluation/scripts/auto_eval_gpt4.py` — helper for automated metrics

### Suggested next steps (for reproducibility and thesis)
1. Re-run `extract_first10_cited_found.py` and `find_unmatched_citations.py` after excluding Section "1" from retrieved sections to get cleaner overlap metrics.
2. Add a short README in `MTech Thesis/` describing the commands used to generate each figure/table and where PNG/TeX exports will be saved.
3. Manually confirm cases listed in `gpt4_citation_mismatches.csv` and add a `manual_confirm` column with 'yes'/'no' and a one-line note.

## 2) Non-finetuned baseline (BGE / non-FT index)

- Goal: Compare retrieval when using a base embedding model (non finetuned) to the finetuned index.
- Script: `evaluation/run_system3_100_nonft.py`
- Output: `evaluation/results/system3_results_100_nonft_bge.csv` and a JSONL of raw results (non-FT)
- Notes:
  - Index built from `bns_comparison/sections.csv` limited to BNS_2023 sections.
  - Use median/mean hit-rate and MRR for comparison versus primary system.

## 3) Rerank / rerank_experiment

- Goal: (If applicable) rerank retrieval candidates to improve grounding and section precision.
- Location: `rerank_experiment/` (see README)
- Outputs: experiment-specific CSVs (refer to `rerank_experiment/run_100.py`)
- Notes: Reranking can reduce fabricated_section_count and improve section_precision metrics.

## 4) Manual evaluation protocol

- Source: `evaluation/EVAL_AGENT_INSTRUCTIONS.md` (rubric: definitions of relevance, grounding, justifications)
- Procedure: The selected LLM (assistant) read each JSONL case, examined `answer`, `retrieved_chunks`, and `cited_sections`, then assigned scores and wrote a one-sentence justification per case. Evaluations were batched and saved as CSV.
- Files produced during evaluation:
  - `evaluation/results/gpt4_eval_results_cycle5_manual.csv` (cases 41–50 example)
  - `evaluation/results/gpt4_eval_results_cycle6_manual.csv` (cases 51–100 example)
  - Final consolidated `gpt4_eval_results_manual_100.csv`

## 5) Diagnostics and utilities

- Scripts created:
  - `evaluation/scripts/find_unmatched_citations.py` — finds cases with no overlap between cited and retrieved sections.
  - `evaluation/scripts/extract_first10_cited_found.py` — extracts first 10 cited vs found sections.
  - `evaluation/scripts/auto_eval_gpt4.py` — (helper) automated metric computations (if needed).
- Use these scripts to reproduce diagnostic tables and check for regressions after reranking/index changes.

## How to expand these sections for the thesis

1. For each experiment, add:
   - Detailed methodology (index creation, embedding model, retrieval settings, prompt templates)
   - Evaluation protocol (manual scoring rubric, inter-rater checks if any)
   - Quantitative results (tables and figures referenced in `tables.md` and `figures.md`)
   - Example case studies (1–3 representative cases showing good grounding and 1–2 failure modes)

2. Link to raw outputs:
   - `evaluation/results/system3_raw_results.jsonl`
   - `evaluation/results/gpt4_eval_results_manual_100.csv`
   - `evaluation/results/gpt4_citation_mismatches.csv`

---
If you want, I can now:
- Populate these MD files with graphs and exported table snippets (PNG/CSV -> embedded).
- Remove boilerplate section "1" from the diagnostic outputs and regenerate the first-10 table.

