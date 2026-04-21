# System Names and Descriptions

Use these names consistently in reports, tables, and discussions.

## S1 — Legacy Baseline RAG

- **Description:** Legacy System 3 snapshot used as historical baseline in this repo.
- **Embedding setup:** Prior configured embedding/index stack from archived run.
- **Primary files:**
  - `evaluation/results/system3_results_100_baseline.csv`
  - `evaluation/results/system3_raw_results_baseline.jsonl`

## S2 — Groq-Synthetic FT RAG (Best Validated)

- **Description:** Full System 3 pipeline with Groq-synthetic fine-tuned BGE embeddings.
- **Embedding setup:** Fine-tuned embedding model (`bge-legal-bns-groq`) + FAISS retrieval.
- **Primary files:**
  - `evaluation/results/system3_results_100_groq_ft.csv`
  - `evaluation/results/system3_raw_results_groq_ft.jsonl`

## S3 — Base-BGE Non-FT RAG

- **Description:** Same pipeline as S2 but using non-fine-tuned base BGE embeddings.
- **Embedding setup:** `BAAI/bge-small-en-v1.5` (no fine-tuning), separate non-FT FAISS index.
- **Primary files (clean):**
  - `evaluation/results/system3_results_100_nonft_bge_dedup.csv`
  - `evaluation/results/system3_raw_results_nonft_bge_dedup.jsonl`
- **Retry log files (non-clean, append/resume history):**
  - `evaluation/results/system3_results_100_nonft_bge.csv`
  - `evaluation/results/system3_raw_results_nonft_bge.jsonl`

## S4 — Reranker-Augmented RAG (Experimental)

- **Description:** Dense retrieval plus cross-encoder reranking and citation constraints.
- **Embedding setup:** Current configured embedding model + reranker.
- **Status:** Full 100-case benchmark pending final artifact.
- **Primary files:**
  - `evaluation/results/system3_raw_results_rerank_exp.jsonl` (partial/ongoing)
  - `evaluation/results/system3_results_100_rerank_exp.csv` (target full output)
  - `evaluation/results/system3_results_100_rerank_smoke.csv` (smoke only)
