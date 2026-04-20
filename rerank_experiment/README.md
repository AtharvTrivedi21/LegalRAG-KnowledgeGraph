# Rerank Experiment (Root-Level)

This folder is a standalone experiment package and does **not** modify `bns_comparison`.

## What it adds

- `adapter.py`: System 3 variant with:
  - dense retrieval (existing FAISS + embedding model),
  - cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`),
  - strict citation filtering to only allow sections present in retrieved chunks.
- `run_100.py`: independent 100-case runner with separate output files.

## Run

```bash
python -m rerank_experiment.run_100
```

Optional:

```bash
python -m rerank_experiment.run_100 --start 1 --end 100 --raw-results evaluation/results/system3_raw_results_rerank_exp.jsonl --metrics-csv evaluation/results/system3_results_100_rerank_exp.csv
```

## Notes

- Uses current `bns_comparison` FAISS index and configured embedding model path.
- Respects Groq/Ollama backend auto-selection from existing `_ollama.py` helper.
- Designed for A/B comparison with:
  - `evaluation/results/system3_results_100_baseline.csv`
  - `evaluation/results/system3_results_100_groq_ft.csv`

