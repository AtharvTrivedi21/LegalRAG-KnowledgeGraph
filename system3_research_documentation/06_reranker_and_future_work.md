# Reranker Experiment and Future Work

## Design

The **root-level** package `rerank_experiment/` implements a System 3 variant:

1. **Dense retrieval:** Same FAISS index and bi-encoder as the main pipeline (`FINE_TUNED_MODEL_DIR` in `bns_comparison/config.py`).
2. **Over-fetch:** Retrieve a larger candidate set (e.g. top 24) then **re-rank** with a cross-encoder.
3. **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (MS MARCO–style relevance scoring).
4. **Citation constraint:** The prompt injects an **allowed section list** derived from retrieved chunks so the LLM cannot cite BNS sections that were not in context (reduces fabricated section numbers).

Implementation details: `rerank_experiment/adapter.py` (`System3RerankAdapter`).

**Device selection:** Embedding model may stay on CPU; reranker uses CUDA if `torch.cuda.is_available()` and `RERANK_DEVICE` is `auto` (default). Override with `RERANK_DEVICE=cpu` if debugging on CPU-only torch.

## Why this direction

Embedding fine-tuning on **noisy** or **misaligned** datasets caused **regressions**. A fixed bi-encoder plus a **second-stage ranker** is a standard way to improve **MRR/precision** when dense retrieval is “in the right neighborhood” but ordering is wrong—without entangling everything in one training objective.

## Current status (check before publishing)

As of documentation time:

- `evaluation/results/system3_raw_results_rerank_exp.jsonl` may be **partial** (e.g. only a subset of 100 cases completed).
- `evaluation/results/system3_results_100_rerank_smoke.csv` is a **smoke** run, not a full benchmark.

**To complete the experiment:**

```text
python -m rerank_experiment.run_100 --raw-results evaluation/results/system3_raw_results_rerank_exp.jsonl --metrics-csv evaluation/results/system3_results_100_rerank_exp.csv
```

Then compare means against `system3_results_100_groq_ft.csv` using the same aggregation as [04_results.md](04_results.md).

## Future work (high level)

1. **Finish rerank 100-case** and record metrics next to baseline + Groq FT.
2. **Legal-domain cross-encoder** (if available) or light fine-tuning on (query, section) pairs from Groq synthetic data—MS MARCO is general English.
3. **Complaint dataset:** label cleaning, ambiguity caps, or mixing at low ratio once E1-style filters from `dataset_experiments_v2/EXPERIMENTS_DATASET_PLAN.md` are applied.
4. **External benchmarks:** ILSIC, IL-PCSR (gated)—for **generalization reporting**, not necessarily training dumps.
5. **Benchmark curation:** review gold sections for contested rows; improves interpretability of all metrics.
6. **Neo4j:** System 3 BNS-only path discussed here does not require Neo4j; if other systems in the repo use it, document dependency separately.
