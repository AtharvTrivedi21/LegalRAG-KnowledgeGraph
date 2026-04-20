# System 3 Research Documentation

This folder is the **canonical narrative and reference** for BNS-only **System 3** improvement work: embedding fine-tuning, synthetic and external datasets, evaluation on a 100-case benchmark, Groq/Ollama infrastructure, and the **reranker (second-stage retrieval)** experiment.

| Document | Contents |
|----------|----------|
| [01_background_system3.md](01_background_system3.md) | What System 3 is, main code paths, metric definitions |
| [02_experiment_timeline.md](02_experiment_timeline.md) | Chronological story: what we tried, why, and outcomes |
| [03_scripts_and_artifacts.md](03_scripts_and_artifacts.md) | File paths, commands, datasets, outputs |
| [04_results.md](04_results.md) | Aggregated numbers (baseline vs best embedding run) |
| [05_infrastructure.md](05_infrastructure.md) | Groq key rotation, CUDA/PyTorch, `.gitignore`, evaluation runners |
| [06_reranker_and_future_work.md](06_reranker_and_future_work.md) | Cross-encoder experiment, status, suggested next steps |
| [07_dataset_audit_and_external_sources.md](07_dataset_audit_and_external_sources.md) | HF datasets inspected, literature pointers, complaint artifacts |

**Quick facts**

- **Benchmark:** `bns_comparison/test_cases.py` — 100 citizen-style incident descriptions with gold BNS sections.
- **Best embedding run so far (100-case eval):** fine-tuned BGE on **Groq-generated** incident queries paired with BNS section text; FAISS rebuilt; config points to `phase3_embeddings/bge-legal-bns-groq`.
- **Rerank experiment:** implemented under `rerank_experiment/`; full 100-case CSV may still be pending—see [06_reranker_and_future_work.md](06_reranker_and_future_work.md).

All paths in these docs are relative to the repository root `LegalRAG/` unless stated otherwise.
