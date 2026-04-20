# Scripts, Paths, and Artifacts

Reference tables for reproduction and thesis appendices. Paths are from repo root.

## Core System 3 (BNS comparison)

| Item | Path / name |
|------|-------------|
| Test cases (100) | `bns_comparison/test_cases.py` |
| Full pipeline adapter | `bns_comparison/adapters/full_pipeline_bns.py` |
| LLM + Groq helpers | `bns_comparison/adapters/_ollama.py` |
| Shared config | `bns_comparison/config.py` |
| Metrics | `bns_comparison/metrics.py` |
| FAISS build | `bns_comparison/build_bns_faiss.py` |
| FAISS index | `bns_comparison/faiss_bns_only/faiss.index` |
| Chunk metadata | `bns_comparison/faiss_bns_only/chunk_metadata.pkl` |
| BNS sections CSV | `phase1_output_v2/sections.csv` |

## Evaluation

| Item | Path |
|------|------|
| 100-case runner | `evaluation/run_system3_100.py` |
| Default raw results | `evaluation/config.py` → `SYSTEM3_RAW_RESULTS` (overridable) |
| Baseline metrics CSV (archived name) | `evaluation/results/system3_results_100_baseline.csv` |
| Groq fine-tune run metrics | `evaluation/results/system3_results_100_groq_ft.csv` |
| Raw JSONL (groq ft) | `evaluation/results/system3_raw_results_groq_ft.jsonl` |

**Custom output paths (resume-safe):**

```text
python evaluation/run_system3_100.py --raw-results evaluation/results/<name>.jsonl --metrics-csv evaluation/results/<name>.csv
```

Resume behavior: completed `case_id` values in the JSONL are skipped.

## Embedding fine-tuning

| Item | Path |
|------|------|
| Main training script | `phase3_embeddings/finetune_bge.py` |
| Embedding config (model name, output dir, etc.) | `phase3_embeddings/config.py` |
| Template synthetic JSONL | `phase3_embeddings/bns_synthetic_pairs.jsonl` |
| Groq synthetic JSONL | `phase3_embeddings/bns_groq_synthetic_pairs.jsonl` |
| Mapping pipeline pairs | `phase3_embeddings/bns_mapping_pipeline/bns_mapping_pairs.jsonl` |
| Complaint pairs | `phase3_embeddings/dataset_experiments_v2/datasets/complaint_bns_pairs.jsonl` |
| Complaint hard-negative triplets | `phase3_embeddings/dataset_experiments_v2/datasets/complaint_bns_hardneg_triplets.jsonl` |

**`--dataset` modes** (see docstring in `finetune_bge.py`):  
`indiclegal`, `bns_synthetic`, `combined`, `bns_mapping`, `bns_groq_synthetic`, `combined_groq`, `complaint_bns`, `complaint_bns_hardneg`.

**Losses:** MultipleNegativesRankingLoss for pairs; TripletLoss where triplets are loaded.

## Synthetic / data generation scripts

| Script | Purpose |
|--------|---------|
| `phase3_embeddings/generate_template_synthetic.py` | Template-only queries per section |
| `phase3_embeddings/generate_groq_synthetic.py` | LLM-generated citizen incidents + positives |
| `phase3_embeddings/bns_mapping_pipeline/build_mapping_pairs.py` | IPC→BNS mapping → JSONL pairs |
| `phase3_embeddings/dataset_experiments_v2/build_complaint_training_data.py` | HF complaint dataset → pairs + triplets |

## Utilities

| Script | Purpose |
|--------|---------|
| `phase3_embeddings/check_groq_keys.py` | Verify each configured Groq key responds |

## Reranker experiment (standalone package)

| Item | Path |
|------|------|
| Package | `rerank_experiment/` |
| Adapter | `rerank_experiment/adapter.py` |
| 100-case runner | `rerank_experiment/run_100.py` |
| Package README | `rerank_experiment/README.md` |
| Example raw output | `evaluation/results/system3_raw_results_rerank_exp.jsonl` |
| Smoke metrics CSV | `evaluation/results/system3_results_100_rerank_smoke.csv` |

**Run:**

```text
python -m rerank_experiment.run_100 --raw-results evaluation/results/system3_raw_results_rerank_exp.jsonl --metrics-csv evaluation/results/system3_results_100_rerank_exp.csv
```

## Environment variables (typical)

| Variable | Role |
|----------|------|
| `GROQ_API_KEY` | Single-key fallback |
| `GROQ_API_KEY_1` … `GROQ_API_KEY_4` | Rotating pool (see `config.py`) |
| `RERANK_DEVICE` | `auto` (default) uses CUDA for cross-encoder if available |
| `SYS3_FAST_MODE` | Optional fast path in rerank adapter (see code) |

Secrets live in `.env` (not committed).

## Ignored / local-only artifacts

See root `.gitignore`: model weights (`*.safetensors`, `*.bin`, checkpoints), large FAISS files under certain paths, backup result folders, etc. Local fine-tuned trees such as `phase3_embeddings/bge-legal-bns-groq/` may be ignored—rebuild from training + `build_bns_faiss.py` on a new machine.
