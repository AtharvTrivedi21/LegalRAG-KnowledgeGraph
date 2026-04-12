# sys4 — isolated System 4 (LQ-RAG-style + Neo4j graph)

This folder is self-contained: it does **not** change `bns_comparison` defaults or `requirements.txt` at the repo root. Systems 1–3 keep working exactly as before via `python -m bns_comparison.compare_one`.

## What it is

- Hybrid retrieval: BM25 + dense FAISS (RRF) + cross-encoder re-ranking  
- Neo4j: filter sections to `BNS_2023`, expand via `REFERENCES`, citation counts on `CITES`  
- Self-evaluation loop (Ollama `llama3:8b`) to improve grounding / IPC hygiene  

## Setup

From the project root (with venv activated):

```powershell
pip install -r sys4/requirements.txt
```

Same prerequisites as the main pipeline: BNS FAISS + fine-tuned BGE (`python -m bns_comparison.build_bns_faiss --system bge`), Ollama with `llama3:8b`, optional Neo4j for graph features.

## Commands

**Single query (sys4 only):**

```powershell
python -m sys4.run_compare_one
python -m sys4.run_compare_one --query "Someone stole my phone"
```

**Full 10-case benchmark (CSV under `sys4/results/`):**

```powershell
python -m sys4.run_benchmark
```

**Optional: one query vs systems 1–3 and sys4** (imports existing adapters; does not edit them):

```powershell
python -m sys4.compare_with_baselines --query "..." --systems 1,2,3,4
```

## Files

| File | Role |
|------|------|
| `lqrag_adapter.py` | `LQRAGAdapter` (implements `bns_comparison.adapters.base.BaseAdapter`) |
| `graph_helpers.py` | Neo4j Cypher helpers |
| `run_compare_one.py` | Smoke test one query |
| `run_benchmark.py` | 10 cases → `results/comparison_sys4.csv` |
| `compare_with_baselines.py` | Side-by-side with systems 1–3 |
| `requirements.txt` | Extra deps (`rank-bm25`) |
