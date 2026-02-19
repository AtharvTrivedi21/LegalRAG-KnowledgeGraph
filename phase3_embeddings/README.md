# Phase 3: Embeddings and Vector Retrieval

Phase 3 builds the embedding and vector retrieval layer for LegalRAG. It chunks the legal corpus (cases, sections, articles), fine-tunes BGE on IndicLegalQA, builds a FAISS index for semantic search, and provides a validation script to verify all outputs.

## Prerequisites

1. Phase 1 completed: `phase1_output/` must contain `cases.csv`, `sections.csv`, `articles.csv`.
2. IndicLegalQA dataset: `Datasets/IndicLegalQA Dataset_10K_Revised.json` (required for fine-tuning).
3. Virtual environment with dependencies:

```powershell
cd c:\Users\ATHARV\LegalRAG
venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline Steps

### Step 1: Chunking

Chunk cases, sections, and articles into overlapping token-based chunks (~500 tokens, ~100 overlap).

```powershell
python -m phase3_embeddings.chunk_corpus
```

**Output:** `phase3_embeddings/output/chunks.pkl`

### Step 2: Fine-tuning (Required)

Fine-tune BAAI/bge-small-en-v1.5 on IndicLegalQA using MultipleNegativesRankingLoss. Evaluation runs on a 20% held-out split with MRR@10, NDCG@10, Recall@10.

```powershell
python -m phase3_embeddings.finetune_bge --epochs 2 --batch-size 32
```

**Options:**
- `--epochs N` (default: 2; lower to avoid overfitting)
- `--batch-size N` (default: 32)
- `--lr N` (default: 1e-5; lower LR reduces overfitting)
- `--output-dir PATH` (default: phase3_embeddings/models/bge-legal)

**Note:** If fine-tuned metrics regress vs. baseline (Recall@10 < 0.67), the baseline model is saved instead so build_faiss always gets a usable model.

**Output:** Fine-tuned model saved to `phase3_embeddings/models/bge-legal/`

**Evaluation metrics (satisfactory thresholds):**

| Metric    | Satisfactory | Above-Average |
| --------- | ------------ | ------------- |
| MRR@10    | ≥ 0.50       | ≥ 0.65        |
| NDCG@10   | ≥ 0.55       | ≥ 0.70        |
| Recall@10 | ≥ 0.60       | ≥ 0.75        |

The fine-tuned model must outperform the baseline (unfinetuned BGE). If metrics are below threshold, try:
- Increasing epochs (e.g. `--epochs 5`)
- Using a larger model (e.g. bge-base-en-v1.5) via config
- Adjusting learning rate

### Step 3: Build FAISS Index

Embed all chunks with the fine-tuned model and build a FAISS index.

```powershell
python -m phase3_embeddings.build_faiss
```

**Output:**
- `phase3_embeddings/output/faiss.index`
- `phase3_embeddings/output/chunk_metadata.pkl`

### Step 4: Validate Phase 3

Run the validation script to verify all Phase 3 outputs and retrieval:

```powershell
python -m phase3_embeddings.validate_phase3
```

**Checks performed:**
- Phase 1 prerequisites (cases.csv, sections.csv, articles.csv)
- Chunks: `chunks.pkl` exists, loads, and has expected structure (chunk_id, source_type, source_id, text)
- IndicLegalQA dataset present (for re-runs of fine-tuning)
- Fine-tuned model: directory exists and model loads; embedding dimension
- FAISS index and chunk_metadata: exist, load, and lengths match
- Index–model dimension match (FAISS vectors same dim as model)
- Retrieval sanity check (one query, top-k results)

Exit code 0 if all checks pass; 1 otherwise.

## Retrieval

Use the retrieval API to search for relevant chunks:

```python
from phase3_embeddings.retrieve import load_index, search

index, metadata, model = load_index()
results = search("What is Article 14?", k=5)
for r in results:
    print(r["chunk_id"], r["source_type"], r["score"])
    print(r["text"][:300])
```

Or from the command line:

```powershell
python -m phase3_embeddings.retrieve "What is Article 14?" 5
```

## Memory and GPU

- Chunking and embedding can be memory-heavy for large corpora. Batch encoding (default 64) helps.
- Fine-tuning benefits from GPU. If no GPU, training will run on CPU (slower).
- FAISS IndexFlatIP is exact search; for very large corpora (millions of chunks), consider IndexIVFFlat.

## Configuration

Edit `phase3_embeddings/config.py` to change:
- `PHASE1_OUTPUT` – path to Phase 1 CSVs
- `CHUNK_SIZE`, `CHUNK_OVERLAP` – chunking params
- `BGE_MODEL` – base model
- `METRICS_SATISFACTORY` – evaluation thresholds
