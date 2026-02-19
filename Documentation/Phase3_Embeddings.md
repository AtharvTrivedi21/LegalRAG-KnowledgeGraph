# Phase 3: Embeddings and Vector Retrieval

## What

Phase 3 **chunks the legal corpus** (cases, sections, articles from Phase 1), **fine-tunes a sentence encoder** (BGE) on the IndicLegalQA Q/A dataset for legal-domain retrieval, **builds a FAISS index** over chunk embeddings, and exposes a **retrieval API** used by Phase 4. It produces: `chunks.pkl`, a fine-tuned model at `phase3_embeddings/models/bge-legal`, `faiss.index`, and `chunk_metadata.pkl`.

## How

### Data flow

1. **Chunk corpus** — Read `phase1_output/cases.csv`, `sections.csv`, `articles.csv`; tokenize with BGE tokenizer; split into overlapping chunks (CHUNK_SIZE=500, CHUNK_OVERLAP=100, STRIDE=400). Save list of `{chunk_id, source_type, source_id, text}` to `chunks.pkl`.
2. **Fine-tune BGE** — Load IndicLegalQA JSON (question/answer pairs); 80/20 train/eval split; train with MultipleNegativesRankingLoss; evaluate with InformationRetrievalEvaluator (MRR@10, NDCG@10, Recall@10, etc.). Save best checkpoint to `FINE_TUNED_MODEL_DIR`.
3. **Build FAISS** — Load `chunks.pkl` and fine-tuned model; encode all chunk texts (normalized); build IndexFlatIP (cosine similarity); save `faiss.index` and `chunk_metadata.pkl`.
4. **Retrieve** — Phase 4 calls `load_index()` and `search(query, k)` to get top-k chunks by similarity.

### Components

| Component | File | Role |
|-----------|------|------|
| Config | `phase3_embeddings/config.py` | PHASE1_OUTPUT, OUTPUT_DIR, CHUNKS_PATH, FAISS_INDEX_PATH, CHUNK_METADATA_PATH; CHUNK_SIZE/OVERLAP/STRIDE; BGE_MODEL, FINE_TUNED_MODEL_DIR; INDIC_LEGAL_QA_PATH; TRAIN_EVAL_SPLIT, METRICS_SATISFACTORY. |
| Chunking | `phase3_embeddings/chunk_corpus.py` | Reads Phase 1 CSVs; token-based chunking via BGE tokenizer; writes chunks.pkl. |
| Fine-tuning | `phase3_embeddings/finetune_bge.py` | Loads IndicLegalQA; baseline eval; trains SentenceTransformer with MultipleNegativesRankingLoss; eval each epoch; saves best model by Recall@10. |
| FAISS build | `phase3_embeddings/build_faiss.py` | Loads chunks + fine-tuned model; encodes; builds IndexFlatIP; writes faiss.index and chunk_metadata.pkl. |
| Retrieval API | `phase3_embeddings/retrieve.py` | load_index() → (index, metadata, model); search(query, k) → list of {chunk_id, source_type, source_id, text, score}. |

### Datasets used

- **Phase 1 output:** `phase1_output/cases.csv`, `sections.csv`, `articles.csv` (read by `chunk_corpus.py`).
- **IndicLegalQA:** `Datasets/IndicLegalQA Dataset_10K_Revised.json` (read by `finetune_bge.py`). Structure: list of objects with `question` and `answer`; used for training and evaluation only.

### Training setup (reproducibility)

- **Base model:** BAAI/bge-small-en-v1.5  
- **Epochs:** 2  
- **Learning rate:** 1e-5  
- **Batch size:** 32  
- **Loss:** MultipleNegativesRankingLoss  
- **Evaluator:** InformationRetrievalEvaluator (cosine; MRR@10, NDCG@10, Recall@10, accuracy@k, etc.)  
- **Train/eval split:** 80/20, RANDOM_SEED=42  
- **Best model:** Selected by `IndicLegalQA_eval_cosine_recall@10`; saved to `phase3_embeddings/models/bge-legal`.

## Why

- **Token-based chunking:** Aligns with BGE’s tokenizer and keeps chunk sizes consistent for embedding quality.
- **Fine-tuning on IndicLegalQA:** Improves retrieval for Indian legal Q/A over the base BGE model.
- **FAISS IndexFlatIP:** Exact nearest-neighbor with cosine similarity (embeddings normalized); no approximate trade-off for this corpus size.
- **Chunk metadata:** Keeps source_type/source_id/text for filtering and display in Phase 4 (e.g. graph-constrained retrieval).

## Results

Results below are from **phase3_embeddings/results.txt** (Baseline + three fine-tuning runs). Use these tables for reports and PPTs.

### Baseline (unfinetuned BGE)

| Metric | Value |
|--------|--------|
| IndicLegalQA_eval_cosine_accuracy@1 | 0.4535 |
| IndicLegalQA_eval_cosine_accuracy@3 | 0.5695 |
| IndicLegalQA_eval_cosine_accuracy@5 | 0.6135 |
| IndicLegalQA_eval_cosine_accuracy@10 | 0.6735 |
| IndicLegalQA_eval_cosine_map@100 | 0.5296 |
| **IndicLegalQA_eval_cosine_mrr@10** | **0.5233** |
| **IndicLegalQA_eval_cosine_ndcg@10** | **0.5593** |
| IndicLegalQA_eval_cosine_precision@10 | 0.0673 |
| **IndicLegalQA_eval_cosine_recall@10** | **0.6735** |

### Run 1 – Fine-tune (earlier run)

| Metric | Value |
|--------|--------|
| accuracy@1 | 0.6475 |
| accuracy@3 | 0.7620 |
| accuracy@5 | 0.7940 |
| accuracy@10 | 0.8375 |
| map@100 | 0.7170 |
| **MRR@10** | **0.7117** (delta +0.1884 vs baseline) |
| **NDCG@10** | **0.7422** (delta +0.1829 vs baseline) |
| **Recall@10** | **0.8375** (delta +0.1640 vs baseline) |

### Run 2 – Fine-tune (moderate improvement)

| Metric | Value |
|--------|--------|
| **MRR@10** | **0.5972** (delta +0.0739 vs baseline) |
| **NDCG@10** | **0.6354** (delta +0.0761 vs baseline) |
| **Recall@10** | **0.7570** (delta +0.0835 vs baseline) |

### Run 3 – Fine-tune (best run, current model)

| Metric | Value |
|--------|--------|
| accuracy@1 | 0.7135 |
| accuracy@3 | 0.8145 |
| accuracy@5 | 0.8455 |
| accuracy@10 | 0.8845 |
| map@100 | 0.7753 |
| **MRR@10** | **0.7713** (delta +0.2480 vs baseline) |
| **NDCG@10** | **0.7987** (delta +0.2394 vs baseline) |
| precision@10 | 0.0885 |
| **Recall@10** | **0.8845** (delta +0.2110 vs baseline) |

**Summary:** Run 3 is the best and corresponds to the model saved at `phase3_embeddings/models/bge-legal`. All three fine-tuned runs improve over the baseline on MRR@10, NDCG@10, and Recall@10.
