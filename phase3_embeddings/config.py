"""
Configuration for Phase 3: Embeddings and Vector Retrieval.
Paths, chunk params, model names, evaluation thresholds.
"""
from pathlib import Path

# Phase 1 output (corpus CSVs)
PHASE1_OUTPUT = Path("phase1_output")

# Phase 3 output directory
OUTPUT_DIR = Path("phase3_embeddings/output")
CHUNKS_PATH = OUTPUT_DIR / "chunks.pkl"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss.index"
CHUNK_METADATA_PATH = OUTPUT_DIR / "chunk_metadata.pkl"

# Chunking parameters (token-based)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP  # 400

# Model
BGE_MODEL = "BAAI/bge-small-en-v1.5"
FINE_TUNED_MODEL_DIR = Path("phase3_embeddings/models/bge-legal")

# IndicLegalQA for fine-tuning
INDIC_LEGAL_QA_PATH = Path("Datasets/IndicLegalQA Dataset_10K_Revised.json")

# Train/eval split
TRAIN_EVAL_SPLIT = 0.8
RANDOM_SEED = 42

# Evaluation thresholds (satisfactory; target Recall@10 ~0.67 to match baseline)
METRICS_SATISFACTORY = {
    "mrr@10": 0.50,
    "ndcg@10": 0.55,
    "recall@10": 0.60,
}

# Allow saving model even if metrics don't meet thresholds (default: False)
ALLOW_SAVE_BELOW_THRESHOLD = False
