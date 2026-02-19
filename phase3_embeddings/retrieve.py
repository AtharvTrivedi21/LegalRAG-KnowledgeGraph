"""
Simple retrieval API: load FAISS index and search for top-k chunks.
"""
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_embeddings.config import (
    CHUNK_METADATA_PATH,
    FAISS_INDEX_PATH,
    FINE_TUNED_MODEL_DIR,
)


def load_index():
    """
    Load FAISS index, chunk metadata, and embedding model.
    Returns (index, metadata, model).
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Run build_faiss.py first.")
    if not CHUNK_METADATA_PATH.exists():
        raise FileNotFoundError(f"Chunk metadata not found at {CHUNK_METADATA_PATH}. Run build_faiss.py first.")
    if not FINE_TUNED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found at {FINE_TUNED_MODEL_DIR}. Run finetune_bge.py first.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNK_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))

    return index, metadata, model


def search(query: str, k: int = 5, index=None, metadata=None, model=None):
    """
    Search for top-k most similar chunks.

    Args:
        query: Search query text.
        k: Number of results to return.
        index: FAISS index (optional, loaded if not provided).
        metadata: Chunk metadata list (optional, loaded if not provided).
        model: SentenceTransformer model (optional, loaded if not provided).

    Returns:
        List of dicts: [{chunk_id, source_type, source_id, text, score}, ...]
    """
    if index is None or metadata is None or model is None:
        index, metadata, model = load_index()

    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        meta = metadata[idx].copy()
        meta["score"] = float(score)
        results.append(meta)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m phase3_embeddings.retrieve <query> [k=5]")
        sys.exit(1)

    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Query: {q}\nTop-{k} results:\n")
    for r in search(q, k=k):
        print(f"  [{r['score']:.4f}] {r['chunk_id']} ({r['source_type']})")
        text_preview = r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        print(f"    {text_preview}")
        print()
