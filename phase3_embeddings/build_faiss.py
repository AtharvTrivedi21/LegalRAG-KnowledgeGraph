"""
Embed all chunks and build FAISS index.
Uses the fine-tuned BGE model. Persists faiss.index and chunk_metadata.pkl.
"""
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_embeddings.config import (
    CHUNK_METADATA_PATH,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    FINE_TUNED_MODEL_DIR,
    OUTPUT_DIR,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found. Run chunk_corpus.py first.")
        sys.exit(1)

    if not FINE_TUNED_MODEL_DIR.exists():
        print(f"Error: Fine-tuned model not found at {FINE_TUNED_MODEL_DIR}")
        print("Run finetune_bge.py first.")
        sys.exit(1)

    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks")
    texts = [c["text"] for c in chunks]

    print(f"Loading model from {FINE_TUNED_MODEL_DIR}...")
    model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))

    batch_size = 64
    all_embeddings = []

    print("Encoding chunks...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")

    print("Building FAISS index (IndexFlatIP for cosine similarity)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")

    # chunk_metadata: list of dicts parallel to index
    metadata = [
        {"chunk_id": c["chunk_id"], "source_type": c["source_type"], "source_id": c["source_id"], "text": c["text"]}
        for c in chunks
    ]
    with open(CHUNK_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved chunk metadata to {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()
