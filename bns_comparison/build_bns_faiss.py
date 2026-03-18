"""
Build BNS-only FAISS indexes for the comparison framework.

Builds two indexes:
  1. BGE-based index (Systems 2 & 3): uses fine-tuned BGE model, structured v2 sections
  2. Old-Work-style index (System 1): uses nomic-embed-text via Ollama, PDF-style chunking

Run from project root:
    python -m bns_comparison.build_bns_faiss [--system {bge,oldwork,both}]
"""
import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.config import (
    BNS_FAISS_INDEX_PATH,
    BNS_CHUNK_METADATA_PATH,
    FINE_TUNED_MODEL_DIR,
    SECTIONS_CSV,
    OLD_WORK_FAISS_INDEX_PATH,
    OLD_WORK_CHUNK_METADATA_PATH,
    OLD_WORK_BNS_PDF,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TIMEOUT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from phase3_embeddings.chunk_corpus import chunk_text, get_tokenizer
from phase3_embeddings.config import CHUNK_SIZE as BGE_CHUNK_SIZE


# ---------------------------------------------------------------------------
# System 1 (Old-Work): PDF chunking + nomic-embed-text via Ollama
# ---------------------------------------------------------------------------

def _split_text_char(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Character-based splitter matching Old-Work's RecursiveCharacterTextSplitter."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        # Advance by (chunk_size - overlap), ensuring forward progress
        stride = max(chunk_size - overlap, 1)
        start += stride
    return [c for c in chunks if c.strip()]


def _embed_ollama(texts: List[str], model: str = OLLAMA_EMBED_MODEL) -> np.ndarray:
    """Embed texts using Ollama's nomic-embed-text model."""
    embeddings = []
    for text in tqdm(texts, desc=f"Embedding with {model}"):
        payload = {"model": model, "input": text}
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embed error {resp.status_code}: {resp.text}")
        data = resp.json()
        # Ollama /api/embed returns {"embeddings": [[...]]}
        emb = data.get("embeddings") or data.get("embedding")
        if emb is None:
            raise RuntimeError(f"No embedding in response: {data}")
        if isinstance(emb[0], list):
            emb = emb[0]
        embeddings.append(emb)
    return np.array(embeddings, dtype="float32")


def build_old_work_index() -> None:
    """Build Old-Work-style FAISS index from BNS PDF using nomic-embed-text."""
    if not OLD_WORK_BNS_PDF.exists():
        print(f"[ERROR] BNS PDF not found at {OLD_WORK_BNS_PDF}")
        print("  Copy Datasets/1_BNS.pdf to Old-Work/data/bns.pdf first.")
        sys.exit(1)

    print(f"[Old-Work] Loading BNS PDF from {OLD_WORK_BNS_PDF}...")
    try:
        import pypdf
        reader = pypdf.PdfReader(str(OLD_WORK_BNS_PDF))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(pages)
        print(f"[Old-Work] Extracted {len(pages)} pages, {len(full_text)} chars")
    except ImportError:
        print("[ERROR] pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    print("[Old-Work] Splitting into character-based chunks...")
    raw_chunks = _split_text_char(full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"[Old-Work] Created {len(raw_chunks)} chunks")

    metadata = []
    for i, text in enumerate(raw_chunks):
        metadata.append({
            "chunk_id": f"old_work_chunk_{i}",
            "source_type": "pdf_chunk",
            "source_id": f"bns_pdf_p{i}",
            "act_id": "BNS_2023",
            "text": text,
        })

    print("[Old-Work] Embedding with nomic-embed-text via Ollama...")
    texts = [m["text"] for m in metadata]
    embeddings = _embed_ollama(texts, model=OLLAMA_EMBED_MODEL)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = (embeddings / norms).astype("float32")

    print("[Old-Work] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    OLD_WORK_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OLD_WORK_FAISS_INDEX_PATH))
    with open(OLD_WORK_CHUNK_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[Old-Work] Saved FAISS index ({index.ntotal} vectors) -> {OLD_WORK_FAISS_INDEX_PATH}")
    print(f"[Old-Work] Saved metadata -> {OLD_WORK_CHUNK_METADATA_PATH}")


# ---------------------------------------------------------------------------
# Systems 2 & 3: BGE embeddings from v2 structured sections
# ---------------------------------------------------------------------------

def build_bge_index() -> None:
    """Build BGE-based FAISS index from v2 BNS sections."""
    if not SECTIONS_CSV.exists():
        print(f"[ERROR] sections.csv not found at {SECTIONS_CSV}")
        sys.exit(1)
    if not FINE_TUNED_MODEL_DIR.exists():
        print(f"[ERROR] Fine-tuned BGE model not found at {FINE_TUNED_MODEL_DIR}")
        print("  Run phase3_embeddings/finetune_bge.py first.")
        sys.exit(1)

    print(f"[BGE] Loading sections from {SECTIONS_CSV}...")
    df = pd.read_csv(SECTIONS_CSV)
    bns_df = df[df["act_id"] == "BNS_2023"].copy()
    print(f"[BGE] Found {len(bns_df)} BNS_2023 sections")

    print("[BGE] Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("[BGE] Chunking sections...")
    all_chunks: List[Dict] = []
    for _, row in tqdm(bns_df.iterrows(), total=len(bns_df), desc="Sections"):
        section_id = str(row["section_id"])
        act_id = str(row.get("act_id", "BNS_2023"))
        text = str(row.get("full_text", "")).strip()
        if not text:
            continue
        chunks = chunk_text(text, tokenizer, section_id, "section", act_id=act_id)
        all_chunks.extend(chunks)

    # Deduplicate chunk_ids
    seen = set()
    for c in all_chunks:
        base = c["chunk_id"]
        idx = 0
        while c["chunk_id"] in seen:
            c["chunk_id"] = f"{base}_{idx}"
            idx += 1
        seen.add(c["chunk_id"])

    print(f"[BGE] Total chunks: {len(all_chunks)}")

    print(f"[BGE] Loading model from {FINE_TUNED_MODEL_DIR}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))

    texts = [c["text"] for c in all_chunks]
    batch_size = 64
    all_embeddings = []
    print("[BGE] Encoding chunks...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")

    print("[BGE] Building FAISS index (IndexFlatIP)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    BNS_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(BNS_FAISS_INDEX_PATH))

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "source_type": c["source_type"],
            "source_id": c["source_id"],
            "act_id": c.get("act_id", "BNS_2023"),
            "text": c["text"],
        }
        for c in all_chunks
    ]
    with open(BNS_CHUNK_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[BGE] Saved FAISS index ({index.ntotal} vectors) -> {BNS_FAISS_INDEX_PATH}")
    print(f"[BGE] Saved metadata -> {BNS_CHUNK_METADATA_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build BNS-only FAISS indexes")
    parser.add_argument(
        "--system",
        choices=["bge", "oldwork", "both"],
        default="both",
        help="Which index to build (default: both)",
    )
    args = parser.parse_args()

    if args.system in ("bge", "both"):
        print("\n=== Building BGE index (Systems 2 & 3) ===")
        build_bge_index()

    if args.system in ("oldwork", "both"):
        print("\n=== Building Old-Work index (System 1) ===")
        build_old_work_index()

    print("\n[Done] All requested indexes built.")


if __name__ == "__main__":
    main()
