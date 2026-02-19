"""
Chunk cases, sections, and articles from Phase 1 output into overlapping token-based chunks.
Output: chunks.pkl with metadata (chunk_id, source_type, source_id, text).
"""
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_embeddings.config import (
    CHUNK_SIZE,
    CHUNKS_PATH,
    OUTPUT_DIR,
    PHASE1_OUTPUT,
    STRIDE,
)


def get_tokenizer():
    """Load BGE tokenizer for token-based chunking."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def chunk_text(
    text: str, tokenizer, source_id: str, source_type: str, chunk_id_prefix: Optional[str] = None
) -> list[dict]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE tokens with ~CHUNK_OVERLAP overlap.
    Returns list of dicts: {chunk_id, source_type, source_id, text}.
    """
    if not text or not str(text).strip():
        return []

    # Use sanitized source_id for chunk_id to avoid special chars
    prefix = chunk_id_prefix if chunk_id_prefix is not None else _sanitize_id(source_id)
    text = str(text).strip()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= CHUNK_SIZE:
        chunk_id = f"{prefix}_chunk_0"
        return [
            {
                "chunk_id": chunk_id,
                "source_type": source_type,
                "source_id": source_id,
                "text": text,
            }
        ]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk_text_str.strip():
            chunk_id = f"{prefix}_chunk_{chunk_idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "text": chunk_text_str.strip(),
                }
            )
            chunk_idx += 1

        start += STRIDE
        if start >= len(tokens):
            break

    return chunks


def _sanitize_id(s: str) -> str:
    """Sanitize source ID for use in chunk_id (replace problematic chars)."""
    return re.sub(r"[^\w\-.]", "_", str(s))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_path = PHASE1_OUTPUT / "cases.csv"
    sections_path = PHASE1_OUTPUT / "sections.csv"
    articles_path = PHASE1_OUTPUT / "articles.csv"

    for p in [cases_path, sections_path, articles_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run Phase 1 first.")
            sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    all_chunks = []

    # Cases
    print("Chunking cases...")
    cases_df = pd.read_csv(cases_path)
    for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Cases"):
        case_id = str(row["case_id"])
        text = row["judgment_text"]
        chunks = chunk_text(text, tokenizer, case_id, "case")
        all_chunks.extend(chunks)

    # Sections
    print("Chunking sections...")
    sections_df = pd.read_csv(sections_path)
    for _, row in tqdm(sections_df.iterrows(), total=len(sections_df), desc="Sections"):
        section_id = str(row["section_id"])
        text = row["full_text"]
        chunks = chunk_text(text, tokenizer, section_id, "section")
        all_chunks.extend(chunks)

    # Articles
    print("Chunking articles...")
    articles_df = pd.read_csv(articles_path)
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Articles"):
        article_id = str(row["article_id"])
        text = row["full_text"]
        chunks = chunk_text(text, tokenizer, article_id, "article")
        all_chunks.extend(chunks)

    # Ensure unique chunk_ids (in case source_id had duplicates)
    seen = set()
    for c in all_chunks:
        base = c["chunk_id"]
        idx = 0
        while c["chunk_id"] in seen:
            c["chunk_id"] = f"{base}_{idx}"
            idx += 1
        seen.add(c["chunk_id"])

    print(f"Total chunks: {len(all_chunks)}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
