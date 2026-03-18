"""
Chunk cases, sections, and articles from Phase 1 output into overlapping token-based chunks.
Output: chunks.pkl with metadata (chunk_id, source_type, source_id, act_id, text).
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
    text: str,
    tokenizer,
    source_id: str,
    source_type: str,
    act_id: Optional[str] = None,
    chunk_id_prefix: Optional[str] = None,
) -> list[dict]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE tokens with ~CHUNK_OVERLAP overlap.
    Returns list of dicts: {chunk_id, source_type, source_id, act_id, text}.
    act_id is included so the retriever and LLM know which Act each chunk belongs to.
    """
    if not text or not str(text).strip():
        return []

    # Use sanitized source_id for chunk_id to avoid special chars
    prefix = chunk_id_prefix if chunk_id_prefix is not None else _sanitize_id(source_id)
    text = str(text).strip()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    base_meta = {
        "source_type": source_type,
        "source_id": source_id,
        "act_id": act_id or "",
    }

    if len(tokens) <= CHUNK_SIZE:
        chunk_id = f"{prefix}_chunk_0"
        return [{**base_meta, "chunk_id": chunk_id, "text": text}]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk_text_str.strip():
            chunk_id = f"{prefix}_chunk_{chunk_idx}"
            chunks.append({**base_meta, "chunk_id": chunk_id, "text": chunk_text_str.strip()})
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

    sections_path = PHASE1_OUTPUT / "sections.csv"
    articles_path = PHASE1_OUTPUT / "articles.csv"

    # v2 uses two separate case files; fall back to single cases.csv for v1 compatibility
    cases_iltur_path = PHASE1_OUTPUT / "cases_iltur_neo4j.csv"
    cases_sc_path = PHASE1_OUTPUT / "cases_sc_neo4j.csv"
    cases_legacy_path = PHASE1_OUTPUT / "cases.csv"

    for p in [sections_path, articles_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run Phase 1 first.")
            sys.exit(1)

    has_v2_cases = cases_iltur_path.exists() or cases_sc_path.exists()
    has_legacy_cases = cases_legacy_path.exists()
    if not has_v2_cases and not has_legacy_cases:
        print(f"Error: No cases CSV found in {PHASE1_OUTPUT}. Run Phase 1 first.")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    all_chunks = []

    # Cases — combine v2 files or fall back to legacy cases.csv
    print("Chunking cases...")
    case_dfs = []
    if has_v2_cases:
        if cases_iltur_path.exists():
            case_dfs.append(pd.read_csv(cases_iltur_path))
            print(f"  Loaded {cases_iltur_path.name}")
        if cases_sc_path.exists():
            case_dfs.append(pd.read_csv(cases_sc_path))
            print(f"  Loaded {cases_sc_path.name}")
    else:
        case_dfs.append(pd.read_csv(cases_legacy_path))
        print(f"  Loaded {cases_legacy_path.name}")

    cases_df = pd.concat(case_dfs, ignore_index=True)
    for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Cases"):
        case_id = str(row["case_id"])
        text = row.get("judgment_text", "")
        chunks = chunk_text(text, tokenizer, case_id, "case", act_id=None)
        all_chunks.extend(chunks)

    # Sections — include act_id in metadata
    print("Chunking sections...")
    sections_df = pd.read_csv(sections_path)
    for _, row in tqdm(sections_df.iterrows(), total=len(sections_df), desc="Sections"):
        section_id = str(row["section_id"])
        act_id = str(row.get("act_id", "")) if pd.notna(row.get("act_id", "")) else ""
        text = row["full_text"]
        chunks = chunk_text(text, tokenizer, section_id, "section", act_id=act_id)
        all_chunks.extend(chunks)

    # Articles — include act_id in metadata
    print("Chunking articles...")
    articles_df = pd.read_csv(articles_path)
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Articles"):
        article_id = str(row["article_id"])
        act_id = str(row.get("act_id", "")) if pd.notna(row.get("act_id", "")) else ""
        text = row["full_text"]
        chunks = chunk_text(text, tokenizer, article_id, "article", act_id=act_id)
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
