"""
Test the retrieval pipeline and diagnose chunk ID mismatches.
"""
import sys
import pickle
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load chunk metadata
meta_path = Path("phase3_embeddings/output/chunk_metadata.pkl")
print(f"Loading chunk metadata from {meta_path}...")
with open(meta_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks: {len(chunks)}")

# Sample chunk structure
if chunks:
    sample = chunks[0]
    print(f"\nSample chunk keys: {list(sample.keys())}")
    print(f"Sample chunk: {str(sample)[:300]}")

# Count by source type
source_types = Counter()
act_ids = Counter()
for ch in chunks:
    src = ch.get("source_type") or ch.get("type") or "unknown"
    source_types[src] += 1
    act = ch.get("act_id") or ch.get("act") or "unknown"
    act_ids[act] += 1

print(f"\nChunks by source_type: {dict(source_types.most_common(10))}")
print(f"Chunks by act_id: {dict(act_ids.most_common(10))}")

# Check chunk IDs
chunk_ids = [ch.get("chunk_id") or ch.get("id") or ch.get("section_id") or "?" for ch in chunks[:10]]
print(f"\nSample chunk IDs: {chunk_ids}")

# Test a FAISS query
print("\n--- Testing FAISS retrieval ---")
from phase4_rag.vector_retriever_v3 import retrieve_chunks

result = retrieve_chunks("punishment for murder", k=5)
print(f"Error: {result.get('error')}")
print(f"Chunks returned: {len(result.get('chunks', []))}")
for ch in result.get("chunks", [])[:5]:
    print(f"  chunk_id={ch.get('chunk_id') or ch.get('id')}, score={ch.get('score', 0):.3f}, text={str(ch.get('text',''))[:80]}")
