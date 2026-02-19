"""
Validate Phase 3 outputs: chunks, fine-tuned model, FAISS index, and retrieval.
Run from project root: python -m phase3_embeddings.validate_phase3
"""
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_embeddings.config import (
    CHUNK_METADATA_PATH,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    FINE_TUNED_MODEL_DIR,
    INDIC_LEGAL_QA_PATH,
    OUTPUT_DIR,
    PHASE1_OUTPUT,
)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_phase1_prerequisites() -> bool:
    """Check Phase 1 outputs exist (needed for chunking)."""
    print("\n--- Phase 1 prerequisites ---")
    required = ["cases.csv", "sections.csv", "articles.csv"]
    all_ok = True
    for name in required:
        p = PHASE1_OUTPUT / name
        if p.exists():
            ok(f"{name} found")
        else:
            fail(f"{name} not found at {p}")
            all_ok = False
    return all_ok


def check_chunks() -> bool:
    """Check chunks.pkl exists, is loadable, and has expected structure."""
    print("\n--- Chunks (chunk_corpus output) ---")
    if not CHUNKS_PATH.exists():
        fail(f"chunks.pkl not found at {CHUNKS_PATH}. Run chunk_corpus first.")
        return False
    try:
        import pickle

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        fail(f"Could not load chunks.pkl: {e}")
        return False
    if not isinstance(chunks, list) or len(chunks) == 0:
        fail("chunks.pkl is empty or not a list.")
        return False
    required_keys = {"chunk_id", "source_type", "source_id", "text"}
    sample = chunks[0]
    if not isinstance(sample, dict):
        fail("Chunk is not a dict.")
        return False
    missing = required_keys - set(sample.keys())
    if missing:
        fail(f"Chunk missing keys: {missing}")
        return False
    ok(f"chunks.pkl: {len(chunks)} chunks, structure valid")
    return True


def check_model() -> bool:
    """Check fine-tuned model exists and loads."""
    print("\n--- Fine-tuned model ---")
    if not FINE_TUNED_MODEL_DIR.exists():
        fail(f"Model dir not found at {FINE_TUNED_MODEL_DIR}. Run finetune_bge first.")
        return False
    # Check for expected model files (sentence-transformers save config + model)
    expected = ["config.json", "config_sentence_transformers.json"]
    for f in expected:
        if (FINE_TUNED_MODEL_DIR / f).exists():
            break
    else:
        fail("Model dir does not look like a SentenceTransformer model (missing config).")
        return False
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))
        dim = model.get_sentence_embedding_dimension()
        ok(f"Model loads; embedding dim = {dim}")
        return True
    except Exception as e:
        fail(f"Could not load model: {e}")
        return False


def check_faiss_and_metadata() -> bool:
    """Check FAISS index and chunk_metadata exist and are consistent."""
    print("\n--- FAISS index & chunk metadata ---")
    if not FAISS_INDEX_PATH.exists():
        fail(f"faiss.index not found at {FAISS_INDEX_PATH}. Run build_faiss first.")
        return False
    if not CHUNK_METADATA_PATH.exists():
        fail(f"chunk_metadata.pkl not found at {CHUNK_METADATA_PATH}. Run build_faiss first.")
        return False
    try:
        import pickle

        import faiss

        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(CHUNK_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        fail(f"Could not load index or metadata: {e}")
        return False
    n_index = index.ntotal
    n_meta = len(metadata)
    if n_index != n_meta:
        fail(f"Index size ({n_index}) != metadata length ({n_meta})")
        return False
    required_keys = {"chunk_id", "source_type", "source_id", "text"}
    sample = metadata[0]
    missing = required_keys - set(sample.keys())
    if missing:
        fail(f"Metadata entry missing keys: {missing}")
        return False
    ok(f"faiss.index: {n_index} vectors; chunk_metadata.pkl: {n_meta} entries; consistent")
    return True


def check_index_model_dim_match() -> bool:
    """Check FAISS index dimension matches model embedding dimension."""
    print("\n--- Index–model dimension match ---")
    try:
        import faiss

        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(FAISS_INDEX_PATH))
        model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))
        idx_dim = index.d
        model_dim = model.get_sentence_embedding_dimension()
        if idx_dim != model_dim:
            fail(f"Index dim ({idx_dim}) != model dim ({model_dim})")
            return False
        ok(f"Dimension match: {idx_dim}")
        return True
    except Exception as e:
        fail(str(e))
        return False


def check_retrieval() -> bool:
    """Run a quick retrieval to verify end-to-end search works."""
    print("\n--- Retrieval sanity check ---")
    try:
        from phase3_embeddings.retrieve import load_index, search

        index, metadata, model = load_index()
        results = search("What is Article 14?", k=3, index=index, metadata=metadata, model=model)
        if not results:
            fail("Search returned no results.")
            return False
        if len(results[0]) < 4 or "score" not in results[0]:
            fail("Result entry missing expected keys (e.g. score, text).")
            return False
        ok(f"Retrieval works; sample score = {results[0]['score']:.4f}")
        return True
    except Exception as e:
        fail(f"Retrieval failed: {e}")
        return False


def check_indic_legal_qa() -> bool:
    """Check IndicLegalQA dataset exists (for reference / re-run fine-tuning)."""
    print("\n--- IndicLegalQA dataset (optional for validation) ---")
    if INDIC_LEGAL_QA_PATH.exists():
        ok(f"IndicLegalQA found at {INDIC_LEGAL_QA_PATH}")
        return True
    fail(f"IndicLegalQA not found at {INDIC_LEGAL_QA_PATH}")
    return False


def main():
    print("Phase 3 validation")
    print("Working directory (project root):", Path.cwd())

    results = []
    results.append(("Phase 1 prerequisites", check_phase1_prerequisites()))
    results.append(("Chunks", check_chunks()))
    results.append(("IndicLegalQA", check_indic_legal_qa()))
    results.append(("Fine-tuned model", check_model()))
    results.append(("FAISS & metadata", check_faiss_and_metadata()))
    results.append(("Index–model dim", check_index_model_dim_match()))
    results.append(("Retrieval", check_retrieval()))

    print("\n--- Summary ---")
    passed = sum(1 for _, v in results if v)
    total = len(results)
    for name, v in results:
        print(f"  {'[OK]' if v else '[FAIL]'} {name}")
    print(f"\nPassed: {passed}/{total}")
    if passed == total:
        print("Phase 3 validation passed.")
        return 0
    print("Some checks failed. Complete the pipeline steps (chunk_corpus, finetune_bge, build_faiss) and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
