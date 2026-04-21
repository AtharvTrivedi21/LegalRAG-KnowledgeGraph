"""
Shared configuration for the BNS-Only RAG Comparison framework.
All paths are relative to the project root (c:\\Users\\ATHARV\\LegalRAG).
"""
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# BNS-only FAISS index (built by build_bns_faiss.py)
BNS_FAISS_INDEX_PATH = PROJECT_ROOT / "bns_comparison" / "faiss_bns_only" / "faiss.index"
BNS_CHUNK_METADATA_PATH = PROJECT_ROOT / "bns_comparison" / "faiss_bns_only" / "chunk_metadata.pkl"

# Fine-tuned BGE model (shared with main pipeline)
# FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "phase3_embeddings" / "models" / "bge-legal"
# FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "phase3_embeddings" / "models" / "bge-legal-bns"
# FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "phase3_embeddings" / "models" / "bge-legal-bns-mapping"
FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "phase3_embeddings" / "bge-legal-bns-groq"

# v2 sections CSV (source of BNS sections)
SECTIONS_CSV = PROJECT_ROOT / "phase1_output_v2" / "sections.csv"

# Old-Work FAISS index (built by Old-Work/ingest_bns.py reimplemented below)
OLD_WORK_FAISS_INDEX_PATH = PROJECT_ROOT / "bns_comparison" / "faiss_bns_only" / "old_work_faiss.index"
OLD_WORK_CHUNK_METADATA_PATH = PROJECT_ROOT / "bns_comparison" / "faiss_bns_only" / "old_work_chunk_metadata.pkl"
OLD_WORK_BNS_PDF = PROJECT_ROOT / "Old-Work" / "data" / "bns.pdf"

# Results output
RESULTS_DIR = PROJECT_ROOT / "bns_comparison" / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"

# Ollama settings (shared)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT = 300

# Groq Cloud settings (use only the last configured key)
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_GROQ_KEY_CANDIDATES = [
    v for v in (
        os.environ.get("GROQ_API_KEY_1", "").strip(),
        os.environ.get("GROQ_API_KEY_2", "").strip(),
        os.environ.get("GROQ_API_KEY_3", "").strip(),
        os.environ.get("GROQ_API_KEY_4", "").strip(),
        os.environ.get("GROQ_API_KEY_5", "").strip(),
    )
    if v
]
if _GROQ_KEY_CANDIDATES:
    GROQ_API_KEYS = [_GROQ_KEY_CANDIDATES[-1]]
elif GROQ_API_KEY:
    GROQ_API_KEYS = [GROQ_API_KEY]
else:
    GROQ_API_KEYS = []
GROQ_MODEL = "llama-3.1-8b-instant"  # same family as llama3:8b (Llama 3.1 8B)
GROQ_RPM_LIMIT = 30  # free tier: 30 requests per minute

# Retrieval settings
TOP_K = 8
CHUNK_SIZE = 1000   # chars for Old-Work (matches original ingest_bns.py)
CHUNK_OVERLAP = 200
