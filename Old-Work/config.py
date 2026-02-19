"""
Configuration for BNS Mitra clone.

We start with the BASELINE setup that replicates the paper:
- LLM: llama2 (via Ollama)
- Embeddings: nomic-embed-text (via Ollama)
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    llm_model_name: str
    embedding_model_name: str
    config_name: str

# --- Baseline config: matches paper setup ---
BASELINE_CONFIG = ModelConfig(
    llm_model_name="llama2",           # Meta LLaMA 2 via Ollama
    embedding_model_name="nomic-embed-text",
    config_name="baseline_llama2_nomic",
)

# --- Future improved config (for later) ---
IMPROVED_CONFIG = ModelConfig(
    llm_model_name="llama3:8b-instruct",   # example; adjust to what you can run
    embedding_model_name="nomic-embed-text",
    config_name="improved_llama3_nomic",
)

# Toggle here: start with baseline
ACTIVE_CONFIG = BASELINE_CONFIG

# Paths
BNS_PDF_PATH = "data/bns.pdf"
VECTORSTORE_DIR = "vectorstore/bns_faiss"