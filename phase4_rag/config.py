from __future__ import annotations

"""
Central configuration for Phase 4: Graph-Constrained LegalRAG.

All values are kept local-only and can be overridden via environment
variables. Defaults are chosen to work out-of-the-box on a standard
developer machine where:
- Neo4j Desktop runs on bolt://localhost:7687
- Ollama runs on http://localhost:11434 with a pulled llama3:8b model
- Phase 3 artifacts live at the paths defined in phase3_embeddings.config
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from phase3_embeddings import config as phase3_config

# Load .env from project root (parent of phase4_rag/) so NEO4J_PASSWORD etc. are set
_load_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env_path)


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    request_timeout: int = 60  # seconds


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 8
    # When constraints are present, we can over-retrieve and filter
    # down to top_k after applying graph constraints.
    constrained_multiplier: int = 3


@dataclass(frozen=True)
class Phase4Settings:
    neo4j: Neo4jSettings
    ollama: OllamaSettings
    retrieval: RetrievalSettings
    # Phase 3 artifacts (paths imported from phase3_embeddings.config)
    faiss_index_path: str
    chunk_metadata_path: str
    fine_tuned_model_dir: str


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def load_settings() -> Phase4Settings:
    """
    Load Phase 4 settings from environment variables with sensible defaults.
    """
    neo4j = Neo4jSettings(
        uri=_env("NEO4J_URI", "bolt://localhost:7687"),
        user=_env("NEO4J_USER", "neo4j"),
        password=_env("NEO4J_PASSWORD", "neo4j"),
    )

    ollama = OllamaSettings(
        base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=_env("OLLAMA_MODEL", "llama3:8b"),
        request_timeout=int(_env("OLLAMA_TIMEOUT", "60")),
    )

    retrieval = RetrievalSettings(
        top_k=int(_env("PHASE4_TOP_K", "8")),
        constrained_multiplier=int(_env("PHASE4_CONSTRAINED_MULTIPLIER", "3")),
    )

    return Phase4Settings(
        neo4j=neo4j,
        ollama=ollama,
        retrieval=retrieval,
        faiss_index_path=str(phase3_config.FAISS_INDEX_PATH),
        chunk_metadata_path=str(phase3_config.CHUNK_METADATA_PATH),
        fine_tuned_model_dir=str(phase3_config.FINE_TUNED_MODEL_DIR),
    )


# A module-level singleton for convenience. Code that wants to read
# configuration can import `settings` instead of calling load_settings()
# repeatedly. This is safe because configuration is static during a run.
settings: Phase4Settings = load_settings()

