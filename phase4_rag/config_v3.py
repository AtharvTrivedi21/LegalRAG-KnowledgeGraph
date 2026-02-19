from __future__ import annotations

"""
Configuration for Phase 4 V3 (Act-aware disambiguation and Act classification).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from phase3_embeddings import config as phase3_config

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
    request_timeout: int = 300


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 8
    constrained_multiplier: int = 3
    diversity_multiplier: int = 4
    min_sections_per_query: int = 2
    min_articles_per_query: int = 2


@dataclass(frozen=True)
class Phase4Settings:
    neo4j: Neo4jSettings
    ollama: OllamaSettings
    retrieval: RetrievalSettings
    faiss_index_path: str
    chunk_metadata_path: str
    fine_tuned_model_dir: str


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def load_settings() -> Phase4Settings:
    neo4j = Neo4jSettings(
        uri=_env("NEO4J_URI", "bolt://localhost:7687"),
        user=_env("NEO4J_USER", "neo4j"),
        password=_env("NEO4J_PASSWORD", "neo4j"),
    )
    ollama = OllamaSettings(
        base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=_env("OLLAMA_MODEL", "llama3:8b"),
        request_timeout=max(int(_env("OLLAMA_TIMEOUT") or "300"), 180),
    )
    retrieval = RetrievalSettings(
        top_k=int(_env("PHASE4_TOP_K", "8")),
        constrained_multiplier=int(_env("PHASE4_CONSTRAINED_MULTIPLIER", "3")),
        diversity_multiplier=int(_env("PHASE4_DIVERSITY_MULTIPLIER", "4")),
        min_sections_per_query=int(_env("PHASE4_MIN_SECTIONS", "2")),
        min_articles_per_query=int(_env("PHASE4_MIN_ARTICLES", "2")),
    )
    return Phase4Settings(
        neo4j=neo4j,
        ollama=ollama,
        retrieval=retrieval,
        faiss_index_path=str(phase3_config.FAISS_INDEX_PATH),
        chunk_metadata_path=str(phase3_config.CHUNK_METADATA_PATH),
        fine_tuned_model_dir=str(phase3_config.FINE_TUNED_MODEL_DIR),
    )


settings: Phase4Settings = load_settings()
