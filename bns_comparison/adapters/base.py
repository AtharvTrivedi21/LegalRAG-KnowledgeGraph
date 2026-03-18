"""
Abstract base class for all RAG system adapters.
Each adapter wraps one system and exposes a common answer_query() interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAdapter(ABC):
    """Common interface for all three RAG systems."""

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Human-readable name for this system."""

    @abstractmethod
    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user query.

        Returns:
            {
                "system_name": str,
                "rephrased_query": str,
                "answer": str,
                "retrieved_chunks": List[dict],  # [{text, source_id, score}]
                "cited_sections": List[str],      # BNS section numbers from answer
                "context_text": str,              # concatenated retrieved text
                "timings": {
                    "rephrase_sec": float,
                    "retrieval_sec": float,
                    "generation_sec": float,
                    "total_sec": float,
                },
            }
        """
