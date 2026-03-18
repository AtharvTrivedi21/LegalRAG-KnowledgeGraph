"""
System 1 Adapter: Old-Work Baseline
- BNS PDF chunked with character-based splitter (1000 chars, 200 overlap)
- Embeddings: nomic-embed-text via Ollama
- Retrieval: FAISS top-k=4 (matching original Old-Work k=4)
- LLM: llama3:8b via Ollama
- No graph, no structured metadata
"""
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bns_comparison.adapters.base import BaseAdapter
from bns_comparison.adapters._ollama import ollama_chat, extract_section_numbers
from bns_comparison.config import (
    OLD_WORK_FAISS_INDEX_PATH,
    OLD_WORK_CHUNK_METADATA_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TIMEOUT,
)

REPHRASE_PROMPT = (
    "You are an expert in Indian criminal law and Bharatiya Nyaya Sanhita (BNS).\n"
    "Rewrite the user's informal incident description into a concise, formal legal query "
    "using appropriate legal terminology. Do not answer the question, only rewrite it.\n\n"
    "User incident description:\n{user_query}\n\nFormal legal query:"
)

QA_PROMPT = (
    "You are an AI-powered legal assistant specialized in Bharatiya Nyaya Sanhita (BNS).\n\n"
    "You are given:\n"
    "1. The original user incident description.\n"
    "2. A rephrased formal legal query.\n"
    "3. Relevant BNS sections with their text.\n\n"
    "Using ONLY the information in the BNS context, answer the user's question.\n"
    "Always:\n"
    "- Mention the most relevant BNS section numbers (if present in the context).\n"
    "- Briefly explain why these sections apply.\n"
    "- Use clear, simple language understandable by a layperson.\n"
    "- If you are unsure or the context is insufficient, say so and suggest consulting a human lawyer.\n\n"
    "User description:\n{user_query}\n\n"
    "Formal legal query:\n{legal_query}\n\n"
    "BNS context:\n{context}\n\n"
    "Answer:"
)


class OldWorkAdapter(BaseAdapter):
    """System 1: Old-Work baseline — PDF + nomic-embed-text + FAISS."""

    def __init__(self):
        self._index = None
        self._metadata: List[Dict] = []
        self._loaded = False

    @property
    def system_name(self) -> str:
        return "System1_OldWork"

    def _load(self) -> None:
        if self._loaded:
            return
        if not OLD_WORK_FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Old-Work FAISS index not found at {OLD_WORK_FAISS_INDEX_PATH}. "
                "Run: python -m bns_comparison.build_bns_faiss --system oldwork"
            )
        self._index = faiss.read_index(str(OLD_WORK_FAISS_INDEX_PATH))
        with open(OLD_WORK_CHUNK_METADATA_PATH, "rb") as f:
            self._metadata = pickle.load(f)
        self._loaded = True

    def _embed_query(self, text: str) -> np.ndarray:
        payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embed error {resp.status_code}: {resp.text}")
        data = resp.json()
        emb = data.get("embeddings") or data.get("embedding")
        if isinstance(emb[0], list):
            emb = emb[0]
        arr = np.array(emb, dtype="float32")
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.reshape(1, -1)

    def _retrieve(self, query: str, k: int = 4) -> List[Dict]:
        self._load()
        q_emb = self._embed_query(query)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        t0 = time.time()

        # Rephrase
        t_rephrase_start = time.time()
        legal_query = ollama_chat(
            [{"role": "user", "content": REPHRASE_PROMPT.format(user_query=user_query)}]
        )
        rephrase_sec = time.time() - t_rephrase_start

        # Retrieve
        t_retrieval_start = time.time()
        chunks = self._retrieve(legal_query, k=4)
        retrieval_sec = time.time() - t_retrieval_start

        # Build context
        context = "\n\n".join(
            f"[DOC {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
        )

        # Generate
        t_gen_start = time.time()
        answer = ollama_chat([
            {
                "role": "user",
                "content": QA_PROMPT.format(
                    user_query=user_query,
                    legal_query=legal_query,
                    context=context,
                ),
            }
        ])
        generation_sec = time.time() - t_gen_start

        total_sec = time.time() - t0

        return {
            "system_name": self.system_name,
            "rephrased_query": legal_query,
            "answer": answer,
            "retrieved_chunks": [
                {"text": c["text"], "source_id": c.get("source_id", ""), "score": c.get("score", 0.0)}
                for c in chunks
            ],
            "cited_sections": extract_section_numbers(answer),
            "context_text": context,
            "timings": {
                "rephrase_sec": rephrase_sec,
                "retrieval_sec": retrieval_sec,
                "generation_sec": generation_sec,
                "total_sec": total_sec,
            },
        }
