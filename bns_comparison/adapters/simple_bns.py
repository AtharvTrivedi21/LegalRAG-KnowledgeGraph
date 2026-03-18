"""
System 2 Adapter: Simple-BNS
- BNS sections from v2 structured CSV (100 sections)
- Embeddings: fine-tuned BGE model
- Retrieval: FAISS top-k=8, no graph constraints
- LLM: llama3:8b via Ollama
- Same improved prompts as current pipeline (BNS/BNSS/BSA aware)
- No Neo4j, no graph enrichment
"""
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bns_comparison.adapters.base import BaseAdapter
from bns_comparison.adapters._ollama import ollama_chat, extract_section_numbers
from bns_comparison.config import (
    BNS_FAISS_INDEX_PATH,
    BNS_CHUNK_METADATA_PATH,
    FINE_TUNED_MODEL_DIR,
    TOP_K,
)

REPHRASE_PROMPT = (
    "You are an expert in Indian criminal law. "
    "IMPORTANT: The Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and Indian Evidence Act (IEA) "
    "have been REPEALED and replaced by the following NEW laws effective 2024: "
    "Bharatiya Nyaya Sanhita 2023 (BNS) replaces IPC, "
    "Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) replaces CrPC, "
    "Bharatiya Sakshya Adhiniyam 2023 (BSA) replaces IEA. "
    "Rewrite the user's informal question or incident description into a concise, formal legal query "
    "using the NEW codes (BNS/BNSS/BSA) and appropriate legal terminology. "
    "Do NOT reference IPC, CrPC, or IEA sections. Do not answer the question, only rewrite it. "
    "Keep any explicit references to Article or Section numbers unchanged.\n\n"
    "User input:\n{user_query}\n\nFormal legal query:"
)

QA_SYSTEM = (
    "You are a legal research assistant for Indian law. "
    "Your knowledge base contains ONLY the Bharatiya Nyaya Sanhita 2023 (BNS_2023). "
    "The Indian Penal Code (IPC) is REPEALED. Do NOT cite IPC sections. "
    "Answer the user's question using ONLY the provided BNS context. "
    "Cite the relevant BNS section numbers in your answer. "
    "Structure your answer with: ## Summary, ## Applicable BNS provisions, ## Recommendation."
)

QA_USER = (
    "User description: {user_query}\n\n"
    "Formal legal query: {legal_query}\n\n"
    "BNS context:\n{context}\n\n"
    "Answer:"
)


class SimpleBNSAdapter(BaseAdapter):
    """System 2: Simple FAISS-only BNS RAG with BGE embeddings."""

    def __init__(self):
        self._index = None
        self._metadata: List[Dict] = []
        self._model = None
        self._loaded = False

    @property
    def system_name(self) -> str:
        return "System2_SimpleBNS"

    def _load(self) -> None:
        if self._loaded:
            return
        if not BNS_FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BNS FAISS index not found at {BNS_FAISS_INDEX_PATH}. "
                "Run: python -m bns_comparison.build_bns_faiss --system bge"
            )
        self._index = faiss.read_index(str(BNS_FAISS_INDEX_PATH))
        with open(BNS_CHUNK_METADATA_PATH, "rb") as f:
            self._metadata = pickle.load(f)
        self._model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR))
        self._loaded = True

    def _retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        self._load()
        q_emb = self._model.encode([query], normalize_embeddings=True).astype("float32")
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
        chunks = self._retrieve(legal_query, k=TOP_K)
        retrieval_sec = time.time() - t_retrieval_start

        # Build context
        context_parts = []
        for i, c in enumerate(chunks):
            sid = c.get("source_id", "unknown")
            act = c.get("act_id", "BNS_2023")
            context_parts.append(f"[SECTION {i+1}] {sid} (Act: {act})\n{c['text']}")
        context = "\n\n".join(context_parts)

        # Generate
        t_gen_start = time.time()
        answer = ollama_chat([
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": QA_USER.format(
                    user_query=user_query,
                    legal_query=legal_query,
                    context=context,
                ),
            },
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
