"""
Standalone System 3 experiment adapter with:
- Dense retrieval (FAISS + bi-encoder)
- Cross-encoder reranking
- Citation constraints (final citations must be from retrieved sections)
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.adapters._ollama import extract_section_numbers, ollama_chat
from bns_comparison.config import (
    BNS_CHUNK_METADATA_PATH,
    BNS_FAISS_INDEX_PATH,
    FINE_TUNED_MODEL_DIR,
    TOP_K,
)


REPHRASE_PROMPT = (
    "You are an expert in Indian criminal law. Rewrite the user's informal incident into a concise legal query "
    "using BNS/BNSS/BSA terminology. Do not answer.\n\n"
    "User input:\n{user_query}\n\nFormal legal query:"
)

QA_SYSTEM = (
    "You are a legal research assistant for Indian law.\n"
    "Use ONLY the provided BNS context.\n"
    "Do NOT cite IPC sections.\n"
    "In 'Applicable BNS provisions', cite only sections from this allowed list: {allowed_sections}.\n"
    "Format citations strictly as 'Section NNN'.\n"
    "Structure:\n"
    "## Summary\n"
    "## Applicable BNS provisions\n"
    "## Recommendation / next steps"
)

QA_USER = (
    "User description: {user_query}\n\n"
    "Formal legal query: {legal_query}\n\n"
    "BNS context:\n{context}\n\n"
    "Answer:"
)

_SOURCE_ID_SECTION_RE = re.compile(r"_s(\d+[A-Za-z]?)$")


def _section_num_from_source_id(sid: str) -> str:
    m = _SOURCE_ID_SECTION_RE.search(sid or "")
    return m.group(1).upper() if m else ""


class System3RerankAdapter:
    def __init__(self):
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._embed_model = None
        self._reranker = None
        self._loaded = False
        self._fast_mode = os.getenv("SYS3_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def system_name(self) -> str:
        return "System3_Rerank_CitationConstrained"

    def _load(self) -> None:
        if self._loaded:
            return
        if not BNS_FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {BNS_FAISS_INDEX_PATH}")
        if not BNS_CHUNK_METADATA_PATH.exists():
            raise FileNotFoundError(f"Chunk metadata not found at {BNS_CHUNK_METADATA_PATH}")
        if not FINE_TUNED_MODEL_DIR.exists():
            raise FileNotFoundError(f"Embedding model not found at {FINE_TUNED_MODEL_DIR}")

        self._index = faiss.read_index(str(BNS_FAISS_INDEX_PATH))
        with open(BNS_CHUNK_METADATA_PATH, "rb") as f:
            self._metadata = pickle.load(f)

        # Keep embedding model on CPU, use GPU for cross-encoder if available.
        self._embed_model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR), device="cpu")
        prefer_cuda = os.getenv("RERANK_DEVICE", "auto").strip().lower() in {"auto", "cuda"}
        rerank_device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=rerank_device)
        self._loaded = True

    def _retrieve_dense(self, query: str, k_overfetch: int = 24) -> List[Dict[str, Any]]:
        self._load()
        k = min(k_overfetch, self._index.ntotal)
        q_emb = self._embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, k)

        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self._metadata[idx].copy()
            item["dense_score"] = float(score)
            out.append(item)
        return out

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        scored = []
        for c, s in zip(candidates, scores):
            item = c.copy()
            item["rerank_score"] = float(s)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        t0 = time.time()

        # 1) Rephrase
        t1 = time.time()
        if self._fast_mode:
            legal_query = user_query
        else:
            legal_query = ollama_chat(
                [{"role": "user", "content": REPHRASE_PROMPT.format(user_query=user_query)}]
            )
        rephrase_sec = time.time() - t1

        # 2) Dense retrieve
        t2 = time.time()
        dense = self._retrieve_dense(legal_query, k_overfetch=max(24, TOP_K * 3))
        retrieval_sec = time.time() - t2

        # 3) Rerank
        t3 = time.time()
        chunks = self._rerank(legal_query, dense, top_k=TOP_K)
        rerank_sec = time.time() - t3

        # 4) Build context + allowed citation set
        context_parts = []
        allowed_sections = []
        for c in chunks:
            sid = c.get("source_id", "")
            snum = _section_num_from_source_id(sid)
            if snum:
                allowed_sections.append(snum)
            header = f"[BNS Section {snum}] {sid}"
            context_parts.append(f"{header}\n{c.get('text', '')}")
        allowed_sections = sorted(set(allowed_sections), key=lambda x: int(re.sub(r"[A-Za-z]", "", x) or "0"))
        context = "\n\n".join(context_parts)

        # 5) Generate answer
        t4 = time.time()
        answer = ollama_chat([
            {"role": "system", "content": QA_SYSTEM.format(allowed_sections=", ".join(allowed_sections))},
            {"role": "user", "content": QA_USER.format(user_query=user_query, legal_query=legal_query, context=context)},
        ])
        generation_sec = time.time() - t4

        # 6) Hard citation constraint at output struct level
        cited_sections = extract_section_numbers(answer)
        cited_sections = [s for s in cited_sections if s in set(allowed_sections)]

        total_sec = time.time() - t0
        return {
            "system_name": self.system_name,
            "rephrased_query": legal_query,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "text": c.get("text", ""),
                    "source_id": c.get("source_id", ""),
                    "score": c.get("rerank_score", c.get("dense_score", 0.0)),
                }
                for c in chunks
            ],
            "cited_sections": cited_sections,
            "context_text": context,
            "graph_sections": [],
            "timings": {
                "rephrase_sec": rephrase_sec,
                "retrieval_sec": retrieval_sec,
                "generation_sec": generation_sec,
                "rerank_sec": rerank_sec,
                "total_sec": total_sec,
            },
        }

