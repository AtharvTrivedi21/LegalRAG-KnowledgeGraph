"""
System 3 Adapter: Full-Pipeline-BNS
- BNS sections from v2 structured CSV (100 sections)
- Embeddings: fine-tuned BGE model
- Retrieval: FAISS top-k=8 with diversity (min 3 sections)
- Graph enrichment: Neo4j lookup of retrieved section IDs
- LLM: llama3:8b via Ollama
- Full improved prompts (BNS/BNSS/BSA aware, corpus description)
"""
import pickle
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

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
    "The Indian Penal Code (IPC) is REPEALED. Do NOT cite IPC sections — they no longer apply. "
    "Answer the user's question using ONLY the provided context (BNS sections). "
    "In 'Applicable BNS provisions', cite ONLY sections that appear in context headers as [BNS Section NNN]. "
    "Use the format 'Section NNN' (example: Section 303). "
    "Under 'Applicable BNS provisions', list every relevant context section exactly once. "
    "Do not fabricate citations not present in the context. "
    "Structure your answer with: ## Summary, ## Applicable BNS provisions, "
    "## Recommendation / next steps."
)

QA_USER = (
    "User description: {user_query}\n\n"
    "Formal legal query: {legal_query}\n\n"
    "BNS context:\n{context}\n\n"
    "Answer:"
)

EVAL_USER = (
    "You review a draft legal answer. Context headers use the exact form [BNS Section NUMBER].\n\n"
    "Reply with EXACTLY three lines in this format:\n"
    "GROUNDED: YES or NO\n"
    "IPC: YES or NO\n"
    "RELEVANT: YES or NO\n\n"
    "GROUNDED is YES only if every Section number cited in ANSWER appears in CONTEXT headers.\n"
    "IPC is YES if answer cites IPC/CrPC/IEA or treats repealed codes as current law.\n\n"
    "CONTEXT:\n{ctx}\n\n"
    "ANSWER:\n{answer}\n"
)


def _section_num_from_source_id(sid: str) -> str:
    m = re.search(r"_s(\d+[A-Za-z]?)$", sid or "")
    return m.group(1) if m else ""


def _parse_eval(text: str) -> tuple[bool, bool, bool]:
    grounded_ok, ipc_ok, relevant_ok = True, True, True
    for line in (text or "").splitlines():
        s = line.strip()
        us = s.upper()
        if us.startswith("GROUNDED:"):
            grounded_ok = s.split(":", 1)[-1].strip().upper().startswith("YES")
        elif us.startswith("IPC:"):
            ipc_ok = s.split(":", 1)[-1].strip().upper().startswith("NO")
        elif us.startswith("RELEVANT:"):
            relevant_ok = s.split(":", 1)[-1].strip().upper().startswith("YES")
    return grounded_ok, ipc_ok, relevant_ok


def _try_neo4j_enrich(section_ids: List[str]) -> List[Dict]:
    """Look up section metadata from Neo4j. Returns empty list if Neo4j unavailable."""
    try:
        from phase4_rag.neo4j_client_v3 import get_sections_by_ids
        return get_sections_by_ids(section_ids)
    except Exception:
        return []


class FullPipelineBNSAdapter(BaseAdapter):
    """System 3: Full pipeline with BGE + FAISS + diversity + Neo4j graph enrichment."""

    def __init__(self):
        self._index = None
        self._metadata: List[Dict] = []
        self._model = None
        self._loaded = False
        self._fast_mode = os.getenv("SYS3_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def system_name(self) -> str:
        return "System3_FullPipelineBNS"

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
        # Force CPU so GPU is free for Ollama llama3:8b
        self._model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR), device="cpu")
        self._loaded = True

    def _retrieve_diverse(self, query: str, k: int = TOP_K, min_sections: int = 3) -> List[Dict]:
        """Retrieve with diversity — guarantee at least min_sections section chunks."""
        self._load()
        k_over = min(k * 10, self._index.ntotal)
        q_emb = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, k_over)

        all_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx].copy()
            meta["score"] = float(score)
            all_results.append(meta)

        # Guarantee min_sections section chunks in top-k
        sections = [r for r in all_results if r.get("source_type") == "section"]
        chosen: List[Dict] = []
        seen_ids: Set[str] = set()

        for r in sorted(sections, key=lambda x: x.get("score", 0.0), reverse=True):
            if len(chosen) >= min_sections:
                break
            cid = r.get("chunk_id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                chosen.append(r)

        rest = sorted(
            [r for r in all_results if r.get("chunk_id") not in seen_ids],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        for r in rest:
            if len(chosen) >= k:
                break
            chosen.append(r)

        return chosen[:k]

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        t0 = time.time()

        # Rephrase (skip in fast mode to cut one LLM call)
        t_rephrase_start = time.time()
        if self._fast_mode:
            legal_query = user_query
        else:
            legal_query = ollama_chat(
                [{"role": "user", "content": REPHRASE_PROMPT.format(user_query=user_query)}]
            )
        rephrase_sec = time.time() - t_rephrase_start

        # Retrieve with diversity
        t_retrieval_start = time.time()
        chunks = self._retrieve_diverse(legal_query, k=TOP_K, min_sections=3)

        # Graph enrichment: look up section metadata from Neo4j
        section_ids = list({c["source_id"] for c in chunks if c.get("source_type") == "section"})
        graph_sections = _try_neo4j_enrich(section_ids)
        graph_section_map = {s["section_id"]: s for s in graph_sections}
        retrieval_sec = time.time() - t_retrieval_start

        # Build context with act_id from graph enrichment
        context_parts = []
        for i, c in enumerate(chunks):
            sid = c.get("source_id", "unknown")
            snum = _section_num_from_source_id(sid)
            act = c.get("act_id", "BNS_2023")
            if sid in graph_section_map:
                gs = graph_section_map[sid]
                act = gs.get("act_id", act)
                heading = gs.get("heading", "")
                header = f"[BNS Section {snum}] {sid} (Act: {act})"
                if heading:
                    header += f" — {heading}"
            else:
                header = f"[BNS Section {snum}] {sid} (Act: {act})"
            context_parts.append(f"{header}\n{c['text']}")
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

        if not self._fast_mode:
            # Quality mode: one self-check pass and optional rewrite.
            eval_text = ollama_chat(
                [
                    {
                        "role": "user",
                        "content": EVAL_USER.format(ctx=context[:7000], answer=answer),
                    }
                ]
            )
            grounded_ok, ipc_ok, relevant_ok = _parse_eval(eval_text)
            if not (grounded_ok and ipc_ok and relevant_ok):
                feedback = []
                if not grounded_ok:
                    feedback.append("Cite ONLY section numbers present in [BNS Section NNN] headers.")
                if not ipc_ok:
                    feedback.append("Do not cite IPC/CrPC/IEA; use only BNS sections from context.")
                if not relevant_ok:
                    feedback.append("Focus directly on the user incident and legal applicability.")
                answer = ollama_chat([
                    {"role": "system", "content": QA_SYSTEM},
                    {
                        "role": "user",
                        "content": QA_USER.format(
                            user_query=user_query,
                            legal_query=legal_query,
                            context=context,
                        ) + "\n\nRewrite the full answer. " + " ".join(feedback),
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
            "graph_sections": graph_sections,
            "timings": {
                "rephrase_sec": rephrase_sec,
                "retrieval_sec": retrieval_sec,
                "generation_sec": generation_sec,
                "total_sec": total_sec,
            },
        }
