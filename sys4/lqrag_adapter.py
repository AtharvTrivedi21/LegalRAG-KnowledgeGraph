"""
System 4 (sys4): LQ-RAG-inspired + graph-expanded BNS RAG.
Hybrid BM25 + dense FAISS (RRF), cross-encoder re-ranking, Neo4j filter/expand, self-eval.
"""
from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from bns_comparison.adapters.base import BaseAdapter
from bns_comparison.adapters._ollama import ollama_chat, extract_section_numbers
from bns_comparison.config import (
    BNS_CHUNK_METADATA_PATH,
    BNS_FAISS_INDEX_PATH,
    FINE_TUNED_MODEL_DIR,
    TOP_K,
)
from sys4.graph_helpers import (
    TARGET_ACT,
    expand_reference_sections,
    filter_section_ids_by_act,
    get_citation_counts,
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
    "Follow this structure:\n"
    "1) Brief reasoning (do NOT cite section numbers in this step).\n"
    "2) Then output the user-facing answer with headers:\n"
    "## Summary\n## Applicable BNS provisions\n## Recommendation / next steps\n"
    "In Applicable BNS provisions, cite ONLY sections that appear in the context headers "
    "as [BNS Section NNN] or [GRAPH-DISCOVERED]. Use the form 'Section NNN'.\n"
    "Do not invent section numbers."
)

QA_USER = (
    "User description: {user_query}\n\n"
    "Formal legal query: {legal_query}\n\n"
    "BNS context:\n{context}\n\n"
    "Answer:"
)

EVAL_USER = (
    "You review a draft legal answer. Each statute block in CONTEXT starts with "
    "[BNS Section NUMBER] or [GRAPH-DISCOVERED Section NUMBER].\n\n"
    "Reply with EXACTLY three lines in this format:\n"
    "GROUNDED: YES or NO\n"
    "IPC: YES or NO\n"
    "RELEVANT: YES or NO\n\n"
    "GROUNDED is YES only if every Section number cited in the ANSWER appears in the CONTEXT headers.\n"
    "IPC is YES if the answer cites IPC/CrPC/IEA section numbers or treats repealed codes as current law.\n\n"
    "CONTEXT (excerpt):\n{ctx}\n\n"
    "ANSWER:\n{answer}\n"
)

RRF_K = 60
BM25_TOP = 30
FAISS_TOP = 30
RERANK_POOL = 20
MIN_SECTIONS = 3
MAX_GRAPH_EXPAND = 3
MAX_EVAL_RETRIES = 2
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _section_num_from_source_id(sid: str) -> str:
    m = re.search(r"_s(\d+[A-Za-z]?)$", sid or "")
    return m.group(1) if m else ""


def _reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for rlist in rankings:
        for rank, doc_idx in enumerate(rlist):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _try_neo4j_enrich(section_ids: List[str]) -> List[Dict]:
    try:
        from phase4_rag.neo4j_client_v3 import get_sections_by_ids

        return get_sections_by_ids(section_ids)
    except Exception:
        return []


def _parse_eval(text: str) -> Tuple[bool, bool, bool]:
    """Returns (grounded_ok, ipc_ok, relevant_ok). Default True if parse fails."""
    g_ok, i_ok, r_ok = True, True, True
    for line in (text or "").splitlines():
        u = line.strip()
        ul = u.upper()
        if ul.startswith("GROUNDED:"):
            tail = u.split(":", 1)[-1].strip().upper()
            g_ok = tail.startswith("YES")
        elif ul.startswith("IPC:"):
            tail = u.split(":", 1)[-1].strip().upper()
            i_ok = tail.startswith("NO")
        elif ul.startswith("RELEVANT:"):
            tail = u.split(":", 1)[-1].strip().upper()
            r_ok = tail.startswith("YES")
    return g_ok, i_ok, r_ok


class LQRAGAdapter(BaseAdapter):
    """sys4: Hybrid + rerank + graph + self-eval."""

    def __init__(self) -> None:
        self._index = None
        self._metadata: List[Dict] = []
        self._bi_model: SentenceTransformer | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus_tokens: List[List[str]] | None = None
        self._cross_encoder: CrossEncoder | None = None
        self._loaded = False

    @property
    def system_name(self) -> str:
        return "System4_LQRAG_Graph"

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
        self._bi_model = SentenceTransformer(str(FINE_TUNED_MODEL_DIR), device="cpu")
        texts = [m.get("text") or "" for m in self._metadata]
        self._bm25_corpus_tokens = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)
        self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
        self._loaded = True

    def _dense_top_indices(self, query: str, k: int) -> List[int]:
        assert self._index is not None and self._bi_model is not None
        k = min(k, self._index.ntotal)
        q_emb = self._bi_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, k)
        return [int(i) for i in indices[0] if i >= 0]

    def _bm25_top_indices(self, query: str, k: int) -> List[int]:
        assert self._bm25 is not None and self._metadata is not None
        q_tok = _tokenize(query)
        scores = self._bm25.get_scores(q_tok)
        order = np.argsort(scores)[::-1][:k]
        return [int(i) for i in order if i < len(self._metadata)]

    def _rerank_candidates(self, query: str, indices: List[int]) -> List[Tuple[int, float]]:
        assert self._cross_encoder is not None and self._metadata is not None
        pairs = []
        for idx in indices:
            text = self._metadata[idx].get("text") or ""
            pairs.append((query, text[:4000]))
        if not pairs:
            return []
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def _retrieve_hybrid_rerank_diverse(self, legal_query: str, k: int = TOP_K) -> List[Dict]:
        self._load()
        assert self._metadata is not None

        d_list = self._dense_top_indices(legal_query, FAISS_TOP)
        b_list = self._bm25_top_indices(legal_query, BM25_TOP)
        rrf_scores = _reciprocal_rank_fusion([d_list, b_list])
        fused_pool = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:RERANK_POOL]

        reranked = self._rerank_candidates(legal_query, fused_pool)

        all_results: List[Dict] = []
        for idx, ce_score in reranked:
            meta = self._metadata[idx].copy()
            meta["score"] = float(ce_score)
            meta["_rrf"] = rrf_scores.get(idx, 0.0)
            all_results.append(meta)

        sections = [r for r in all_results if r.get("source_type") == "section"]
        chosen: List[Dict] = []
        seen_chunk: Set[str] = set()

        for r in sections:
            if len(chosen) >= MIN_SECTIONS:
                break
            cid = r.get("chunk_id")
            if cid and cid not in seen_chunk:
                seen_chunk.add(cid)
                chosen.append(r)

        rest = sorted(
            [r for r in all_results if r.get("chunk_id") not in seen_chunk],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        for r in rest:
            if len(chosen) >= k:
                break
            chosen.append(r)

        return chosen[:k]

    def _filter_by_act(self, chunks: List[Dict]) -> List[Dict]:
        sids = [c["source_id"] for c in chunks if c.get("source_type") == "section"]
        valid = filter_section_ids_by_act(sids, TARGET_ACT)
        if not valid:
            return [c for c in chunks if c.get("act_id", TARGET_ACT) == TARGET_ACT]
        out = []
        for c in chunks:
            if c.get("source_type") != "section":
                out.append(c)
                continue
            sid = c.get("source_id", "")
            if sid in valid:
                out.append(c)
        return out if out else chunks

    def _build_context(
        self,
        chunks: List[Dict],
        graph_rows: List[Dict],
        expanded: List[Dict[str, Any]],
        cite_counts: Dict[str, int],
    ) -> str:
        parts: List[str] = []
        gmap = {g["section_id"]: g for g in graph_rows}

        for i, c in enumerate(chunks):
            sid = c.get("source_id", "unknown")
            snum = _section_num_from_source_id(sid)
            act = c.get("act_id", TARGET_ACT)
            if sid in gmap:
                act = gmap[sid].get("act_id", act)
            cc = cite_counts.get(sid, 0)
            cc_s = f" (cited by {cc} Supreme Court cases)" if cc else ""
            header = f"[BNS Section {snum}] source: {sid} (Act: {act}){cc_s}"
            parts.append(f"{header}\n{c.get('text', '')}")

        for ex in expanded:
            snum = ex.get("section_number") or _section_num_from_source_id(ex.get("section_id", ""))
            sid = ex.get("section_id", "")
            header = (
                f"[GRAPH-DISCOVERED Section {snum}] source: {sid} "
                f"(related via REFERENCES from {ex.get('seed_from', '')})"
            )
            parts.append(f"{header}\n{ex.get('full_text', '')}")

        return "\n\n".join(parts)

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        t0 = time.time()
        eval_iterations = 0

        t_rephrase_start = time.time()
        legal_query = ollama_chat(
            [{"role": "user", "content": REPHRASE_PROMPT.format(user_query=user_query)}]
        )
        rephrase_sec = time.time() - t_rephrase_start

        t_retrieval_start = time.time()
        chunks = self._retrieve_hybrid_rerank_diverse(legal_query, k=TOP_K)
        chunks = self._filter_by_act(chunks)

        section_ids = list({c["source_id"] for c in chunks if c.get("source_type") == "section"})
        graph_sections = _try_neo4j_enrich(section_ids)
        cite_counts = get_citation_counts(section_ids)
        expanded = expand_reference_sections(section_ids, TARGET_ACT, limit=MAX_GRAPH_EXPAND)

        context = self._build_context(chunks, graph_sections, expanded, cite_counts)
        retrieval_sec = time.time() - t_retrieval_start

        t_gen_start = time.time()
        answer = ollama_chat(
            [
                {"role": "system", "content": QA_SYSTEM},
                {
                    "role": "user",
                    "content": QA_USER.format(
                        user_query=user_query,
                        legal_query=legal_query,
                        context=context,
                    ),
                },
            ]
        )

        ctx_excerpt = context[:7000]
        for attempt in range(MAX_EVAL_RETRIES):
            eval_iterations = attempt + 1
            eval_text = ollama_chat(
                [
                    {
                        "role": "user",
                        "content": EVAL_USER.format(ctx=ctx_excerpt, answer=answer),
                    }
                ]
            )
            g_ok, ipc_ok, rel_ok = _parse_eval(eval_text)
            if g_ok and ipc_ok and rel_ok:
                break
            feedback_parts = []
            if not g_ok:
                feedback_parts.append(
                    "You cited section numbers that do not appear in the context headers. "
                    "Remove or fix them; cite ONLY sections listed in the context."
                )
            if not ipc_ok:
                feedback_parts.append(
                    "Do not cite IPC/CrPC/IEA. Use only BNS (Bharatiya Nyaya Sanhita 2023)."
                )
            if not rel_ok:
                feedback_parts.append("Focus on the user's specific situation.")
            fb = " ".join(feedback_parts)
            answer = ollama_chat(
                [
                    {"role": "system", "content": QA_SYSTEM},
                    {
                        "role": "user",
                        "content": QA_USER.format(
                            user_query=user_query,
                            legal_query=legal_query,
                            context=context,
                        )
                        + f"\n\nPrevious draft was insufficient. {fb}\nRewrite the full answer.",
                    },
                ]
            )

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
            "graph_expanded": expanded,
            "eval_iterations": eval_iterations,
            "timings": {
                "rephrase_sec": rephrase_sec,
                "retrieval_sec": retrieval_sec,
                "generation_sec": generation_sec,
                "total_sec": total_sec,
            },
        }
