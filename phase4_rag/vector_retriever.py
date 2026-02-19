from __future__ import annotations

"""
Vector retrieval for Phase 4, reusing Phase 3 artifacts.

This module:
- Lazily loads the FAISS index, chunk metadata, and fine-tuned BGE model
  via `phase3_embeddings.retrieve.load_index`.
- Exposes `retrieve_chunks` which optionally applies graph constraints
  (case_ids / section_ids / article_ids) before returning top-k chunks.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from phase3_embeddings.retrieve import load_index, search

from .config import settings


@dataclass
class GraphConstraints:
    allowed_case_ids: List[str] = field(default_factory=list)
    allowed_section_ids: List[str] = field(default_factory=list)
    allowed_article_ids: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.allowed_case_ids or self.allowed_section_ids or self.allowed_article_ids)


_index = None
_metadata: Optional[List[Dict]] = None
_model = None
_init_error: Optional[str] = None


def init_vector_store() -> Tuple[bool, Optional[str]]:
    """
    Load FAISS index, metadata, and model once.

    Returns (ok, error_message). On success, error_message is None.
    """
    global _index, _metadata, _model, _init_error

    if _index is not None and _metadata is not None and _model is not None:
        return True, None

    try:
        _index, _metadata, _model = load_index()
        _init_error = None
        return True, None
    except Exception as exc:  # file not found, model issues, faiss errors
        _index = None
        _metadata = None
        _model = None
        _init_error = str(exc)
        return False, _init_error


def vector_store_status() -> Dict[str, Optional[str]]:
    """
    Lightweight status helper for the UI to show whether the vector store
    is ready.
    """
    ok, _ = init_vector_store()
    return {
        "ok": ok,
        "error": _init_error,
        "index_size": int(_index.ntotal) if ok and _index is not None else 0,
    }


def get_chunks_by_source_ids(source_ids: List[str], max_chunks: int = 10) -> List[Dict]:
    """
    Return chunks from the index whose source_id is in source_ids (e.g. when the
    graph says "this article/section was asked for"). No semantic score; we assign
    score=1.0 so they sort first. Use this to ensure the cited article/section text
    is always in context even if semantic search ranks it low.
    """
    ok, _ = init_vector_store()
    if not ok or not _metadata or not source_ids:
        return []
    allowed = set(source_ids)
    out: List[Dict] = []
    for meta in _metadata:
        if meta.get("source_id") in allowed:
            out.append({**meta, "score": 1.0})
            if len(out) >= max_chunks:
                break
    return out


def _apply_constraints(
    results: List[Dict],
    constraints: GraphConstraints,
) -> List[Dict]:
    if constraints.is_empty:
        return results

    allowed_cases = set(constraints.allowed_case_ids)
    allowed_sections = set(constraints.allowed_section_ids)
    allowed_articles = set(constraints.allowed_article_ids)

    filtered: List[Dict] = []
    for r in results:
        stype = r.get("source_type")
        sid = r.get("source_id")
        if stype == "case" and sid in allowed_cases:
            filtered.append(r)
        elif stype == "section" and sid in allowed_sections:
            filtered.append(r)
        elif stype == "article" and sid in allowed_articles:
            filtered.append(r)

    return filtered


def retrieve_chunks(
    query: str,
    k: Optional[int] = None,
    constraints: Optional[GraphConstraints] = None,
) -> Dict:
    """
    Run semantic search and optionally apply graph constraints.

    Returns a dict with:
    - chunks: List[dict]  (each has chunk_id, source_type, source_id, text, score)
    - used_constraints: bool
    - used_fallback_unconstrained: bool
    """
    ok, err = init_vector_store()
    if not ok:
        return {
            "chunks": [],
            "used_constraints": False,
            "used_fallback_unconstrained": False,
            "error": f"Vector store not available: {err}",
        }

    top_k = k or settings.retrieval.top_k
    used_constraints = constraints is not None and not constraints.is_empty

    if not used_constraints:
        base_results = search(query, k=top_k, index=_index, metadata=_metadata, model=_model)
        return {
            "chunks": base_results,
            "used_constraints": False,
            "used_fallback_unconstrained": False,
            "error": None,
        }

    # Must-include: when the user asked for specific articles/sections, always
    # pull those chunks from the index so the model sees the actual law text.
    must_include_ids = list(constraints.allowed_article_ids) + list(constraints.allowed_section_ids)
    must_include = get_chunks_by_source_ids(must_include_ids, max_chunks=10) if must_include_ids else []
    seen_chunk_ids = {c["chunk_id"] for c in must_include}

    # Over-retrieve and filter down by constraints.
    k_base = max(top_k * settings.retrieval.constrained_multiplier, top_k)
    base_results = search(query, k=k_base, index=_index, metadata=_metadata, model=_model)
    constrained = _apply_constraints(base_results, constraints)
    # Add semantic hits that aren't already in must-include
    constrained = [r for r in constrained if r.get("chunk_id") not in seen_chunk_ids]

    if must_include or constrained:
        # Prefer must-include (actual article/section text), then best semantic hits.
        combined = must_include + sorted(constrained, key=lambda r: r.get("score", 0.0), reverse=True)[: max(0, top_k - len(must_include))]
        return {
            "chunks": combined[:top_k],
            "used_constraints": True,
            "used_fallback_unconstrained": False,
            "error": None,
        }

    # No hits that satisfy the constraint – fall back to unconstrained search
    fallback_results = search(query, k=top_k, index=_index, metadata=_metadata, model=_model)
    return {
        "chunks": must_include + fallback_results[: max(0, top_k - len(must_include))],
        "used_constraints": True,
        "used_fallback_unconstrained": True,
        "error": None,
    }


def group_by_source(chunks: Iterable[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Group retrieved chunks by (source_type, source_id) for easier
    prompt construction and UI display.

    Returns a nested dict:
    {
      "case": {
         "<case_id>": {"source_id": ..., "source_type": ..., "chunks": [...], "max_score": ...},
         ...
      },
      "section": { ... },
      "article": { ... },
    }
    """
    grouped: Dict[str, Dict[str, Dict]] = {}
    for ch in chunks:
        stype = ch.get("source_type") or "unknown"
        sid = ch.get("source_id") or "unknown"
        grouped.setdefault(stype, {})
        bucket = grouped[stype].setdefault(
            sid,
            {
                "source_type": stype,
                "source_id": sid,
                "chunks": [],
                "max_score": float(ch.get("score", 0.0)),
            },
        )
        bucket["chunks"].append(ch)
        score = float(ch.get("score", 0.0))
        if score > bucket["max_score"]:
            bucket["max_score"] = score

    return grouped

