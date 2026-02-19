from __future__ import annotations

"""
Vector retrieval for Phase 4 V3: adds top_faiss_similarity for confidence guard.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from phase3_embeddings.retrieve import load_index, search

from .config_v3 import settings


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
    global _index, _metadata, _model, _init_error
    if _index is not None and _metadata is not None and _model is not None:
        return True, None
    try:
        _index, _metadata, _model = load_index()
        _init_error = None
        return True, None
    except Exception as exc:
        _index = None
        _metadata = None
        _model = None
        _init_error = str(exc)
        return False, _init_error


def vector_store_status() -> Dict[str, Optional[str]]:
    ok, _ = init_vector_store()
    return {
        "ok": ok,
        "error": _init_error,
        "index_size": int(_index.ntotal) if ok and _index is not None else 0,
    }


def get_chunks_by_source_ids(source_ids: List[str], max_chunks: int = 10) -> List[Dict]:
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


def _diversify_by_source_type(
    results: List[Dict],
    top_k: int,
    min_sections: int,
    min_articles: int,
) -> List[Dict]:
    by_type: Dict[str, List[Dict]] = {"section": [], "article": [], "case": []}
    for r in results:
        stype = r.get("source_type") or "case"
        if stype not in by_type:
            by_type[stype] = []
        by_type.setdefault(stype, []).append(r)
    chosen: List[Dict] = []
    seen_ids: Set[str] = set()

    def add_best(typ: str, limit: int) -> None:
        nonlocal chosen, seen_ids
        lst = by_type.get(typ, [])
        lst_sorted = sorted(lst, key=lambda x: x.get("score", 0.0), reverse=True)
        n = 0
        for r in lst_sorted:
            if n >= limit or len(chosen) >= top_k:
                return
            cid = r.get("chunk_id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                chosen.append(r)
                n += 1

    add_best("section", min_sections)
    add_best("article", min_articles)
    rest = [r for r in results if r.get("chunk_id") not in seen_ids]
    rest_sorted = sorted(rest, key=lambda x: x.get("score", 0.0), reverse=True)
    for r in rest_sorted:
        if len(chosen) >= top_k:
            break
        chosen.append(r)
    return chosen[:top_k]


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
    Returns dict with chunks, used_constraints, used_fallback_unconstrained,
    top_faiss_similarity (max score from initial FAISS search).
    """
    ok, err = init_vector_store()
    if not ok:
        return {
            "chunks": [],
            "used_constraints": False,
            "used_fallback_unconstrained": False,
            "top_faiss_similarity": 0.0,
            "error": f"Vector store not available: {err}",
        }

    top_k = k or settings.retrieval.top_k
    used_constraints = constraints is not None and not constraints.is_empty

    if not used_constraints:
        k_over = min(
            top_k * getattr(settings.retrieval, "diversity_multiplier", 4),
            _index.ntotal if _index is not None else 100,
        )
        k_over = max(k_over, top_k)
        base_results = search(query, k=k_over, index=_index, metadata=_metadata, model=_model)
        top_faiss_similarity = max((r.get("score", 0.0) for r in base_results), default=0.0)
        min_sec = getattr(settings.retrieval, "min_sections_per_query", 2)
        min_art = getattr(settings.retrieval, "min_articles_per_query", 2)
        diversified = _diversify_by_source_type(base_results, top_k, min_sec, min_art)
        return {
            "chunks": diversified,
            "used_constraints": False,
            "used_fallback_unconstrained": False,
            "top_faiss_similarity": top_faiss_similarity,
            "error": None,
        }

    must_include_ids = list(constraints.allowed_article_ids) + list(constraints.allowed_section_ids)
    must_include = get_chunks_by_source_ids(must_include_ids, max_chunks=10) if must_include_ids else []
    seen_chunk_ids = {c["chunk_id"] for c in must_include}

    k_base = max(top_k * settings.retrieval.constrained_multiplier, top_k)
    base_results = search(query, k=k_base, index=_index, metadata=_metadata, model=_model)
    top_faiss_similarity = max((r.get("score", 0.0) for r in base_results), default=0.0)
    constrained = _apply_constraints(base_results, constraints)
    constrained = [r for r in constrained if r.get("chunk_id") not in seen_chunk_ids]

    if must_include or constrained:
        combined = must_include + sorted(constrained, key=lambda r: r.get("score", 0.0), reverse=True)[: max(0, top_k - len(must_include))]
        return {
            "chunks": combined[:top_k],
            "used_constraints": True,
            "used_fallback_unconstrained": False,
            "top_faiss_similarity": top_faiss_similarity,
            "error": None,
        }

    fallback_results = search(query, k=top_k, index=_index, metadata=_metadata, model=_model)
    fallback_top = max((r.get("score", 0.0) for r in fallback_results), default=0.0)
    return {
        "chunks": must_include + fallback_results[: max(0, top_k - len(must_include))],
        "used_constraints": True,
        "used_fallback_unconstrained": True,
        "top_faiss_similarity": max(top_faiss_similarity, fallback_top),
        "error": None,
    }


def group_by_source(chunks: Iterable[Dict]) -> Dict[str, Dict[str, Dict]]:
    grouped: Dict[str, Dict[str, Dict]] = {}
    for ch in chunks:
        stype = ch.get("source_type") or "unknown"
        sid = ch.get("source_id") or "unknown"
        grouped.setdefault(stype, {})
        bucket = grouped[stype].setdefault(
            sid,
            {"source_type": stype, "source_id": sid, "chunks": [], "max_score": float(ch.get("score", 0.0))},
        )
        bucket["chunks"].append(ch)
        score = float(ch.get("score", 0.0))
        if score > bucket["max_score"]:
            bucket["max_score"] = score
    return grouped
