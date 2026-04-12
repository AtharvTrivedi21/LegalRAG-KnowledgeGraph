"""
Neo4j helpers for sys4: act filtering, REFERENCES expansion, citation counts.

All functions degrade gracefully when Neo4j is unavailable (empty dict / passthrough).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from phase4_rag.neo4j_client import Neo4jUnavailableError

TARGET_ACT = "BNS_2023"


def _run_query(query: str, parameters: dict) -> List[Dict[str, Any]]:
    from phase4_rag.neo4j_client import _run_query as rq

    return rq(query, parameters)


def filter_section_ids_by_act(section_ids: List[str], act_id: str = TARGET_ACT) -> Set[str]:
    """
    Return subset of section_ids that exist in Neo4j under the given act.
    If Neo4j fails, returns all non-empty ids (no filtering).
    """
    ids = [i for i in section_ids if i]
    if not ids:
        return set()
    try:
        rows = _run_query(
            """
            MATCH (s:Section)-[:IN_ACT]->(a:Act)
            WHERE s.section_id IN $ids AND a.act_id = $act_id
            RETURN s.section_id AS section_id
            """,
            {"ids": ids, "act_id": act_id},
        )
        return {r["section_id"] for r in rows if r.get("section_id")}
    except Neo4jUnavailableError:
        return set(ids)
    except Exception:
        return set(ids)


def get_citation_counts(section_ids: List[str]) -> Dict[str, int]:
    """
    Count Case-[:CITES]->Section per section_id.
    """
    ids = list({i for i in section_ids if i})
    if not ids:
        return {}
    try:
        rows = _run_query(
            """
            MATCH (c:Case)-[:CITES]->(s:Section)
            WHERE s.section_id IN $ids
            RETURN s.section_id AS section_id, count(c) AS cite_count
            """,
            {"ids": ids},
        )
        return {r["section_id"]: int(r["cite_count"]) for r in rows}
    except (Neo4jUnavailableError, Exception):
        return {}


def expand_reference_sections(
    seed_section_ids: List[str],
    act_id: str = TARGET_ACT,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    For each seed section, follow Section-[:REFERENCES]->Section within the same act.
    Returns up to `limit` unique related sections (dicts with section_id, section_number,
    full_text, act_id, act_name, seed_from).
    """
    seeds = list({s for s in seed_section_ids if s})
    if not seeds:
        return []
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set(seeds)

    try:
        for sid in seeds:
            if len(out) >= limit:
                break
            rows = _run_query(
                """
                MATCH (s:Section {section_id: $sid})-[r:REFERENCES]->(t:Section)-[:IN_ACT]->(a:Act)
                WHERE a.act_id = $act_id
                RETURN t.section_id AS section_id,
                       t.section_number AS section_number,
                       t.full_text AS full_text,
                       a.act_id AS act_id,
                       a.act_name AS act_name
                LIMIT 5
                """,
                {"sid": sid, "act_id": act_id},
            )
            for r in rows:
                tid = r.get("section_id")
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                out.append(
                    {
                        "section_id": tid,
                        "section_number": str(r.get("section_number", "")),
                        "full_text": (r.get("full_text") or "")[:8000],
                        "act_id": r.get("act_id", act_id),
                        "act_name": r.get("act_name", ""),
                        "seed_from": sid,
                    }
                )
                if len(out) >= limit:
                    break
    except (Neo4jUnavailableError, Exception):
        return []

    return out[:limit]
