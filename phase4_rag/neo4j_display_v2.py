"""
Phase 4 v2: Neo4j display helpers for the Streamlit UI.

Fetch act names and case details for the Cited references panels.
Uses its own driver from config; does not modify neo4j_client.py.
On connection/query failure, returns empty lists so the UI can still
show retrieval-based citations.
"""

from __future__ import annotations

from typing import Dict, List

from neo4j import GraphDatabase, basic_auth, Driver

from .config import settings

_driver: Driver | None = None


def _get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=basic_auth(settings.neo4j.user, settings.neo4j.password),
        )
    return _driver


def _run_query(query: str, parameters: dict) -> List[Dict]:
    try:
        driver = _get_driver()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
    except Exception:
        return []


def get_acts_by_ids(act_ids: List[str]) -> List[Dict]:
    """
    Fetch Act nodes by act_id for the Cited Acts panel.

    Returns list of {act_id, act_name, act_type}. Returns [] on error.
    """
    ids = list({i for i in act_ids if i})
    if not ids:
        return []
    query = """
    MATCH (a:Act)
    WHERE a.act_id IN $ids
    RETURN a.act_id AS act_id,
           a.act_name AS act_name,
           a.act_type AS act_type
    """
    return _run_query(query, {"ids": ids})


def get_case_details(case_ids: List[str], snippet_len: int = 300) -> List[Dict]:
    """
    Fetch Case nodes with a short judgment_text snippet for the Cited Cases panel.

    Returns list of {case_id, year, snippet}. Returns [] on error.
    """
    ids = list({i for i in case_ids if i})
    if not ids:
        return []
    query = """
    MATCH (c:Case)
    WHERE c.case_id IN $ids
    RETURN c.case_id AS case_id,
           c.year AS year,
           c.judgment_text AS judgment_text
    """
    rows = _run_query(query, {"ids": ids})
    out = []
    for r in rows:
        text = (r.get("judgment_text") or "").strip()
        snippet = text[:snippet_len] + "..." if len(text) > snippet_len else text
        out.append({
            "case_id": r.get("case_id"),
            "year": r.get("year"),
            "snippet": snippet,
        })
    return out
