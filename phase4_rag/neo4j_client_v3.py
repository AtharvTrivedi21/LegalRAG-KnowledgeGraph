from __future__ import annotations

"""
Neo4j client for Phase 4 V3: Act-aware section disambiguation.

Extends base client with act_id filtering and act_name in results.
"""

from typing import Iterable, List, Dict, Optional

from .neo4j_client import Neo4jUnavailableError, get_cases_citing_ids, _run_query


def get_articles_by_numbers(
    article_numbers: Iterable[str],
    act_id: Optional[str] = None,
) -> List[Dict]:
    """
    Resolve Articles by article_number. When act_id is provided, filter to that Act.
    Returns dicts with: article_id, article_number, act_id, act_name, full_text.
    """
    nums = list({n for n in article_numbers if n})
    if not nums:
        return []

    query = """
    MATCH (ar:Article)-[:IN_ACT]->(a:Act)
    WHERE ar.article_number IN $nums
      AND ($act_id IS NULL OR a.act_id = $act_id)
    RETURN ar.article_id AS article_id,
           ar.article_number AS article_number,
           a.act_id AS act_id,
           a.act_name AS act_name,
           ar.full_text AS full_text
    """
    return _run_query(query, {"nums": nums, "act_id": act_id})


def get_sections_by_numbers(
    section_numbers: Iterable[str],
    act_id: Optional[str] = None,
) -> List[Dict]:
    """
    Resolve Sections by section_number. When act_id is provided, filter to that Act.
    Returns dicts with: section_id, section_number, act_id, act_name, full_text.
    """
    nums = list({n for n in section_numbers if n})
    if not nums:
        return []

    query = """
    MATCH (s:Section)-[:IN_ACT]->(a:Act)
    WHERE s.section_number IN $nums
      AND ($act_id IS NULL OR a.act_id = $act_id)
    RETURN s.section_id AS section_id,
           s.section_number AS section_number,
           a.act_id AS act_id,
           a.act_name AS act_name,
           s.full_text AS full_text
    """
    return _run_query(query, {"nums": nums, "act_id": act_id})
