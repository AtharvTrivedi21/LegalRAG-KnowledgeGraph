from __future__ import annotations

"""
Thin Neo4j client wrapper for Phase 4.

Responsibilities:
- Manage a singleton Neo4j driver using credentials from config.
- Provide small, typed helpers to fetch:
  * Articles by article_number
  * Sections by section_number
  * Cases that cite a given set of section/article IDs

All methods are safe to call from the LangGraph nodes. Connection errors
are converted into Neo4jUnavailableError so callers can degrade gracefully.
"""

from dataclasses import dataclass
from typing import Iterable, List, Dict

from neo4j import GraphDatabase, basic_auth, Driver
from neo4j.exceptions import ServiceUnavailable

from .config import settings


class Neo4jUnavailableError(RuntimeError):
    """Raised when Neo4j cannot be reached."""


_driver: Driver | None = None


def _get_driver() -> Driver:
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=basic_auth(settings.neo4j.user, settings.neo4j.password),
            )
        except Exception as exc:  # connection or auth errors
            raise Neo4jUnavailableError(str(exc)) from exc
    return _driver


def _run_query(query: str, parameters: dict) -> List[Dict]:
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
    except Exception as exc:
        # Any driver-level issue (unavailable, auth, protocol) is treated as
        # Neo4j being unavailable from the perspective of the RAG pipeline so
        # callers can degrade gracefully to unconstrained retrieval.
        raise Neo4jUnavailableError(str(exc)) from exc


def get_articles_by_numbers(article_numbers: Iterable[str]) -> List[Dict]:
    """
    Resolve Articles by their article_number property.

    Returns dicts with at least: article_id, article_number, act_id, full_text.
    """
    nums = list({n for n in article_numbers if n})
    if not nums:
        return []

    query = """
    MATCH (a:Article)
    WHERE a.article_number IN $nums
    RETURN a.article_id AS article_id,
           a.article_number AS article_number,
           a.act_id AS act_id,
           a.full_text AS full_text
    """
    return _run_query(query, {"nums": nums})


def get_sections_by_numbers(section_numbers: Iterable[str]) -> List[Dict]:
    """
    Resolve Sections by their section_number property.

    Returns dicts with at least: section_id, section_number, act_id, full_text.
    """
    nums = list({n for n in section_numbers if n})
    if not nums:
        return []

    query = """
    MATCH (s:Section)
    WHERE s.section_number IN $nums
    RETURN s.section_id AS section_id,
           s.section_number AS section_number,
           s.act_id AS act_id,
           s.full_text AS full_text
    """
    return _run_query(query, {"nums": nums})


def get_cases_citing_ids(target_ids: Iterable[str]) -> List[Dict]:
    """
    Find cases that cite any of the given Section/Article IDs.

    Input IDs should be canonical section_id/article_id strings.
    Returns dicts with at least: case_id, year.
    """
    ids = list({i for i in target_ids if i})
    if not ids:
        return []

    query = """
    MATCH (c:Case)-[r:CITES]->(t)
    WHERE coalesce(t.section_id, t.article_id) IN $ids
    RETURN DISTINCT c.case_id AS case_id,
                    c.year     AS year
    """
    return _run_query(query, {"ids": ids})


def close_driver() -> None:
    """
    Close the shared driver instance. Useful for tests or graceful shutdown.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None

