from __future__ import annotations

"""
Query parser for Phase 4.

Responsibilities:
- Detect explicit references to Articles and Sections in the user query.
- Return a structured ParsedQuery object used by the LangGraph workflow.

This parser is intentionally simple and deterministic, using regex patterns
aligned with Phase 1 citation extraction.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


ARTICLE_PATTERN = re.compile(r"Article\s+(\d+[A-Z]?)", flags=re.IGNORECASE)
SECTION_PATTERN = re.compile(r"Section\s+(\d+(?:\(\d+\))?)", flags=re.IGNORECASE)


@dataclass
class ParsedQuery:
    raw_query: str
    article_numbers: List[str] = field(default_factory=list)
    section_numbers: List[str] = field(default_factory=list)
    explicit_ids: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def has_explicit_refs(self) -> bool:
        return bool(self.article_numbers or self.section_numbers or self.explicit_ids)


def parse_query(text: str) -> ParsedQuery:
    """
    Parse a user query for explicit Article/Section mentions.

    For now we:
    - Capture raw article/section numbers using regex.
    - Do not hard-code act-specific IDs here; mapping to concrete
      article_id/section_id is handled in the graph_retriever node
      via Neo4j lookups.
    """
    article_nums = [m.group(1).strip() for m in ARTICLE_PATTERN.finditer(text)]
    section_nums = [m.group(1).strip() for m in SECTION_PATTERN.finditer(text)]

    return ParsedQuery(
        raw_query=text,
        article_numbers=article_nums,
        section_numbers=section_nums,
        explicit_ids={},
    )
