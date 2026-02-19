from __future__ import annotations

"""
Query parser for Phase 4 V3.

Responsibilities:
- Detect explicit references to Articles and Sections in the user query.
- Detect Act mentions (BNS, BNSS, BSA, Constitution) for act-aware disambiguation.
- Return a structured ParsedQuery object used by the LangGraph workflow.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


ARTICLE_PATTERN = re.compile(r"Article\s+(\d+[A-Z]?)", flags=re.IGNORECASE)
SECTION_PATTERN = re.compile(r"Section\s+(\d+(?:\(\d+\))?)", flags=re.IGNORECASE)

# Act detection: canonical act_ids and patterns to detect them in text
ACT_ALIASES = {
    "BNS": [
        r"\bBNS\b",
        r"bharatiya\s+nyaya\s+sanhita",
        r"bharatiya\s+nyaya\s+sanhita\s*\(?\s*bns\s*\)?",
    ],
    "BNSS": [
        r"\bBNSS\b",
        r"bharatiya\s+nagarik\s+suraksha\s+sanhita",
        r"bharatiya\s+nagrik\s+suraksha\s+sanhita",
        r"bharatiya\s+nagarik\s+suraksha\s+sanhita\s*\(?\s*bnss\s*\)?",
    ],
    "BSA": [
        r"\bBSA\b",
        r"bharatiya\s+sakshya\s+adhiniyam",
        r"bharatiya\s+sakshya\s+adhiniyam\s*\(?\s*bsa\s*\)?",
    ],
    "Constitution": [
        r"\bconstitution\s+of\s+india\b",
        r"article\s+\d+[A-Z]?\s+(?:of\s+)?(?:the\s+)?constitution",
        r"\bconstitution\b",
    ],
}


def _detect_act(text: str) -> Optional[str]:
    """Detect explicit Act mention in query. Returns canonical act_id or None."""
    text_lower = text.lower().strip()
    for act_id, patterns in ACT_ALIASES.items():
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                return act_id
    return None


def _detect_section_act(text: str) -> Optional[str]:
    """Act mentioned in context of sections (e.g. 'Section 302 of BNS')."""
    if not SECTION_PATTERN.search(text):
        return None
    return _detect_act(text)


def _detect_article_act(text: str) -> Optional[str]:
    """Act mentioned in context of articles (e.g. 'Article 14 of Constitution')."""
    if not ARTICLE_PATTERN.search(text):
        return None
    return _detect_act(text)


@dataclass
class ParsedQuery:
    raw_query: str
    article_numbers: List[str] = field(default_factory=list)
    section_numbers: List[str] = field(default_factory=list)
    explicit_ids: Dict[str, List[str]] = field(default_factory=dict)
    section_act_id: Optional[str] = None
    article_act_id: Optional[str] = None

    @property
    def has_explicit_refs(self) -> bool:
        return bool(self.article_numbers or self.section_numbers or self.explicit_ids)


def parse_query(text: str) -> ParsedQuery:
    """
    Parse a user query for explicit Article/Section mentions and Act hints.
    """
    article_nums = [m.group(1).strip() for m in ARTICLE_PATTERN.finditer(text)]
    section_nums = [m.group(1).strip() for m in SECTION_PATTERN.finditer(text)]

    section_act = _detect_section_act(text) if section_nums else None
    article_act = _detect_article_act(text) if article_nums else None

    if section_nums and section_act is None:
        section_act = _detect_act(text)
    if article_nums and article_act is None:
        article_act = _detect_act(text)

    return ParsedQuery(
        raw_query=text,
        article_numbers=article_nums,
        section_numbers=section_nums,
        explicit_ids={},
        section_act_id=section_act,
        article_act_id=article_act,
    )
