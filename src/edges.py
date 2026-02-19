"""
Extract Section/Article references from judgment text and build citation edges.
Edges: source_case_id | target_section_or_article | relation (CITES / REFERS)
"""
import re
from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple

import pandas as pd

# Regex to find Section X and Article X in text
SECTION_REF = re.compile(r"Section\s+(\d+(?:\(\d+\))?)", re.IGNORECASE)
ARTICLE_REF = re.compile(r"Article\s+(\d+(?:\(\d+\))?)", re.IGNORECASE)


def _normalize_num(num: str) -> str:
    """Normalize section/article number for matching (e.g. 41(1) -> 41(1))."""
    return num.strip()


def extract_citations_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract unique Section and Article references from judgment text.
    Returns (list of section nums, list of article nums).
    """
    sections = set()
    articles = set()
    for m in SECTION_REF.finditer(text):
        sections.add(_normalize_num(m.group(1)))
    for m in ARTICLE_REF.finditer(text):
        articles.add(_normalize_num(m.group(1)))
    return list(sections), list(articles)


def build_target_ids(
    section_nums: list[str],
    article_nums: list[str],
    section_num_to_ids: DefaultDict[str, List[str]],
    valid_article_ids: Set[str],
) -> List[Tuple[str, str]]:
    """
    Map raw numbers to target IDs (Constitution_Art_X, BNS_Sec_X, etc.).
    Only includes targets that exist in valid_*_ids.
    Returns list of (target_id, relation).
    """
    edges = []
    for num in article_nums:
        # Constitution articles: Constitution_Art_14, Constitution_Art_32(1)
        aid = f"Constitution_Art_{num}"
        if aid in valid_article_ids:
            edges.append((aid, "CITES"))
    for num in section_nums:
        # Sections can exist in BNS, BNSS, BSA - add edge for each matching act
        for sid in section_num_to_ids.get(num, []):
            edges.append((sid, "CITES"))
    return edges


def build_edges_df(
    cases_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    sections_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build edges from cases to cited sections/articles.
    Only cites targets that exist in articles_df or sections_df.
    """
    valid_article_ids = (
        set(articles_df["article_id"].astype(str)) if len(articles_df) > 0 else set()
    )

    section_num_to_ids: DefaultDict[str, List[str]] = defaultdict(list)
    if len(sections_df) > 0:
        for _, srow in sections_df.iterrows():
            num = str(srow.get("section_number", "")).strip()
            sid = str(srow.get("section_id", "")).strip()
            if num and sid:
                section_num_to_ids[num].append(sid)

    rows = []
    for _, row in cases_df.iterrows():
        case_id = row["case_id"]
        text = str(row.get("judgment_text", ""))
        section_nums, article_nums = extract_citations_from_text(text)
        targets = build_target_ids(
            section_nums, article_nums, section_num_to_ids, valid_article_ids
        )
        for target_id, relation in targets:
            rows.append({
                "source_case_id": case_id,
                "target_section_or_article": target_id,
                "relation": relation,
            })

    return pd.DataFrame(rows)
