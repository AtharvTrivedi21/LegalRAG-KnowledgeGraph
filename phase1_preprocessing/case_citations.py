"""
Act-aware citation extraction from case text.
Supports section/s./u/s/sec./read with/r/w and article/art.; resolves act from context.
Output: case_cites_section.csv, case_cites_article.csv, unresolved_case_cites.csv.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

# Allow very large CSV fields (full_text in cases_sc.csv can be several MB)
try:
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
except OverflowError:
    csv.field_size_limit(sys.maxsize)

# Act name patterns -> canonical act_id
ACT_PATTERNS = [
    (re.compile(r"bharatiya\s+nyaya\s+sanhita|BNS\b", re.I), "BNS_2023"),
    (re.compile(r"bharatiya\s+nagarik\s+suraksha\s+sanhita|bharatiya\s+nagrik\s+suraksha|BNSS\b", re.I), "BNSS_2023"),
    (re.compile(r"bharatiya\s+sakshya\s+adhiniyam|BSA\b", re.I), "BSA_2023"),
    (re.compile(r"indian\s+penal\s+code|IPC\b|I\.P\.C", re.I), "IPC_1860"),
    (re.compile(r"code\s+of\s+criminal\s+procedure|CrPC|Cr\.?P\.?C", re.I), "CrPC_1973"),
    (re.compile(r"indian\s+evidence\s+act|IEA\b", re.I), "IEA_1872"),
    (re.compile(r"constitution\s+of\s+india|constitution\b", re.I), "CONST_1950"),
]

# Section: "section 302", "s. 64", "u/s 376", "sec. 2"
SECTION_REF_RE = re.compile(
    r"(?:section|s\.|sec\.|u/s|under\s+section)\s*(\d+[A-Z]?(?:\s*\([^)]+\))*)",
    re.IGNORECASE,
)
# Article: "article 21", "art. 14"
ARTICLE_REF_RE = re.compile(
    r"(?:article|art\.)\s*(\d+[A-Z]?)",
    re.IGNORECASE,
)


def _detect_act_in_context(text: str, start: int, window: int = 150) -> str | None:
    """Return act_id if act mentioned in text around start."""
    s = max(0, start - window)
    e = min(len(text), start + window)
    chunk = text[s:e]
    for pat, act_id in ACT_PATTERNS:
        if pat.search(chunk):
            return act_id
    return None


def _normalize_section_num(s: str) -> str:
    return re.sub(r"\s+", "", s.strip()).upper()


def extract_citations_from_case(
    case_id: str,
    text: str,
    valid_section_ids: set[str],
    valid_article_ids: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Extract section and article citations; resolve act from context.
    Returns (section_cites, article_cites, unresolved).
    """
    section_cites = []
    article_cites = []
    unresolved = []
    seen_sec = set()
    seen_art = set()
    for m in SECTION_REF_RE.finditer(text):
        num = _normalize_section_num(m.group(1))
        act_id = _detect_act_in_context(text, m.start())
        if not act_id:
            act_id = "BNS_2023"  # default for criminal
        if act_id == "CONST_1950":
            sid = f"CONST_1950_Art{num}"
            if sid in valid_article_ids and (case_id, sid) not in seen_art:
                seen_art.add((case_id, sid))
                article_cites.append({"case_id": case_id, "article_id": sid, "context": text[max(0, m.start()-80):m.end()+80].replace("\n", " ")[:300]})
            else:
                unresolved.append({"case_id": case_id, "raw": m.group(0), "target_type": "article", "resolved_id": sid})
        else:
            sid = f"{act_id}_s{num}"
            if sid in valid_section_ids and (case_id, sid) not in seen_sec:
                seen_sec.add((case_id, sid))
                section_cites.append({"case_id": case_id, "section_id": sid, "context": text[max(0, m.start()-80):m.end()+80].replace("\n", " ")[:300]})
            else:
                unresolved.append({"case_id": case_id, "raw": m.group(0), "target_type": "section", "resolved_id": sid})
    for m in ARTICLE_REF_RE.finditer(text):
        num = _normalize_section_num(m.group(1))
        aid = f"CONST_1950_Art{num}"
        if aid in valid_article_ids and (case_id, aid) not in seen_art:
            seen_art.add((case_id, aid))
            article_cites.append({"case_id": case_id, "article_id": aid, "context": text[max(0, m.start()-80):m.end()+80].replace("\n", " ")[:300]})
        else:
            unresolved.append({"case_id": case_id, "raw": m.group(0), "target_type": "article", "resolved_id": aid})
    return section_cites, article_cites, unresolved


def run_case_citation_extraction(
    cases_sc_csv: Path,
    cases_iltur_csv: Path | None,
    sections_csv: Path,
    articles_csv: Path,
    output_dir: Path,
) -> dict[str, int]:
    """
    Load cases from cases_sc.csv and optionally cases_iltur.csv; load valid section/article IDs;
    extract citations; write case_cites_section.csv, case_cites_article.csv, unresolved_case_cites.csv.
    Returns dict with resolved_section, resolved_article, unresolved counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_sections = set()
    valid_articles = set()
    with open(sections_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            valid_sections.add(row.get("section_id", "").strip())
    with open(articles_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            valid_articles.add(row.get("article_id", "").strip())
    all_sec = []
    all_art = []
    all_unres = []
    def process_file(path: Path, text_col: str):
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row.get("case_id", "")
                text = row.get(text_col, "") or row.get("judgment_text", "") or ""
                if not text or len(text) < 50:
                    continue
                s, a, u = extract_citations_from_case(case_id, text, valid_sections, valid_articles)
                all_sec.extend(s)
                all_art.extend(a)
                all_unres.extend(u)
    process_file(cases_sc_csv, "full_text")
    if cases_iltur_csv:
        process_file(cases_iltur_csv, "judgment_text")
    _write_csv(output_dir / "case_cites_section.csv", all_sec, ["case_id", "section_id", "context"])
    _write_csv(output_dir / "case_cites_article.csv", all_art, ["case_id", "article_id", "context"])
    _write_csv(output_dir / "unresolved_case_cites.csv", all_unres, ["case_id", "raw", "target_type", "resolved_id"])
    return {"resolved_section": len(all_sec), "resolved_article": len(all_art), "unresolved": len(all_unres)}


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
