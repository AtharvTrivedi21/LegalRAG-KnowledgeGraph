"""
Parse Constitution (CONST_1950) raw page JSONL into Part → Article hierarchy.
Articles: "1. Name and territory" style; Parts: "PART I", "PART II".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Article: number optionally followed by letter (21A, 370) then dot and space and a capital letter (heading)
# Avoid matching "1. Subs. by" or "2. Ins. by" by requiring capital after the number.
ARTICLE_START_RE = re.compile(r"^(\d+[A-Z]?)\.\s+[A-Z]", re.MULTILINE)
PART_RE = re.compile(
    r"PART\s+([IVXLC\d]+)\s*\n",
    re.MULTILINE | re.IGNORECASE,
)


def load_raw_pages(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load JSONL of page records."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def full_text_from_pages(pages: list[dict[str, Any]], skip_toc: bool = True) -> str:
    """Concatenate text from pages; optionally skip is_toc."""
    parts = []
    for p in pages:
        if skip_toc and p.get("is_toc"):
            continue
        t = (p.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def parse_constitution_articles(act_id: str, full_text: str) -> list[dict[str, Any]]:
    """Split by article number at line start. Keep only main articles 1-395 and amendment suffixes (e.g. 21A)."""
    articles = []
    matches = list(ARTICLE_START_RE.finditer(full_text))
    seen_numbers = set()
    for i, m in enumerate(matches):
        num = m.group(1).strip()
        # Allow 1-395 or amendment style 21A, 32A (digits + single optional letter)
        num_digits = re.match(r"^\d+", num)
        if not num_digits:
            continue
        n = int(num_digits.group(0))
        if n < 1 or n > 395:
            continue
        if num in seen_numbers:
            continue
        seen_numbers.add(num)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        first_line = block.split("\n")[0].strip()
        if ".—" in first_line:
            heading = first_line.split(".—")[0].strip()
        else:
            heading = first_line[:300]
        article_id = f"{act_id}_Art{num}"
        articles.append({
            "article_id": article_id,
            "article_number": num,
            "heading": heading[:500] if heading else "",
            "full_text": block[:100000],
            "part_number": None,
        })
    _assign_part(full_text, articles)
    return articles


def _assign_part(full_text: str, articles: list[dict[str, Any]]) -> None:
    """Set part_number from last PART before each article."""
    current_part = None
    pos = 0
    for art in articles:
        needle = f"{art['article_number']}. "
        idx = full_text.find(needle, pos)
        if idx > 0:
            before = full_text[:idx]
            for m in PART_RE.finditer(before):
                current_part = m.group(1).strip()
        art["part_number"] = current_part
        pos = idx + 1 if idx >= 0 else pos


def parse_constitution_parts(act_id: str, full_text: str) -> list[dict[str, Any]]:
    """Extract PART headers."""
    parts = []
    for m in PART_RE.finditer(full_text):
        num = m.group(1).strip()
        # Title often on next line(s)
        start = m.end()
        end = full_text.find("\n\n", start)
        if end == -1:
            end = min(start + 200, len(full_text))
        title = full_text[start:end].replace("\n", " ").strip()[:300]
        parts.append({
            "part_id": f"{act_id}_PART_{num}",
            "part_number": num,
            "part_title": title or f"Part {num}",
            "act_id": act_id,
        })
    return parts


def parse_constitution(
    act_id: str,
    raw_pages: list[dict[str, Any]],
    source_file: str,
) -> dict[str, Any]:
    """Full parse of Constitution. Returns dict with act, parts, articles."""
    # Use all pages (skip_toc=False) so we have full text for article extraction
    full_text = full_text_from_pages(raw_pages, skip_toc=False)
    parts = parse_constitution_parts(act_id, full_text)
    articles = parse_constitution_articles(act_id, full_text)
    part_by_num = {p["part_number"]: p["part_id"] for p in parts}
    for a in articles:
        a["act_id"] = act_id
        a["part_id"] = part_by_num.get(a["part_number"])
    act_meta = {
        "act_id": act_id,
        "short_title": "Constitution of India",
        "year": 1950,
        "act_number": None,
        "act_type": "Constitutional",
        "source_file": source_file,
        "enforcement_date": "1950-01-26",
    }
    return {
        "act": act_meta,
        "parts": parts,
        "chapters": [],
        "articles": articles,
        "sections": [],
    }
