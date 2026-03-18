"""
Parse BNS/BNSS/BSA raw page JSONL into Act → Part → Chapter → Section hierarchy.
Section numbers: standalone "1.", "2.", "64." at line start; PART/CHAPTER headers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Section start: number optional letter then dot and space (e.g. "1. ", "64. ", "34A. ")
SECTION_SPLIT_RE = re.compile(
    r"(?=^(\d+[A-Z]?)\.\s+)",
    re.MULTILINE,
)
CHAPTER_RE = re.compile(
    r"CHAPTER\s+([IVXLC\d]+)\s*\n\s*(.+?)(?=\n\d+[A-Z]?\.|CHAPTER\s|PART\s|$)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
PART_RE = re.compile(
    r"PART\s+([IVXLC\d]+)\s*\n\s*(.+?)(?=\n(?:CHAPTER\s|PART\s|\d+[A-Z]?\.)|$)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)

# Split sections by "N. " at line start (so we get section number and rest until next section)
SECTION_SPLIT_RE = re.compile(
    r"(?=^(\d+[A-Z]?)\.\s+)",
    re.MULTILINE,
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
    """Concatenate text from pages; optionally skip is_toc pages."""
    parts = []
    for p in pages:
        if skip_toc and p.get("is_toc"):
            continue
        t = (p.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def parse_sanhita_sections(act_id: str, full_text: str) -> list[dict[str, Any]]:
    """
    Parse sections from BNS/BNSS/BSA full text.
    Split by lines that start with "N. " (section number). Return list of section dicts.
    """
    sections = []
    # Find all match positions for "number. " at line start
    pattern = re.compile(r"^(\d+[A-Z]?)\.\s+", re.MULTILINE)
    matches = list(pattern.finditer(full_text))
    for i, m in enumerate(matches):
        num = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        # First line(s) until first "(" or ".—" is often the heading
        first_line = block.split("\n")[0].strip()
        if ".—" in first_line:
            heading = first_line.split(".—")[0].strip()
        else:
            heading = first_line[:300]
        section_id = f"{act_id}_s{num}"
        sections.append({
            "section_id": section_id,
            "section_number": num,
            "heading": heading[:500] if heading else "",
            "full_text": block[:100000],
            "chapter_number": None,
            "part_number": None,
            "chapter_id": None,
            "part_id": None,
        })
    _assign_chapter_part(full_text, sections)
    return sections


def _assign_chapter_part(full_text: str, sections: list[dict[str, Any]]) -> None:
    """Set chapter_number and part_number for each section from context in full_text."""
    current_part = None
    current_chapter = None
    pos = 0
    for sec in sections:
        # Find this section's position in full_text
        needle = f"{sec['section_number']}. {sec['heading'][:50]}"
        idx = full_text.find(needle, pos)
        if idx == -1:
            idx = full_text.find(sec["section_number"] + ".", pos)
        if idx > 0:
            before = full_text[:idx]
            # Last PART before this section
            for m in PART_RE.finditer(before):
                current_part = m.group(1).strip()
            # Last CHAPTER before this section
            for m in CHAPTER_RE.finditer(before):
                current_chapter = m.group(1).strip()
        sec["part_number"] = current_part
        sec["chapter_number"] = current_chapter
        # Set chapter_id and part_id in place (caller will have part/chapter lists)
        pos = idx + 1 if idx >= 0 else pos


def _assign_part_to_chapters(
    full_text: str,
    parts: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
) -> None:
    """
    Assign each chapter to its parent part based on document position.
    Walks the text sequentially: whenever a PART header is seen, all following
    CHAPTER headers (until the next PART) belong to that part.
    """
    if not parts or not chapters:
        return

    # Build a sorted list of (text_position, part_id) from PART_RE matches
    part_positions: list[tuple[int, str]] = []
    for m in PART_RE.finditer(full_text):
        num = m.group(1).strip()
        part_id = f"{parts[0]['act_id']}_PART_{num}"
        part_positions.append((m.start(), part_id))
    part_positions.sort(key=lambda x: x[0])

    if not part_positions:
        return

    # Build a sorted list of (text_position, chapter_id) from CHAPTER_RE matches
    chapter_positions: list[tuple[int, str]] = []
    for m in CHAPTER_RE.finditer(full_text):
        num = m.group(1).strip()
        chapter_id = f"{parts[0]['act_id']}_CH_{num}"
        chapter_positions.append((m.start(), chapter_id))
    chapter_positions.sort(key=lambda x: x[0])

    # Map chapter_id -> part_id: for each chapter, the parent part is the last
    # PART header that appears before the chapter in the document.
    ch_to_part: dict[str, str] = {}
    for ch_pos, ch_id in chapter_positions:
        current_part_id = None
        for p_pos, p_id in part_positions:
            if p_pos <= ch_pos:
                current_part_id = p_id
            else:
                break
        if current_part_id:
            ch_to_part[ch_id] = current_part_id

    # Apply the mapping to the chapter dicts in-place
    for ch in chapters:
        ch["part_id"] = ch_to_part.get(ch["chapter_id"])


def parse_parts_chapters(act_id: str, full_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract Part and Chapter headers. Returns (parts, chapters) with part_id assigned."""
    parts = []
    chapters = []
    for m in PART_RE.finditer(full_text):
        num = m.group(1).strip()
        title = m.group(2).strip().split("\n")[0].strip()[:300]
        parts.append({
            "part_id": f"{act_id}_PART_{num}",
            "part_number": num,
            "part_title": title,
            "act_id": act_id,
        })
    for m in CHAPTER_RE.finditer(full_text):
        num = m.group(1).strip()
        title = m.group(2).strip().split("\n")[0].strip()[:300]
        chapters.append({
            "chapter_id": f"{act_id}_CH_{num}",
            "chapter_number": num,
            "chapter_title": title,
            "act_id": act_id,
            "part_id": None,
        })
    # Assign each chapter to its parent part by document position
    _assign_part_to_chapters(full_text, parts, chapters)
    return parts, chapters


def parse_act(
    act_id: str,
    raw_pages: list[dict[str, Any]],
    source_file: str,
) -> dict[str, Any]:
    """
    Full parse of one Sanhita (BNS/BNSS/BSA). Returns dict with acts, parts, chapters, sections.
    """
    full_text = full_text_from_pages(raw_pages, skip_toc=True)
    parts, chapters = parse_parts_chapters(act_id, full_text)
    sections = parse_sanhita_sections(act_id, full_text)
    # Build chapter_id and act_id for each section
    ch_by_num = {c["chapter_number"]: c["chapter_id"] for c in chapters}
    part_by_num = {p["part_number"]: p["part_id"] for p in parts}
    for s in sections:
        s["act_id"] = act_id
        s["chapter_id"] = ch_by_num.get(s["chapter_number"]) if s["chapter_number"] else None
        s["part_id"] = part_by_num.get(s["part_number"]) if s["part_number"] else None
    act_meta = {
        "act_id": act_id,
        "short_title": _act_short_title(act_id),
        "year": 2023,
        "act_number": _act_number(act_id),
        "act_type": _act_type(act_id),
        "source_file": source_file,
        "enforcement_date": "2024-07-01",
    }
    return {
        "act": act_meta,
        "parts": parts,
        "chapters": chapters,
        "sections": sections,
    }


def _act_short_title(act_id: str) -> str:
    return {
        "BNS_2023": "Bharatiya Nyaya Sanhita",
        "BNSS_2023": "Bharatiya Nagarik Suraksha Sanhita",
        "BSA_2023": "Bharatiya Sakshya Adhiniyam",
    }.get(act_id, act_id)


def _act_number(act_id: str) -> int | None:
    return {"BNS_2023": 45, "BNSS_2023": 46, "BSA_2023": 47}.get(act_id)


def _act_type(act_id: str) -> str:
    return {"BNS_2023": "Penal", "BNSS_2023": "Procedure", "BSA_2023": "Evidence"}.get(act_id, "General")
