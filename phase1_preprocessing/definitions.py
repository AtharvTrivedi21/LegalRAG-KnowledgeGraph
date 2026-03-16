"""
Extract defined terms from definition-heavy sections (e.g. "In this Act, unless...").
Output: definitions.csv, section_defines_term.csv (edges).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

DEFINITION_SECTION_RE = re.compile(
    r"[Ii]n this [Aa]ct.{0,80}(?:unless|—)",
)
# "(a) "word" means ..." or "(1) "expression" includes ..."
DEFINED_TERM_RE = re.compile(
    r'[(\[]?\s*([a-z0-9]+)\s*[)\]]\s*["\']([^"\']+)["\']\s+(?:means|includes|shall mean|shall include)\s+(.+?)(?=\n\s*[(\[]\s*[a-z0-9]|$)',
    re.DOTALL | re.IGNORECASE,
)
# Simpler: "word" means / "expression" includes
QUOTED_TERM_RE = re.compile(
    r'"([^"]+)"\s+(?:means|includes|shall mean|shall include)\s+(.+?)(?=\n\s*\([a-z]\)|\n\s*"\w|$)',
    re.DOTALL | re.IGNORECASE,
)


def extract_definitions_from_section(
    section_text: str,
    act_id: str,
    section_id: str,
) -> list[dict[str, Any]]:
    """Extract defined terms from a section's full_text. Returns list of definition dicts."""
    if not DEFINITION_SECTION_RE.search(section_text):
        return []
    definitions = []
    # Split by (a), (b), (c) clauses
    clauses = re.split(r'\n\s*\(([a-z])\)\s+', section_text)
    for i, clause in enumerate(clauses):
        if i > 0 and i % 2 == 1:
            continue  # skip the letter we split on
        for m in QUOTED_TERM_RE.finditer(clause):
            term = m.group(1).strip()
            defined_text = m.group(2).strip()[:5000]
            term_slug = term.lower().replace(" ", "_").replace('"', "")
            def_id = f"{act_id}_DEF_{term_slug}"
            definitions.append({
                "def_id": def_id,
                "term": term,
                "defined_text": defined_text,
                "act_id": act_id,
                "section_id": section_id,
            })
    return definitions


def run_definitions_extraction(
    sections_csv: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Read sections.csv, extract definitions from each section, write definitions.csv and section_defines_term.csv.
    Returns (definitions, edges).
    """
    output_dir = Path(output_dir)
    definitions = []
    edges = []
    with open(sections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section_id = row.get("section_id", "")
            act_id = row.get("act_id", "")
            full_text = row.get("full_text", "")
            if not full_text or not section_id:
                continue
            defs = extract_definitions_from_section(full_text, act_id, section_id)
            for d in defs:
                definitions.append(d)
                edges.append({"section_id": section_id, "def_id": d["def_id"]})
    # Dedupe by def_id (keep first)
    seen = set()
    unique_defs = []
    for d in definitions:
        if d["def_id"] in seen:
            continue
        seen.add(d["def_id"])
        unique_defs.append(d)
    edges = [e for e in edges if e["def_id"] in seen]
    _write_csv(output_dir / "definitions.csv", unique_defs, ["def_id", "term", "defined_text", "act_id", "section_id"])
    _write_csv(output_dir / "section_defines_term.csv", edges, ["section_id", "def_id"])
    return unique_defs, edges


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
