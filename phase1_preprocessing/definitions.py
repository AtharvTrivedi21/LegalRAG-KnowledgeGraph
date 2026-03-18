"""
Extract defined terms from definition-heavy sections (e.g. "In this Act, unless...").
Output: definitions.csv, section_defines_term.csv (edges).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

# Guard: detect definition sections. Indian legal PDFs use "Sanhita" (BNS/BNSS)
# and "Adhiniyam" (BSA) instead of "Act". The preamble always says
# "In this Sanhita/Act/Adhiniyam, unless the context otherwise requires"
# so matching both phrases anywhere in the section is sufficient.
DEFINITION_SECTION_RE = re.compile(
    r"[Ii]n this (?:[Aa]ct|[Ss]anhita|[Aa]dhiniyam)",
)

# Handles both straight ASCII quotes and Unicode curly/smart quotes.
# Also handles "denotes" which BNS s.2 uses for some terms.
_VERBS = r"(?:means|includes|denotes|shall mean|shall include)"

# Quote character class built with explicit alternation to avoid range issues.
# BNS/BNSS use U+201C (left double quotation mark) and U+201D (right double
# quotation mark). BSA and Constitution may use straight " as well.
_AQ = r'(?:\u201c|\u201d|\u2018|\u2019|"|\u2032|\u2033)'

# Clause-numbered: (a) "word" means ... / (1) "expression" includes ...
DEFINED_TERM_RE = re.compile(
    rf'[(\[]?\s*([a-zA-Z0-9]+)\s*[)\]]\s*{_AQ}([^\u201c\u201d"\n]{{1,80}}){_AQ}\s+{_VERBS}\s+(.+?)(?=\n\s*[(\[]\s*[a-z0-9]|$)',
    re.DOTALL | re.IGNORECASE,
)

# Primary: "word" means / "expression" includes — works on BNS/BNSS/BSA s.2
# Sub-clauses are numbered like (1) "act" denotes... or lettered (a) "bail" means...
# Lookahead stops at the next sub-clause opener or the start of another quoted term.
QUOTED_TERM_RE = re.compile(
    rf'{_AQ}([^\u201c\u201d"\n]{{1,80}}){_AQ}\s+{_VERBS}\s+(.+?)(?=\n\s*\([a-zA-Z0-9]+\)|\n\s*{_AQ}|$)',
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
    seen_terms: set[str] = set()

    # Apply QUOTED_TERM_RE directly over the full section text. The regex
    # lookahead already stops at the next sub-clause marker, so splitting first
    # is not necessary and can drop terms when clause markers are missing.
    for m in QUOTED_TERM_RE.finditer(section_text):
        term = m.group(1).strip()
        defined_text = m.group(2).strip()[:5000]
        if not term or term.lower() in seen_terms:
            continue
        seen_terms.add(term.lower())
        term_slug = re.sub(r'[^a-z0-9]+', '_', term.lower()).strip('_')
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
