"""
Extract intra-act and cross-act section references from section full_text.
Output: section_references_section.csv (from_section_id, to_section_id, context, reference_type, target_act_id).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

# Within-act: "section 64", "sub-section (1) of section 64"
INTRA_REF_RE = re.compile(
    r'(?:sub-section\s*\([^)]+\)\s+of\s+)?'
    r'section\s+(\d+[A-Z]?(?:\s*\([^)]+\))*)',
    re.IGNORECASE,
)
# Cross-act: "section 2 of the Bharatiya Nagarik Suraksha Sanhita"
CROSS_ACT_REF_RE = re.compile(
    r'(?:section|s\.)\s*(\d+[A-Z]?)\s+of\s+the\s+'
    r'(Bharatiya Nyaya Sanhita|Bharatiya Nagarik Suraksha Sanhita|'
    r'Bharatiya Sakshya Adhiniyam|Indian Penal Code|Code of Criminal Procedure|'
    r'Indian Evidence Act|Constitution of India)',
    re.IGNORECASE,
)
ACT_NAME_TO_ID = {
    "bharatiya nyaya sanhita": "BNS_2023",
    "bharatiya nagarik suraksha sanhita": "BNSS_2023",
    "bharatiya sakshya adhiniyam": "BSA_2023",
    "indian penal code": "IPC_1860",
    "code of criminal procedure": "CrPC_1973",
    "indian evidence act": "IEA_1872",
    "constitution of india": "CONST_1950",
}


def _normalize_section_num(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def extract_references_from_section(
    section_text: str,
    act_id: str,
    section_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract intra-act and cross-act references. Returns (intra_refs, cross_refs).
    Intra: from_section_id, to_section_id (same act), context snippet.
    Cross: from_section_id, to_act_id, to_section_number, context.
    """
    intra = []
    cross = []
    for m in INTRA_REF_RE.finditer(section_text):
        num = _normalize_section_num(m.group(1))
        to_section_id = f"{act_id}_s{num}"
        start = max(0, m.start() - 50)
        end = min(len(section_text), m.end() + 50)
        context = section_text[start:end].replace("\n", " ").strip()
        intra.append({
            "from_section_id": section_id,
            "to_section_id": to_section_id,
            "context": context[:500],
            "reference_type": "see_also",
            "target_act_id": act_id,
        })
    for m in CROSS_ACT_REF_RE.finditer(section_text):
        num = _normalize_section_num(m.group(1))
        act_name = m.group(2).strip().lower()
        to_act_id = ACT_NAME_TO_ID.get(act_name)
        if not to_act_id:
            continue
        to_section_id = f"{to_act_id}_s{num}" if to_act_id != "CONST_1950" else f"{to_act_id}_Art{num}"
        start = max(0, m.start() - 50)
        end = min(len(section_text), m.end() + 50)
        context = section_text[start:end].replace("\n", " ").strip()
        cross.append({
            "from_section_id": section_id,
            "to_section_id": to_section_id,
            "context": context[:500],
            "reference_type": "cross_act",
            "target_act_id": to_act_id,
        })
    return intra, cross


def run_citations_extraction(
    sections_csv: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """
    Read sections.csv, extract references from each section, write section_references_section.csv.
    Returns list of all reference rows.
    """
    output_dir = Path(output_dir)
    all_refs = []
    with open(sections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section_id = row.get("section_id", "")
            act_id = row.get("act_id", "")
            full_text = row.get("full_text", "")
            if not full_text or not section_id:
                continue
            intra, cross = extract_references_from_section(full_text, act_id, section_id)
            all_refs.extend(intra)
            all_refs.extend(cross)
    fieldnames = ["from_section_id", "to_section_id", "context", "reference_type", "target_act_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "section_references_section.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_refs)
    return all_refs
