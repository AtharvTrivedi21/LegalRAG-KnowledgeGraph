"""
Load raw JSONL, run act/constitution parsers, write Neo4j-ready CSVs and verify counts.
Then run enrichment: definitions, citations, act index.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .act_parser import load_raw_pages, parse_act
from .act_index_parser import run_act_index_parsing
from .citations import run_citations_extraction
from .constitution_parser import parse_constitution
from .definitions import run_definitions_extraction

# Expected approximate section/article counts (low, high) for verification
EXPECTED_COUNTS = {
    "BNS_2023": ("sections", 358, 200, 500),
    "BNSS_2023": ("sections", 531, 300, 700),
    "BSA_2023": ("sections", 170, 50, 250),
    "CONST_1950": ("articles", 395, 300, 500),
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_structure_parse_and_export(
    raw_pdf_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Load raw_pdf_text/*.jsonl, parse each act, write all CSVs to output_dir.
    Returns summary dict with counts and any verification warnings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_acts = []
    all_parts = []
    all_chapters = []
    all_sections = []
    all_articles = []
    act_part_edges = []
    part_chapter_edges = []
    chapter_section_edges = []
    act_section_edges = []
    act_article_edges = []
    warnings = []

    act_ids_sanhita = ("BNS_2023", "BNSS_2023", "BSA_2023")
    for act_id in act_ids_sanhita:
        jsonl_path = raw_pdf_dir / f"{act_id}.jsonl"
        if not jsonl_path.exists():
            warnings.append(f"Missing raw JSONL: {jsonl_path}")
            continue
        pages = load_raw_pages(jsonl_path)
        source_file = pages[0].get("source_file", "") if pages else ""
        parsed = parse_act(act_id, pages, source_file)
        all_acts.append(parsed["act"])
        all_parts.extend(parsed["parts"])
        all_chapters.extend(parsed["chapters"])
        all_sections.extend(parsed["sections"])
        for p in parsed["parts"]:
            act_part_edges.append({"act_id": p["act_id"], "part_id": p["part_id"]})
        for c in parsed["chapters"]:
            if c.get("part_id"):
                part_chapter_edges.append({"part_id": c["part_id"], "chapter_id": c["chapter_id"]})
        for s in parsed["sections"]:
            act_section_edges.append({"act_id": s["act_id"], "section_id": s["section_id"]})
            if s.get("chapter_id"):
                chapter_section_edges.append({"chapter_id": s["chapter_id"], "section_id": s["section_id"]})

    # Constitution
    const_path = raw_pdf_dir / "CONST_1950.jsonl"
    if const_path.exists():
        pages = load_raw_pages(const_path)
        source_file = pages[0].get("source_file", "") if pages else ""
        parsed = parse_constitution("CONST_1950", pages, source_file)
        all_acts.append(parsed["act"])
        all_parts.extend(parsed["parts"])
        all_articles.extend(parsed["articles"])
        for p in parsed["parts"]:
            act_part_edges.append({"act_id": p["act_id"], "part_id": p["part_id"]})
        for a in parsed["articles"]:
            act_article_edges.append({"act_id": a["act_id"], "article_id": a["article_id"]})

    # Dedupe act_part (keep only act_id -> part_id)
    act_part_edges = [e for e in act_part_edges if e.get("part_id")]

    # Write CSVs
    _write_csv(
        output_dir / "acts.csv",
        all_acts,
        ["act_id", "short_title", "year", "act_number", "act_type", "source_file", "enforcement_date"],
    )
    _write_csv(
        output_dir / "parts.csv",
        all_parts,
        ["part_id", "part_number", "part_title", "act_id"],
    )
    _write_csv(
        output_dir / "chapters.csv",
        all_chapters,
        ["chapter_id", "chapter_number", "chapter_title", "act_id", "part_id"],
    )
    _write_csv(
        output_dir / "sections.csv",
        all_sections,
        ["section_id", "act_id", "chapter_id", "section_number", "heading", "full_text"],
    )
    _write_csv(
        output_dir / "articles.csv",
        all_articles,
        ["article_id", "act_id", "article_number", "heading", "full_text"],
    )
    _write_csv(output_dir / "act_part.csv", act_part_edges, ["act_id", "part_id"])
    _write_csv(output_dir / "part_chapter.csv", part_chapter_edges, ["part_id", "chapter_id"])
    _write_csv(output_dir / "chapter_section.csv", chapter_section_edges, ["chapter_id", "section_id"])
    _write_csv(output_dir / "act_section.csv", act_section_edges, ["act_id", "section_id"])
    _write_csv(output_dir / "act_article.csv", act_article_edges, ["act_id", "article_id"])

    # Verification
    for act_id, (kind, expected, low, high) in EXPECTED_COUNTS.items():
        if kind == "sections":
            count = sum(1 for s in all_sections if s.get("act_id") == act_id)
        else:
            count = sum(1 for a in all_articles if a.get("act_id") == act_id)
        if count < low or count > high:
            warnings.append(f"{act_id}: {kind} count {count} outside expected range [{low}, {high}] (expected ~{expected})")

    # --- Enrichment: definitions, citations, act index ---
    defs, _ = run_definitions_extraction(output_dir / "sections.csv", output_dir)
    refs = run_citations_extraction(output_dir / "sections.csv", output_dir)
    try:
        import config as cfg
        base_path = Path(cfg.BASE_PATH)
        if not base_path.is_absolute():
            base_path = output_dir.parent / base_path
        alp = getattr(cfg, "PDF_ACT_INDEX_ALPHABETICAL", "Albhabetical List of Central Acts.pdf")
        chron = getattr(cfg, "PDF_ACT_INDEX_CHRONOLOGICAL", "Chronological List of Central Acts.pdf")
        idx_rows = run_act_index_parsing(base_path, alp, chron, output_dir)
    except Exception:
        idx_rows = []
    summary = {
        "acts": len(all_acts),
        "parts": len(all_parts),
        "chapters": len(all_chapters),
        "sections": len(all_sections),
        "articles": len(all_articles),
        "warnings": warnings,
        "definitions": len(defs),
        "section_refs": len(refs),
        "act_index_rows": len(idx_rows),
    }
    return summary
