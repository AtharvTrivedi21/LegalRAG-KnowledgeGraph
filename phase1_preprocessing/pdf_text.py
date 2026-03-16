"""
Robust statutory PDF preprocessing using PyMuPDF.
Extracts text per page with header/footer removal, dehyphenation, TOC detection.
Outputs raw page JSONL per act for downstream structural parsing.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]


# Canonical act_id for output filenames (phase1_output_v2/raw_pdf_text/{act_id}.jsonl)
CONFIG_KEY_TO_ACT_ID = {
    "constitution": "CONST_1950",
    "bns": "BNS_2023",
    "bnss": "BNSS_2023",
    "bsa": "BSA_2023",
}

# Lines that look like headers/footers (repeated across many pages)
HEADER_FOOTER_PATTERNS = (
    r"^\s*\d+\s*$",  # page number alone
    r"^THE BHARATIYA NYAYA SANHITA",
    r"^THE BHARATIYA NAGARIK SURAKSHA SANHITA",
    r"^THE BHARATIYA SAKSHYA ADHINIYAM",
    r"^THE CONSTITUTION OF INDIA",
    r"^\s*SECTIONS\s*$",
    r"^\s*\(?\d+\)\s*$",
    r"^Page\s+\d+",
    r"^-\s*\d+\s*-",
)
HEADER_FOOTER_RE = re.compile("|".join(f"({p})" for p in HEADER_FOOTER_PATTERNS), re.IGNORECASE | re.MULTILINE)

# TOC indicators: skip or mark pages that are mostly TOC
TOC_INDICATORS = (
    "CONTENTS",
    "Contents",
    "ARTICLES",
    "SECTIONS",
    "CHAPTER",
    "PART ",
)
# Dotted leader pattern (e.g. "1. Short title .............. 1")
DOTTED_LEADER_RE = re.compile(r"\.\s{2,}\d+\s*$", re.MULTILINE)


def _dehyphenate(text: str) -> str:
    """Join words broken across lines by hyphen (e.g. 'law-\nful' -> 'lawful')."""
    # Replace hyphen-newline-optional spaces with nothing (join the word)
    text = re.sub(r"-\s*\n\s*", "", text)
    # Normalize whitespace: collapse multiple spaces/newlines to single space, keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _strip_headers_footers(lines: list[str], line_counts: Counter[str] | None) -> list[str]:
    """
    Remove lines that appear on many pages (likely header/footer).
    If line_counts is None, we only remove lines matching common header/footer patterns.
    """
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        if HEADER_FOOTER_RE.match(stripped):
            continue
        if line_counts is not None and len(line_counts) > 0:
            if line_counts.get(stripped, 0) > 3:  # repeated on more than 3 pages
                continue
        out.append(line)
    return out


def _is_likely_toc_page(text: str, min_toc_score: float = 0.15) -> bool:
    """
    Heuristic: page is TOC if it has CONTENTS/ARTICLES/SECTIONS and many dotted leaders or short lines.
    """
    text_upper = text.upper()
    has_toc_marker = any(m in text_upper for m in ("CONTENTS", "ARTICLES", "SECTIONS"))
    dotted = len(DOTTED_LEADER_RE.findall(text))
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short_lines = sum(1 for l in lines if len(l) < 80)
    score = 0.0
    if has_toc_marker:
        score += 0.3
    if lines:
        score += 0.4 * min(1.0, dotted / max(1, len(lines)))
        score += 0.3 * min(1.0, short_lines / max(1, len(lines)))
    return score >= min_toc_score


def extract_pages_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract text per page using PyMuPDF. No header/footer or TOC filtering here.
    Returns list of {page_num, text, char_start, char_end}.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
    doc = fitz.open(str(pdf_path))
    pages = []
    char_offset = 0
    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text("text")
        text = _dehyphenate(text)
        start = char_offset
        char_offset += len(text) + 1
        pages.append({
            "page_num": page_num,
            "text": text,
            "char_start": start,
            "char_end": char_offset - 1,
        })
    doc.close()
    return pages


def compute_line_counts(pages: list[dict[str, Any]]) -> Counter[str]:
    """Count how often each non-empty line appears across all pages (for header/footer detection)."""
    counter: Counter[str] = Counter()
    for p in pages:
        for line in p["text"].splitlines():
            s = line.strip()
            if s:
                counter[s] += 1
    return counter


def process_pages(
    pages: list[dict[str, Any]],
    *,
    strip_headers_footers: bool = True,
    skip_toc_pages: bool = True,
) -> list[dict[str, Any]]:
    """
    Apply header/footer removal and optionally skip TOC pages.
    Returns list of {page_num, text, char_start, char_end, is_toc}.
    """
    line_counts = compute_line_counts(pages) if strip_headers_footers else None
    result = []
    for p in pages:
        text = p["text"]
        is_toc = _is_likely_toc_page(text)
        if skip_toc_pages and is_toc:
            result.append({
                "page_num": p["page_num"],
                "text": "",
                "char_start": p["char_start"],
                "char_end": p["char_end"],
                "is_toc": True,
            })
            continue
        if strip_headers_footers and line_counts:
            lines = text.splitlines()
            filtered = _strip_headers_footers(lines, line_counts)
            text = "\n".join(filtered)
            text = _dehyphenate(text)
        result.append({
            "page_num": p["page_num"],
            "text": text,
            "char_start": p["char_start"],
            "char_end": p["char_end"],
            "is_toc": False,
        })
    return result


def extract_act_pdf(
    pdf_path: Path,
    act_id: str,
    source_file: str,
    *,
    strip_headers_footers: bool = True,
    skip_toc_pages: bool = True,
) -> list[dict[str, Any]]:
    """
    Full pipeline: load PDF, extract pages, preprocess, return list of page dicts.
    """
    pages = extract_pages_pymupdf(pdf_path)
    processed = process_pages(
        pages,
        strip_headers_footers=strip_headers_footers,
        skip_toc_pages=skip_toc_pages,
    )
    for p in processed:
        p["act_id"] = act_id
        p["source_file"] = source_file
    return processed


def write_raw_jsonl(
    page_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Write one JSON object per line (JSONL)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in page_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_statute_preprocessing(
    base_path: Path,
    pdf_config: dict[str, str],
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """
    Process all statutory PDFs from pdf_config; write raw_pdf_text/{act_id}.jsonl.
    Constitution: do NOT skip TOC pages (so we have full text for article extraction).
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_pdf_text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for key, filename in pdf_config.items():
        act_id = CONFIG_KEY_TO_ACT_ID.get(key)
        if not act_id:
            continue
        pdf_path = base_path / filename
        if not pdf_path.exists():
            continue
        # Constitution: keep all pages (don't skip TOC) so we have full text for articles
        skip_toc = act_id != "CONST_1950"
        page_records = extract_act_pdf(
            pdf_path,
            act_id=act_id,
            source_file=filename,
            strip_headers_footers=True,
            skip_toc_pages=skip_toc,
        )
        out_path = raw_dir / f"{act_id}.jsonl"
        write_raw_jsonl(page_records, out_path)
        all_results[act_id] = page_records
    return all_results
