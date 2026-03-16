"""
Parse Central Acts index PDFs (Alphabetical and Chronological) into act_index.csv.
Uses pdfplumber for table extraction. Output: index_id, short_title, year, act_number, category.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


def _classify_act(title: str) -> str:
    """Classify act into category from title keywords."""
    if not title:
        return "General"
    t = title.lower()
    if any(w in t for w in ["penal", "criminal", "punishment", "offence"]):
        return "Criminal"
    if any(w in t for w in ["tax", "income", "customs", "excise", "gst"]):
        return "Tax"
    if any(w in t for w in ["civil", "procedure", "arbitration"]):
        return "Civil Procedure"
    if any(w in t for w in ["constitution", "amendment"]):
        return "Constitutional"
    if any(w in t for w in ["company", "corporation", "trade"]):
        return "Commercial"
    return "General"


def _extract_year(val: Any) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    m = re.search(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return y if 1800 <= y <= 2030 else None
    return None


def _extract_act_number(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    m = re.search(r"(\d+)\s*of", s) or re.search(r"Act\s*(\d+)", s, re.I) or re.search(r"^(\d+)$", s)
    return m.group(1) if m else s[:20]


def parse_index_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract table rows from index PDF. Returns list of {short_title, year, act_number, category}."""
    if pdfplumber is None:
        return []
    rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            # Assume header row; data rows have at least title and year/number
            for i, row in enumerate(table):
                if not row or not any(cell and str(cell).strip() for cell in row):
                    continue
                cells = [str(c).strip() if c else "" for c in row]
                # Skip header
                if i == 0 and any("S.No" in c or "No." in c for c in cells):
                    continue
                title = ""
                year_val = None
                act_num = ""
                for j, c in enumerate(cells):
                    if not c:
                        continue
                    if re.match(r"^\d{4}$", c) or re.search(r"\d{4}", c):
                        year_val = _extract_year(c)
                    elif re.search(r"Act\s*\d+|^\d+\s*of", c, re.I) or (len(c) <= 5 and c.isdigit()):
                        act_num = _extract_act_number(c) or act_num
                    elif len(c) > 10 and not title:
                        title = c[:300]
                if not title and cells:
                    title = cells[0] or "Unknown"
                if title and ("S.No" not in title and "No." not in title):
                    index_id = f"IDX_{_extract_year(title) or 0}_{re.sub(r'[^A-Za-z0-9]', '_', title[:30])}"
                    rows.append({
                        "index_id": index_id[:80],
                        "short_title": title[:500],
                        "year": year_val or 0,
                        "act_number": act_num or "",
                        "category": _classify_act(title),
                    })
    return rows


def run_act_index_parsing(
    base_path: Path,
    alphabetical_filename: str,
    chronological_filename: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """
    Parse both index PDFs, merge and dedupe by title+year, write act_index.csv.
    """
    output_dir = Path(output_dir)
    all_rows = []
    for fname in (alphabetical_filename, chronological_filename):
        path = base_path / fname
        if path.exists():
            all_rows.extend(parse_index_pdf(path))
    # Dedupe by (short_title, year)
    seen = set()
    unique = []
    for r in all_rows:
        key = (r.get("short_title", "")[:200], r.get("year"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    fieldnames = ["index_id", "short_title", "year", "act_number", "category"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "act_index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(unique)
    return unique
