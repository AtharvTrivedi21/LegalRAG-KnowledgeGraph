"""
Triage and normalize SC judgment PDFs from Datasets/SC_Judgements-16-25/<year>/*.pdf.
Output: cases_sc.csv with case_id, year, full_text, source_file, source=sc_pdf, text_extractable.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def triage_pdf(pdf_path: Path) -> dict[str, Any]:
    """Return {file, status, page_count, text_extractable, first_chars}."""
    out = {"file": pdf_path.name, "status": "ok", "page_count": 0, "text_extractable": False, "first_chars": ""}
    if not fitz:
        out["status"] = "error: PyMuPDF not installed"
        return out
    try:
        doc = fitz.open(str(pdf_path))
        out["page_count"] = len(doc)
        if len(doc) > 0:
            text = doc[0].get_text("text").strip()
            out["text_extractable"] = len(text) > 50
            out["first_chars"] = text[:100]
        doc.close()
    except Exception as e:
        out["status"] = f"error: {e}"
    return out


def extract_text_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF."""
    if not fitz:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        parts = [doc[i].get_text("text") for i in range(len(doc))]
        doc.close()
        return "\n\n".join(parts)
    except Exception:
        return ""


def run_sc_pdf_normalize(
    extracted_root: Path,
    output_dir: Path,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Scan extracted_root/<year>/*.pdf, triage, extract text, write cases_sc.csv.
    Returns (cases_list, triage_list).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    triage = []
    count = 0
    for year_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        year = 0
        if year_dir.name.isdigit() and len(year_dir.name) == 4:
            try:
                year = int(year_dir.name)
            except ValueError:
                pass
        for pdf_path in sorted(year_dir.glob("*.pdf")) + sorted(year_dir.glob("*.PDF")):
            if limit is not None and count >= limit:
                break
            rec = triage_pdf(pdf_path)
            triage.append(rec)
            case_id = f"sc_{year}_{pdf_path.stem}"
            text = extract_text_pdf(pdf_path) if rec.get("text_extractable") else ""
            cases.append({
                "case_id": case_id,
                "year": year,
                "full_text": text[:500000],
                "source_file": pdf_path.name,
                "source": "sc_pdf",
                "text_extractable": rec.get("text_extractable", False),
            })
            count += 1
        if limit is not None and count >= limit:
            break
    fieldnames = ["case_id", "year", "full_text", "source_file", "source", "text_extractable"]
    with open(output_dir / "cases_sc.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(cases)
    with open(output_dir / "sc_triage.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "status", "page_count", "text_extractable", "first_chars"], extrasaction="ignore")
        w.writeheader()
        w.writerows(triage)
    return cases, triage
