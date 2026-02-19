"""
Extract Articles and Sections from legal PDFs using pdfplumber and regex.
Produces structured tables: act_name, article_or_section_number, full_text.
"""
import re
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd
import pdfplumber

# Regex patterns for Indian legal documents
ARTICLE_PATTERN = re.compile(
    r"Article\s+(\d+(?:\(\d+\))?)\s*[-–:]?\s*(.*?)(?=Article\s+\d|$)",
    re.DOTALL | re.IGNORECASE,
)
SECTION_PATTERN = re.compile(
    r"Section\s+(\d+(?:\(\d+\))?)\s*[-–:]?\s*(.*?)(?=Section\s+\d|$)",
    re.DOTALL | re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from PDF pages using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def extract_structured_from_pdf(
    pdf_path: Path,
    act_name: str,
    pattern_type: Literal["article", "section"],
) -> pd.DataFrame:
    """
    Extract Articles or Sections from PDF using regex.
    Returns DataFrame: act_name, article_or_section_number, full_text
    """
    full_text = extract_text_from_pdf(pdf_path)
    pattern = ARTICLE_PATTERN if pattern_type == "article" else SECTION_PATTERN
    num_col = "article_number" if pattern_type == "article" else "section_number"

    rows = []
    for m in pattern.finditer(full_text):
        num = m.group(1).strip()
        body = m.group(2).strip() if m.lastindex >= 2 else ""
        body = re.sub(r"\s+", " ", body).strip()
        if num and (body or len(num) < 20):
            rows.append({num_col: num, "full_text": body, "act_name": act_name})

    if not rows:
        # Fallback: split by heading lines only (simpler pattern)
        fallback_pattern = (
            re.compile(r"Article\s+(\d+(?:\(\d+\))?)", re.IGNORECASE)
            if pattern_type == "article"
            else re.compile(r"Section\s+(\d+(?:\(\d+\))?)", re.IGNORECASE)
        )
        matches = list(fallback_pattern.finditer(full_text))
        for i, m in enumerate(matches):
            num = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start:end]
            body = re.sub(r"\s+", " ", body).strip()[:10000]
            rows.append({num_col: num, "full_text": body, "act_name": act_name})

    return pd.DataFrame(rows)


def build_acts_sections_articles(
    base_path: Path,
    pdf_config: dict,
    act_names: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process all PDFs and build acts, sections, articles DataFrames.
    Returns (acts_df, sections_df, articles_df).
    """
    acts_rows = []
    sections_rows = []
    articles_rows = []

    # Constitution -> articles
    constitution_path = base_path / pdf_config.get("constitution", "Constitution Of India.pdf")
    if constitution_path.exists():
        articles_df = extract_structured_from_pdf(
            constitution_path,
            act_names.get("constitution", "Constitution Of India"),
            "article",
        )
        act_id = "Constitution"
        acts_rows.append({
            "act_id": act_id,
            "act_name": act_names.get("constitution", "Constitution Of India"),
            "act_type": "article",
            "source_file": constitution_path.name,
        })
        for _, row in articles_df.iterrows():
            articles_rows.append({
                "article_id": f"Constitution_Art_{row['article_number']}",
                "act_id": act_id,
                "article_number": row["article_number"],
                "full_text": row["full_text"],
            })

    # BNS, BNSS, BSA -> sections
    section_acts = [
        ("bns", "BNS"),
        ("bnss", "BNSS"),
        ("bsa", "BSA"),
    ]
    for key, act_id in section_acts:
        fname = pdf_config.get(key)
        if not fname:
            continue
        pdf_path = base_path / fname
        if not pdf_path.exists():
            continue
        sec_df = extract_structured_from_pdf(
            pdf_path,
            act_names.get(key, act_id),
            "section",
        )
        acts_rows.append({
            "act_id": act_id,
            "act_name": act_names.get(key, act_id),
            "act_type": "section",
            "source_file": fname,
        })
        for _, row in sec_df.iterrows():
            sections_rows.append({
                "section_id": f"{act_id}_Sec_{row['section_number']}",
                "act_id": act_id,
                "section_number": row["section_number"],
                "full_text": row["full_text"],
            })

    acts_df = pd.DataFrame(acts_rows)
    sections_df = pd.DataFrame(sections_rows)
    articles_df = pd.DataFrame(articles_rows)

    return acts_df, sections_df, articles_df
