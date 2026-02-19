"""
Load and preprocess Supreme Court judgments from zip or CSV.
Inspect schema, map to case_id/judgment_text/year, compute stats.

Supports: (1) Zip with CSV, (2) Zip with PDFs (SC_Judgements-16-25 structure),
         (3) Fallback CSV (legal_data_train).
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pdfplumber

# Column mapping candidates (case-insensitive)
ID_COLUMNS = ["case_id", "id", "case_number", "case_no", "caseid"]
TEXT_COLUMNS = ["judgment_text", "text", "judgment", "case_text", "content"]
YEAR_COLUMNS = ["year", "judgment_date", "date", "judgement_date", "decision_date"]


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return first column name that exists (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _parse_year(val) -> Optional[int]:
    """Extract year from date string or integer."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        y = int(val)
        return y if 1900 <= y <= 2100 else None
    s = str(val).strip()
    # Try common date patterns
    m = re.search(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return y if 1900 <= y <= 2100 else None
    return None


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def _extract_text_from_pdf_path(pdf_path: Path) -> str:
    """Extract text from a PDF file path using pdfplumber."""
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def load_judgments_from_extracted_folder(
    extracted_root: Path, limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load judgments from an already-extracted folder structure:

      extracted_root/
        2016/*.PDF
        2017/*.PDF
        ...

    Uses parent folder name as year and PDF filename stem as case_id.
    """
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted judgments folder not found: {extracted_root}")

    rows = []
    count = 0

    # Iterate year folders in sorted order for reproducibility
    for year_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        year = 0
        if year_dir.name.isdigit():
            y = int(year_dir.name)
            if 1900 <= y <= 2100:
                year = y

        # Match both .pdf and .PDF etc.
        pdf_files = sorted(
            [p for p in year_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        )
        for pdf_path in pdf_files:
            case_id = pdf_path.stem
            try:
                text = _extract_text_from_pdf_path(pdf_path)
            except Exception:
                text = ""
            rows.append({"case_id": case_id, "judgment_text": text, "year": year})
            count += 1
            if limit is not None and count >= limit:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def load_judgments_from_zip_pdfs(zip_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load judgments from zip containing PDFs (e.g. SC_Judgements-16-25/YYYY/case.PDF).
    Extracts year from path, case_id from filename, text via pdfplumber.
    """
    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
        if limit:
            pdf_names = pdf_names[:limit]
        for i, name in enumerate(pdf_names):
            # Year from path: SC_Judgements-16-25/2016/... -> 2016
            parts = name.replace("\\", "/").split("/")
            year = 0
            for p in parts:
                if p.isdigit() and len(p) == 4 and 2010 <= int(p) <= 2030:
                    year = int(p)
                    break
            case_id = Path(name).stem
            try:
                with zf.open(name) as f:
                    data = f.read()
                text = _extract_text_from_pdf_bytes(data)
            except Exception:
                text = ""
            rows.append({"case_id": case_id, "judgment_text": text, "year": year})
    return pd.DataFrame(rows)


def load_judgments_from_zip(zip_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Unzip and load: CSV if present, else PDFs (SC_Judgements structure).
    Returns raw or prepared DataFrame depending on source.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if csv_names:
            first_csv = csv_names[0]
            with zf.open(first_csv) as f:
                content = f.read().decode("utf-8", errors="replace")
            raw = pd.read_csv(io.StringIO(content), low_memory=False)
            return raw
    return load_judgments_from_zip_pdfs(zip_path, limit=limit)


def load_judgments_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load judgments from a single CSV (fallback when zip not available)."""
    return pd.read_csv(csv_path, low_memory=False)


def inspect_schema(df: pd.DataFrame) -> dict:
    """Return schema info: shape, columns, dtypes, nulls."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
    }


def prepare_judgments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map columns to case_id, judgment_text, year.
    Keep only those three. Assign case_id if missing.
    """
    id_col = _find_column(df, ID_COLUMNS)
    text_col = _find_column(df, TEXT_COLUMNS)
    year_col = _find_column(df, YEAR_COLUMNS)

    if not text_col:
        raise ValueError(
            f"Could not find judgment text column. Available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["judgment_text"] = df[text_col].astype(str).replace("nan", "")

    if id_col:
        out["case_id"] = df[id_col].astype(str)
    else:
        out["case_id"] = [f"case_{i}" for i in range(len(df))]

    if year_col:
        years = df[year_col].apply(_parse_year)
        out["year"] = years
        # Fill missing years with mode or 0 for now
        if out["year"].isna().all():
            out["year"] = 0
        else:
            mode_val = out["year"].mode()
            if len(mode_val) > 0:
                out["year"] = out["year"].fillna(mode_val.iloc[0])
            out["year"] = out["year"].astype(int)
    else:
        out["year"] = 0

    return out[["case_id", "judgment_text", "year"]]


def get_judgment_stats(df: pd.DataFrame) -> dict:
    """Cases per year, avg text length."""
    stats = {}
    stats["cases_per_year"] = df.groupby("year").size().to_dict()
    stats["avg_text_length"] = df["judgment_text"].str.len().mean()
    stats["total_cases"] = len(df)
    return stats


def load_and_prepare_judgments(
    base_path: Path,
    sc_extracted_dir: str,
    limit: Optional[int] = None,
    source: str = "pdf",
) -> Tuple[pd.DataFrame, dict]:
    """
    Load judgments from extracted folder (pdf) or from legal_data_train.csv (csv).
    Returns (cases_df, schema_info).
    """
    source = (source or "pdf").lower().strip()

    # Fast path: CSV mode for quick prototype runs
    if source == "csv":
        csv_path = base_path / "legal_data_train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV mode requested but not found: {csv_path}")
        raw = pd.read_csv(csv_path, usecols=["Text"], low_memory=False)
        if limit is not None:
            raw = raw.head(limit)
        raw = raw.rename(columns={"Text": "judgment_text"})
        raw["case_id"] = [f"case_{i}" for i in range(len(raw))]
        raw["year"] = 0
        df = raw[["case_id", "judgment_text", "year"]].copy()
        schema_info = {"source": "legal_data_train_csv", "shape": raw.shape, "columns": list(raw.columns)}
        return df, schema_info

    extracted_root = base_path / sc_extracted_dir
    if extracted_root.exists():
        cases_df = load_judgments_from_extracted_folder(extracted_root, limit=limit)
        schema_info = {
            "source": "extracted_pdfs",
            "shape": cases_df.shape,
            "root": str(extracted_root),
        }
        return cases_df[["case_id", "judgment_text", "year"]], schema_info

    # Fallback: try legal_data_train.csv if extracted folder missing
    fallback = base_path / "legal_data_train.csv"
    if fallback.exists():
        raw = pd.read_csv(fallback, usecols=["Text"], low_memory=False)
        raw = raw.rename(columns={"Text": "judgment_text", "Summary": "summary"})
        raw["case_id"] = [f"case_{i}" for i in range(len(raw))]
        raw["year"] = 0
        df = raw[["case_id", "judgment_text", "year"]].copy()
        schema_info = {"source": "fallback_csv", "shape": raw.shape}
        return df, schema_info

    raise FileNotFoundError(
        f"Missing extracted judgments folder ({extracted_root}) and fallback CSV ({fallback})."
    )
