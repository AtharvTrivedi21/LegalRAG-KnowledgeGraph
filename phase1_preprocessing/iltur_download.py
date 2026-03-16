"""
Download IL-TUR dataset (summarization + classification) via HuggingFace datasets.
Fallback: allow manual drop-in at Datasets/iltur_raw/ or use legacy legal_data.csv.
Output: cases_iltur.csv with case_id, judgment_text, summary, label, source=iltur, year.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

def try_download_iltur(
    output_dir: Path,
    iltur_raw_dir: Path,
    use_legacy_csv: Path | None,
) -> list[dict[str, Any]]:
    """
    Try load_dataset("Exploration-Lab/IL-TUR", task_name, revision="script") for
    summarization and classification. If HF unavailable, try iltur_raw_dir (manual drop-in)
    or use_legacy_csv. Returns list of case dicts for cases_iltur.csv.
    """
    rows = []
    # 1) Try HuggingFace
    try:
        from datasets import load_dataset
        for task in ("summarization", "classification"):
            try:
                ds = load_dataset("Exploration-Lab/IL-TUR", task, revision="script")
                if ds and hasattr(ds, "get") and "train" in ds:
                    train = ds["train"]
                    for i, ex in enumerate(train):
                        text = ex.get("text") or ex.get("input") or ex.get("source") or ""
                        summary = ex.get("summary") or ex.get("target") or ex.get("output") or ""
                        label = ex.get("label") or ex.get("labels") or ""
                        case_id = f"iltur_{task}_{i}"
                        rows.append({
                            "case_id": case_id,
                            "judgment_text": str(text)[:500000],
                            "summary": str(summary)[:10000] if summary else "",
                            "label": str(label)[:200] if label else "",
                            "source": "iltur",
                            "year": None,
                        })
            except Exception as e:
                pass  # task not available or network error
        if rows:
            return rows
    except ImportError:
        pass
    except Exception as e:
        print(f"[IL-TUR] HuggingFace load failed: {e}. Try manual drop-in or legacy CSV.")
    # 2) Manual drop-in: Datasets/iltur_raw/*.json or *.jsonl
    if iltur_raw_dir and iltur_raw_dir.exists():
        for f in iltur_raw_dir.glob("*.jsonl"):
            with open(f, "r", encoding="utf-8") as fp:
                for i, line in enumerate(fp):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        text = ex.get("text") or ex.get("input") or ""
                        summary = ex.get("summary") or ex.get("target") or ""
                        label = ex.get("label") or ""
                        rows.append({
                            "case_id": f"iltur_manual_{f.stem}_{i}",
                            "judgment_text": str(text)[:500000],
                            "summary": str(summary)[:10000] if summary else "",
                            "label": str(label)[:200] if label else "",
                            "source": "iltur",
                            "year": None,
                        })
                    except Exception:
                        pass
        for f in iltur_raw_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for i, ex in enumerate(data):
                        text = ex.get("text") or ex.get("input") or ""
                        summary = ex.get("summary") or ex.get("target") or ""
                        rows.append({
                            "case_id": f"iltur_manual_{f.stem}_{i}",
                            "judgment_text": str(text)[:500000],
                            "summary": str(summary)[:10000] if summary else "",
                            "label": str(ex.get("label", ""))[:200],
                            "source": "iltur",
                            "year": None,
                        })
            except Exception:
                pass
        if rows:
            return rows
    # 3) Legacy CSV (legal_data.csv) with Text/Summary
    if use_legacy_csv and use_legacy_csv.exists():
        import pandas as pd
        df = pd.read_csv(use_legacy_csv, nrows=100000, low_memory=False)
        text_col = "Text" if "Text" in df.columns else "text"
        if text_col not in df.columns:
            return rows
        summary_col = "Summary" if "Summary" in df.columns else "summary"
        for i, r in df.iterrows():
            text = str(r.get(text_col, ""))
            if not text or text == "nan":
                continue
            rows.append({
                "case_id": f"iltur_legacy_{i}",
                "judgment_text": text[:500000],
                "summary": str(r.get(summary_col, ""))[:10000] if summary_col in df.columns else "",
                "label": "",
                "source": "iltur_legacy",
                "year": None,
            })
    return rows


def run_iltur_normalize(
    output_dir: Path,
    base_path: Path,
    allow_legacy_csv: bool = True,
) -> int:
    """Try download/normalize IL-TUR; write cases_iltur.csv. Returns row count."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iltur_raw = base_path / "iltur_raw"
    legacy = (base_path / "legal_data.csv") if allow_legacy_csv else None
    rows = try_download_iltur(output_dir, iltur_raw, legacy)
    if not rows:
        return 0
    path = output_dir / "cases_iltur.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "judgment_text", "summary", "label", "source", "year"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)
