"""
Print shape, schema, and head() for each dataset actually used in LegalRAG.
Run from project root: python scripts/doc_dataset_samples.py

Optionally writes a summary to Documentation/dataset_samples.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    out_lines: list[str] = []
    def log(s: str = ""):
        print(s)
        out_lines.append(s)

    try:
        import config
        import pandas as pd
    except ImportError as e:
        print(f"Import error: {e}. Run from project root with dependencies installed.")
        return 1

    base_path = config.BASE_PATH if config.BASE_PATH.is_absolute() else PROJECT_ROOT / config.BASE_PATH
    phase1_output = PROJECT_ROOT / "phase1_output"

    # ---- legal_data_train.csv ----
    log("=" * 60)
    log("1. legal_data_train.csv (Phase 1)")
    log("=" * 60)
    csv_path = base_path / "legal_data_train.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, nrows=100, low_memory=False)
            log(f"Path: {csv_path}")
            log(f"Shape (first 100 rows): {df.shape}")
            log(f"Columns: {list(df.columns)}")
            log("head(5):")
            log(df.head(5).to_string())
        except Exception as e:
            log(f"Error: {e}")
    else:
        log(f"Not found: {csv_path}")
        log("Expected: at least column 'Text' (judgment text).")
    log()

    # ---- Phase 1 output CSVs ----
    for name in ["cases.csv", "sections.csv", "articles.csv", "acts.csv", "edges.csv"]:
        log("=" * 60)
        log(f"Phase 1 output: {name}")
        log("=" * 60)
        p = phase1_output / name
        if p.exists():
            try:
                df = pd.read_csv(p, nrows=500, low_memory=False)
                log(f"Path: {p}")
                log(f"Shape (first 500 rows): {df.shape}")
                log(f"Columns: {list(df.columns)}")
                log("head(5):")
                log(df.head(5).to_string())
            except Exception as e:
                log(f"Error: {e}")
        else:
            log(f"Not found: {p}. Run Phase 1 first.")
        log()

    # ---- IndicLegalQA JSON ----
    log("=" * 60)
    log("IndicLegalQA Dataset_10K_Revised.json (Phase 3)")
    log("=" * 60)
    try:
        from phase3_embeddings import config as p3config
        json_path = p3config.INDIC_LEGAL_QA_PATH if p3config.INDIC_LEGAL_QA_PATH.is_absolute() else PROJECT_ROOT / p3config.INDIC_LEGAL_QA_PATH
    except ImportError:
        json_path = PROJECT_ROOT / "Datasets" / "IndicLegalQA Dataset_10K_Revised.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log(f"Path: {json_path}")
            log(f"Type: {type(data).__name__}, len: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                log(f"First item keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}")
                for i, item in enumerate(data[:3]):
                    if isinstance(item, dict):
                        q = item.get("question", "")[:80]
                        a = item.get("answer", "")[:80]
                        log(f"  [{i}] question: {q}...")
                        log(f"      answer: {a}...")
            else:
                log(f"First 3 items: {str(data[:3])[:200]}...")
        except Exception as e:
            log(f"Error: {e}")
    else:
        log(f"Not found: {json_path}")
    log()

    # Optional: write to Documentation/dataset_samples.txt
    doc_dir = PROJECT_ROOT / "Documentation"
    out_file = doc_dir / "dataset_samples.txt"
    try:
        doc_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        print(f"Wrote summary to {out_file}")
    except Exception as e:
        print(f"Could not write {out_file}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
