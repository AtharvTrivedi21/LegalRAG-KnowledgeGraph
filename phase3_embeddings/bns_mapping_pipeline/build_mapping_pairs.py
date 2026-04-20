"""
Build retrieval fine-tuning pairs from IPC->BNS mappings.

Inputs:
  - phase3_embeddings/bns_mapping_pipeline/ipc_bns_mapping.csv
  - Datasets/IndicLegalQA Dataset_10K_Revised.json
  - phase1_output_v2/sections.csv

Output:
  - phase3_embeddings/bns_mapping_pipeline/bns_mapping_pairs.jsonl
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_CSV = ROOT / "phase3_embeddings" / "bns_mapping_pipeline" / "ipc_bns_mapping.csv"
HF_MAPPING_CSV = ROOT / "phase3_embeddings" / "bns_mapping_pipeline" / "ipc_bns_mapping_hf.csv"
MAPPING_TEMPLATE = ROOT / "phase3_embeddings" / "bns_mapping_pipeline" / "ipc_bns_mapping_template.csv"
INDIC_QA_PATH = ROOT / "Datasets" / "IndicLegalQA Dataset_10K_Revised.json"
SECTIONS_CSV = ROOT / "phase1_output_v2" / "sections.csv"
OUTPUT_JSONL = ROOT / "phase3_embeddings" / "bns_mapping_pipeline" / "bns_mapping_pairs.jsonl"

IPC_SECTION_RE = re.compile(r"(?:ipc\s*)?(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)


def _normalize_section(sec: str) -> str:
    return str(sec).strip().upper().replace("SECTION", "").replace("SEC.", "").strip()


def _load_mapping(mapping_csv: Path) -> pd.DataFrame:
    if not mapping_csv.exists():
        raise FileNotFoundError(
            f"Missing mapping CSV: {mapping_csv}\n"
            f"Copy the template and fill mappings: {MAPPING_TEMPLATE}"
        )
    df = pd.read_csv(mapping_csv)
    lower = {c.lower().strip(): c for c in df.columns}
    if "ipc_section" not in lower or "bns_section" not in lower:
        raise ValueError("Mapping CSV must contain columns: ipc_section, bns_section")
    df = df.rename(
        columns={
            lower["ipc_section"]: "ipc_section",
            lower["bns_section"]: "bns_section",
        }
    )
    df["ipc_section"] = df["ipc_section"].map(_normalize_section)
    df["bns_section"] = df["bns_section"].map(_normalize_section)
    return df[["ipc_section", "bns_section"]].dropna().drop_duplicates()


def _load_bns_sections() -> pd.DataFrame:
    df = pd.read_csv(SECTIONS_CSV)
    bns = df[df["act_id"] == "BNS_2023"].copy()
    bns["section_number"] = bns["section_number"].astype(str).map(_normalize_section)
    return bns.drop_duplicates("section_id")


def _extract_ipc_sections(text: str) -> List[str]:
    if not text:
        return []
    return [_normalize_section(m) for m in IPC_SECTION_RE.findall(text)]


def _load_indic_queries_by_ipc() -> Dict[str, List[str]]:
    if not INDIC_QA_PATH.exists():
        return {}
    data = json.loads(INDIC_QA_PATH.read_text(encoding="utf-8"))
    queries_by_ipc: Dict[str, List[str]] = {}
    for item in data:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        secs = _extract_ipc_sections(q)
        for s in secs:
            queries_by_ipc.setdefault(s, []).append(q)
    return queries_by_ipc


def _build_hard_negatives(
    bns_sections: pd.DataFrame, target_bns: str, max_negatives: int = 3
) -> List[str]:
    row = bns_sections[bns_sections["section_number"] == target_bns]
    if row.empty:
        return []
    chapter_id = row.iloc[0].get("chapter_id")
    try:
        num = int(re.sub(r"[^0-9]", "", target_bns))
    except ValueError:
        num = None

    cands = []
    if chapter_id and pd.notna(chapter_id):
        same_chapter = bns_sections[
            (bns_sections["chapter_id"] == chapter_id)
            & (bns_sections["section_number"] != target_bns)
        ]
        cands.extend(same_chapter["section_number"].astype(str).tolist())

    if num is not None:
        for delta in (1, 2, 3):
            cands.append(str(num - delta))
            cands.append(str(num + delta))

    seen = set()
    uniq = []
    for c in cands:
        c = _normalize_section(c)
        if c == target_bns or c in seen:
            continue
        if not (bns_sections["section_number"] == c).any():
            continue
        seen.add(c)
        uniq.append(c)
        if len(uniq) >= max_negatives:
            break
    return uniq


def _section_text(df: pd.DataFrame, sec_num: str) -> str:
    row = df[df["section_number"] == sec_num]
    if row.empty:
        return ""
    r = row.iloc[0]
    heading = str(r.get("heading", "")).strip()
    full_text = str(r.get("full_text", "")).strip()
    return f"Section {sec_num} - {heading}\n{full_text}".strip()


def main():
    parser = argparse.ArgumentParser(description="Build IPC->BNS mapping training pairs")
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=HF_MAPPING_CSV if HF_MAPPING_CSV.exists() else DEFAULT_MAPPING_CSV,
        help="Mapping CSV path with columns ipc_section,bns_section[,notes]",
    )
    args = parser.parse_args()

    mapping = _load_mapping(args.mapping_csv)
    bns_sections = _load_bns_sections()
    queries_by_ipc = _load_indic_queries_by_ipc()

    records = []
    for _, m in mapping.iterrows():
        ipc_sec = m["ipc_section"]
        bns_sec = m["bns_section"]
        pos_text = _section_text(bns_sections, bns_sec)
        if not pos_text:
            continue

        q_candidates = queries_by_ipc.get(ipc_sec, [])
        if not q_candidates:
            q_candidates = [
                f"What is the legal provision equivalent to IPC section {ipc_sec} under BNS?",
                f"What is punishment for conduct covered under IPC section {ipc_sec} in the new law?",
            ]
            source = "template"
        else:
            q_candidates = q_candidates[:3]
            source = "indiclegal"

        neg_secs = _build_hard_negatives(bns_sections, bns_sec, max_negatives=3)
        neg_texts = [_section_text(bns_sections, s) for s in neg_secs if _section_text(bns_sections, s)]

        for q in q_candidates:
            records.append(
                {
                    "query": q,
                    "positive": pos_text,
                    "negatives": neg_texts,
                    "meta": {
                        "ipc_section": ipc_sec,
                        "bns_section": bns_sec,
                        "query_source": source,
                    },
                }
            )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} mapping pairs -> {OUTPUT_JSONL}")
    print(f"Unique mappings used: {mapping.shape[0]}")
    print(f"IndicLegal IPC query buckets: {len(queries_by_ipc)}")
    print(f"Mapping source: {args.mapping_csv}")


if __name__ == "__main__":
    main()

