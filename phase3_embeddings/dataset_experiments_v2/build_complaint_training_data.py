"""
Build complaint-style training data for BNS embedding fine-tuning.

Inputs:
  - HF dataset: navaneeth005/complaint-relevant-bns
  - BNS sections CSV: phase1_output_v2/sections.csv

Outputs:
  - complaint_bns_pairs.jsonl
      {"query": "...", "positive": "...", "meta": {...}}
  - complaint_bns_hardneg_triplets.jsonl
      {"query": "...", "positive": "...", "negative": "...", "meta": {...}}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
SECTIONS_CSV = ROOT / "phase1_output_v2" / "sections.csv"
OUT_DIR = ROOT / "phase3_embeddings" / "dataset_experiments_v2" / "datasets"
PAIRS_JSONL = OUT_DIR / "complaint_bns_pairs.jsonl"
TRIPLETS_JSONL = OUT_DIR / "complaint_bns_hardneg_triplets.jsonl"

SECTION_RE = re.compile(r"\d+(?:\(\d+\))?")


def _extract_base_sections(output_text: str) -> list[int]:
    nums = []
    for token in SECTION_RE.findall(output_text or ""):
        base = int(token.split("(")[0])
        nums.append(base)
    # Keep order, remove duplicates
    seen = set()
    ordered = []
    for n in nums:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


def _load_sections_map() -> tuple[dict[int, str], dict[int, str], dict[int, int]]:
    df = pd.read_csv(SECTIONS_CSV)
    bns = df[df["act_id"] == "BNS_2023"].drop_duplicates("section_number").copy()
    bns["section_number"] = bns["section_number"].astype(int)
    bns = bns.sort_values("section_number")

    section_text = {}
    chapter_by_section = {}
    index_by_section = {}

    for idx, (_, row) in enumerate(bns.iterrows()):
        sec = int(row["section_number"])
        heading = str(row.get("heading", "")).strip()
        full_text = str(row.get("full_text", "")).strip()
        chapter = str(row.get("chapter_id", "")).strip()
        if not full_text:
            continue
        section_text[sec] = f"Section {sec} - {heading}\n{full_text}"
        chapter_by_section[sec] = chapter
        index_by_section[sec] = idx
    return section_text, chapter_by_section, index_by_section


def _pick_hard_negative(
    pos_sec: int,
    positive_set: set[int],
    section_text: dict[int, str],
    chapter_by_section: dict[int, str],
    index_by_section: dict[int, int],
) -> int | None:
    # 1) Prefer same chapter, excluding positives
    pos_chapter = chapter_by_section.get(pos_sec, "")
    same_chapter = [
        s for s, ch in chapter_by_section.items()
        if ch == pos_chapter and s not in positive_set and s in section_text
    ]
    if same_chapter:
        same_chapter.sort(key=lambda s: abs(index_by_section[s] - index_by_section.get(pos_sec, 0)))
        return same_chapter[0]

    # 2) Fallback: nearest section number in global order
    if pos_sec not in index_by_section:
        return None
    pos_idx = index_by_section[pos_sec]
    candidates = [
        s for s in section_text.keys()
        if s not in positive_set
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda s: abs(index_by_section[s] - pos_idx))
    return candidates[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    section_text, chapter_by_section, index_by_section = _load_sections_map()

    ds = load_dataset("navaneeth005/complaint-relevant-bns")["train"]

    n_rows = 0
    n_pairs = 0
    n_triplets = 0
    n_filtered = 0

    with PAIRS_JSONL.open("w", encoding="utf-8") as fpairs, TRIPLETS_JSONL.open("w", encoding="utf-8") as ftrip:
        for row in ds:
            n_rows += 1
            query = str(row.get("input", "")).strip()
            if not query:
                n_filtered += 1
                continue

            sections = _extract_base_sections(str(row.get("output", "")))
            sections = [s for s in sections if s in section_text]
            if not sections:
                n_filtered += 1
                continue

            # Filter very noisy examples with too many labels.
            if len(sections) > 8:
                n_filtered += 1
                continue

            positive_set = set(sections)
            for pos_sec in sections:
                pos_text = section_text[pos_sec]
                pair = {
                    "query": query,
                    "positive": pos_text,
                    "meta": {
                        "source": "complaint-relevant-bns",
                        "positive_section": pos_sec,
                        "all_sections": sections,
                    },
                }
                fpairs.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_pairs += 1

                neg_sec = _pick_hard_negative(
                    pos_sec=pos_sec,
                    positive_set=positive_set,
                    section_text=section_text,
                    chapter_by_section=chapter_by_section,
                    index_by_section=index_by_section,
                )
                if neg_sec is None:
                    continue

                triplet = {
                    "query": query,
                    "positive": pos_text,
                    "negative": section_text[neg_sec],
                    "meta": {
                        "source": "complaint-relevant-bns",
                        "positive_section": pos_sec,
                        "negative_section": neg_sec,
                        "all_sections": sections,
                    },
                }
                ftrip.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                n_triplets += 1

    print(f"Rows read: {n_rows}")
    print(f"Filtered rows: {n_filtered}")
    print(f"Pairs written: {n_pairs} -> {PAIRS_JSONL}")
    print(f"Triplets written: {n_triplets} -> {TRIPLETS_JSONL}")


if __name__ == "__main__":
    main()
