"""
Build BNS synthetic training JSONL from generated queries + actual section text.

Reads:
  - phase3_embeddings/synthetic_queries.py  (dict: section_number -> [list of query strings])
  - phase1_output_v2/sections.csv           (BNS section full text)

Writes:
  - phase3_embeddings/bns_synthetic_pairs.jsonl

Usage:
    python -m phase3_embeddings.build_synthetic_jsonl
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTIONS_CSV = PROJECT_ROOT / "phase1_output_v2" / "sections.csv"
QUERIES_FILE = PROJECT_ROOT / "phase3_embeddings" / "synthetic_queries.py"
OUTPUT_FILE = PROJECT_ROOT / "phase3_embeddings" / "bns_synthetic_pairs.jsonl"


def main():
    if not QUERIES_FILE.exists():
        print(f"ERROR: {QUERIES_FILE} not found.")
        print("Generate it first (see instructions in the plan).")
        sys.exit(1)

    # Load the synthetic queries dictionary
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    # Extract the dict — file should contain SYNTHETIC_QUERIES = { ... }
    ns = {}
    exec(content, ns)
    queries_dict = ns.get("SYNTHETIC_QUERIES")
    if not queries_dict:
        print("ERROR: SYNTHETIC_QUERIES dict not found in synthetic_queries.py")
        sys.exit(1)

    # Load BNS section text
    df = pd.read_csv(SECTIONS_CSV)
    bns = df[df["act_id"] == "BNS_2023"].drop_duplicates("section_id")
    section_text = {}
    for _, row in bns.iterrows():
        sec_num = int(row["section_number"])
        text = str(row.get("full_text", "")).strip()
        heading = str(row.get("heading", "")).strip()
        if text:
            section_text[sec_num] = f"Section {sec_num} - {heading}\n{text}"

    # Build pairs
    pairs = []
    missing_sections = []
    for sec_num, queries in sorted(queries_dict.items()):
        sec_num = int(sec_num)
        if sec_num not in section_text:
            missing_sections.append(sec_num)
            continue
        positive = section_text[sec_num]
        for q in queries:
            q = q.strip()
            if q:
                pairs.append({"query": q, "positive": positive})

    # Write JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written {len(pairs)} training pairs to {OUTPUT_FILE}")
    print(f"Sections covered: {len(queries_dict)}")
    if missing_sections:
        print(f"WARNING: {len(missing_sections)} section numbers not found in CSV: {missing_sections}")

    # Stats
    queries_per_section = [len(v) for v in queries_dict.values()]
    print(f"Avg queries per section: {sum(queries_per_section)/len(queries_per_section):.1f}")
    print(f"Min: {min(queries_per_section)}, Max: {max(queries_per_section)}")


if __name__ == "__main__":
    main()
