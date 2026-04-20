"""
Generate synthetic (query, positive) training pairs from BNS sections using templates.

This avoids external API calls by programmatically creating 3 informal incident
descriptions per crime-related section (section_number >= 45).

Writes:
  - phase3_embeddings/bns_synthetic_pairs.jsonl

Usage:
  python -m phase3_embeddings.generate_template_synthetic
"""
from pathlib import Path
import json
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTIONS_CSV = PROJECT_ROOT / "phase1_output_v2" / "sections.csv"
OUTPUT_JSONL = PROJECT_ROOT / "phase3_embeddings" / "bns_synthetic_pairs.jsonl"

TEMPLATES = [
    "Someone {verb} {obj} {where}.",
    "My {obj} was {verb_ed} {where} and the person ran away.",
    "I was {verb_ing} and {someone} {verb_ed} my {obj}.",
]

VERB_MAP = {
    "theft": ("stole", "stolen", "stealing"),
    "snatching": ("snatched", "snatched", "snatching"),
    "robbery": ("robbed", "robbed", "robbing"),
    "dacoity": ("attacked and took", "attacked and took", "attacking and taking"),
    "extortion": ("threatened and made me pay", "extorted", "extorting"),
    "assault": ("assaulted", "assaulted", "assaulting"),
    "murder": ("killed", "killed", "killing"),
    "kidnapping": ("took", "taken", "taking"),
    "forgery": ("used a fake document to steal", "forged", "forging"),
    "fraud": ("cheated me out of", "cheated", "cheating"),
    "trespass": ("broke into", "broke into", "breaking into"),
    "house-trespass": ("broke into my house and took", "broke into", "breaking into"),
}

def choose_verb(heading: str):
    h = heading.lower()
    for key, forms in VERB_MAP.items():
        if key in h:
            return forms
    # default
    return ("did something to", "done something to", "doing something to")

def sanitize_obj(heading: str):
    # pick a short noun phrase from heading
    # remove punctuation and parentheticals
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", heading)
    parts = s.split()
    if not parts:
        return "my property"
    # try to pick a meaningful word
    for w in parts:
        if len(w) > 3:
            return "my " + w.lower()
    return "my property"

def gen_queries_for_section(sec_num: int, heading: str):
    verb, verb_ed, verb_ing = choose_verb(heading)
    obj = sanitize_obj(heading)
    where = "near my house"
    qlist = []
    # simple variations
    q1 = f"Someone {verb} {obj} {where}."
    q2 = f"My {obj} was {verb_ed} yesterday while I was at work."
    q3 = f"I noticed {verb_ing} of {obj} when I returned home."
    qlist.extend([q1, q2, q3])
    # ensure uniqueness and length
    uniq = []
    for q in qlist:
        q = re.sub(r"\s+", " ", q).strip()
        if q not in uniq:
            uniq.append(q)
    return uniq

def main():
    if not SECTIONS_CSV.exists():
        print(f"Error: sections CSV not found at {SECTIONS_CSV}")
        return
    df = pd.read_csv(SECTIONS_CSV)
    bns = df[df["act_id"] == "BNS_2023"].drop_duplicates("section_id")
    pairs = []
    covered = 0
    for _, row in bns.iterrows():
        try:
            sec_num = int(row["section_number"])
        except Exception:
            continue
        if sec_num < 45:
            continue
        heading = str(row.get("heading", "")).strip()
        queries = gen_queries_for_section(sec_num, heading)
        positive = str(row.get("full_text", "")).strip()
        if not positive:
            continue
        for q in queries:
            pairs.append({"query": q, "positive": positive})
        covered += 1

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} pairs for {covered} sections -> {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

