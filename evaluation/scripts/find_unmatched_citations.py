import json
from pathlib import Path

INPUT = Path("evaluation/results/system3_raw_results.jsonl")
OUTPUT = Path("evaluation/results/gpt4_citation_mismatches.csv")

def normalize_cited_sections(cited):
    # cited may be numbers as strings or ints; normalize to strings without whitespace
    return {str(x).strip() for x in cited}

def normalize_retrieved_sections(retrieved_chunks):
    secs = set()
    for c in retrieved_chunks:
        sid = c.get("source_id") or c.get("source") or ""
        # try to extract numeric suffix like BNS_2023_s314 -> 314
        if isinstance(sid, str):
            parts = sid.split("_")
            for p in parts[::-1]:
                if p.startswith("s") and p[1:].isdigit():
                    secs.add(p[1:])
                    break
                if p.isdigit():
                    secs.add(p)
                    break
    return secs

def main():
    unmatched = []
    with INPUT.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = obj.get("case_id") or obj.get("result", {}).get("case_id")
            result = obj.get("result", {})
            cited = result.get("cited_sections", [])
            retrieved = result.get("retrieved_chunks", [])
            cited_set = normalize_cited_sections(cited)
            retrieved_set = normalize_retrieved_sections(retrieved)
            # If there is no overlap at all
            if cited_set and retrieved_set and cited_set.isdisjoint(retrieved_set):
                unmatched.append({
                    "case_id": case_id,
                    "cited_sections": "|".join(sorted(cited_set)),
                    "retrieved_sections": "|".join(sorted(retrieved_set))
                })
    # write CSV
    with OUTPUT.open("w", encoding="utf-8") as out:
        out.write("case_id,cited_sections,retrieved_sections\n")
        for r in unmatched:
            out.write(f'{r["case_id"]},"{r["cited_sections"]}","{r["retrieved_sections"]}"\n')
    print(f"Wrote {len(unmatched)} unmatched cases to {OUTPUT}")

if __name__ == "__main__":
    main()

