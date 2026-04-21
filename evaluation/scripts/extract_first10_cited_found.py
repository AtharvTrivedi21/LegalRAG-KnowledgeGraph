import json
from pathlib import Path

INPUT = Path("evaluation/results/system3_raw_results.jsonl")
OUTPUT = Path("evaluation/results/first10_cited_found.csv")

def extract_retrieved_sections(retrieved_chunks):
    secs = set()
    for c in retrieved_chunks:
        sid = c.get("source_id","")
        if isinstance(sid, str):
            # try to extract trailing number like BNS_2023_s314 or ..._s314
            parts = sid.split("_")
            for p in parts[::-1]:
                if p.startswith("s") and p[1:].isdigit():
                    secs.add(p[1:])
                    break
                if p.isdigit():
                    secs.add(p)
                    break
        # also try to find section numbers in 'text' like [BNS Section 314]
        text = c.get("text","")
        for token in ["Section","Section","Article","Article"]:
            pass
    return sorted(secs, key=lambda x: int(x))

def main():
    out_lines = ["case_id,cited_sections,found_sections"]
    with INPUT.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i>=10:
                break
            obj = json.loads(line)
            cid = obj.get("case_id")
            res = obj.get("result",{})
            cited = res.get("cited_sections") or res.get("cited_sections", [])
            cited_str = "|".join(str(x) for x in (cited or []))
            retrieved = res.get("retrieved_chunks", [])
            found = extract_retrieved_sections(retrieved)
            found_str = "|".join(found)
            out_lines.append(f'{cid},"{cited_str}","{found_str}"')
    OUTPUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")

if __name__ == "__main__":
    main()

