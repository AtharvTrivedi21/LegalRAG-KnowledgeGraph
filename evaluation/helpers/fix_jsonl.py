"""Remove entries with empty/error results from the raw JSONL."""
import json
from pathlib import Path

JSONL = Path(__file__).resolve().parent.parent / "results" / "system3_raw_results.jsonl"

lines = JSONL.read_text(encoding="utf-8").strip().split("\n")
kept = []
removed = []

for line in lines:
    if not line.strip():
        continue
    obj = json.loads(line)
    answer = obj.get("result", {}).get("answer", "")
    rq = obj.get("result", {}).get("rephrased_query", "")
    if not answer and rq == "ERROR":
        removed.append(obj["case_id"])
    elif not answer and not rq:
        removed.append(obj["case_id"])
    else:
        kept.append(line)

JSONL.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"Kept {len(kept)} valid entries, removed {len(removed)} error entries")
print(f"Removed case IDs: {removed}")
