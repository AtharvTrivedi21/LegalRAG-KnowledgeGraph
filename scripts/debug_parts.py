"""Debug: check part_chapter.csv and what parts/chapters actually exist."""
import csv
from pathlib import Path

print("=== act_part.csv ===")
with open("phase1_output_v2/act_part.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total act_part rows: {len(rows)}")
for r in rows[:20]:
    print(f"  {r}")

print("\n=== part_chapter.csv ===")
with open("phase1_output_v2/part_chapter.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total part_chapter rows: {len(rows)}")
for r in rows:
    print(f"  {r}")

print("\n=== parts.csv sample ===")
with open("phase1_output_v2/parts.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total parts rows: {len(rows)}")
from collections import Counter
by_act = Counter(r["act_id"] for r in rows)
for act, cnt in by_act.most_common():
    print(f"  {act}: {cnt} parts")

print("\n=== chapters.csv sample ===")
with open("phase1_output_v2/chapters.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total chapters rows: {len(rows)}")
by_act = Counter(r["act_id"] for r in rows)
for act, cnt in by_act.most_common():
    print(f"  {act}: {cnt} chapters")
has_part = sum(1 for r in rows if r.get("part_id"))
print(f"  Chapters with part_id: {has_part}")
