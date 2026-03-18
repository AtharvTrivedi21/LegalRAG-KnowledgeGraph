"""Count rows in case CSV files."""
import csv
import sys
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

files = [
    "cases_iltur.csv",
    "cases_iltur_neo4j.csv",
    "cases_sc.csv",
    "cases_sc_neo4j.csv",
]

for f in files:
    p = Path("phase1_output_v2") / f
    if not p.exists():
        print(f"{f}: MISSING")
        continue
    with open(p, encoding="utf-8") as fin:
        n = sum(1 for _ in csv.DictReader(fin))
    print(f"{f}: {n} rows")
