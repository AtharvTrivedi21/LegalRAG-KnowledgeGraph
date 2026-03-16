"""
Write slim case CSVs for Neo4j import (no full judgment text) to avoid LOAD CSV limits.
Reads cases_sc.csv, cases_iltur.csv from phase1_output_v2; writes cases_sc_neo4j.csv, cases_iltur_neo4j.csv.
Run after Phase 1 v2. Then copy the _neo4j.csv files to Neo4j import/ and use them in 02_load_nodes_v2.cypher.
"""
import csv
import sys
from pathlib import Path

try:
    csv.field_size_limit(10 * 1024 * 1024)
except OverflowError:
    csv.field_size_limit(sys.maxsize)

SNIPPET_LEN = 800

def _snippet(text: str, limit: int = SNIPPET_LEN) -> str:
    t = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "..."

def main():
    base = Path(__file__).resolve().parent.parent / "phase1_output_v2"
    if not base.exists():
        print("phase1_output_v2 not found")
        return 1

    # cases_sc: keep a short judgment_text snippet for UI display
    sc_in = base / "cases_sc.csv"
    sc_out = base / "cases_sc_neo4j.csv"
    if sc_in.exists():
        count = 0
        with open(sc_in, "r", encoding="utf-8") as fin, open(sc_out, "w", encoding="utf-8", newline="") as fout:
            r = csv.DictReader(fin)
            w = csv.DictWriter(fout, fieldnames=["case_id", "year", "source_file", "source", "judgment_text"])
            w.writeheader()
            for row in r:
                w.writerow({
                    "case_id": row.get("case_id", ""),
                    "year": row.get("year", ""),
                    "source_file": row.get("source_file", ""),
                    "source": row.get("source", "sc_pdf"),
                    "judgment_text": _snippet(row.get("full_text", "")),
                })
                count += 1
        print(f"cases_sc_neo4j.csv: {count} rows")
    else:
        print("cases_sc.csv not found, skipping")

    # cases_iltur: keep a short judgment_text snippet for UI display
    iltur_in = base / "cases_iltur.csv"
    iltur_out = base / "cases_iltur_neo4j.csv"
    if iltur_in.exists():
        count = 0
        with open(iltur_in, "r", encoding="utf-8") as fin, open(iltur_out, "w", encoding="utf-8", newline="") as fout:
            r = csv.DictReader(fin)
            w = csv.DictWriter(fout, fieldnames=["case_id", "year", "source", "judgment_text"])
            w.writeheader()
            for row in r:
                w.writerow({
                    "case_id": row.get("case_id", ""),
                    "year": row.get("year", ""),
                    "source": row.get("source", "iltur"),
                    "judgment_text": _snippet(row.get("judgment_text", "")),
                })
                count += 1
        print(f"cases_iltur_neo4j.csv: {count} rows")
    else:
        print("cases_iltur.csv not found, skipping")

    print("Copy cases_sc_neo4j.csv and cases_iltur_neo4j.csv to Neo4j import/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
