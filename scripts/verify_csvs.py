"""Verify Phase 1 output CSV counts after pipeline re-run."""
import csv
from pathlib import Path

OUTPUT_DIR = Path("phase1_output_v2")

def count_rows(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

checks = [
    ("acts.csv",                      4,   4),
    ("parts.csv",                    48,  48),
    ("chapters.csv",                 49,  49),
    ("sections.csv",                600, 900),
    ("articles.csv",               300, 600),
    ("definitions.csv",              1,   None),  # at least 1 (pipeline uses existing CSV data)
    ("section_defines_term.csv",     1,   None),
    ("cases_sc_neo4j.csv",        7000,  None),
    ("cases_iltur_neo4j.csv",     7000,  None),
    ("act_part.csv",                48,  48),
    ("part_chapter.csv",             1,  None),  # BSA: 3 rows, at minimum
    ("chapter_section.csv",        600, 900),
    ("act_section.csv",            600, 900),
    ("act_article.csv",            300, 600),
    ("section_references_section.csv", 44, None),
    ("case_cites_section.csv",   25000, None),
    ("case_cites_article.csv",   25000, None),
]

failures = []
print(f"{'CSV':<42} {'Rows':>8}  {'Expected':>12}  Status")
print("-" * 75)
for filename, low, high in checks:
    count = count_rows(filename)
    if count is None:
        status = "MISSING"
        failures.append(f"MISSING: {filename}")
    elif count < low:
        status = f"LOW (want >={low})"
        failures.append(f"{filename}: {count} rows < expected {low}")
    elif high is not None and count > high:
        status = f"HIGH (want <={high})"
        failures.append(f"{filename}: {count} rows > expected {high}")
    else:
        expected_str = f">={low}" + (f" <={high}" if high else "")
        status = "OK"
    expected_str = f">={low}" + (f", <={high}" if high else "+")
    print(f"  {filename:<40} {count if count is not None else 'N/A':>8}  {expected_str:>12}  {status}")

print()
if failures:
    print("=== FAILURES ===")
    for f in failures:
        print(f"  {f}")
    exit(1)
else:
    print("=== ALL CSV COUNTS OK ===")
