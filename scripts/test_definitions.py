"""Unit test for definitions extraction fix (Phase 1)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force fresh import
for key in list(sys.modules.keys()):
    if 'definitions' in key:
        del sys.modules[key]

from phase1_preprocessing.definitions import (
    DEFINITION_SECTION_RE,
    QUOTED_TERM_RE,
    extract_definitions_from_section,
    run_definitions_extraction,
)

SECTIONS_CSV = Path("phase1_output_v2/sections.csv")
OUTPUT_DIR = Path("phase1_output_v2")

# --- Step 1: read target sections (first occurrence only -- CSV may have duplicates) ---
target_ids = {"BNS_2023_s2", "BNSS_2023_s2"}
sections = {}
with open(SECTIONS_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sid = row["section_id"]
        if sid in target_ids and sid not in sections:
            sections[sid] = (row["act_id"], row["full_text"])

print(f"Sections loaded: {list(sections.keys())}")

# --- Step 2: verify guard regex fires ---
failures = []
for sid, (act_id, text) in sections.items():
    guard = bool(DEFINITION_SECTION_RE.search(text))
    print(f"Guard for {sid}: {guard}, text starts: {repr(text[:50])}")
    if not guard:
        failures.append(f"FAIL: guard regex did not fire for {sid}")
    else:
        print(f"OK: guard fired for {sid}")

# --- Step 3: extract and count definitions ---
for sid, (act_id, text) in sections.items():
    # Direct regex test
    raw_matches = list(QUOTED_TERM_RE.finditer(text[:1000]))
    print(f"\n{sid}: QUOTED_TERM_RE found {len(raw_matches)} raw matches in first 1000 chars")
    for m in raw_matches[:5]:
        print(f"  raw: {repr(m.group(0)[:60])}")

    defs = extract_definitions_from_section(text, act_id, sid)
    terms = [d["term"] for d in defs]
    print(f"{sid}: {len(defs)} definitions via extract fn")
    for d in defs[:15]:
        print(f"  - {d['term']!r:30s} -> {d['defined_text'][:60]!r}")

    if sid == "BNS_2023_s2":
        # We found 3 terms from debug (act, animal, child) — 'person' is in later part
        if len(defs) < 3:
            failures.append(f"FAIL: BNS_2023_s2 only {len(defs)} defs, expected >=3")
        else:
            print(f"OK: BNS_2023_s2 has {len(defs)} defs")

    if sid == "BNSS_2023_s2":
        if len(defs) < 5:
            failures.append(f"FAIL: BNSS_2023_s2 only {len(defs)} defs, expected >=5")
        else:
            print(f"OK: BNSS_2023_s2 has {len(defs)} defs")

# --- Step 4: run full extraction and print totals ---
print("\n--- Running full extraction over all sections ---")
unique_defs, edges = run_definitions_extraction(SECTIONS_CSV, OUTPUT_DIR)
print(f"Total unique definitions: {len(unique_defs)}")
print(f"Total section->def edges: {len(edges)}")
by_act = {}
for d in unique_defs:
    by_act[d["act_id"]] = by_act.get(d["act_id"], 0) + 1
for act, cnt in sorted(by_act.items()):
    print(f"  {act}: {cnt} definitions")

if len(unique_defs) < 10:
    failures.append(f"FAIL: total definitions {len(unique_defs)} < 10")
else:
    print(f"OK: {len(unique_defs)} total definitions written")

# --- Result ---
print("\n=== RESULT ===")
if failures:
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("ALL ASSERTIONS PASSED")
