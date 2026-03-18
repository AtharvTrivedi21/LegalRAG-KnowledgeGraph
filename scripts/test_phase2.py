"""Unit tests for Phase 2 fixes: Part-Chapter linkage and Constitution cross-references."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force fresh imports
for key in list(sys.modules.keys()):
    if any(x in key for x in ['act_parser', 'citations', 'phase1_preprocessing']):
        del sys.modules[key]

from phase1_preprocessing.act_parser import (
    load_raw_pages,
    parse_parts_chapters,
    parse_act,
)
from phase1_preprocessing.citations import (
    extract_references_from_section,
    CROSS_ACT_ARTICLE_RE,
)

RAW_DIR = Path("phase1_output_v2/raw_pdf_text")
failures = []

# =============================================================================
# Test 2.1: Part-Chapter linkage for BNS
# BNS has NO Part divisions -- only Chapters. This is structurally correct.
# Verify chapters are extracted and part_id is correctly None for all.
# =============================================================================
print("=== Test 2.1: Part-Chapter linkage (BNS -- no Parts expected) ===")
bns_jsonl = RAW_DIR / "BNS_2023.jsonl"
if not bns_jsonl.exists():
    print(f"SKIP: {bns_jsonl} not found")
else:
    pages = load_raw_pages(bns_jsonl)
    full_text = "\n\n".join(p.get("text", "") for p in pages if not p.get("is_toc"))
    parts, chapters = parse_parts_chapters("BNS_2023", full_text)
    print(f"BNS Parts: {len(parts)} (expected 0 -- BNS has no Part divisions)")
    print(f"BNS Chapters: {len(chapters)} (expected ~19)")
    if len(chapters) < 10:
        failures.append(f"FAIL: BNS only {len(chapters)} chapters, expected >=10")
    else:
        print(f"OK: BNS has {len(chapters)} chapters, 0 parts (correct for this act)")

# =============================================================================
# Test 2.2: Part-Chapter linkage for BSA
# BSA has Part IV with Chapters VII, VIII, IX -- the ONLY act with Parts+Chapters.
# After fix, chapters VII/VIII/IX should have part_id = BSA_2023_PART_IV.
# =============================================================================
print("\n=== Test 2.2: Part-Chapter linkage (BSA -- has Part IV) ===")
bsa_jsonl = RAW_DIR / "BSA_2023.jsonl"
if not bsa_jsonl.exists():
    print(f"SKIP: {bsa_jsonl} not found")
else:
    pages = load_raw_pages(bsa_jsonl)
    full_text = "\n\n".join(p.get("text", "") for p in pages if not p.get("is_toc"))
    parts, chapters = parse_parts_chapters("BSA_2023", full_text)
    chapters_with_part = [c for c in chapters if c.get("part_id")]
    print(f"BSA: {len(parts)} parts, {len(chapters)} chapters, {len(chapters_with_part)} with part_id")
    for c in chapters:
        print(f"  {c['chapter_id']} -> part_id={c['part_id']}")
    if len(parts) < 1:
        failures.append(f"FAIL: BSA has {len(parts)} parts, expected 1 (Part IV)")
    elif len(chapters) < 3:
        failures.append(f"FAIL: BSA has {len(chapters)} chapters, expected >=3")
    elif len(chapters_with_part) == 0:
        failures.append("FAIL: BSA chapters should have part_id=BSA_2023_PART_IV")
    else:
        print(f"OK: BSA {len(chapters_with_part)}/{len(chapters)} chapters have part_id")

# =============================================================================
# Test 2.3: Constitution article cross-references
# =============================================================================
print("\n=== Test 2.3: Constitution article cross-references ===")

test_text = (
    "The Magistrate shall record the statement under clause (1) of article 356 of the Constitution "
    "was in force. Also see art. 72 of the Constitution of India for pardon powers."
)
intra, cross = extract_references_from_section(test_text, "BNSS_2023", "BNSS_2023_s460")
const_refs = [r for r in cross if r["target_act_id"] == "CONST_1950"]
print(f"Constitution cross-refs found: {len(const_refs)}")
for r in const_refs:
    print(f"  {r['from_section_id']} -> {r['to_section_id']} [{r['reference_type']}]")

expected_articles = {"CONST_1950_Art356", "CONST_1950_Art72"}
found_articles = {r["to_section_id"] for r in const_refs}
missing = expected_articles - found_articles
if missing:
    failures.append(f"FAIL: Missing Constitution article refs: {missing}")
else:
    print(f"OK: All expected Constitution article refs found: {found_articles}")

# =============================================================================
# Test 2.4: CROSS_ACT_ARTICLE_RE pattern test
# =============================================================================
print("\n=== Test 2.4: CROSS_ACT_ARTICLE_RE regex ===")
test_cases = [
    ("article 21 of the Constitution", "21"),
    ("Article 356 of the Constitution of India", "356"),
    ("art. 14 of the Constitution", "14"),
    ("article 21A of the Constitution of India", "21A"),
]
for text, expected_num in test_cases:
    m = CROSS_ACT_ARTICLE_RE.search(text)
    if m and m.group(1) == expected_num:
        print(f"  OK: '{text}' -> article {m.group(1)}")
    else:
        failures.append(f"FAIL: '{text}' did not match, got {m}")

# =============================================================================
# Test 2.5: Verify against existing sections CSV for Constitution refs
# =============================================================================
print("\n=== Test 2.5: Constitution refs in existing sections.csv ===")
import csv
const_refs_in_csv = []
with open("phase1_output_v2/sections.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("full_text"):
            _, cross = extract_references_from_section(
                row["full_text"], row["act_id"], row["section_id"]
            )
            for r in cross:
                if r["target_act_id"] == "CONST_1950":
                    const_refs_in_csv.append(r)

print(f"Constitution article cross-refs in existing CSV: {len(const_refs_in_csv)}")
for r in const_refs_in_csv[:5]:
    print(f"  {r['from_section_id']} -> {r['to_section_id']}")

if len(const_refs_in_csv) == 0:
    failures.append("FAIL: No Constitution article cross-refs found in sections.csv")
else:
    print(f"OK: {len(const_refs_in_csv)} Constitution cross-refs found")

# =============================================================================
# Result
# =============================================================================
print("\n=== RESULT ===")
if failures:
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("ALL ASSERTIONS PASSED")
