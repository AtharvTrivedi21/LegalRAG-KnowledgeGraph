"""Debug: check what 'In this Sanhita' looks like in BNS s.2 in the CSV."""
import csv
import re

DEFINITION_SECTION_RE = re.compile(
    r"[Ii]n this (?:[Aa]ct|[Ss]anhita|[Aa]dhiniyam)",
)

with open('phase1_output_v2/sections.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['section_id'] == 'BNS_2023_s2':
            text = row['full_text']
            print("Guard search:", DEFINITION_SECTION_RE.search(text))
            print("Text starts with:", repr(text[:50]))

            # Find every 'in this' ignoring case
            for m in re.finditer(r'[Ii]n this', text):
                snippet = text[m.start():m.start()+30]
                print("Found 'in this':", repr(snippet))
            break

# Also check what sections actually contain definitions - search for 'denotes' or 'means'
print("\n=== Sections that match guard ===")
with open('phase1_output_v2/sections.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if DEFINITION_SECTION_RE.search(row['full_text']):
            print(f"  {row['section_id']}: {repr(row['full_text'][:60])}")
