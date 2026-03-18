"""Debug: test QUOTED_TERM_RE directly on BNS s.2 text."""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if 'phase1_preprocessing.definitions' in sys.modules:
    del sys.modules['phase1_preprocessing.definitions']

from phase1_preprocessing.definitions import QUOTED_TERM_RE, DEFINITION_SECTION_RE, extract_definitions_from_section

with open('phase1_output_v2/sections.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['section_id'] == 'BNS_2023_s2':
            text = row['full_text']
            print("Guard fires:", bool(DEFINITION_SECTION_RE.search(text)))
            print("QUOTED_TERM_RE pattern:", QUOTED_TERM_RE.pattern[:120])
            print()

            # Test regex on text
            matches = list(QUOTED_TERM_RE.finditer(text[:800]))
            print(f"Regex matches in first 800 chars: {len(matches)}")
            for m in matches[:10]:
                print(f"  term={repr(m.group(1))}, def_start={repr(m.group(2)[:40])}")

            print()
            # Try a simple raw pattern directly
            simple = re.compile(r'\u201c([^\u201c\u201d\n]{1,80})\u201d\s+(?:means|includes|denotes|shall mean|shall include)')
            simple_matches = list(simple.finditer(text[:800]))
            print(f"Simple pattern matches: {len(simple_matches)}")
            for m in simple_matches[:10]:
                print(f"  {repr(m.group(0)[:60])}")

            print()
            defs = extract_definitions_from_section(text, 'BNS_2023', 'BNS_2023_s2')
            print(f"extract_definitions_from_section result: {len(defs)} defs")
            for d in defs[:10]:
                print(f"  {d['term']!r}")
            break
