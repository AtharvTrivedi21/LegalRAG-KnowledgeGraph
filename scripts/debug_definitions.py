"""Debug script to inspect actual quote chars in BNS s.2 and test regex matching."""
import csv
import re

with open('phase1_output_v2/sections.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['section_id'] == 'BNS_2023_s2':
            text = row['full_text']
            print("=== First 200 chars repr:")
            print(repr(text[:200]))
            print()

            print("=== Non-ASCII chars in first 400 chars:")
            seen = set()
            for c in text[:400]:
                if ord(c) > 127 and c not in seen:
                    seen.add(c)
                    print(f"  U+{ord(c):04X} {repr(c)}")
            print()

            # Find chars immediately around first term "act"
            # The text shows (1) <QUOTE>act<QUOTE> denotes
            idx = text.find('act')
            print(f"=== Chars around first 'act' at idx={idx}:")
            for i, c in enumerate(text[max(0, idx-3):idx+6]):
                print(f"  [{i}] U+{ord(c):04X} {repr(c)}")
            print()

            # Build quote set from what we found
            Q_chars = [c for c in text[:500] if ord(c) > 127 and 'QUOTE' in '' or ord(c) in (
                0x201C, 0x201D, 0x2018, 0x2019, 0x2032, 0x2033, 0x0022, 0x0027,
                # check what's actually there:
                ord(text[idx-1]) if idx > 0 else 0,
                ord(text[idx-2]) if idx > 1 else 0,
            )]

            # Get actual open/close quote chars from the text
            open_q = text[idx-1] if idx > 0 else ''
            close_q_idx = text.find('denotes', idx)
            if close_q_idx > 0:
                close_q = text[close_q_idx - 2]  # char before space before verb
                print(f"Open quote char: U+{ord(open_q):04X} {repr(open_q)}")
                print(f"Close quote char: U+{ord(close_q):04X} {repr(close_q)}")
            print()

            # Try a direct regex with the actual chars
            # Build pattern using actual char codes found
            actual_quotes = set()
            for c in text[:500]:
                co = ord(c)
                if co in (0x201C, 0x201D, 0x2018, 0x2019, 0x2032, 0x2033, 0x22, 0x27):
                    actual_quotes.add(co)
            print("Actual quote code points in text:", [hex(x) for x in sorted(actual_quotes)])

            # Now try matching
            Q = ''.join(chr(c) for c in sorted(actual_quotes))
            pat_str = rf'[{re.escape(Q)}]([^{"".join(chr(c) for c in sorted(actual_quotes))}\n]{{1,80}})[{re.escape(Q)}]\s+(?:means|includes|denotes|shall mean|shall include)'
            print("Pattern:", repr(pat_str[:100]))
            pat = re.compile(pat_str, re.IGNORECASE)
            matches = list(pat.finditer(text[:500]))
            print(f"Matches: {len(matches)}")
            for m in matches[:5]:
                print(f"  {repr(m.group(0)[:80])}")
            break
