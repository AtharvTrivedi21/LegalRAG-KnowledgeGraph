"""
Extract BNS_2023 section numbers and headings from sections.csv.

Outputs a clean numbered list for use in generating test cases.
Usage:
    python -m evaluation.helpers.extract_bns_headings
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SECTIONS_CSV = PROJECT_ROOT / "phase1_output_v2" / "sections.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "bns_section_headings.txt"


def extract():
    if not SECTIONS_CSV.exists():
        print(f"ERROR: {SECTIONS_CSV} not found")
        sys.exit(1)

    bns_sections = []
    seen = set()
    with open(SECTIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("act_id") != "BNS_2023":
                continue
            sec_num = row.get("section_number", "").strip()
            if not sec_num or sec_num in seen:
                continue
            heading = row.get("heading", "").strip()
            # Truncate heading at the first '––' or '—' to get just the title
            for sep in ["––", ".––", "—", ".—"]:
                if sep in heading:
                    heading = heading.split(sep)[0].strip()
                    break
            # Remove trailing period
            heading = heading.rstrip(".")
            # Skip rows where heading looks like commentary (no legal heading)
            if len(heading) > 120 or heading[0:1].islower():
                continue
            seen.add(sec_num)
            bns_sections.append((int(sec_num) if sec_num.isdigit() else sec_num, sec_num, heading))

    bns_sections.sort(key=lambda x: (int(x[0]) if isinstance(x[0], int) else 9999))

    lines = []
    lines.append("=" * 70)
    lines.append("BNS 2023 (Bharatiya Nyaya Sanhita) — All Sections")
    lines.append(f"Total: {len(bns_sections)} sections")
    lines.append("=" * 70)
    lines.append("")

    for _, sec_num, heading in bns_sections:
        lines.append(f"Section {sec_num}: {heading}")

    lines.append("")
    lines.append("=" * 70)

    output = "\n".join(lines)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"Written {len(bns_sections)} BNS sections to: {OUTPUT_FILE}")
    print()
    print(output)


if __name__ == "__main__":
    extract()
