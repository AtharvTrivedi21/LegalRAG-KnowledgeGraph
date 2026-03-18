"""Re-run the full Phase 1 pipeline to regenerate all CSVs with fixed code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase1_preprocessing.structure_export import run_structure_parse_and_export

result = run_structure_parse_and_export(
    Path("phase1_output_v2/raw_pdf_text"),
    Path("phase1_output_v2"),
)

print("=== Pipeline Result ===")
for k, v in result.items():
    if k != "warnings":
        print(f"  {k}: {v}")
warnings = result.get("warnings", [])
print(f"  warnings ({len(warnings)}):")
for w in warnings:
    print(f"    - {w}")
