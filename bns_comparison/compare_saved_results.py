"""
Compare separately saved system CSV outputs and write a summary text report.

Usage:
  python -m bns_comparison.compare_saved_results
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bns_comparison" / "results"

INPUTS = [
    ("System1_OldWork", RESULTS_DIR / "comparison_system1.csv"),
    ("System2_SimpleBNS", RESULTS_DIR / "comparison_system2.csv"),
    ("System3_FullPipelineBNS", RESULTS_DIR / "comparison_system3.csv"),
]

OUT_TXT = RESULTS_DIR / "comparison_separate_runs_summary.txt"

METRICS = [
    "section_f1",
    "correct_act_cited",
    "ipc_reference_count",
    "hallucination_flag",
    "grounding_score",
    "offense_keyword_coverage",
    "completeness_score",
    "total_latency_sec",
]


def _read_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _avg(rows: List[Dict], key: str) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, 0.0)))
        except (TypeError, ValueError):
            continue
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    per_system_rows: Dict[str, List[Dict]] = {}
    for sname, csv_path in INPUTS:
        per_system_rows[sname] = _read_rows(csv_path)

    lines: List[str] = []
    lines.append("BNS Comparison Summary (separate system runs)")
    lines.append("=" * 72)
    lines.append("")

    for sname, rows in per_system_rows.items():
        lines.append(f"{sname}: cases={len(rows)}")
    lines.append("")

    col_w = 24
    systems = [name for name, _ in INPUTS]
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:<{col_w}}" for s in systems)
    lines.append(header)
    lines.append("-" * len(header))

    for m in METRICS:
        row = f"{m:<{col_w}}"
        for s in systems:
            row += f"{_avg(per_system_rows[s], m):<{col_w}.4f}"
        lines.append(row)

    # System 3 deltas vs baseline System 1
    lines.append("")
    lines.append("Delta: System3_FullPipelineBNS - System1_OldWork")
    lines.append("-" * 72)
    s1_rows = per_system_rows.get("System1_OldWork", [])
    s3_rows = per_system_rows.get("System3_FullPipelineBNS", [])
    for m in METRICS:
        delta = _avg(s3_rows, m) - _avg(s1_rows, m)
        lines.append(f"{m:<40}{delta:+.4f}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Done] Wrote summary: {OUT_TXT}")


if __name__ == "__main__":
    main()
