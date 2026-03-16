"""
Phase 1 v2 validation: duplicate check, min text length, citation resolution stats.
Output: phase1_output_v2/validation_report.md
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

MIN_SECTION_TEXT_LEN = 20
MIN_ARTICLE_TEXT_LEN = 20


def run_validation(output_dir: Path) -> dict[str, Any]:
    """Run all validation gates; write validation_report.md. Returns summary dict."""
    output_dir = Path(output_dir)
    errors = []
    warnings = []
    counts = {}
    # Duplicate ID check
    for name, id_col in [
        ("acts.csv", "act_id"),
        ("parts.csv", "part_id"),
        ("chapters.csv", "chapter_id"),
        ("sections.csv", "section_id"),
        ("articles.csv", "article_id"),
        ("definitions.csv", "def_id"),
    ]:
        path = output_dir / name
        if not path.exists():
            continue
        ids = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(row.get(id_col, ""))
        counts[name] = len(ids)
        dup = len(ids) - len(set(ids))
        if dup > 0:
            errors.append(f"{name}: {dup} duplicate {id_col} values")
    # Min text length
    for name, text_col, min_len in [
        ("sections.csv", "full_text", MIN_SECTION_TEXT_LEN),
        ("articles.csv", "full_text", MIN_ARTICLE_TEXT_LEN),
    ]:
        path = output_dir / name
        if not path.exists():
            continue
        short = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len((row.get(text_col) or "").strip()) < min_len:
                    short += 1
        if short > 0:
            warnings.append(f"{name}: {short} rows with {text_col} length < {min_len}")
    # Citation resolution (if case citation CSVs exist)
    res_sec = res_art = unres = 0
    if (output_dir / "case_cites_section.csv").exists():
        with open(output_dir / "case_cites_section.csv", "r", encoding="utf-8") as f:
            res_sec = sum(1 for _ in csv.DictReader(f))
    if (output_dir / "case_cites_article.csv").exists():
        with open(output_dir / "case_cites_article.csv", "r", encoding="utf-8") as f:
            res_art = sum(1 for _ in csv.DictReader(f))
    if (output_dir / "unresolved_case_cites.csv").exists():
        with open(output_dir / "unresolved_case_cites.csv", "r", encoding="utf-8") as f:
            unres = sum(1 for _ in csv.DictReader(f))
    total_cites = res_sec + res_art + unres
    if total_cites > 0:
        rate = (res_sec + res_art) / total_cites * 100
        counts["citation_resolution_rate_pct"] = round(rate, 1)
        counts["resolved_section_cites"] = res_sec
        counts["resolved_article_cites"] = res_art
        counts["unresolved_cites"] = unres
        if rate < 50:
            warnings.append(f"Citation resolution rate low: {rate:.1f}%")
    # Write report
    lines = [
        "# Phase 1 v2 Validation Report",
        "",
        "## Counts",
        "| File | Count |",
        "|------|-------|",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Errors", ""])
    lines.extend(errors if errors else ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(warnings if warnings else ["None"])
    lines.append("")
    report_path = output_dir / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"errors": errors, "warnings": warnings, "counts": counts, "passed": len(errors) == 0}
