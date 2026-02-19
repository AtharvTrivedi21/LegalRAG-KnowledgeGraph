from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


EXPECTED = {
    "acts.csv": ["act_id", "act_name", "act_type", "source_file"],
    "sections.csv": ["section_id", "act_id", "section_number", "full_text"],
    "articles.csv": ["article_id", "act_id", "article_number", "full_text"],
    "cases.csv": ["case_id", "judgment_text", "year"],
    "edges.csv": ["source_case_id", "target_section_or_article", "relation"],
}


def _read_csv(path: Path) -> pd.DataFrame:
    # Prefer the default C engine (fast, supports low_memory).
    # Fall back to the python engine if needed for tricky quoting/newlines.
    try:
        return pd.read_csv(
            path,
            engine="c",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
            low_memory=False,
        )
    except Exception:
        return pd.read_csv(
            path,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="error",
        )


def _check_columns(name: str, df: pd.DataFrame) -> list[str]:
    expected = EXPECTED[name]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    out = []
    if missing:
        out.append(f"missing columns: {missing}")
    if extra:
        out.append(f"unexpected columns: {extra}")
    return out


def _dup_report(df: pd.DataFrame, col: str, max_examples: int = 10) -> tuple[int, list[str]]:
    s = df[col].astype(str)
    vc = s.value_counts(dropna=False)
    dups = vc[vc > 1]
    if dups.empty:
        return 0, []
    examples = [f"{idx} (x{int(cnt)})" for idx, cnt in dups.head(max_examples).items()]
    return int(dups.sum() - len(dups)), examples


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Phase-1 outputs under phase1_output/")
    ap.add_argument(
        "--output-dir",
        default="phase1_output",
        help="Output directory to verify (default: phase1_output)",
    )
    ap.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Treat duplicate IDs/edges as an error (default: warn only).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: output dir not found: {out_dir.resolve()}")
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    print(f"Verifying Phase-1 outputs in: {out_dir.resolve()}")
    print()

    # 1) Required files
    for fname in EXPECTED:
        path = out_dir / fname
        if not path.exists():
            errors.append(f"missing file: {fname}")

    # 2) Optional plot files (warn if missing)
    expected_plots = ["top_cited_sections.png", "top_cited_articles.png"]
    for p in expected_plots:
        if not (out_dir / p).exists():
            warnings.append(f"missing plot: {p}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
        return 2

    # 3) Read and validate CSVs
    dfs: dict[str, pd.DataFrame] = {}
    for fname in EXPECTED:
        path = out_dir / fname
        try:
            df = _read_csv(path)
        except Exception as e:
            errors.append(f"failed reading {fname}: {e}")
            continue

        dfs[fname] = df
        col_issues = _check_columns(fname, df)
        if col_issues:
            errors.append(f"{fname}: " + "; ".join(col_issues))

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
        return 2

    # 4) Basic row counts
    counts = {k: len(v) for k, v in dfs.items()}
    print("Row counts:")
    for k in ["cases.csv", "acts.csv", "sections.csv", "articles.csv", "edges.csv"]:
        print(f"- {k}: {counts[k]:,}")
    print()

    # 5) Integrity checks
    # Acts
    acts = dfs["acts.csv"]
    if acts["act_id"].isna().any():
        errors.append("acts.csv: act_id contains nulls")
    if len(acts) == 0:
        errors.append("acts.csv: no rows")

    # Sections / Articles IDs
    sections = dfs["sections.csv"]
    articles = dfs["articles.csv"]
    for fname, df, idcol in [
        ("sections.csv", sections, "section_id"),
        ("articles.csv", articles, "article_id"),
    ]:
        if df[idcol].isna().any():
            errors.append(f"{fname}: {idcol} contains nulls")

        dup_count, examples = _dup_report(df, idcol)
        if dup_count > 0:
            msg = f"{fname}: duplicate {idcol} values detected (extra rows due to dups: {dup_count:,}). Examples: {examples}"
            if args.fail_on_duplicates:
                errors.append(msg)
            else:
                warnings.append(msg)

    # Cases
    cases = dfs["cases.csv"]
    if cases["case_id"].isna().any():
        errors.append("cases.csv: case_id contains nulls")
    if (cases["judgment_text"].astype(str).str.len() == 0).all():
        warnings.append("cases.csv: all judgment_text values are empty strings")
    if not pd.api.types.is_numeric_dtype(cases["year"]):
        warnings.append("cases.csv: year column is not numeric")

    dup_count, examples = _dup_report(cases, "case_id")
    if dup_count > 0:
        msg = f"cases.csv: duplicate case_id values detected (extra rows due to dups: {dup_count:,}). Examples: {examples}"
        if args.fail_on_duplicates:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Edges
    edges = dfs["edges.csv"]
    required_rel = {"CITES"}
    rel_vals = set(edges["relation"].dropna().astype(str).unique())
    if not rel_vals.issubset(required_rel):
        warnings.append(f"edges.csv: unexpected relation values: {sorted(rel_vals - required_rel)}")

    if edges[["source_case_id", "target_section_or_article"]].isna().any(axis=None):
        errors.append("edges.csv: nulls found in source_case_id/target_section_or_article")

    # Edge targets should reference sections or articles (best-effort)
    target_ids = set(edges["target_section_or_article"].astype(str).unique())
    legal_ids = set(sections["section_id"].astype(str).unique()) | set(
        articles["article_id"].astype(str).unique()
    )
    missing_targets = sorted(list(target_ids - legal_ids))
    if missing_targets:
        # show only a few; could be due to duplicate IDs or parsing mismatches
        warnings.append(
            f"edges.csv: {len(missing_targets):,} target IDs not found in sections/articles. Examples: {missing_targets[:10]}"
        )

    # Duplicate edges (same tuple repeated)
    edge_dups = edges.duplicated(subset=["source_case_id", "target_section_or_article", "relation"]).sum()
    if edge_dups:
        msg = f"edges.csv: duplicate edges detected (exact duplicate rows: {int(edge_dups):,})"
        if args.fail_on_duplicates:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Summary
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")
        print()

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
        print()
        return 2

    print("OK: Phase-1 outputs look structurally correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

