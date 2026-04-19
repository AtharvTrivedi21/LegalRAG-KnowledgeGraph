"""
Merge all evaluation metrics and produce thesis-ready tables and statistics.

Combines:
  - Rule-based metrics from system3_results_100.csv
  - GPT-4.1 judge scores from gpt4_eval_results.csv

Usage:
    python -m evaluation.analysis
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.config import (
    SYSTEM3_METRICS_CSV,
    GPT4_EVAL_RESULTS,
    FINAL_EVAL_CSV,
    RESULTS_DIR,
)


def _load_csv(path: Path) -> dict:
    """Load CSV as dict keyed by case_id (int)."""
    data = {}
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["case_id"])
                data[cid] = row
            except (KeyError, ValueError):
                pass
    return data


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def merge_results():
    """Merge rule-based and GPT-4.1 metrics into a single CSV."""
    rule_data = _load_csv(SYSTEM3_METRICS_CSV)
    gpt_data = _load_csv(GPT4_EVAL_RESULTS)

    if not rule_data:
        print(f"ERROR: No rule-based metrics found at {SYSTEM3_METRICS_CSV}")
        print("Run evaluation.run_system3_100 first.")
        return

    all_case_ids = sorted(rule_data.keys())

    gpt_fields = ["answer_relevance", "context_relevance", "groundedness", "avg_relevance", "justification"]
    merged_fields = list(csv.DictReader(open(SYSTEM3_METRICS_CSV, "r", encoding="utf-8")).fieldnames or [])
    for gf in gpt_fields:
        if gf not in merged_fields:
            merged_fields.append(gf)

    rows = []
    for cid in all_case_ids:
        row = dict(rule_data[cid])
        if cid in gpt_data:
            for gf in gpt_fields:
                row[gf] = gpt_data[cid].get(gf, "")
        else:
            for gf in gpt_fields:
                row[gf] = ""
        rows.append(row)

    with open(FINAL_EVAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Merge] Final evaluation CSV: {FINAL_EVAL_CSV} ({len(rows)} rows)")
    return rows


def compute_summary(rows):
    """Compute and print aggregate statistics for thesis tables."""
    numeric_metrics = [
        ("hit_rate", "Hit Rate"),
        ("mrr", "MRR"),
        ("section_precision", "Section Precision"),
        ("section_recall", "Section Recall"),
        ("section_f1", "Section F1"),
        ("correct_act_cited", "Correct Act Cited"),
        ("grounding_score", "Grounding Score"),
        ("ipc_reference_count", "IPC References"),
        ("fabricated_section_count", "Fabricated Sections"),
        ("hallucination_flag", "Hallucination Flag"),
        ("offense_category_hit", "Offense Category Hit"),
        ("offense_keyword_coverage", "Keyword Coverage"),
        ("completeness_score", "Completeness"),
        ("key_issue_coverage", "Key Issue Coverage"),
        ("answer_relevance_score", "Answer Relevance (rule)"),
        ("context_relevance_score", "Context Relevance (rule)"),
        ("total_latency_sec", "Total Latency (s)"),
        ("answer_length_words", "Answer Length (words)"),
        ("has_safety_disclaimer", "Safety Disclaimer"),
        ("answer_relevance", "Answer Relevance (GPT-4.1)"),
        ("context_relevance", "Context Relevance (GPT-4.1)"),
        ("groundedness", "Groundedness (GPT-4.1)"),
        ("avg_relevance", "Avg Relevance (GPT-4.1)"),
    ]

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION SUMMARY (100 queries)")
    print("=" * 70)
    print(f"{'Metric':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for key, label in numeric_metrics:
        vals = [_safe_float(r.get(key), default=None) for r in rows]
        vals = [v for v in vals if v is not None and v >= 0]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        mn, mx = min(vals), max(vals)
        print(f"{label:<35} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    print("=" * 70)


def compute_category_breakdown(rows):
    """Per offense-category breakdown of key metrics."""
    categories = defaultdict(list)
    for r in rows:
        cat = r.get("offense_category", "unknown")
        categories[cat].append(r)

    key_metrics = ["section_f1", "hit_rate", "grounding_score", "answer_relevance", "groundedness"]

    print("\n" + "=" * 90)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 90)
    header = f"{'Category':<35} {'N':>3}"
    for km in key_metrics:
        header += f" {km[:12]:>12}"
    print(header)
    print("-" * 90)

    for cat in sorted(categories.keys()):
        cat_rows = categories[cat]
        line = f"{cat[:34]:<35} {len(cat_rows):>3}"
        for km in key_metrics:
            vals = [_safe_float(r.get(km), default=None) for r in cat_rows]
            vals = [v for v in vals if v is not None and v >= 0]
            avg = sum(vals) / len(vals) if vals else 0.0
            line += f" {avg:>12.4f}"
        print(line)

    print("=" * 90)


def find_failure_cases(rows):
    """Identify cases where GPT-4.1 scores indicate poor performance."""
    print("\n" + "=" * 70)
    print("FAILURE CASES (GPT-4.1 score < 0.5 on any metric)")
    print("=" * 70)

    failures = []
    for r in rows:
        ar = _safe_float(r.get("answer_relevance"), -1)
        cr = _safe_float(r.get("context_relevance"), -1)
        g = _safe_float(r.get("groundedness"), -1)
        if ar < 0 or cr < 0 or g < 0:
            continue
        if ar < 0.5 or cr < 0.5 or g < 0.5:
            failures.append(r)

    if not failures:
        print("No failure cases found.")
    else:
        print(f"Found {len(failures)} failure cases:\n")
        for r in failures:
            cid = r.get("case_id", "?")
            cat = r.get("offense_category", "?")
            ar = _safe_float(r.get("answer_relevance"))
            cr = _safe_float(r.get("context_relevance"))
            g = _safe_float(r.get("groundedness"))
            just = r.get("justification", "")[:100]
            print(f"  Case {cid} [{cat}]: AR={ar:.2f} CR={cr:.2f} G={g:.2f}")
            if just:
                print(f"    Justification: {just}")

    print("=" * 70)


def compute_correlation(rows):
    """Compute Pearson correlation between rule-based and GPT-4.1 grounding scores."""
    pairs = []
    for r in rows:
        rule_g = _safe_float(r.get("grounding_score"), None)
        gpt_g = _safe_float(r.get("groundedness"), None)
        if rule_g is not None and gpt_g is not None and gpt_g >= 0:
            pairs.append((rule_g, gpt_g))

    if len(pairs) < 5:
        print("\n[Correlation] Not enough data points for correlation analysis.")
        return

    n = len(pairs)
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / n
    std_x = (sum((x - x_mean) ** 2 for x in x_vals) / n) ** 0.5
    std_y = (sum((y - y_mean) ** 2 for y in y_vals) / n) ** 0.5

    if std_x > 0 and std_y > 0:
        pearson = cov / (std_x * std_y)
    else:
        pearson = 0.0

    print(f"\n[Correlation] Rule-based grounding vs GPT-4.1 groundedness:")
    print(f"  Pearson r = {pearson:.4f} (n={n})")


def main():
    rows = merge_results()
    if not rows:
        return

    compute_summary(rows)
    compute_category_breakdown(rows)
    find_failure_cases(rows)
    compute_correlation(rows)

    print(f"\n[Analysis] All outputs saved to {RESULTS_DIR}")
    print("[Analysis] Use final_evaluation.csv for thesis tables.")


if __name__ == "__main__":
    main()
