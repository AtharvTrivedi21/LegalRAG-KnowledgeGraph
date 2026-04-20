"""
Run System 3 (Full Pipeline BNS) on all 100 test cases with incremental saving.

Saves full results to JSONL (one line per case) for crash-safe resume.
After completion, exports a CSV with all rule-based metrics.

Usage:
    python -m evaluation.run_system3_100 [--start ID] [--end ID]

Resume: just re-run — completed case IDs are skipped automatically.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so GROQ_API_KEY is picked up automatically
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from bns_comparison.test_cases import TEST_CASES
from bns_comparison.metrics import compute_metrics
from bns_comparison.adapters.full_pipeline_bns import FullPipelineBNSAdapter
from evaluation.config import SYSTEM3_RAW_RESULTS, SYSTEM3_METRICS_CSV, RESULTS_DIR

CSV_FIELDNAMES = [
    "case_id", "case_description", "offense_category",
    "rephrased_query", "cited_sections", "gold_sections",
    # Retrieval
    "hit_rate", "mrr",
    # Accuracy
    "section_precision", "section_recall", "section_f1", "correct_act_cited",
    # Hallucination
    "ipc_reference_count", "fabricated_section_count", "grounding_score", "hallucination_flag",
    # Speed
    "rephrase_latency_sec", "retrieval_latency_sec", "generation_latency_sec", "total_latency_sec",
    # Answer Quality
    "offense_category_hit", "offense_keyword_coverage", "completeness_score",
    "key_issue_coverage", "answer_relevance_score", "context_relevance_score",
    "answer_length_words", "has_safety_disclaimer",
]


def _load_completed_ids(raw_results_path: Path) -> set:
    """Read already-completed case IDs from the JSONL file."""
    completed = set()
    if raw_results_path.exists():
        with open(raw_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    completed.add(obj["case_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def _export_metrics_csv(raw_results_path: Path, metrics_csv_path: Path):
    """Read all JSONL results and export rule-based metrics to CSV."""
    if not raw_results_path.exists():
        return

    rows = []
    with open(raw_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case = next((c for c in TEST_CASES if c["id"] == obj["case_id"]), None)
            if case is None:
                continue
            metrics = compute_metrics(case, obj["result"])
            row = {
                "case_id": case["id"],
                "case_description": case["description"],
                "offense_category": case.get("offense_category", ""),
                "rephrased_query": obj["result"].get("rephrased_query", ""),
                "cited_sections": "|".join(obj["result"].get("cited_sections", [])),
                "gold_sections": "|".join(case.get("expected_bns_sections", [])),
                **metrics,
            }
            rows.append(row)

    rows.sort(key=lambda r: r["case_id"])

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Export] Metrics CSV written: {metrics_csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Run System 3 on 100 test cases")
    parser.add_argument("--start", type=int, default=1, help="Start case ID (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End case ID (inclusive)")
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=SYSTEM3_RAW_RESULTS,
        help="Path to JSONL raw results file (default: evaluation.config.SYSTEM3_RAW_RESULTS)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=SYSTEM3_METRICS_CSV,
        help="Path to metrics CSV output file (default: evaluation.config.SYSTEM3_METRICS_CSV)",
    )
    args = parser.parse_args()

    cases = [c for c in TEST_CASES if args.start <= c["id"] <= args.end]
    completed = _load_completed_ids(args.raw_results)

    remaining = [c for c in cases if c["id"] not in completed]
    print(f"[System3-100] Total cases: {len(cases)}, Already done: {len(completed)}, Remaining: {len(remaining)}")

    if not remaining:
        print("[System3-100] All cases complete. Exporting metrics CSV...")
        _export_metrics_csv(args.raw_results, args.metrics_csv)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.raw_results.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[System3-100] Loading System 3 adapter...")
    adapter = FullPipelineBNSAdapter()
    print("[System3-100] Adapter loaded. Starting runs...\n")

    total_target = len(completed) + len(remaining)
    wall_times = []
    run_start = time.time()

    for i, case in enumerate(remaining):
        done_so_far = len(completed) + i + 1
        case_id = case["id"]

        # ETA calculation
        if wall_times:
            avg_wall = sum(wall_times) / len(wall_times)
            eta_sec = avg_wall * (len(remaining) - i)
            eta_min = eta_sec / 60
            eta_str = f" | ETA: {eta_min:.0f} min"
        else:
            eta_str = ""

        print(f"\n{'='*60}")
        print(f"[{done_so_far}/{total_target}] Case {case_id}{eta_str}")
        print(f"  Query: {case['description'][:80]}...")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            result = adapter.answer_query(case["description"])
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "system_name": "System3_FullPipelineBNS",
                "rephrased_query": "ERROR",
                "answer": "",
                "retrieved_chunks": [],
                "cited_sections": [],
                "context_text": "",
                "graph_sections": [],
                "timings": {"rephrase_sec": 0, "retrieval_sec": 0, "generation_sec": 0, "total_sec": 0},
            }
        wall_time = time.time() - t_start
        wall_times.append(wall_time)

        record = {
            "case_id": case_id,
            "result": result,
            "wall_time_sec": round(wall_time, 2),
        }

        with open(args.raw_results, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = compute_metrics(case, result)
        cited = result.get("cited_sections", [])
        gold = case.get("expected_bns_sections", [])
        elapsed_total = time.time() - run_start
        print(
            f"  RESULT: hr={metrics['hit_rate']} mrr={metrics['mrr']:.2f} "
            f"f1={metrics['section_f1']:.2f} grounding={metrics['grounding_score']:.2f}"
        )
        print(
            f"  TIMING: case={wall_time:.1f}s  total_elapsed={elapsed_total/60:.1f} min"
        )
        print(f"  CITED:  {cited}")
        print(f"  GOLD:   {gold}")
        print(f"  PROGRESS: {done_so_far}/{total_target} complete ({done_so_far*100//total_target}%)")

    print(f"\n{'='*60}")
    total_elapsed = time.time() - run_start
    print(f"[System3-100] ALL RUNS COMPLETE")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Avg per case: {total_elapsed/len(remaining):.1f}s")
    print(f"  Raw results: {args.raw_results}")
    print(f"{'='*60}")
    _export_metrics_csv(args.raw_results, args.metrics_csv)


if __name__ == "__main__":
    main()
