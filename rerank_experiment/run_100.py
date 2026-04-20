"""
Run standalone reranker experiment on all 100 test cases.
Outputs to experiment-specific result files to avoid clobbering existing runs.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.metrics import compute_metrics
from bns_comparison.test_cases import TEST_CASES
from evaluation.run_system3_100 import CSV_FIELDNAMES
from rerank_experiment.adapter import System3RerankAdapter


DEFAULT_RAW = Path("evaluation/results/system3_raw_results_rerank_exp.jsonl")
DEFAULT_CSV = Path("evaluation/results/system3_results_100_rerank_exp.csv")


def _load_completed_ids(raw_results_path: Path) -> set:
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
                except Exception:
                    pass
    return completed


def _export_metrics_csv(raw_results_path: Path, metrics_csv_path: Path):
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
            rows.append({
                "case_id": case["id"],
                "case_description": case["description"],
                "offense_category": case.get("offense_category", ""),
                "rephrased_query": obj["result"].get("rephrased_query", ""),
                "cited_sections": "|".join(obj["result"].get("cited_sections", [])),
                "gold_sections": "|".join(case.get("expected_bns_sections", [])),
                **metrics,
            })
    rows.sort(key=lambda r: r["case_id"])
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Export] Metrics CSV written: {metrics_csv_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Run rerank experiment on 100 test cases")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--raw-results", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    cases = [c for c in TEST_CASES if args.start <= c["id"] <= args.end]
    completed = _load_completed_ids(args.raw_results)
    remaining = [c for c in cases if c["id"] not in completed]
    print(f"[Rerank-100] Total cases: {len(cases)}, Already done: {len(completed)}, Remaining: {len(remaining)}")
    if not remaining:
        _export_metrics_csv(args.raw_results, args.metrics_csv)
        return

    args.raw_results.parent.mkdir(parents=True, exist_ok=True)
    adapter = System3RerankAdapter()

    run_start = time.time()
    wall_times = []
    total_target = len(completed) + len(remaining)

    for i, case in enumerate(remaining):
        case_id = case["id"]
        done_so_far = len(completed) + i + 1
        eta_str = ""
        if wall_times:
            eta_sec = (sum(wall_times) / len(wall_times)) * (len(remaining) - i)
            eta_str = f" | ETA: {eta_sec/60:.0f} min"

        print(f"\n{'='*60}")
        print(f"[{done_so_far}/{total_target}] Case {case_id}{eta_str}")
        print(f"  Query: {case['description'][:80]}...")
        print(f"{'='*60}")

        t_case = time.time()
        try:
            result = adapter.answer_query(case["description"])
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "system_name": "System3_Rerank_CitationConstrained",
                "rephrased_query": "ERROR",
                "answer": "",
                "retrieved_chunks": [],
                "cited_sections": [],
                "context_text": "",
                "graph_sections": [],
                "timings": {"rephrase_sec": 0, "retrieval_sec": 0, "generation_sec": 0, "total_sec": 0},
            }
        wall = time.time() - t_case
        wall_times.append(wall)

        record = {"case_id": case_id, "result": result, "wall_time_sec": round(wall, 2)}
        with open(args.raw_results, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = compute_metrics(case, result)
        print(
            f"  RESULT: hr={metrics['hit_rate']} mrr={metrics['mrr']:.2f} "
            f"f1={metrics['section_f1']:.2f} grounding={metrics['grounding_score']:.2f}"
        )
        print(f"  TIMING: case={wall:.1f}s  total_elapsed={(time.time()-run_start)/60:.1f} min")
        print(f"  CITED:  {result.get('cited_sections', [])}")
        print(f"  GOLD:   {case.get('expected_bns_sections', [])}")
        print(f"  PROGRESS: {done_so_far}/{total_target} complete ({done_so_far*100//total_target}%)")

    print(f"\n{'='*60}")
    print("[Rerank-100] ALL RUNS COMPLETE")
    print(f"  Total time: {(time.time()-run_start)/60:.1f} minutes")
    print(f"  Raw results: {args.raw_results}")
    print(f"{'='*60}")
    _export_metrics_csv(args.raw_results, args.metrics_csv)


if __name__ == "__main__":
    main()

