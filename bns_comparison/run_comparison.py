"""
BNS-Only RAG Comparison Runner.

Runs all three systems on the 10 shared test cases, computes all metrics,
and outputs a CSV + terminal summary table.

Usage (from project root, with venv active):
    python -m bns_comparison.run_comparison [--systems 1,2,3] [--cases 1-10]

Prerequisites:
    1. Build FAISS indexes:
       python -m bns_comparison.build_bns_faiss --system both
    2. Ensure Ollama is running with llama3:8b and nomic-embed-text pulled
    3. Ensure Neo4j is running (for System 3; gracefully skipped if unavailable)
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.test_cases import TEST_CASES
from bns_comparison.metrics import compute_metrics, summarize_metrics
from bns_comparison.config import RESULTS_DIR, COMPARISON_CSV


def _load_adapters(system_ids: List[int]):
    """Lazily import and instantiate the requested adapters."""
    adapters = {}
    if 1 in system_ids:
        try:
            from bns_comparison.adapters.old_work import OldWorkAdapter
            adapters[1] = OldWorkAdapter()
            print("[Init] System 1 (Old-Work) loaded.")
        except Exception as e:
            print(f"[WARN] System 1 failed to load: {e}")
    if 2 in system_ids:
        try:
            from bns_comparison.adapters.simple_bns import SimpleBNSAdapter
            adapters[2] = SimpleBNSAdapter()
            print("[Init] System 2 (Simple-BNS) loaded.")
        except Exception as e:
            print(f"[WARN] System 2 failed to load: {e}")
    if 3 in system_ids:
        try:
            from bns_comparison.adapters.full_pipeline_bns import FullPipelineBNSAdapter
            adapters[3] = FullPipelineBNSAdapter()
            print("[Init] System 3 (Full-Pipeline-BNS) loaded.")
        except Exception as e:
            print(f"[WARN] System 3 failed to load: {e}")
    return adapters


def _run_one(adapter, case: Dict) -> Optional[Dict]:
    """Run adapter on a single case. Returns result dict or None on error."""
    try:
        return adapter.answer_query(case["description"])
    except Exception as e:
        print(f"    [ERROR] {adapter.system_name} case {case['id']}: {e}")
        traceback.print_exc()
        return None


def _build_csv_row(system_name: str, case: Dict, result: Dict, metrics: Dict) -> Dict:
    return {
        "system_name": system_name,
        "case_id": case["id"],
        "case_description": case["description"],
        "offense_category": case.get("offense_category", ""),
        "rephrased_query": result.get("rephrased_query", ""),
        "cited_sections": "|".join(result.get("cited_sections", [])),
        "gold_sections": "|".join(case.get("expected_bns_sections", [])),
        **metrics,
    }


CSV_FIELDNAMES = [
    "system_name", "case_id", "case_description", "offense_category",
    "rephrased_query", "cited_sections", "gold_sections",
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


def _print_summary_table(all_rows: List[Dict]) -> None:
    """Print a per-system average metrics table to the terminal."""
    systems = {}
    for row in all_rows:
        sname = row["system_name"]
        systems.setdefault(sname, []).append(row)

    key_metrics = [
        "section_f1", "correct_act_cited", "ipc_reference_count",
        "hallucination_flag", "grounding_score", "offense_keyword_coverage",
        "completeness_score", "total_latency_sec",
    ]

    col_w = 28
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:<{col_w}}" for s in systems)
    print("\n" + "=" * (col_w * (1 + len(systems))))
    print("COMPARISON SUMMARY (averages across 10 test cases)")
    print("=" * (col_w * (1 + len(systems))))
    print(header)
    print("-" * (col_w * (1 + len(systems))))

    for metric in key_metrics:
        row_str = f"{metric:<{col_w}}"
        for sname, rows in systems.items():
            vals = [r.get(metric, 0) for r in rows if isinstance(r.get(metric), (int, float))]
            avg = sum(vals) / len(vals) if vals else 0.0
            row_str += f"{avg:<{col_w}.4f}"
        print(row_str)

    print("=" * (col_w * (1 + len(systems))))


def main():
    parser = argparse.ArgumentParser(description="Run BNS RAG comparison")
    parser.add_argument(
        "--systems",
        type=str,
        default="1,2,3",
        help="Comma-separated system IDs to run (default: 1,2,3)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case IDs to run (default: all 10)",
    )
    args = parser.parse_args()

    system_ids = [int(s.strip()) for s in args.systems.split(",") if s.strip()]
    case_ids = (
        {int(c.strip()) for c in args.cases.split(",") if c.strip()}
        if args.cases
        else None
    )
    cases = [c for c in TEST_CASES if case_ids is None or c["id"] in case_ids]

    print(f"\n[Comparison] Systems: {system_ids}")
    print(f"[Comparison] Test cases: {[c['id'] for c in cases]}")

    adapters = _load_adapters(system_ids)
    if not adapters:
        print("[ERROR] No adapters loaded. Check FAISS indexes and Ollama.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for sys_id, adapter in sorted(adapters.items()):
            print(f"\n{'='*60}")
            print(f"Running {adapter.system_name} on {len(cases)} cases...")
            print(f"{'='*60}")

            for case in cases:
                print(f"\n  [Case {case['id']}] {case['description'][:70]}...")
                t_start = time.time()
                result = _run_one(adapter, case)
                elapsed = time.time() - t_start

                if result is None:
                    # Write a blank error row
                    error_row = {
                        "system_name": adapter.system_name,
                        "case_id": case["id"],
                        "case_description": case["description"],
                        "offense_category": case.get("offense_category", ""),
                        "rephrased_query": "ERROR",
                        "cited_sections": "",
                        "gold_sections": "|".join(case.get("expected_bns_sections", [])),
                    }
                    for fn in CSV_FIELDNAMES:
                        error_row.setdefault(fn, 0)
                    writer.writerow(error_row)
                    all_rows.append(error_row)
                    continue

                metrics = compute_metrics(case, result)
                row = _build_csv_row(adapter.system_name, case, result, metrics)
                writer.writerow(row)
                all_rows.append(row)

                print(
                    f"    sec_f1={metrics['section_f1']:.2f}  "
                    f"correct_act={metrics['correct_act_cited']}  "
                    f"ipc_refs={metrics['ipc_reference_count']}  "
                    f"halluc={metrics['hallucination_flag']}  "
                    f"latency={metrics['total_latency_sec']:.1f}s"
                )
                print(
                    f"    cited={result.get('cited_sections', [])}  "
                    f"gold={case['expected_bns_sections']}"
                )

    print(f"\n[Done] Results written to: {COMPARISON_CSV}")
    _print_summary_table(all_rows)


if __name__ == "__main__":
    main()
