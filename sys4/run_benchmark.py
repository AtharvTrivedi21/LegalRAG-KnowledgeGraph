"""
Run the 10 bns_comparison test cases against sys4 only; write CSV under sys4/results/.

Usage (from project root):
  python -m sys4.run_benchmark
  python -m sys4.run_benchmark --cases 1,2,3
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bns_comparison.metrics import compute_metrics
from bns_comparison.test_cases import TEST_CASES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_CSV = RESULTS_DIR / "comparison_sys4.csv"

CSV_FIELDNAMES = [
    "system_name", "case_id", "case_description", "offense_category",
    "rephrased_query", "cited_sections", "gold_sections",
    "section_precision", "section_recall", "section_f1", "correct_act_cited",
    "ipc_reference_count", "fabricated_section_count", "grounding_score", "hallucination_flag",
    "rephrase_latency_sec", "retrieval_latency_sec", "generation_latency_sec", "total_latency_sec",
    "offense_category_hit", "offense_keyword_coverage", "completeness_score",
    "key_issue_coverage", "answer_relevance_score", "context_relevance_score",
    "answer_length_words", "has_safety_disclaimer",
]


def _build_row(system_name: str, case: Dict, result: Dict, metrics: Dict) -> Dict:
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


def _run(adapter, case: Dict) -> Optional[Dict]:
    try:
        return adapter.answer_query(case["description"])
    except Exception as e:
        print(f"    [ERROR] case {case['id']}: {e}")
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sys4 on TEST_CASES")
    parser.add_argument("--cases", type=str, default="", help="Comma case ids (default: all)")
    args = parser.parse_args()
    case_ids = (
        {int(c.strip()) for c in args.cases.split(",") if c.strip()}
        if args.cases
        else None
    )
    cases = [c for c in TEST_CASES if case_ids is None or c["id"] in case_ids]

    from sys4.lqrag_adapter import LQRAGAdapter

    print(f"[sys4] Cases: {[c['id'] for c in cases]}")
    adapter = LQRAGAdapter()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for case in cases:
            print(f"\n  [Case {case['id']}] {case['description'][:70]}...")
            result = _run(adapter, case)
            if result is None:
                err = {
                    "system_name": adapter.system_name,
                    "case_id": case["id"],
                    "case_description": case["description"],
                    "offense_category": case.get("offense_category", ""),
                    "rephrased_query": "ERROR",
                    "cited_sections": "",
                    "gold_sections": "|".join(case.get("expected_bns_sections", [])),
                }
                for fn in CSV_FIELDNAMES:
                    err.setdefault(fn, 0)
                w.writerow(err)
                rows.append(err)
                f.flush()
                continue
            m = compute_metrics(case, result)
            row = _build_row(adapter.system_name, case, result, m)
            w.writerow(row)
            rows.append(row)
            f.flush()
            print(
                f"    sec_f1={m['section_f1']:.2f}  latency={m['total_latency_sec']:.1f}s  "
                f"cited={result.get('cited_sections', [])}"
            )

    del adapter
    gc.collect()
    print(f"\n[Done] Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
