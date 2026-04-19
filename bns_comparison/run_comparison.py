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

Note: Each system is run in isolation (one at a time) to avoid GPU OOM when
      SentenceTransformer and Ollama llama3:8b compete for GPU memory.
"""
import argparse
import csv
import gc
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.test_cases import TEST_CASES
from bns_comparison.metrics import compute_metrics, summarize_metrics
from bns_comparison.config import RESULTS_DIR, COMPARISON_CSV


def _load_single_adapter(system_id: int):
    """Import and instantiate a single adapter (one at a time to avoid GPU OOM)."""
    if system_id == 1:
        from bns_comparison.adapters.old_work import OldWorkAdapter
        return OldWorkAdapter()
    elif system_id == 2:
        from bns_comparison.adapters.simple_bns import SimpleBNSAdapter
        return SimpleBNSAdapter()
    elif system_id == 3:
        from bns_comparison.adapters.full_pipeline_bns import FullPipelineBNSAdapter
        return FullPipelineBNSAdapter()
    raise ValueError(f"Unknown system_id: {system_id}")


def _load_adapters(system_ids: List[int]):
    """Load all adapters (kept for backward compatibility)."""
    adapters = {}
    for sid in system_ids:
        try:
            adapters[sid] = _load_single_adapter(sid)
            print(f"[Init] System {sid} loaded.")
        except Exception as e:
            print(f"[WARN] System {sid} failed to load: {e}")
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


def _print_summary_table(all_rows: List[Dict], case_count: int) -> None:
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
    print(f"COMPARISON SUMMARY (averages across {case_count} test cases)")
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
    t_run_start = time.time()
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for sys_id in sorted(system_ids):
            # Load one system at a time to avoid GPU OOM
            print(f"\n{'='*60}")
            print(f"Loading System {sys_id}...")
            t_sys_start = time.time()
            t_load_start = time.time()
            try:
                adapter = _load_single_adapter(sys_id)
            except Exception as e:
                print(f"[WARN] System {sys_id} failed to load: {e}")
                continue
            load_sec = time.time() - t_load_start

            print(f"Running {adapter.system_name} on {len(cases)} cases...")
            print(f"Load time: {load_sec:.2f}s")
            print(f"{'='*60}")

            for case in cases:
                print(f"\n  [Case {case['id']}] {case['description'][:70]}...")
                t_case_start = time.time()
                result = _run_one(adapter, case)

                if result is None:
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
                    f.flush()
                    all_rows.append(error_row)
                    case_wall = time.time() - t_case_start
                    print(f"    case_wall_time={case_wall:.2f}s (failed)")
                    continue

                metrics = compute_metrics(case, result)
                row = _build_csv_row(adapter.system_name, case, result, metrics)
                writer.writerow(row)
                f.flush()
                all_rows.append(row)
                case_wall = time.time() - t_case_start

                print(
                    f"    sec_f1={metrics['section_f1']:.2f}  "
                    f"correct_act={metrics['correct_act_cited']}  "
                    f"ipc_refs={metrics['ipc_reference_count']}  "
                    f"halluc={metrics['hallucination_flag']}  "
                    f"latency={metrics['total_latency_sec']:.1f}s"
                )
                print(
                    f"    timing_breakdown: load(reused) + "
                    f"rephrase={metrics['rephrase_latency_sec']:.2f}s, "
                    f"retrieval={metrics['retrieval_latency_sec']:.2f}s, "
                    f"generation={metrics['generation_latency_sec']:.2f}s, "
                    f"case_total={metrics['total_latency_sec']:.2f}s, "
                    f"case_wall={case_wall:.2f}s"
                )
                print(
                    f"    cited={result.get('cited_sections', [])}  "
                    f"gold={case['expected_bns_sections']}"
                )

            # Explicitly unload the adapter to free memory before loading the next system
            del adapter
            gc.collect()
            sys_total = time.time() - t_sys_start
            print(f"\n[System {sys_id}] Done. Memory released.")
            print(f"[System {sys_id}] total_time={sys_total:.2f}s (including load + all cases)")
            time.sleep(5)  # Brief pause to let Ollama settle

    print(f"\n[Done] Results written to: {COMPARISON_CSV}")
    run_total = time.time() - t_run_start
    print(f"[Done] End-to-end total_time={run_total:.2f}s")
    _print_summary_table(all_rows, case_count=len(cases))


if __name__ == "__main__":
    main()
