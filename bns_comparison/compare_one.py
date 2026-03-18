"""
Single-query BNS RAG comparison.

Runs one query through all three systems and prints a side-by-side comparison.
Much faster than run_comparison.py (which runs 10 cases × 3 systems).

Usage (from project root, with venv active):
    python -m bns_comparison.compare_one
    python -m bns_comparison.compare_one --query "Someone stole my phone"
    python -m bns_comparison.compare_one --systems 1,2   # skip System 3

Prerequisites:
    python -m bns_comparison.build_bns_faiss --system both
"""
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bns_comparison.metrics import compute_metrics
from bns_comparison.config import RESULTS_DIR

DEFAULT_QUERY = "Someone broke into my home and stole my property, also broke my windows and door."

# Minimal gold annotations for the default query (housebreaking + theft)
DEFAULT_GOLD = {
    "id": 0,
    "description": DEFAULT_QUERY,
    "offense_category": "housebreaking and theft",
    "expected_bns_sections": ["330", "331", "333", "334", "303"],
    "offense_keywords": ["housebreaking", "house-breaking", "trespass", "theft", "burglary"],
    "key_issues": [
        "entry into dwelling house",
        "without permission",
        "stealing property",
        "dishonest intention",
        "damage to property",
    ],
}


def _load_adapter(system_id: int):
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


def _run_system(system_id: int, query: str, gold: Dict) -> Optional[Dict]:
    """Load one system, run the query, compute metrics, unload."""
    print(f"\n{'='*65}")
    print(f"  System {system_id}: Loading...")
    try:
        adapter = _load_adapter(system_id)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to load System {system_id}: {e}")
        return None

    print(f"  System {system_id}: {adapter.system_name} — running query...")
    try:
        result = adapter.answer_query(query)
    except Exception as e:
        print(f"  [ERROR] System {system_id} query failed: {e}")
        traceback.print_exc()
        return None
    finally:
        del adapter
        gc.collect()

    metrics = compute_metrics(gold, result)
    return {"result": result, "metrics": metrics}


def _print_result(system_id: int, data: Optional[Dict]) -> None:
    sep = "=" * 65
    if data is None:
        print(f"\n{sep}")
        print(f"  SYSTEM {system_id}: NOT AVAILABLE")
        print(sep)
        return

    r = data["result"]
    m = data["metrics"]

    print(f"\n{sep}")
    print(f"  SYSTEM {system_id}: {r['system_name']}")
    print(sep)
    print(f"  Rephrased query:\n    {r['rephrased_query']}")
    print(f"\n  Answer:\n{r['answer']}")
    print(f"\n  Cited sections : {r.get('cited_sections', [])}")
    print(f"  Gold sections  : {data.get('gold_sections', [])}")
    print(f"\n  --- Metrics ---")
    print(f"  section_f1         : {m['section_f1']:.3f}  "
          f"(precision={m['section_precision']:.2f}, recall={m['section_recall']:.2f})")
    print(f"  correct_act_cited  : {m['correct_act_cited']}  "
          f"(BNS cited, no IPC)")
    print(f"  ipc_reference_count: {m['ipc_reference_count']}")
    print(f"  hallucination_flag : {m['hallucination_flag']}  "
          f"(grounding={m['grounding_score']:.2f})")
    print(f"  offense_kw_coverage: {m['offense_keyword_coverage']:.2f}")
    print(f"  completeness_score : {m['completeness_score']:.2f}")
    print(f"  answer_length_words: {m['answer_length_words']}")
    print(f"  total_latency_sec  : {m['total_latency_sec']:.1f}s  "
          f"(rephrase={m['rephrase_latency_sec']:.1f}s, "
          f"retrieval={m['retrieval_latency_sec']:.1f}s, "
          f"generation={m['generation_latency_sec']:.1f}s)")


def _print_summary(results: Dict[int, Optional[Dict]]) -> None:
    print(f"\n{'='*65}")
    print("  SUMMARY TABLE")
    print(f"{'='*65}")
    headers = ["System", "sec_f1", "correct_act", "ipc_refs", "halluc", "latency_s"]
    col_w = 13
    print("  " + "".join(f"{h:<{col_w}}" for h in headers))
    print("  " + "-" * (col_w * len(headers)))
    for sid, data in sorted(results.items()):
        if data is None:
            row = [f"Sys{sid}", "N/A", "N/A", "N/A", "N/A", "N/A"]
        else:
            m = data["metrics"]
            row = [
                f"Sys{sid}",
                f"{m['section_f1']:.3f}",
                str(m['correct_act_cited']),
                str(m['ipc_reference_count']),
                str(m['hallucination_flag']),
                f"{m['total_latency_sec']:.1f}",
            ]
        print("  " + "".join(f"{v:<{col_w}}" for v in row))
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Run a single query through all BNS RAG systems")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=DEFAULT_QUERY,
        help="The incident description or legal query to test",
    )
    parser.add_argument(
        "--systems", "-s",
        type=str,
        default="1,2,3",
        help="Comma-separated system IDs to run (default: 1,2,3)",
    )
    args = parser.parse_args()

    system_ids = [int(s.strip()) for s in args.systems.split(",") if s.strip()]
    query = args.query.strip()

    # Build gold annotations for the query
    gold = dict(DEFAULT_GOLD)
    gold["description"] = query

    print(f"\nQuery: {query}")
    print(f"Systems: {system_ids}")

    all_results: Dict[int, Optional[Dict]] = {}

    for sid in system_ids:
        data = _run_system(sid, query, gold)
        if data:
            data["gold_sections"] = gold["expected_bns_sections"]
        all_results[sid] = data
        # Brief pause between systems to let Ollama release GPU memory
        if sid != system_ids[-1]:
            print(f"\n  [Pause 5s before next system...]")
            time.sleep(5)

    # Print full results
    for sid in system_ids:
        _print_result(sid, all_results[sid])

    # Print summary table
    _print_summary(all_results)


if __name__ == "__main__":
    main()
