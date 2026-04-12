"""
Single-query test for sys4 only (isolated from bns_comparison.compare_one).

Usage (from project root, venv active):
  pip install -r sys4/requirements.txt
  python -m sys4.run_compare_one
  python -m sys4.run_compare_one --query "Someone stole my phone"
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bns_comparison.metrics import compute_metrics

DEFAULT_QUERY = "Someone broke into my home and stole my property, also broke my windows and door."

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sys4 LQRAG on one query")
    parser.add_argument("--query", "-q", type=str, default=DEFAULT_QUERY)
    args = parser.parse_args()
    query = args.query.strip()
    gold = dict(DEFAULT_GOLD)
    gold["description"] = query

    print(f"\nQuery: {query}")
    print("System: sys4 (LQRAG + graph)\n")

    from sys4.lqrag_adapter import LQRAGAdapter

    try:
        adapter = LQRAGAdapter()
    except Exception as e:
        print(f"[ERROR] Failed to load sys4: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        result = adapter.answer_query(query)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        del adapter
        gc.collect()

    metrics = compute_metrics(gold, result)
    sep = "=" * 65
    print(sep)
    print(f"  {result['system_name']}")
    print(sep)
    print(f"  Rephrased query:\n    {result['rephrased_query']}")
    print(f"\n  Answer:\n{result['answer']}")
    print(f"\n  Cited sections : {result.get('cited_sections', [])}")
    print(f"  Gold sections  : {gold['expected_bns_sections']}")
    print(f"\n  --- Metrics ---")
    print(f"  section_f1         : {metrics['section_f1']:.3f}")
    print(f"  correct_act_cited  : {metrics['correct_act_cited']}")
    print(f"  ipc_reference_count: {metrics['ipc_reference_count']}")
    print(f"  hallucination_flag : {metrics['hallucination_flag']}  (grounding={metrics['grounding_score']:.2f})")
    print(f"  eval_iterations    : {result.get('eval_iterations', 0)}")
    print(f"  total_latency_sec  : {metrics['total_latency_sec']:.1f}s")
    print(sep + "\n")


if __name__ == "__main__":
    main()
