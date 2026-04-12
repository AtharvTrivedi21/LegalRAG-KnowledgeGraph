"""
Optional: one query through systems 1,2,3 (bns_comparison) and sys4 — without changing
bns_comparison defaults or files.

Usage (from project root):
  pip install -r sys4/requirements.txt
  python -m sys4.compare_with_baselines --query "..."
"""
from __future__ import annotations

import argparse
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


def _load_adapter(sid: int):
    if sid == 1:
        from bns_comparison.adapters.old_work import OldWorkAdapter
        return OldWorkAdapter()
    if sid == 2:
        from bns_comparison.adapters.simple_bns import SimpleBNSAdapter
        return SimpleBNSAdapter()
    if sid == 3:
        from bns_comparison.adapters.full_pipeline_bns import FullPipelineBNSAdapter
        return FullPipelineBNSAdapter()
    if sid == 4:
        from sys4.lqrag_adapter import LQRAGAdapter
        return LQRAGAdapter()
    raise ValueError(f"Unknown system_id: {sid}")


def _run(sid: int, query: str, gold: Dict) -> Optional[Dict]:
    print(f"\n{'='*65}\n  System {sid}: Loading...")
    try:
        adapter = _load_adapter(sid)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return None
    print(f"  {adapter.system_name} — running...")
    try:
        result = adapter.answer_query(query)
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return None
    finally:
        del adapter
        gc.collect()
    metrics = compute_metrics(gold, result)
    return {"result": result, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baselines 1–3 with sys4")
    parser.add_argument("--query", "-q", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--systems", "-s", type=str, default="1,2,3,4")
    args = parser.parse_args()
    sids = [int(x.strip()) for x in args.systems.split(",") if x.strip()]
    query = args.query.strip()
    gold = dict(DEFAULT_GOLD)
    gold["description"] = query

    print(f"\nQuery: {query}\nSystems: {sids}\n")

    results: Dict[int, Optional[Dict]] = {}
    for sid in sids:
        results[sid] = _run(sid, query, gold)
        if sid != sids[-1]:
            print("\n  [Pause 5s...]")
            time.sleep(5)

    for sid in sids:
        data = results[sid]
        sep = "=" * 65
        if data is None:
            print(f"\n{sep}\n  SYSTEM {sid}: NOT AVAILABLE\n{sep}")
            continue
        r, m = data["result"], data["metrics"]
        print(f"\n{sep}\n  SYSTEM {sid}: {r['system_name']}\n{sep}")
        print(f"  section_f1={m['section_f1']:.3f}  ipc_refs={m['ipc_reference_count']}  "
              f"grounding={m['grounding_score']:.2f}  latency={m['total_latency_sec']:.1f}s")
        print(f"  cited={r.get('cited_sections')}  gold={gold['expected_bns_sections']}")


if __name__ == "__main__":
    main()
