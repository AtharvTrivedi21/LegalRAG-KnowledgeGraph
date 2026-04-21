"""
Run System 3 with a strictly non-finetuned embedding model.

This script creates a separate FAISS index + metadata using a base
SentenceTransformer model (default: BAAI/bge-small-en-v1.5), then runs the
same System 3 pipeline on test cases with isolated output files.

It does NOT overwrite the default finetuned index/results.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from sentence_transformers import SentenceTransformer

from bns_comparison.metrics import compute_metrics
from bns_comparison.test_cases import TEST_CASES
import bns_comparison.config as bns_config
import bns_comparison.adapters.full_pipeline_bns as sys3_module
from bns_comparison.adapters.full_pipeline_bns import FullPipelineBNSAdapter
from phase3_embeddings.chunk_corpus import chunk_text, get_tokenizer

CSV_FIELDNAMES = [
    "case_id",
    "case_description",
    "offense_category",
    "rephrased_query",
    "cited_sections",
    "gold_sections",
    "hit_rate",
    "mrr",
    "section_precision",
    "section_recall",
    "section_f1",
    "correct_act_cited",
    "ipc_reference_count",
    "fabricated_section_count",
    "grounding_score",
    "hallucination_flag",
    "rephrase_latency_sec",
    "retrieval_latency_sec",
    "generation_latency_sec",
    "total_latency_sec",
    "offense_category_hit",
    "offense_keyword_coverage",
    "completeness_score",
    "key_issue_coverage",
    "answer_relevance_score",
    "context_relevance_score",
    "answer_length_words",
    "has_safety_disclaimer",
]


def _load_latest_results(raw_results_path: Path) -> dict:
    """Return latest saved result per case_id from JSONL."""
    latest = {}
    if raw_results_path.exists():
        with open(raw_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    latest[int(obj["case_id"])] = obj.get("result", {}) or {}
                except (json.JSONDecodeError, KeyError):
                    pass
    return latest


def _is_valid_result(result: dict) -> bool:
    """A result is valid if core generated fields are present and non-empty."""
    if not isinstance(result, dict):
        return False
    answer = (result.get("answer") or "").strip()
    context = (result.get("context_text") or "").strip()
    chunks = result.get("retrieved_chunks") or []
    rephrased = (result.get("rephrased_query") or "").strip()
    if rephrased == "ERROR":
        return False
    return bool(answer) and bool(context) and bool(chunks)


def _export_metrics_csv(raw_results_path: Path, metrics_csv_path: Path) -> None:
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
            rows.append(
                {
                    "case_id": case["id"],
                    "case_description": case["description"],
                    "offense_category": case.get("offense_category", ""),
                    "rephrased_query": obj["result"].get("rephrased_query", ""),
                    "cited_sections": "|".join(obj["result"].get("cited_sections", [])),
                    "gold_sections": "|".join(case.get("expected_bns_sections", [])),
                    **metrics,
                }
            )

    rows.sort(key=lambda r: r["case_id"])
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Export] Metrics CSV written: {metrics_csv_path} ({len(rows)} rows)")


def _build_nonft_index(base_model: str, sections_csv: Path, index_path: Path, metadata_path: Path) -> None:
    if index_path.exists() and metadata_path.exists():
        print(f"[Index] Reusing existing non-FT index: {index_path}")
        return

    if not sections_csv.exists():
        raise FileNotFoundError(f"sections.csv not found: {sections_csv}")

    print(f"[Index] Building non-FT FAISS index with base model: {base_model}")
    df = pd.read_csv(sections_csv)
    bns_df = df[df["act_id"] == "BNS_2023"].copy()
    tokenizer = get_tokenizer()

    all_chunks = []
    for _, row in bns_df.iterrows():
        section_id = str(row["section_id"])
        act_id = str(row.get("act_id", "BNS_2023"))
        text = str(row.get("full_text", "")).strip()
        if not text:
            continue
        all_chunks.extend(chunk_text(text, tokenizer, section_id, "section", act_id=act_id))

    seen = set()
    for c in all_chunks:
        base = c["chunk_id"]
        idx = 0
        while c["chunk_id"] in seen:
            c["chunk_id"] = f"{base}_{idx}"
            idx += 1
        seen.add(c["chunk_id"])

    model = SentenceTransformer(base_model, device="cuda")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(
            [
                {
                    "chunk_id": c["chunk_id"],
                    "source_type": c["source_type"],
                    "source_id": c["source_id"],
                    "act_id": c.get("act_id", "BNS_2023"),
                    "text": c["text"],
                }
                for c in all_chunks
            ],
            f,
        )
    print(f"[Index] Built vectors={idx.ntotal} -> {index_path}")


def _patch_system3_paths(index_path: Path, metadata_path: Path, model_name_or_path: str) -> None:
    # Patch config module values
    bns_config.BNS_FAISS_INDEX_PATH = index_path
    bns_config.BNS_CHUNK_METADATA_PATH = metadata_path
    bns_config.FINE_TUNED_MODEL_DIR = model_name_or_path
    # Patch already-imported adapter module globals
    sys3_module.BNS_FAISS_INDEX_PATH = index_path
    sys3_module.BNS_CHUNK_METADATA_PATH = metadata_path
    sys3_module.FINE_TUNED_MODEL_DIR = model_name_or_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run System 3 with non-finetuned base BGE model")
    parser.add_argument("--start", type=int, default=1, help="Start case ID (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End case ID (inclusive)")
    parser.add_argument(
        "--base-model",
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformer model id/path for non-finetuned baseline",
    )
    parser.add_argument(
        "--sections-csv",
        type=Path,
        default=bns_config.SECTIONS_CSV,
        help="Path to sections CSV (default: bns_comparison.config.SECTIONS_CSV)",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("bns_comparison/faiss_bns_only_nonft/faiss.index"),
        help="Output FAISS index path for non-FT run",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("bns_comparison/faiss_bns_only_nonft/chunk_metadata.pkl"),
        help="Output metadata path for non-FT run",
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path("evaluation/results/system3_raw_results_nonft_bge.jsonl"),
        help="Output JSONL path (non-FT run)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("evaluation/results/system3_results_100_nonft_bge.csv"),
        help="Output metrics CSV path (non-FT run)",
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="Re-run cases whose latest saved result is empty/error in the same JSONL",
    )
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is configured to run on GPU only.")

    root = Path(__file__).resolve().parent.parent
    index_path = (root / args.index_path).resolve() if not args.index_path.is_absolute() else args.index_path
    metadata_path = (root / args.metadata_path).resolve() if not args.metadata_path.is_absolute() else args.metadata_path
    raw_results = (root / args.raw_results).resolve() if not args.raw_results.is_absolute() else args.raw_results
    metrics_csv = (root / args.metrics_csv).resolve() if not args.metrics_csv.is_absolute() else args.metrics_csv
    sections_csv = (root / args.sections_csv).resolve() if not args.sections_csv.is_absolute() else args.sections_csv

    _build_nonft_index(args.base_model, sections_csv, index_path, metadata_path)
    _patch_system3_paths(index_path, metadata_path, args.base_model)
    os.environ["SYS3_EMBED_DEVICE"] = "cuda"

    cases = [c for c in TEST_CASES if args.start <= c["id"] <= args.end]
    latest_results = _load_latest_results(raw_results)
    completed_ids = {cid for cid, res in latest_results.items() if _is_valid_result(res)}
    invalid_ids = {cid for cid, res in latest_results.items() if not _is_valid_result(res)}

    if args.retry_empty:
        remaining = [c for c in cases if c["id"] in invalid_ids or c["id"] not in completed_ids]
    else:
        remaining = [c for c in cases if c["id"] not in completed_ids]

    print(
        f"[System3-NonFT] Model={args.base_model} | cases={len(cases)} "
        f"valid_done={len(completed_ids)} invalid_saved={len(invalid_ids)} remaining={len(remaining)}"
    )
    if not remaining:
        _export_metrics_csv(raw_results, metrics_csv)
        return

    raw_results.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    adapter = FullPipelineBNSAdapter()
    run_start = time.time()
    wall_times = []
    total_target = len(completed_ids) + len(remaining)

    for i, case in enumerate(remaining):
        done_so_far = len(completed_ids) + i + 1
        case_id = case["id"]
        eta_str = ""
        if wall_times:
            avg_wall = sum(wall_times) / len(wall_times)
            eta_str = f" | ETA: {(avg_wall * (len(remaining) - i))/60:.0f} min"

        print(f"\n{'='*60}")
        print(f"[{done_so_far}/{total_target}] Case {case_id}{eta_str}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = adapter.answer_query(case["description"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = {
                "system_name": "System3_FullPipelineBNS_nonft",
                "rephrased_query": "ERROR",
                "answer": "",
                "retrieved_chunks": [],
                "cited_sections": [],
                "context_text": "",
                "graph_sections": [],
                "timings": {"rephrase_sec": 0, "retrieval_sec": 0, "generation_sec": 0, "total_sec": 0},
            }
        wall = time.time() - t0
        wall_times.append(wall)

        with open(raw_results, "a", encoding="utf-8") as f:
            f.write(json.dumps({"case_id": case_id, "result": result, "wall_time_sec": round(wall, 2)}) + "\n")

        metrics = compute_metrics(case, result)
        cited = result.get("cited_sections", [])
        gold = case.get("expected_bns_sections", [])
        print(
            f"  RESULT: hr={metrics['hit_rate']} mrr={metrics['mrr']:.2f} "
            f"f1={metrics['section_f1']:.2f} grounding={metrics['grounding_score']:.2f}"
        )
        print(f"  TIMING: case={wall:.1f}s total_elapsed={(time.time()-run_start)/60:.1f} min")
        print(f"  CITED:  {cited}")
        print(f"  GOLD:   {gold}")

    total_elapsed = time.time() - run_start
    print(f"\n[System3-NonFT] Complete in {total_elapsed/60:.1f} min")
    _export_metrics_csv(raw_results, metrics_csv)


if __name__ == "__main__":
    main()
