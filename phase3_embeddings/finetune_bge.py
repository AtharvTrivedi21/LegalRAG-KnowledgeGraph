"""
BGE fine-tuning with IndicLegalQA and/or BNS synthetic pairs.
Uses MultipleNegativesRankingLoss and InformationRetrievalEvaluator.

Supports dataset modes:
  --dataset indiclegal    : original IndicLegalQA only (default)
  --dataset bns_synthetic : BNS synthetic pairs only (Exp 7)
  --dataset combined      : both datasets merged (Exp 7B)
  --dataset bns_mapping   : IPC->BNS mapping-derived retrieval pairs
  --dataset complaint_bns : complaint-style BNS pairs
  --dataset complaint_bns_hardneg : complaint triplets with hard negatives
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
import torch

# Compatibility shim for transformers/accelerate versions where Trainer calls
# optimizer.train()/optimizer.eval(), but torch.optim optimizers do not expose
# those methods.
if not hasattr(torch.optim.Optimizer, "train"):
    setattr(torch.optim.Optimizer, "train", lambda self: None)
if not hasattr(torch.optim.Optimizer, "eval"):
    setattr(torch.optim.Optimizer, "eval", lambda self: None)

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_embeddings.config import (
    ALLOW_SAVE_BELOW_THRESHOLD,
    BGE_MODEL,
    FINE_TUNED_MODEL_DIR,
    INDIC_LEGAL_QA_PATH,
    METRICS_SATISFACTORY,
    RANDOM_SEED,
    TRAIN_EVAL_SPLIT,
)

BNS_SYNTHETIC_PATH = Path(__file__).resolve().parent / "bns_synthetic_pairs.jsonl"
BNS_MAPPING_PATH = Path(__file__).resolve().parent / "bns_mapping_pipeline" / "bns_mapping_pairs.jsonl"
BNS_GROQ_SYNTHETIC_PATH = Path(__file__).resolve().parent / "bns_groq_synthetic_pairs.jsonl"
COMPLAINT_BNS_PAIRS_PATH = (
    Path(__file__).resolve().parent / "dataset_experiments_v2" / "datasets" / "complaint_bns_pairs.jsonl"
)
COMPLAINT_BNS_TRIPLETS_PATH = (
    Path(__file__).resolve().parent / "dataset_experiments_v2" / "datasets" / "complaint_bns_hardneg_triplets.jsonl"
)


def load_indic_legal_qa(path: Path) -> list[dict]:
    """Load IndicLegalQA JSON and return list of {question, answer}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


def load_bns_synthetic(path: Path) -> list[dict]:
    """Load BNS synthetic JSONL and return list of {question, answer}."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query", "").strip()
            a = obj.get("positive", "").strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


def load_bns_mapping(path: Path) -> list[dict]:
    """Load IPC->BNS mapping JSONL and return list of {question, answer}."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query", "").strip()
            a = obj.get("positive", "").strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


def load_triplets(path: Path) -> list[dict]:
    """Load JSONL triplets and return list of {question, answer, negative}."""
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query", "").strip()
            a = obj.get("positive", "").strip()
            n = obj.get("negative", "").strip()
            if q and a and n:
                triples.append({"question": q, "answer": a, "negative": n})
    return triples


def split_train_eval(pairs: list, split: float = 0.8, seed: int = 42):
    """Split into train and eval with fixed seed."""
    import random

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * split)
    return shuffled[:n_train], shuffled[n_train:]


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0].strip()


def build_ir_evaluator(eval_pairs: list, name: str = "eval") -> InformationRetrievalEvaluator:
    """Build InformationRetrievalEvaluator from eval (question, answer) pairs."""
    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, pair in enumerate(eval_pairs):
        qid = f"q_{i}"
        doc_id = f"d_{i}"
        queries[qid] = pair["question"]
        corpus[doc_id] = pair["answer"]
        relevant_docs[qid] = {doc_id}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[10],
        show_progress_bar=True,
    )


def evaluate_model(model: SentenceTransformer, evaluator: InformationRetrievalEvaluator) -> dict:
    """Run evaluation and return metrics dict."""
    return evaluator(model)


def _find_metric(metrics: dict, pattern: str) -> Optional[float]:
    """Find first metric key matching pattern (e.g. 'mrr', '10')."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and pattern.lower() in k.lower() and "@10" in k:
            return float(v)
    return None


def check_metrics_satisfactory(metrics: dict):
    """Check if metrics meet satisfactory thresholds. Returns (passed, list of failures)."""
    failures = []

    checks = [
        ("mrr@10", "mrr", METRICS_SATISFACTORY["mrr@10"]),
        ("ndcg@10", "ndcg", METRICS_SATISFACTORY["ndcg@10"]),
        ("recall@10", "recall", METRICS_SATISFACTORY["recall@10"]),
    ]

    for name, pattern, threshold in checks:
        val = _find_metric(metrics, pattern)
        if val is None:
            failures.append(f"{name}: metric not found in results")
        elif val < threshold:
            failures.append(f"{name}: {val:.4f} < {threshold} (threshold)")

    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE on IndicLegalQA with evaluation")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (reduced to avoid overfitting)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--output-dir", type=Path, default=FINE_TUNED_MODEL_DIR)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs (0=only at end)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower to reduce overfitting)")
    parser.add_argument(
        "--dataset",
        choices=[
            "indiclegal",
            "bns_synthetic",
            "combined",
            "bns_mapping",
            "bns_groq_synthetic",
            "combined_groq",
            "complaint_bns",
            "complaint_bns_hardneg",
        ],
        default="indiclegal",
        help="Training dataset mode",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available in this environment.")
        print("Please install CUDA-enabled PyTorch in this virtual environment.")
        sys.exit(1)
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    pairs = []
    triplets = []

    if args.dataset in ("indiclegal", "combined"):
        if not INDIC_LEGAL_QA_PATH.exists():
            print(f"Error: IndicLegalQA dataset not found at {INDIC_LEGAL_QA_PATH}")
            sys.exit(1)
        indic_pairs = load_indic_legal_qa(INDIC_LEGAL_QA_PATH)
        print(f"Loaded {len(indic_pairs)} IndicLegalQA pairs")
        pairs.extend(indic_pairs)

    if args.dataset in ("bns_synthetic", "combined"):
        if not BNS_SYNTHETIC_PATH.exists():
            print(f"Error: BNS synthetic pairs not found at {BNS_SYNTHETIC_PATH}")
            print("Run: python -m phase3_embeddings.build_synthetic_jsonl")
            sys.exit(1)
        bns_pairs = load_bns_synthetic(BNS_SYNTHETIC_PATH)
        print(f"Loaded {len(bns_pairs)} BNS synthetic pairs")
        pairs.extend(bns_pairs)

    if args.dataset == "bns_mapping":
        if not BNS_MAPPING_PATH.exists():
            print(f"Error: BNS mapping pairs not found at {BNS_MAPPING_PATH}")
            print("Run: python -m phase3_embeddings.bns_mapping_pipeline.build_mapping_pairs")
            sys.exit(1)
        map_pairs = load_bns_mapping(BNS_MAPPING_PATH)
        print(f"Loaded {len(map_pairs)} BNS mapping pairs")
        pairs.extend(map_pairs)

    if args.dataset in ("bns_groq_synthetic", "combined_groq"):
        if not BNS_GROQ_SYNTHETIC_PATH.exists():
            print(f"Error: BNS Groq synthetic pairs not found at {BNS_GROQ_SYNTHETIC_PATH}")
            print("Run: python -m phase3_embeddings.generate_groq_synthetic")
            sys.exit(1)
        groq_pairs = load_bns_synthetic(BNS_GROQ_SYNTHETIC_PATH)
        print(f"Loaded {len(groq_pairs)} BNS Groq synthetic pairs")
        pairs.extend(groq_pairs)

    if args.dataset == "combined_groq":
        if not INDIC_LEGAL_QA_PATH.exists():
            print(f"Error: IndicLegalQA dataset not found at {INDIC_LEGAL_QA_PATH}")
            sys.exit(1)
        indic_pairs = load_indic_legal_qa(INDIC_LEGAL_QA_PATH)
        print(f"Loaded {len(indic_pairs)} IndicLegalQA pairs")
        pairs.extend(indic_pairs)

    if args.dataset == "complaint_bns":
        if not COMPLAINT_BNS_PAIRS_PATH.exists():
            print(f"Error: complaint pairs not found at {COMPLAINT_BNS_PAIRS_PATH}")
            print("Run: python -m phase3_embeddings.dataset_experiments_v2.build_complaint_training_data")
            sys.exit(1)
        complaint_pairs = load_bns_synthetic(COMPLAINT_BNS_PAIRS_PATH)
        print(f"Loaded {len(complaint_pairs)} complaint BNS pairs")
        pairs.extend(complaint_pairs)

    if args.dataset == "complaint_bns_hardneg":
        if not COMPLAINT_BNS_TRIPLETS_PATH.exists():
            print(f"Error: complaint triplets not found at {COMPLAINT_BNS_TRIPLETS_PATH}")
            print("Run: python -m phase3_embeddings.dataset_experiments_v2.build_complaint_training_data")
            sys.exit(1)
        triplets = load_triplets(COMPLAINT_BNS_TRIPLETS_PATH)
        print(f"Loaded {len(triplets)} complaint hard-negative triplets")

    # Complaint narratives are very long; trim them for stability on Windows CPU training.
    if args.dataset in ("complaint_bns", "complaint_bns_hardneg"):
        for p in pairs:
            p["question"] = _trim_text(p["question"], 600)
            p["answer"] = _trim_text(p["answer"], 1800)
        for t in triplets:
            t["question"] = _trim_text(t["question"], 600)
            t["answer"] = _trim_text(t["answer"], 1800)
            t["negative"] = _trim_text(t["negative"], 1800)

    if args.dataset == "complaint_bns_hardneg":
        print(f"Total training triplets: {len(triplets)}")
    else:
        print(f"Total training pairs: {len(pairs)}")

    if args.dataset == "complaint_bns_hardneg":
        train_pairs, eval_pairs = split_train_eval(triplets, TRAIN_EVAL_SPLIT, RANDOM_SEED)
    else:
        train_pairs, eval_pairs = split_train_eval(pairs, TRAIN_EVAL_SPLIT, RANDOM_SEED)
    print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Build train dataset
    if args.dataset == "complaint_bns_hardneg":
        train_dataset = Dataset.from_dict({
            "anchor": [p["question"] for p in train_pairs],
            "positive": [p["answer"] for p in train_pairs],
            "negative": [p["negative"] for p in train_pairs],
        })
        eval_for_ir = [{"question": p["question"], "answer": p["answer"]} for p in eval_pairs]
    else:
        train_dataset = Dataset.from_dict({
            "anchor": [p["question"] for p in train_pairs],
            "positive": [p["answer"] for p in train_pairs],
        })
        eval_for_ir = eval_pairs

    # Build evaluator with dataset-specific name
    eval_name = f"{args.dataset}_eval"
    evaluator = build_ir_evaluator(eval_for_ir, name=eval_name)

    # Load base model
    print(f"Loading base model: {BGE_MODEL}")
    model = SentenceTransformer(BGE_MODEL)

    # Baseline evaluation
    print("\n--- Baseline (unfinetuned) evaluation ---")
    baseline_metrics = evaluate_model(model, evaluator)
    for k, v in sorted(baseline_metrics.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print()

    # Loss
    if args.dataset == "complaint_bns_hardneg":
        train_loss = losses.TripletLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Training arguments
    # Use checkpoint_dir for intermediate checkpoints; final model saved to output_dir
    checkpoint_dir = args.output_dir.parent / (args.output_dir.name + "_checkpoints")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        eval_strategy="epoch" if args.eval_every else "no",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Use best eval checkpoint instead of final (avoids overfit last epoch)
        load_best_model_at_end=True,
        metric_for_best_model=f"{eval_name}_cosine_recall@10",
        greater_is_better=True,
    )

    # Trainer and train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    # Final evaluation (model is already the best checkpoint if load_best_model_at_end=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Final (fine-tuned) evaluation ---")
    final_metrics = evaluate_model(model, evaluator)
    for k, v in sorted(final_metrics.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    print()

    # Check thresholds
    passed, failures = check_metrics_satisfactory(final_metrics)
    if passed:
        print("All metrics meet satisfactory thresholds.")
    else:
        print("WARNING: Some metrics below threshold:")
        for f in failures:
            print(f"  - {f}")
        if not ALLOW_SAVE_BELOW_THRESHOLD:
            print("\nModel was saved regardless (fit() saves automatically).")
            print("Set ALLOW_SAVE_BELOW_THRESHOLD=True in config to always save.")
        else:
            print("\nModel saved (ALLOW_SAVE_BELOW_THRESHOLD=True).")

    # Compare to baseline
    print("\n--- Comparison to baseline ---")
    baseline_recall = _find_metric(baseline_metrics, "recall")
    final_recall = _find_metric(final_metrics, "recall")
    use_baseline_fallback = False
    if baseline_recall is not None and final_recall is not None and final_recall < baseline_recall:
        use_baseline_fallback = True
        print("  Fine-tuned model is WORSE than baseline on Recall@10. Using baseline for save.")

    for pattern, label in [("mrr", "MRR@10"), ("ndcg", "NDCG@10"), ("recall", "Recall@10")]:
        b = _find_metric(baseline_metrics, pattern)
        f = _find_metric(final_metrics, pattern)
        if b is not None and f is not None:
            diff = f - b
            status = "OK" if diff >= 0 else "WORSE"
            print(f"  {label}: baseline={b:.4f} -> final={f:.4f} (delta={diff:+.4f}) [{status}]")

    # Save model for build_faiss: use baseline if fine-tuned regressed
    if use_baseline_fallback:
        print("\nLoading baseline model for save (fine-tuned regressed)...")
        model = SentenceTransformer(BGE_MODEL)
    model.save(str(args.output_dir))
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
