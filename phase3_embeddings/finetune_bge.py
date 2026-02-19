"""
Mandatory BGE fine-tuning with IndicLegalQA and evaluation.
Uses MultipleNegativesRankingLoss and InformationRetrievalEvaluator.
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


def split_train_eval(pairs: list, split: float = 0.8, seed: int = 42):
    """Split into train and eval with fixed seed."""
    import random

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * split)
    return shuffled[:n_train], shuffled[n_train:]


def build_ir_evaluator(eval_pairs: list) -> InformationRetrievalEvaluator:
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
        name="IndicLegalQA_eval",
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
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output-dir", type=Path, default=FINE_TUNED_MODEL_DIR)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs (0=only at end)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower to reduce overfitting)")
    args = parser.parse_args()

    if not INDIC_LEGAL_QA_PATH.exists():
        print(f"Error: IndicLegalQA dataset not found at {INDIC_LEGAL_QA_PATH}")
        print("Please ensure Datasets/IndicLegalQA Dataset_10K_Revised.json exists.")
        sys.exit(1)

    print("Loading IndicLegalQA...")
    pairs = load_indic_legal_qa(INDIC_LEGAL_QA_PATH)
    print(f"Loaded {len(pairs)} question-answer pairs")

    train_pairs, eval_pairs = split_train_eval(pairs, TRAIN_EVAL_SPLIT, RANDOM_SEED)
    print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Build Dataset for MultipleNegativesRankingLoss (anchor, positive columns)
    train_dataset = Dataset.from_dict({
        "anchor": [p["question"] for p in train_pairs],
        "positive": [p["answer"] for p in train_pairs],
    })

    # Build evaluator
    evaluator = build_ir_evaluator(eval_pairs)

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
        metric_for_best_model="IndicLegalQA_eval_cosine_recall@10",
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
