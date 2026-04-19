"""
Configuration for the evaluation pipeline.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# Raw System 3 outputs (one JSON object per line, crash-safe)
SYSTEM3_RAW_RESULTS = RESULTS_DIR / "system3_raw_results.jsonl"

# Rule-based metrics CSV (computed from raw results)
SYSTEM3_METRICS_CSV = RESULTS_DIR / "system3_results_100.csv"

# GPT-4.1 evaluation outputs
GPT4_EVAL_RESULTS = RESULTS_DIR / "gpt4_eval_results.csv"
GPT4_EVAL_LOG = RESULTS_DIR / "gpt4_eval_log.jsonl"

# Final merged evaluation
FINAL_EVAL_CSV = RESULTS_DIR / "final_evaluation.csv"

# OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_MODEL = "gpt-4.1"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500

# Context truncation for judge (chars) — keeps cost down
MAX_CONTEXT_CHARS = 8000
