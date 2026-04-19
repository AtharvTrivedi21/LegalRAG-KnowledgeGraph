"""
GPT-4.1 Evaluation: Score System 3 results on Answer Relevance, Context Relevance, Groundedness.

Reads raw results from system3_raw_results.jsonl, calls OpenAI GPT-4.1 API,
and saves scored results incrementally.

Usage:
    python -m evaluation.evaluate_gpt4 [--dry-run] [--limit N]

Prerequisites:
    Set OPENAI_API_KEY environment variable.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.config import (
    SYSTEM3_RAW_RESULTS,
    GPT4_EVAL_RESULTS,
    GPT4_EVAL_LOG,
    RESULTS_DIR,
    OPENAI_API_KEY,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_MAX_TOKENS,
    MAX_CONTEXT_CHARS,
)

RUBRIC_PROMPT = """\
You are an expert legal evaluation agent specializing in Indian criminal law \
(Bharatiya Nyaya Sanhita 2023). Assess the following legal RAG system response \
on three criteria.

QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Score each criterion from 0.0 to 1.0 (use one decimal place):

1. ANSWER_RELEVANCE: Does the answer accurately and completely address the \
legal query? Consider whether the correct legal provisions are identified and \
the advice is actionable.

2. CONTEXT_RELEVANCE: Is the retrieved context pertinent and sufficient for \
answering the query? Are the retrieved BNS sections the right ones for this \
legal scenario?

3. GROUNDEDNESS: Is every factual claim and legal citation in the answer \
supported by the retrieved context? Are all cited BNS section numbers actually \
present in the context? Penalize any hallucinated or fabricated section references.

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{{"answer_relevance": <float>, "context_relevance": <float>, "groundedness": <float>, "justification": "<1-2 sentence explanation>"}}\
"""

EVAL_CSV_FIELDS = [
    "case_id", "answer_relevance", "context_relevance", "groundedness",
    "avg_relevance", "justification",
]


def _load_raw_results():
    """Load all raw results keyed by case_id."""
    results = {}
    if not SYSTEM3_RAW_RESULTS.exists():
        print(f"ERROR: Raw results not found at {SYSTEM3_RAW_RESULTS}")
        print("Run evaluation.run_system3_100 first.")
        sys.exit(1)
    with open(SYSTEM3_RAW_RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results[obj["case_id"]] = obj["result"]
    return results


def _load_completed_eval_ids() -> set:
    """Read already-evaluated case IDs from the eval CSV."""
    completed = set()
    if GPT4_EVAL_RESULTS.exists():
        with open(GPT4_EVAL_RESULTS, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed.add(int(row["case_id"]))
                except (KeyError, ValueError):
                    pass
    return completed


def _call_gpt4(query: str, context: str, answer: str) -> dict:
    """Call OpenAI GPT-4.1 API and return parsed scores."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    truncated_context = context[:MAX_CONTEXT_CHARS]
    prompt = RUBRIC_PROMPT.format(
        query=query,
        context=truncated_context,
        answer=answer,
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    raw_text = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    # Parse JSON from response (handle potential markdown wrapping)
    json_text = raw_text
    if "```" in json_text:
        lines = json_text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or not json_text.startswith("```"):
                json_lines.append(line)
        json_text = "\n".join(json_lines)

    try:
        scores = json.loads(json_text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in the text
        import re
        match = re.search(r'\{[^}]+\}', raw_text, re.DOTALL)
        if match:
            scores = json.loads(match.group())
        else:
            scores = {
                "answer_relevance": -1.0,
                "context_relevance": -1.0,
                "groundedness": -1.0,
                "justification": f"PARSE_ERROR: {raw_text[:200]}",
            }

    return {
        "scores": scores,
        "raw_response": raw_text,
        "usage": usage,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate System 3 results with GPT-4.1")
    parser.add_argument("--dry-run", action="store_true", help="Process only first 2 cases for testing")
    parser.add_argument("--limit", type=int, default=0, help="Max cases to evaluate this run (0=all)")
    args = parser.parse_args()

    raw_results = _load_raw_results()
    completed = _load_completed_eval_ids()
    case_ids = sorted(raw_results.keys())

    remaining = [cid for cid in case_ids if cid not in completed]
    if args.dry_run:
        remaining = remaining[:2]
    elif args.limit > 0:
        remaining = remaining[:args.limit]

    print(f"[GPT-4.1 Eval] Model: {JUDGE_MODEL}")
    print(f"[GPT-4.1 Eval] Total raw results: {len(raw_results)}")
    print(f"[GPT-4.1 Eval] Already evaluated: {len(completed)}")
    print(f"[GPT-4.1 Eval] To evaluate: {len(remaining)}")

    if not remaining:
        print("[GPT-4.1 Eval] All cases already evaluated.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize CSV if needed
    write_header = not GPT4_EVAL_RESULTS.exists() or GPT4_EVAL_RESULTS.stat().st_size == 0
    csv_file = open(GPT4_EVAL_RESULTS, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=EVAL_CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        csv_file.flush()

    total_cost_estimate = 0.0

    try:
        for i, case_id in enumerate(remaining):
            result = raw_results[case_id]
            query = result.get("rephrased_query", "")
            context = result.get("context_text", "")
            answer = result.get("answer", "")

            print(f"\n[{len(completed) + i + 1}/{len(raw_results)}] Evaluating case {case_id}...")

            t_start = time.time()
            try:
                eval_result = _call_gpt4(query, context, answer)
            except Exception as e:
                print(f"  ERROR: {e}")
                eval_result = {
                    "scores": {
                        "answer_relevance": -1.0,
                        "context_relevance": -1.0,
                        "groundedness": -1.0,
                        "justification": f"API_ERROR: {str(e)[:200]}",
                    },
                    "raw_response": "",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            api_time = time.time() - t_start

            scores = eval_result["scores"]
            ar = scores.get("answer_relevance", -1.0)
            cr = scores.get("context_relevance", -1.0)
            g = scores.get("groundedness", -1.0)
            valid_scores = [s for s in [ar, cr, g] if s >= 0]
            avg = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else -1.0

            row = {
                "case_id": case_id,
                "answer_relevance": ar,
                "context_relevance": cr,
                "groundedness": g,
                "avg_relevance": avg,
                "justification": scores.get("justification", ""),
            }
            writer.writerow(row)
            csv_file.flush()

            # Log full response for auditability
            log_entry = {
                "case_id": case_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "api_time_sec": round(api_time, 2),
                **eval_result,
            }
            with open(GPT4_EVAL_LOG, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            usage = eval_result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # GPT-4.1 pricing: $2/1M input, $8/1M output
            cost = (prompt_tokens * 2.0 + completion_tokens * 8.0) / 1_000_000
            total_cost_estimate += cost

            print(
                f"  AR={ar:.2f} CR={cr:.2f} G={g:.2f} avg={avg:.2f} "
                f"time={api_time:.1f}s tokens={prompt_tokens}+{completion_tokens} cost=${cost:.4f}"
            )
    finally:
        csv_file.close()

    print(f"\n[GPT-4.1 Eval] Done. Results: {GPT4_EVAL_RESULTS}")
    print(f"[GPT-4.1 Eval] Full log: {GPT4_EVAL_LOG}")
    print(f"[GPT-4.1 Eval] Estimated total cost: ${total_cost_estimate:.4f}")


if __name__ == "__main__":
    main()
