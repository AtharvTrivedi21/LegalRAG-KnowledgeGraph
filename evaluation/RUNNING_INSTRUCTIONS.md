# Evaluation Pipeline — Running Instructions

## Prerequisites

### 1. Ollama Setup

Make sure Ollama is running and the required models are pulled:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
```

Verify Ollama is running:
```bash
ollama list
```

You should see `llama3:8b` and `nomic-embed-text` in the output.

### 2. Neo4j (Optional)

System 3 uses Neo4j for graph enrichment. If Neo4j is running, it adds section headings and act metadata. If not, it gracefully skips — results are still valid.

### 3. FAISS Index

The BNS FAISS index must exist at `bns_comparison/faiss_bns_only/faiss.index`. If missing, rebuild:
```bash
python -m bns_comparison.build_bns_faiss --system bge
```

---

## Step 1: Run System 3 on 100 Queries

This runs the full RAG pipeline (rephrase + retrieve + generate + self-eval) for each of the 100 test cases and saves results incrementally.

### Option A: Using Groq Cloud (RECOMMENDED — ~15-20 minutes)

Set your free Groq API key (get one at https://console.groq.com):
```bash
set GROQ_API_KEY=gsk_your-key-here
python -m evaluation.run_system3_100
```

Uses the same llama3-8b model but hosted on Groq's fast hardware. Free tier: 30 req/min, 14,400 req/day.

### Option B: Using local Ollama (~7-10 hours)

```bash
python -m evaluation.run_system3_100
```

**Expected runtime (Groq): ~15-20 minutes. Expected runtime (local Ollama): ~7-10 hours.**

**Terminal output:** Shows progress for every case with ETA, metrics, and percentage complete.

**Incremental/Resumable:** If you stop the process (Ctrl+C), just re-run the same command — it skips already-completed cases and picks up where it left off.

**Optional flags:**
```bash
python -m evaluation.run_system3_100 --start 1 --end 50    # run only cases 1-50
python -m evaluation.run_system3_100 --start 51 --end 100  # run cases 51-100
```

**Output files:**
- `evaluation/results/system3_raw_results.jsonl` — full raw results (query, context, answer, chunks, timings)
- `evaluation/results/system3_results_100.csv` — rule-based metrics CSV (exported after all 100 complete)

---

## Step 2: GPT-4.1 Evaluation (Answer Relevance, Context Relevance, Groundedness)

After Step 1 completes, evaluate the results using GPT-4.1 as a judge.

### Option A: Via Cursor Agent (No API Key Needed)

1. Open a **new Cursor chat**
2. Select **GPT-4.1** as the model
3. Send this message:

> Read `@evaluation/EVAL_AGENT_INSTRUCTIONS.md` and follow the instructions. Process all 100 cases from the results file and write the evaluation scores to the output CSV.

The agent will:
- Read `evaluation/results/system3_raw_results.jsonl`
- Score each case on 3 metrics (0.0-1.0)
- Write results to `evaluation/results/gpt4_eval_results.csv`

### Option B: Via OpenAI API (Programmatic, ~$0.50)

```bash
set OPENAI_API_KEY=sk-your-key-here
python -m evaluation.evaluate_gpt4 --dry-run     # test on 2 cases first
python -m evaluation.evaluate_gpt4               # full 100 cases
```

**Output files:**
- `evaluation/results/gpt4_eval_results.csv` — GPT-4.1 scores (AR, CR, G, avg, justification)
- `evaluation/results/gpt4_eval_log.jsonl` — full API response log (Option B only)

---

## Step 3: Generate Final Analysis

Merges rule-based metrics and GPT-4.1 scores, then prints thesis-ready tables.

```bash
python -m evaluation.analysis
```

**Output:**
- `evaluation/results/final_evaluation.csv` — all metrics merged into one file
- Terminal prints: overall summary, per-category breakdown, failure cases, correlation analysis

---

## Quick Reference

| Step | Command | Runtime | Output |
|------|---------|---------|--------|
| Prereq | `ollama pull llama3:8b` | ~5 min | Model downloaded |
| Prereq | `ollama pull nomic-embed-text` | ~1 min | Model downloaded |
| Step 1 (Groq) | `set GROQ_API_KEY=... && python -m evaluation.run_system3_100` | ~15-20 min | `system3_raw_results.jsonl` |
| Step 1 (local) | `python -m evaluation.run_system3_100` | ~7-10 hrs | `system3_raw_results.jsonl` |
| Step 2 | Cursor agent OR `python -m evaluation.evaluate_gpt4` | ~5-10 min | `gpt4_eval_results.csv` |
| Step 3 | `python -m evaluation.analysis` | ~1 sec | `final_evaluation.csv` |

---

## Troubleshooting

- **Ollama timeout:** If cases fail with timeout errors, ensure no other heavy GPU tasks are running. System 3 forces the embedding model to CPU so the GPU is free for llama3:8b.
- **CUDA OOM:** The Ollama helper retries up to 3 times with 10s delays on GPU OOM errors.
- **Partial run:** If Step 1 was interrupted, re-run the same command. It reads the JSONL and skips completed case IDs.
- **Missing context in eval:** If GPT-4.1 scores seem off, check that `system3_raw_results.jsonl` contains non-empty `context_text` for each case.
