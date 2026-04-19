# GPT-4.1 Evaluation Agent Instructions

## Your Task

You are an expert legal evaluation agent. You must evaluate the outputs of a legal RAG system that answers questions about the Bharatiya Nyaya Sanhita 2023 (BNS) — India's new criminal code.

## Input File

Read the file: `evaluation/results/system3_raw_results.jsonl`

Each line is a JSON object with:
- `case_id`: integer (1-100)
- `result`: object containing:
  - `rephrased_query`: the formal legal query
  - `context_text`: the retrieved BNS sections (this is what the system used to generate the answer)
  - `answer`: the system's generated answer

## What to Evaluate

For each case, score these **three metrics** from 0.0 to 1.0 (one decimal place):

### 1. Answer Relevance (answer_relevance)
Does the answer accurately and completely address the legal query?
- 1.0 = Perfectly addresses the query, identifies correct legal provisions, gives actionable advice
- 0.7-0.9 = Mostly correct but missing some relevant provisions or details
- 0.4-0.6 = Partially addresses the query, some errors or omissions
- 0.1-0.3 = Mostly irrelevant or incorrect
- 0.0 = Completely fails to address the query

### 2. Context Relevance (context_relevance)
Is the retrieved context pertinent and sufficient for answering the query?
- 1.0 = All retrieved BNS sections are directly relevant to the legal scenario
- 0.7-0.9 = Most sections relevant, one or two tangential
- 0.4-0.6 = Mixed — some relevant, some irrelevant sections
- 0.1-0.3 = Mostly irrelevant sections retrieved
- 0.0 = No relevant context retrieved

### 3. Groundedness (groundedness)
Is every factual claim and legal citation in the answer supported by the retrieved context?
- 1.0 = Every cited BNS section appears in the context, no fabricated claims
- 0.7-0.9 = Almost all citations grounded, minor unsupported claims
- 0.4-0.6 = Some citations not found in context, or some fabricated information
- 0.1-0.3 = Significant hallucination — many citations not in context
- 0.0 = Answer is entirely fabricated or ungrounded

## Output File

Write results to: `evaluation/results/gpt4_eval_results.csv`

The CSV must have these columns (with header row):
```
case_id,answer_relevance,context_relevance,groundedness,avg_relevance,justification
```

Where `avg_relevance` = mean of the three scores, and `justification` is a brief 1-sentence explanation.

## How to Process

1. Read `evaluation/results/system3_raw_results.jsonl`
2. For each case (process in batches of 10-15 to avoid context overflow):
   - Read the `rephrased_query`, `context_text`, and `answer`
   - Score all three metrics based on the rubric above
   - Write the row to the CSV
3. After processing all cases, confirm the total count matches

## Important Notes

- The system is designed for BNS 2023 (NOT IPC). If the answer cites IPC sections, that should lower the groundedness score.
- Context headers look like `[BNS Section NNN]` — check that cited sections in the answer actually appear in these headers.
- Be consistent in scoring. A case that correctly identifies 2 out of 3 relevant sections should score similarly each time.
- Process ALL 100 cases. Do not skip any.
