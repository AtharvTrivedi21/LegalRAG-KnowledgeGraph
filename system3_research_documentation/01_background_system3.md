# Background: System 3 and the BNS-Only Stack

## Purpose

The goal is to improve **retrieval and generation quality** for a legal assistant that answers using the **Bharatiya Nyaya Sanhita (BNS) 2023** as the primary statute source. System 3 is the **full pipeline** variant used in `bns_comparison`: rephrase user text into a formal query, embed and retrieve from a **BNS-only FAISS** index, generate an answer, and (in the base adapter) run self-eval / rewrite logic as implemented in the adapter.

## High-level pipeline (conceptual)

1. **Rephrase:** informal incident → formal legal-style query (LLM).
2. **Retrieve:** bi-encoder embeddings + FAISS over BNS section chunks.
3. **Generate:** LLM answer conditioned on retrieved context.
4. **Metrics:** compare predicted citations to gold sections; measure grounding and hallucination proxies.

Concrete prompts and behavior live in `bns_comparison/adapters/full_pipeline_bns.py`. The **reranker variant** duplicates a similar flow but adds a cross-encoder stage and stricter citation rules in `rerank_experiment/adapter.py`.

## Key configuration

`bns_comparison/config.py` defines:

- **FAISS index:** `bns_comparison/faiss_bns_only/faiss.index`
- **Chunk metadata:** `bns_comparison/faiss_bns_only/chunk_metadata.pkl`
- **Embedding model directory** (bi-encoder used to build the index and at query time): `FINE_TUNED_MODEL_DIR` → currently `phase3_embeddings/bge-legal-bns-groq`
- **Sections source:** `phase1_output_v2/sections.csv`
- **Groq:** `GROQ_MODEL` (e.g. `llama-3.1-8b-instant`), optional pool `GROQ_API_KEY_1` … `GROQ_API_KEY_4`
- **Retrieval:** `TOP_K` (e.g. 8)

## Metrics (100-case CSV)

The evaluation script `evaluation/run_system3_100.py` exports per-case rows and aggregates **means** over numeric columns. Primary indicators we tracked:

| Metric | Meaning (informal) |
|--------|---------------------|
| `hit_rate` | Whether at least one gold section appears in retrieved/cited set (implementation per `bns_comparison/metrics.py`) |
| `mrr` | Mean reciprocal rank of the first relevant hit |
| `section_precision` / `section_recall` / `section_f1` | Set overlap between cited BNS sections and gold |
| `grounding_score` | How well the answer stays tied to provided context (rule-based) |
| `fabricated_section_count`, `hallucination_flag` | Signals for citations not supported by context |

For exact definitions, see `bns_comparison/metrics.py` and the CSV field list at the top of `evaluation/run_system3_100.py`.

## Test set

- **File:** `bns_comparison/test_cases.py`
- **Size:** 100 cases (`TEST_CASES`), each with narrative description, offense category, and **gold BNS sections**.

**Note:** Gold labels are a benchmark convenience. If individual gold sections are debatable for a narrative, that is a **benchmark quality** issue; improving the benchmark is separate from improving the model. The docs record what was used for comparison, not a claim of statutory perfection for every row.

## Related plans (Cursor)

Higher-level experiment roadmaps lived under the user’s `.cursor/plans/` directory (e.g. `system_3_improvement_experiments_*.plan.md`, `exp7_bns_synthetic_fine-tune_*.plan.md`). This folder documents what was **actually implemented and run** in the repo.
