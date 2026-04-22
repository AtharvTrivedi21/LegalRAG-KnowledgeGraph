# Chapter: Experimentation

This chapter documents the **experimental programme** for LegalRAG: baselines without embedding fine-tuning, **multiple fine-tuning corpora** (IndicLegalQA, templated synthetic, IPC→BNS mapping, **Llama-3.1-generated** incident descriptions, complaint-style HF data with hard negatives), the **best-performing** Llama-3.1-synthetic configuration, and the **cross-encoder reranker** (System S4). We report **development** metrics on legal QA passage ranking and **end-to-end** metrics on a fixed **100-case** BNS benchmark with identical automatic scoring (`bns_comparison/metrics.py`). **Bibliography:** see `content/experiments.md`.

---

## 4.1 Objectives

1. Quantify **dense retrieval** and **answer grounding** when switching from **base BGE** to **fine-tuned** encoders.
2. Determine which **supervision** best matches **incident-style** citizen queries against **BNS** sections.
3. Compare **legacy** pipeline settings (**S1**) against the **Llama-3.1-synthetic** fine-tuned model (**S2**).
4. Assess **two-stage retrieval** (bi-encoder + cross-encoder rerank) as **S4** relative to dense-only **S2**.

---

## 4.2 Experimental platform

### 4.2.1 Models and software

- **Bi-encoder:** `BAAI/bge-small-en-v1.5` (Xiao et al., 2023); training via **sentence-transformers** (Reimers & Gurevych, 2019).
- **Fine-tuning losses:** **MultipleNegativesRankingLoss** for pairs; **TripletLoss** for complaint hard-negative triplets (`phase3_embeddings/finetune_bge.py`).
- **Cross-encoder (S4):** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Nogueira & Cho, 2019).
- **LLM:** Ollama-hosted model for rephrase and answer (configurable; same stack across compared systems except where noted).
- **Index:** FAISS **IndexFlatIP** over BNS section texts for benchmark adapters (`bns_comparison/`).

### 4.2.2 Benchmark

**100 hand-crafted cases** with `expected_bns_sections` (gold), offense metadata, and natural-language descriptions. Runners:

- `evaluation/run_system3_100.py` — dense pipeline with **fine-tuned** index (S2 family).
- `evaluation/run_system3_100_nonft.py` — **non-fine-tuned** BGE index (**S3**).
- `python -m rerank_experiment.run_100` — **S4** (dense + rerank + citation constraints).

Outputs: per-case JSONL traces and CSV aggregates under `evaluation/results/`. Naming: `evaluation/SYSTEM_NAMES.md`.

### 4.2.3 Metrics (summary)

Per-case definitions match **Chapter 3 (Implementation), §3.8.1**: **hit rate**, **MRR**, **section precision/recall/F1**, **grounding score**. All table values below are **means over 100 cases** unless stated otherwise.

---

## 4.3 Experiment E0 — Non-fine-tuned embedding baseline (S3)

**Goal:** Isolate the effect of **embedding adaptation** by keeping the **same** RAG adapter logic and swapping only the FAISS index to **base** `BAAI/bge-small-en-v1.5` embeddings over BNS sections.

**Procedure:** Build or select the non-FT index (`bns_comparison/faiss_bns_only_nonft/` family); run `evaluation/run_system3_100_nonft.py`; use deduplicated artifact `system3_results_100_nonft_bge_dedup.csv`.

**Results (Table 4.1).**

| Metric | Mean (S3) |
|--------|-----------|
| Hit rate | 0.330 |
| MRR | 0.125 |
| Section precision | 0.094 |
| Section recall | 0.154 |
| Section F1 | 0.108 |
| Grounding score | 0.756 |

**Analysis.** **S3** achieves the **lowest** hit rate and MRR among the three full 100-case configurations in Table 4.2, confirming that **BNS-specific fine-tuning** is critical for this benchmark—not merely using a strong off-the-shelf encoder.

---

## 4.4 Experiment E1 — IndicLegalQA fine-tuning (development split)

**Goal:** Establish **legal-domain** adaptation on a **large** Indian QA corpus (Veningston & Mishra, 2025) before BNS-specific tricks.

**Data:** ≈10k pairs, `question` / `answer`, 80/20 split, seed 42.

**Metrics:** **InformationRetrievalEvaluator** on held-out pairs (one relevant passage per query): MRR@10, NDCG@10, Recall@10.

**Results (Table 4.2, from logged training runs / `Documentation/Phase3_Embeddings.md`).**

| Setting | MRR@10 | NDCG@10 | Recall@10 |
|---------|--------|---------|-----------|
| Baseline BGE (no FT) | 0.5233 | 0.5593 | 0.6735 |
| Fine-tune (best logged run) | **0.7713** | **0.7987** | **0.8845** |

**Analysis.** Fine-tuning on IndicLegalQA yields a **large** improvement on the **passage-ranking** dev task. This does **not** automatically equal BNS section hit-rate gains; it motivates **domain** adaptation before adding **incident-shaped** BNS supervision.

---

## 4.5 Experiment E2 — BNS supervision variants

### 4.5.1 Template synthetic

**Method:** Template-generated queries paired with BNS section text (`bns_synthetic_pairs.jsonl`).

**Outcome:** Queries are **unnaturally uniform**; limited utility for the incident benchmark (internal project log).

### 4.5.2 IPC → BNS mapping pairs

**Method:** Structured mapping prompts + BNS positives (`bns_mapping_pipeline/` → `bns_mapping_pairs.jsonl`); train `--dataset bns_mapping`.

**Outcome:** **Distribution mismatch** vs informal victim narratives; deprioritized.

### 4.5.3 Llama 3.1 synthetic incidents — **best method**

**Method:** For sections with numbers **≥ 45**, `llama-3.1-8b-instant` generates **three** short incident descriptions per section; each query is paired with the **true** section text from `phase1_output_v2/sections.csv`. Output is a resumable synthetic pairs JSONL generated by the project pipeline.

**Rationale:** Matches **plain-language** offense descriptions used in the 100-case suite.

**Training:** dedicated **Llama-3.1 synthetic** mode, or a **combined** mode with IndicLegalQA.

**Outcome:** **S2** — best **end-to-end** configuration on the BNS benchmark (Table 4.3).

### 4.5.4 Complaint dataset and hard negatives

**Method:** Hugging Face `navaneeth005/complaint-relevant-bns` processed by `build_complaint_training_data.py` into ≈**1504** pairs and ≈**1504** triplets (chapter-proximity **hard negatives**). Train: `complaint_bns` (MNRL) or `complaint_bns_hardneg` (TripletLoss).

**Outcome:** **Regression** vs the Llama-3.1-synthetic setup on internal checks and the 100-case line—likely **label noise** and ambiguity; reported as a **negative result** useful for thesis discussion.

---

## 4.6 Experiment E3 — Legacy vs Llama-3.1-synthetic (S1 vs S2)

**Goal:** Headline comparison under the **same** generation and retrieval scaffolding, differing primarily in **embedding / index** lineage.

**Results (Table 4.3).** Means from the S1 baseline results CSV and the S2 fine-tuned results CSV in `evaluation/results/`.

| Metric | S1 | S2 | \(\Delta\) |
|--------|----|----|------------|
| Hit rate | 0.460 | **0.590** | +0.130 |
| MRR | 0.193 | **0.345** | +0.152 |
| Section precision | 0.130 | **0.159** | +0.029 |
| Section recall | 0.250 | **0.322** | +0.072 |
| Section F1 | 0.160 | **0.203** | +0.043 |
| Grounding score | 0.746 | **0.796** | +0.051 |

**Analysis.** **S2** improves **retrieval hit and rank** and **citation overlap** (F1) with **higher grounding**, supporting the hypothesis that **distribution-matched synthetic supervision** is effective for this task.

---

## 4.7 Experiment E4 — Cross-encoder reranker (S4)

**Goal:** Add a **second stage** after FAISS: score \((q, \text{passage})\) pairs with a **cross-encoder**, reorder, then generate (`rerank_experiment/adapter.py`).

**Configuration:** `cross-encoder/ms-marco-MiniLM-L-6-v2`; strict citation filtering per package README; timings include `rerank_sec`.

**Status.** A **full** 100-case CSV (`system3_results_100_rerank_exp.csv`) should be produced by completing:

```text
python -m rerank_experiment.run_100
```

At the time of writing, the repository contained a **partial** raw log (`system3_raw_results_rerank_exp.jsonl`, **22** cases). **Thesis recommendation:** run to completion and **insert the S4 row** next to S2 in Table 4.4 for the final bound document.

**Expected use of results:** If S4 improves **MRR** and **section F1** over S2 at similar latency cost, it supports keeping reranking in the **reference architecture**; if gains are small, the **cost–benefit** trade-off should be discussed.

---

## 4.8 Consolidated comparison

**Table 4.4.** All systems with **\(n=100\)** CSVs available in the repository.

| System | Description | Hit rate | MRR | Sec. F1 | Grounding |
|--------|-------------|----------|-----|---------|-----------|
| **S1** | Legacy baseline | 0.460 | 0.193 | 0.160 | 0.746 |
| **S2** | Llama-3.1-synthetic FT | **0.590** | **0.345** | **0.203** | **0.796** |
| **S3** | Non-FT BGE | 0.330 | 0.125 | 0.108 | 0.756 |
| **S4** | + Cross-encoder rerank | *pending full CSV* | *pending* | *pending* | *pending* |

**Ordering by primary retrieval signal:** S2 \(>\) S1 \(>\) S3 on hit rate and MRR; **S3** confirms that **omitting** BNS fine-tuning **hurts** despite reasonable grounding (citations sometimes misaligned to gold).

---

## 4.9 Manual evaluation (complementary)

**File:** `evaluation/results/gpt4_eval_results_manual_100.csv`. Dimensions: **answer relevance**, **context relevance**, **groundedness**, **avg_relevance**, plus short justifications. Use for **qualitative** discussion and failure cases (e.g. citation–retrieval mismatches in `gpt4_citation_mismatches.csv`).

---

## 4.10 Discussion

1. **Fine-tuning signal > generic encoder** on this benchmark (**S2** vs **S3**).
2. **Synthetic diversity** matters: Llama 3.1 **incidents** outperform templates and mapping-style pairs.
3. **External complaints** are **not** a drop-in win—label noise dominated in our runs.
4. **Reranking** is a **logical** next step after strong bi-encoders; **empirical** closure requires the finished S4 table.
5. **Limitations:** \(n=100\); rule-based citation extraction in metrics; LLM judge bias; IPC leakage in some answers despite BNS context.

---

## 4.11 Brief conclusion (experiments)

The experimental track identifies **Llama-3.1-synthetic fine-tuning** of **BGE** as the **most successful** embedding configuration for the BNS incident benchmark, with **clear gains** over both a **legacy** baseline (**S1**) and a **non-fine-tuned** encoder (**S3**). **IndicLegalQA** fine-tuning shows strong **development-set** ranking improvements. **Complaint** supervision **hurt** quality in our setup. **Cross-encoder reranking** is implemented and partially logged; **full S4 numbers** remain the immediate empirical next step. Extended **conclusion and engineering roadmap** appear in the **Implementation** chapter (§3.10).
