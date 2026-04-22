# Experiments and evaluation

> **Thesis draft:** The polished chapter text for submission is in **`chapters/chapter_experimentation.md`**. This file retains the full bibliography and extended RQ/metric notes.

This chapter states the **research questions** addressed by the implementation, describes **embedding fine-tuning experiments** over multiple supervision sources (with quantitative results at both **IR dev** and **end-to-end BNS benchmark** levels), reports **RAG system comparisons**, and defines **evaluation metrics** with formulas aligned to the codebase. **Citations** anchor methods to prior work; the **Bibliography** at the end should be copied into the thesis reference list.

---

## 1. Research questions

The experimental programme is organized around the following questions:

1. **RQ1 (Domain adaptation)** — Does fine-tuning a general-purpose English sentence encoder on **Indian legal Q/A** (IndicLegalQA) improve dense retrieval quality on a held-out IR split?
2. **RQ2 (Distribution match)** — For **BNS section retrieval** from **incident-style** user queries, which **synthetic or external** supervision best matches the evaluation distribution: templated queries, IPC→BNS mapping prompts, **LLM-generated** citizen incidents (Llama 3.1), or complaint-style HF data?
3. **RQ3 (End-to-end effect)** — How do these embedding choices affect **hit rate**, **ranking (MRR)**, **section overlap (F1)**, and **answer grounding** on a fixed **n = 100** case benchmark under the same generation stack?
4. **RQ4 (Human-aligned quality)** — Do **manual** relevance and groundedness scores correlate with automatic grounding, and where do **citation–retrieval mismatches** occur?

---

## 2. Experimental setup (shared)

### 2.1 Base model and framework

All dense retrieval uses the **bi-encoder** backbone **`BAAI/bge-small-en-v1.5`** (Xiao et al., 2024), trained and evaluated with **sentence-transformers** (Reimers & Gurevych, 2019). Fine-tuning objectives are **MultipleNegativesRankingLoss** for `(query, positive passage)` batches and **TripletLoss** for complaint hard-negative triplets (`phase3_embeddings/finetune_bge.py`).

### 2.2 Training-time evaluation (IndicLegalQA-style split)

For each dataset mode, 80% of pairs (seed 42) are used for training and 20% for an **InformationRetrievalEvaluator** task: each held-out query has **exactly one** relevant document (the paired passage). Metrics **MRR@10**, **NDCG@10**, **Recall@10**, and accuracy@\(k\) follow standard definitions (Järvelin & Kekäläinen, 2002; Manning et al., 2008). Formulas are given in **`implementation.md`** §5.4.

> **Note:** These dev metrics measure **passage ranking** on the pair distribution—they are **not** identical to BNS **section-ID** hit rate on the 100-case suite, but they are standard for monitoring **overfitting** and **regression** when swapping training data.

### 2.3 BNS 100-case benchmark (end-to-end)

The primary system comparison uses **FullPipelineBNS**-style adapters over a **BNS section FAISS** index (`bns_comparison/`), with runners `evaluation/run_system3_100.py` (fine-tuned index) and `evaluation/run_system3_100_nonft.py` (base BGE index). **Gold** section sets `expected_bns_sections` come from the benchmark JSON; outputs are aggregated as row-wise means in CSVs under `evaluation/results/`.

**Systems** (see `evaluation/SYSTEM_NAMES.md`):

| ID | Description | Key results file |
|----|-------------|------------------|
| **S1** | Legacy baseline pipeline / embedding configuration | `system3_results_100_baseline.csv` |
| **S2** | **Llama-3.1-synthetic** fine-tuned BGE (best validated) | S2 fine-tuned results CSV in `evaluation/results/` |
| **S3** | Base BGE, non-fine-tuned index | `system3_results_100_nonft_bge_dedup.csv` |
| **S4** | Reranker-augmented (experimental; confirm run completion) | `system3_results_100_rerank_exp.csv` |

### 2.4 Automatic metrics (formulas)

Per-case metrics are computed in **`bns_comparison/metrics.py`**. Let \(G\) = gold BNS section numbers, \(C\) = model-cited sections, and \((s_1,\ldots,s_T)\) = section numbers **in retrieval order** from `retrieved_chunks`.

| Metric | Definition (per case) | Aggregation in reports |
|--------|------------------------|-------------------------|
| **Hit rate** | \(h = \mathbb{1}[G \cap \{s_1,\ldots,s_T\} \neq \emptyset]\) | Mean of \(h\) over \(n=100\) |
| **MRR** | If \(\exists\) smallest \(r\) with \(s_r \in G\): \(1/r\); else \(0\) | Mean MRR |
| **Section precision** | \(|C \cap G|/|C|\) if \(C \neq \emptyset\), else \(0\) | Mean |
| **Section recall** | \(|C \cap G|/|G|\) when \(C,G\) nonempty; \(0\) if \(C=\emptyset\) | Mean |
| **Section F1** | \(2PR/(P+R)\) if \(P+R>0\), else \(0\) | Mean |
| **Grounding score** | With \(S_{\text{ctx}}\) = sections found in **context** text: \(|C \cap S_{\text{ctx}}|/|C|\) if \(C \neq \emptyset\), else \(1\) | Mean |

**Interpretation:** *Hit rate* is a coarse **any-match** retrieval indicator; *MRR* rewards **early** appearance of a gold section; *precision/recall/F1* compare **stated citations** to gold (not to retrieval alone); *grounding score* measures whether **cited** sections are **supported by retrieved context** (a **necessary** condition for faithful quoting).

Additional logged diagnostics include **IPC mention count**, **fabricated section count** (invalid range), and latency splits—see source for thresholds (`_BNS_MAX_SECTION = 358`).

### 2.5 Manual evaluation protocol

A separate **manual** pass (`evaluation/results/gpt4_eval_results_manual_100.csv`) scores each case on:

- **answer_relevance** — Does the answer address the user scenario?
- **context_relevance** — Is retrieved context pertinent?
- **groundedness** — Are claims supported by context (ordinal 0–1)?
- **avg_relevance** — Mean of answer and context relevance.

Rubric text: `evaluation/EVAL_AGENT_INSTRUCTIONS.md`. This follows the **LLM-as-judge** paradigm; limitations (position bias, verbosity bias) are well documented (Zheng et al., 2023).

---

## 3. Experiment A — IndicLegalQA fine-tuning (RQ1)

### 3.1 Data

**IndicLegalQA** (≈10k QA pairs from Indian Supreme Court judgments) supports **broad legal-domain** adaptation (Veningston & Mishra, 2025). Pairs are loaded as `question` / `answer` fields from `Datasets/IndicLegalQA Dataset_10K_Revised.json`.

### 3.2 Procedure

Train with `--dataset indiclegal` (MNRL), 80/20 split, **InformationRetrievalEvaluator** each epoch; best checkpoint by **Recall@10** (`finetune_bge.py`).

### 3.3 Results (IR dev split, reported in repository)

Table 1 reproduces logged **baseline vs fine-tuned** runs from `Documentation/Phase3_Embeddings.md` / `phase3_embeddings/results.txt`. **Run 3** is the strongest reported dev run (MRR@10 = 0.7713 vs baseline 0.5233).

| Setting | MRR@10 | NDCG@10 | Recall@10 |
|---------|--------|---------|-----------|
| Baseline (unfine-tuned BGE) | 0.5233 | 0.5593 | 0.6735 |
| Fine-tune Run 1 | 0.7117 | 0.7422 | 0.8375 |
| Fine-tune Run 2 | 0.5972 | 0.6354 | 0.7570 |
| Fine-tune Run 3 (best logged) | **0.7713** | **0.7987** | **0.8845** |

**Takeaway:** Legal QA supervision **substantially** improves ranking on the **IndicLegalQA-style** dev task, motivating domain adaptation before BNS-specific supervision.

---

## 4. Experiment B — BNS-specific supervision variants (RQ2)

### 4.1 Template synthetic (`bns_synthetic_pairs.jsonl`)

**Method:** Template-generated `(query, BNS section text)` pairs (`generate_template_synthetic.py` / legacy `build_synthetic_jsonl` path).

**Outcome:** Queries are **repetitive** and unlike natural citizen text; **limited** benchmark gains (`system3_research_documentation/02_experiment_timeline.md`). **Thesis role:** negative result on **distribution mismatch**.

### 4.2 IPC → BNS mapping pairs (`bns_mapping_pipeline/`)

**Method:** Structured IPC–BNS mapping produces “meta-legal” questions paired with BNS section text (`build_mapping_pairs.py` → `bns_mapping_pairs.jsonl`). Train with `--dataset bns_mapping`.

**Outcome:** Training distribution **does not match** incident descriptions in the 100-case benchmark; retrieval gains were **not** compelling. **Thesis role:** shows that **statute-reform** style supervision is insufficient for **fact-pattern** queries.

### 4.3 Llama 3.1 incident–BNS pairs (preferred)

**Method:** For crime-relevant sections (implementation: section numbers **≥ 45**), `llama-3.1-8b-instant` generates **three** informal incident descriptions per section; each is paired with the **true** section text from `phase1_output_v2/sections.csv`. Output: resumable synthetic pairs JSONL generated by the project pipeline.

**Rationale:** Aligns query style with **plain-language** offenses in the benchmark (distribution hypothesis).

**Training:** dedicated **Llama-3.1 synthetic** mode or a **combined** mode (IndicLegalQA + Llama-3.1 synthetic pairs).

### 4.4 Complaint dataset (`navaneeth005/complaint-relevant-bns`)

**Method:** `build_complaint_training_data.py` emits:

- `complaint_bns_pairs.jsonl` — ≈ **1504** `(query, positive)` pairs (timeline doc),
- `complaint_bns_hardneg_triplets.jsonl` — ≈ **1504** triplets with **chapter-proximity** hard negatives.

Train modes: `--dataset complaint_bns` (MNRL) or `complaint_bns_hardneg` (TripletLoss). Long narratives are **trimmed** (e.g. 600 / 1800 chars) for training stability on Windows.

**Outcome:** **Regression** vs the Llama-3.1-synthetic setup on internal and 100-case checks—attributed to **label noise**, multi-label ambiguity, or mismatch to hand-crafted gold (`system3_research_documentation/04_results.md`). **Thesis role:** important **negative result**; justifies **data cleaning** or **filtering** before large-scale use.

### 4.5 Summary table (embedding experiments)

| Dataset | CLI `--dataset` | Supervision style | Benchmark outcome (high level) |
|---------|-----------------|-------------------|--------------------------------|
| IndicLegalQA | `indiclegal` | Legal QA pairs | Strong **dev** IR gains (Table 1) |
| Template BNS | `bns_synthetic` | Templated queries | Weak / misaligned |
| IPC→BNS | `bns_mapping` | Reform / mapping prompts | Misaligned with incidents |
| Llama 3.1 incidents | dedicated synthetic mode | LLM citizen text + statute | **Best** end-to-end (S2) |
| Combined | combined synthetic + IndicLegalQA mode | IndicLegalQA + Llama-3.1 synthetic | Use if multi-domain robustness needed |
| Complaint pairs | `complaint_bns` | HF complaints | **Regression** vs Llama-3.1 synthetic |
| Complaint triplets | `complaint_bns_hardneg` | Hard negatives | **Regression** vs Llama-3.1 synthetic |

---

## 5. Experiment C — End-to-end BNS benchmark (RQ3)

### 5.1 S1 vs S2 (primary quantitative result)

Table 2 reports **row-wise means** over **n = 100** cases (`system3_research_documentation/04_results.md`).

| Metric | S1 — Legacy | S2 — Llama-3.1-synthetic FT | Δ |
|--------|-------------|--------------|---|
| Hit rate | 0.460 | 0.590 | **+0.130** |
| MRR | 0.193 | 0.345 | **+0.152** |
| Section precision | 0.130 | 0.159 | +0.029 |
| Section recall | 0.250 | 0.322 | +0.072 |
| Section F1 | 0.160 | 0.203 | +0.043 |
| Grounding score | 0.746 | 0.796 | +0.051 |

**Analysis:** Llama-3.1-synthetic fine-tuning improves **retrieval-centric** measures (hit rate, MRR) and **citation overlap** (F1) under a **fixed** generation stack, consistent with better **query–section alignment**. Absolute F1 remains moderate—**citation extraction** and **IPC vs BNS** wording in answers remain error modes (qualitative logs in `sample_result.txt`).

### 5.2 S3 — Non-fine-tuned baseline

**Purpose:** Isolate embedding adaptation: same pipeline, **base** `BAAI/bge-small-en-v1.5` FAISS index. **Artifact:** `system3_results_100_nonft_bge_dedup.csv`. **Thesis:** report mean metrics beside S2; optionally test significance (bootstrap over cases) if the institute requires it.

### 5.3 S4 — Reranker (optional second stage)

**Motivation:** **Cross-encoder** reranking can reorder top-k dense hits (Nogueira & Cho, 2019) without retraining the bi-encoder—useful when dense scores are noisy.

**Status:** Verify `system3_results_100_rerank_exp.csv` is **complete** before thesis submission; smoke CSVs are not sufficient.

---

## 6. Experiment D — Manual evaluation and citation diagnostics (RQ4)

### 6.1 Consolidated scores

File: `evaluation/results/gpt4_eval_results_manual_100.csv`. Use **`tables.md`** for thesis tables. Optional **results** plots (histograms, heatmaps) are listed under **Appendix A** in **`figures.md`**; the main `figures.md` content is **architecture** (including the reranker figure).

### 6.2 Citation mismatches

`evaluation/results/gpt4_citation_mismatches.csv` lists cases where **cited** sections had **no overlap** with **retrieved** section IDs—useful for **failure analysis**. `evaluation/scripts/find_unmatched_citations.py` reproduces this list.

**Boilerplate Section 1:** Retrieved chunks often include **Section 1** (short title / extent); excluding it in overlap analysis can yield **stricter** grounding statistics (recommended sensitivity analysis in thesis).

---

## 7. Qualitative comparison (`bns_comparison`)

`compare_one.py` / `run_comparison.py` run **multiple adapters** on the same query (e.g. legacy vs simplified BNS pipeline). **`sample_result.txt`** illustrates **rephrasing**, **structured answers**, and occasional **IPC** phrasing despite BNS context—suitable as a **case study** figure in the thesis.

---

## 8. Threats to validity

1. **Single benchmark size** (n = 100) — report confidence intervals if bootstrapping is feasible.
2. **Gold label quality** — Hand-crafted expected sections may omit multi-section answers.
3. **LLM judges** — Manual scores may inherit model biases (Zheng et al., 2023).
4. **Pipeline coupling** — Generation model and prompts fixed across S1/S2; improvements are **not** solely from embeddings if other components drifted between runs—**version** prompts and model IDs in the thesis.
5. **Training vs test leakage** — Ensure Llama-3.1 synthetic generation does not **copy** benchmark text (generation is section-driven, not case-driven; still document procedure).

---

## 9. Thesis figures and tables (cross-reference)

| Asset | Purpose |
|-------|---------|
| `figures.md` | **Architecture** figures (overall pipeline, phases, reranker); optional result plots in Appendix A |
| `tables.md` | Manual eval table, mismatch table, summary statistics |
| `Documentation/Phase*_Diagram.md` | Phase-wise architecture figures |
| Mermaid in `implementation.md` | Overall system Figure 1 source |

---

## Bibliography

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint* arXiv:2312.10997.

Henderson, M., Al-Rfou, R., Strope, B., Sung, Y.-H., Laszlo, A., Guo, S., Kumar, S., Miklos, B., & Kurzweil, R. (2017). Efficient natural language response suggestion for smart reply. *arXiv preprint* arXiv:1708.00640.

Hermans, A., Beyer, L., & Leibe, B. (2017). In defense of the triplet loss for person re-identification. *arXiv preprint* arXiv:1703.07737.

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*, 20(4), 422–446.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

Nogueira, R., & Cho, J. (2019). Passage re-ranking with BERT. *arXiv preprint* arXiv:1901.04085.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*, 3982–3992.

Robinson, I., Webber, J., & Eifrem, E. (2015). *Graph Databases* (2nd ed.). O’Reilly Media.

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 815–823.

Veningston, K., & Mishra, A. (2025). Dataset for legal question answering system in the Indian judiciary context. *Data in Brief*, 58, 111647. https://doi.org/10.1016/j.dib.2025.111647

Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., & Nie, J.-Y. (2023). C-Pack: Packed resources for general Chinese embeddings. *arXiv preprint* arXiv:2309.07597. (FlagEmbedding family; includes English BGE checkpoints such as `bge-small-en-v1.5`.)

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36.

---

*Note:* For LaTeX/BibTeX, map each entry to `@article` / `@inproceedings` / `@book` as appropriate; verify **page numbers and venue** against your library’s style guide (IEEE, ACM, APA). The IndicLegalQA **data descriptor** citation above follows the *Data in Brief* reference from the dataset’s publication record.
