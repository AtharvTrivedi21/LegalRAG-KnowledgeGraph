# Figures (architecture and system design)

This file lists **thesis figures for the implementation / methodology chapters**: overall architecture, per-phase pipelines, and the **reranker** as part of retrieval—not matplotlib charts of evaluation metrics. For optional **results** plots (histograms, heatmaps from CSVs), see **Appendix A** at the bottom.

---

## Figure 1 — End-to-end LegalRAG architecture

- **What to show:** Data sources → Phase 1 CSV export → parallel **Neo4j** (Phase 2) and **chunk + FAISS** (Phase 3) → Phase 4 RAG with an explicit **“Rerank enabled?”** branch (v3 default vs S4).
- **Source:** Mermaid in `content/implementation.md` §2.2 (copy into Mermaid Live, Typora, or your LaTeX Mermaid plugin; export PDF/PNG).
- **Suggested caption:** “End-to-end pipeline: Phase 1 produces shared structured exports; Phase 2 loads the knowledge graph; Phase 3 builds the bi-encoder vector index; Phase 4 resolves queries in Neo4j, retrieves from FAISS, optionally reranks with a cross-encoder (S4), and generates a structured answer.”

---

## Figure 2 — Phase 1: Ingestion and structuring

- **Source:** `Documentation/Phase1_Diagram.md`
- **Caption sketch:** Judgment and statute inputs, extraction steps, and Neo4j-ready CSV outputs.

---

## Figure 3 — Phase 2: Knowledge graph load

- **Source:** `Documentation/Phase2_Diagram.md`
- **Caption sketch:** Import directory, constraint scripts, node and relationship types (`IN_ACT`, `CITES`).

---

## Figure 4 — Phase 3: Chunking, fine-tuning, and FAISS

- **Source:** `Documentation/Phase3_Diagram.md`
- **Caption sketch:** Corpus chunks, optional BGE fine-tuning, embedding, `IndexFlatIP`, retrieval API.

---

## Figure 5 — Phase 4: LangGraph RAG (default v3)

- **Source:** `Documentation/Phase4_Diagram.md`
- **Caption sketch:** Query parser, graph retriever, vector retriever, guard / LLM answer path.

---

## Figure 6 — Two-stage retrieval: bi-encoder + cross-encoder reranker

- **What to show:** Query embedding → FAISS **top-K** → **cross-encoder** scores each \((q, \text{passage})\) → reorder → top-\(k\) context → LLM. Contrasts with v3 (no cross-encoder box).
- **Source:** Mermaid in `content/implementation.md` §2.4; implementation detail in §6.1 (`rerank_experiment/adapter.py`, model `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Suggested caption:** “Two-stage retrieval: the bi-encoder–FAISS stage proposes a shortlist; the cross-encoder reranks candidates before prompt construction (System S4). Production LangGraph v3 skips this stage.”

---

## Figure placement in the thesis

| Chapter / section | Figures |
|-------------------|---------|
| Implementation / system design | Fig. 1 (overall), Fig. 2–5 (phases), Fig. 6 (rerank zoom-in) |
| Experiments / results | Use **tables** and, only if required, appendix plots below |

---

## Appendix A — Optional results plots (experiments chapter only)

These are **not** architecture figures; include only if your examiner expects **empirical** visuals. Generate with Python (pandas + matplotlib/seaborn); save under `MTech Thesis/figures/results/` to keep them separate from design figures.

| Plot | Data | Output filename (suggested) |
|------|------|---------------------------|
| Cited vs retrieved overlap | `evaluation/results/first10_cited_found.csv`, `system3_raw_results*.jsonl` | `results/fig_overlap_hist.png` |
| Manual groundedness distribution | `evaluation/results/gpt4_eval_results_manual_100.csv` | `results/fig_groundedness_hist.png` |
| Answer vs context relevance heatmap | same CSV | `results/fig_relevance_heatmap.png` |
| Groundedness vs avg relevance | same CSV + `gpt4_citation_mismatches.csv` | `results/fig_grounded_scatter.png` |
| Latency boxplots | JSONL or `system3_results_100_*.csv` timing columns | `results/fig_latency_box.png` |

**Reproducibility:** Prefer small scripts under `evaluation/figures/` (create if missing) with pinned random seeds for any bootstrap plots.
