---
name: LegalRAG Documentation Folder
overview: "Create a `Documentation` folder with 9 Markdown files: 4 phase-specific technical docs, 1 datasets reference, 3 architecture docs (overall, LangGraph, Knowledge Graph), plus optional Mermaid diagrams and a script to export architecture images for PPTs."
todos: []
isProject: false
---

# LegalRAG Documentation Plan

## Scope

- **Folder:** `Documentation/` at project root.
- **Deliverables:** 9 MD files (technical, high-level, easy to understand). Only document **datasets actually used** in code. Phase 4 doc covers the design once (no repeated “changes” narrative across phases). Optional: Mermaid diagrams and a script to generate architecture images (PNG/SVG) for PPTs.
- **Sources for content:** Docs may reference any **code file**, **terminal output** (e.g. from running pipeline, smoke tests, `test_three_queries.py`), **[phase3_embeddings/results.txt](phase3_embeddings/results.txt)**, and **agent chat transcripts** where relevant, so the documentation is highly valuable and report/PPT-ready.

---

## 1. Phase-specific docs (4 files)

Each file: **What** (purpose), **How** (components, data flow), **Why** (design choices), and **Results** (where we have showable outcomes). Only reference datasets that are **actually read** in that phase.


| File                                                                     | Phase                  | Datasets to mention                                                                                                                                                                                                                                                                                              | Key components                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Documentation/Phase1_Ingestion.md](Documentation/Phase1_Ingestion.md)   | Ingestion              | `legal_data_train.csv` (when `JUDGMENTS_SOURCE=csv`) **or** PDFs under `BASE_PATH/SC_EXTRACTED_DIR/<year>/*.pdf`; act PDFs from [config.py](config.py) `PDF_FILES`.                                                                                                                                              | [src/run_pipeline.py](src/run_pipeline.py), [src/judgments.py](src/judgments.py), [src/pdf_extractor.py](src/pdf_extractor.py), [src/edges.py](src/edges.py), [src/export.py](src/export.py); config: `BASE_PATH`, `OUTPUT_PATH`, `JUDGMENTS_SOURCE`, `ACT_NAMES`. Output: `phase1_output/*.csv` (cases, sections, articles, acts, edges).                             |
| [Documentation/Phase2_Neo4j.md](Documentation/Phase2_Neo4j.md)           | Neo4j load             | **Input:** Phase 1 CSVs copied to Neo4j `import/`: `acts.csv`, `sections.csv`, `articles.csv`, `cases.csv`, `edges.csv`. No other datasets.                                                                                                                                                                      | [neo4j/cypher/01_constraints.cypher](neo4j/cypher/01_constraints.cypher), [02_load_nodes.cypher](neo4j/cypher/02_load_nodes.cypher), [03_load_edges.cypher](neo4j/cypher/03_load_edges.cypher), [04_smoke_tests.cypher](neo4j/cypher/04_smoke_tests.cypher). Constraints, LOAD CSV, CITES aggregation.                                                                 |
| [Documentation/Phase3_Embeddings.md](Documentation/Phase3_Embeddings.md) | Embeddings & retrieval | **Read:** `phase1_output/cases.csv`, `sections.csv`, `articles.csv` ([chunk_corpus.py](phase3_embeddings/chunk_corpus.py)); `Datasets/IndicLegalQA Dataset_10K_Revised.json` ([finetune_bge.py](phase3_embeddings/finetune_bge.py)). Outputs: `chunks.pkl`, FAISS index, `chunk_metadata.pkl`, fine-tuned model. | [phase3_embeddings/config.py](phase3_embeddings/config.py), chunk_corpus, finetune_bge, build_faiss, retrieve. BGE base model, token chunking (CHUNK_SIZE/OVERLAP), MultipleNegativesRankingLoss, evaluation thresholds.                                                                                                                                               |
| [Documentation/Phase4_RAG.md](Documentation/Phase4_RAG.md)               | RAG (single narrative) | **No direct dataset paths.** Uses Phase 3 artifacts (FAISS, chunk_metadata, fine-tuned model) and Neo4j (populated from Phase 1 CSVs). LLM: Ollama (model from env/config).                                                                                                                                      | [phase4_rag/config_v3.py](phase4_rag/config_v3.py), query_parser_v3, neo4j_client_v3, vector_retriever_v3, langgraph_workflow_v3, llm_ollama. Describe workflow once: query_parser → graph_retriever → query_rephrase → vector_retriever → answer_generator; Act-aware disambiguation, confidence guard, structured answer. No repeated “what changed in v2/v3” lists. |


### Results to include (per phase)

- **Phase 1:** **Results** section with example pipeline output: “Loaded N cases”, “Acts: N”, “Sections: N”, “Articles: N”, “Edges: N”, “Years: min–max”. Source: run `python src/run_pipeline.py` (or capture from terminal/agent logs). Optionally document [scripts/verify_phase1_output.py](scripts/verify_phase1_output.py) output (row counts, duplicate report, edge target coverage) as “Validation results”.
- **Phase 2:** **Results** section describing what **04_smoke_tests.cypher** returns: node counts (cases, acts, sections, articles), relationship counts (IN_ACT, CITES), top 20 cited targets, acts linkage distribution. Include a note: “Run the smoke tests after loading to get your instance’s counts; use these numbers in reports/PPT.”
- **Phase 3:** **Results** section is critical. Use **[phase3_embeddings/results.txt](phase3_embeddings/results.txt)** as the single source of truth:
  - **Baseline (unfinetuned BGE):** Full metric block (accuracy@1/3/5/10, map@100, mrr@10, ndcg@10, precision@10, recall@10) — copy exact numbers from results.txt.
  - **Run 1, Run 2, Run 3:** For each run, document the key IR metrics (MRR@10, NDCG@10, Recall@10) and the **comparison to baseline** (baseline → final, delta). Run 3 is the best (current model at `phase3_embeddings/models/bge-legal`).
  - **Training setup:** 2 epochs, lr=1e-5, batch_size=32, MultipleNegativesRankingLoss, 80/20 train/eval split (from results.txt Notes and [finetune_bge.py](phase3_embeddings/finetune_bge.py)).
  - Present as a clear table or bullet list suitable for reports/PPT (e.g. “Phase 3 – Fine-tuning results: Baseline vs Run 1 / Run 2 / Run 3”).
- **Phase 4:** **Results** section: reference [test_three_queries.py](test_three_queries.py) — the three example queries (“Explain Article 14…”, “What does Section 302 of BNS say?”, “Someone entered my home…”). Document that a run produces: rephrased legal query (if different), final answer, and retrieval mix (counts by source_type). Suggest: “Run `python test_three_queries.py` and paste sample output (or screenshot) for reports/PPT.” No need to invent answer text; the doc explains what the run shows and where to get live results.

---

## 2. Datasets reference (1 file)

**File:** [Documentation/Datasets.md](Documentation/Datasets.md)

- **Contents:** Only datasets **actually used** in the project (as identified above):
  - **legal_data_train.csv** (Phase 1, when using CSV source): path (`BASE_PATH/legal_data_train.csv`), purpose, schema (case_id, judgment text column, year column), and **sample rows** (e.g. `head()` output). If file is missing, document expected schema and note “run script to capture head when available”.
  - **SC judgments PDFs** (Phase 1, when using PDF source): path pattern `BASE_PATH/SC_EXTRACTED_DIR/<year>/*.pdf`, structure, how case_id/year are derived.
  - **Act PDFs** (Phase 1): Constitution, BNS, BNSS, BSA from [config.py](config.py) `PDF_FILES`.
  - **Phase 1 output CSVs** (acts, sections, articles, cases, edges): consumed by Phase 2 and Phase 3; schema (column names and meaning).
  - **IndicLegalQA Dataset_10K_Revised.json** (Phase 3): path, structure (question/answer), usage (fine-tuning and eval split).
- **Script (optional):** Add `scripts/doc_dataset_samples.py` (or under `Documentation/`) that: reads each dataset that exists (legal_data_train.csv, Phase 1 CSVs, IndicLegalQA JSON), prints `head()` or first N items and shape/schema, and optionally writes a snippet to `Documentation/dataset_samples.txt` for pasting into Datasets.md. Document in Datasets.md: “To refresh samples, run: `python scripts/doc_dataset_samples.py`.”

---

## 3. Architecture docs (3 files)

- **Documentation/Architecture.md** – **Whole system**
  - End-to-end flow: Phase 1 → Phase 2 (Neo4j) and Phase 1 → Phase 3 (chunks, FAISS, model) → Phase 4 (LangGraph + Ollama).
  - One Mermaid diagram: high-level boxes (Ingestion, Neo4j KG, Embeddings & FAISS, RAG App) and data flows (CSVs, vectors, graph).
  - Technologies: Python, Neo4j, FAISS, Sentence Transformers, LangGraph, Ollama. Keep high-level and clear.
- **Documentation/LangGraph_Architecture.md** – **RAG workflow only**
  - State (`WorkflowState`), entry point (`query_parser`), nodes (query_parser, graph_retriever, query_rephrase, vector_retriever, answer_generator), edges (linear flow).
  - What each node does (inputs/outputs, key functions). Reference [langgraph_workflow_v3.py](phase4_rag/langgraph_workflow_v3.py) and config.
  - One Mermaid flowchart: nodes and edges with short labels (e.g. “Parse refs”, “Neo4j sections/cases”, “Rephrase”, “FAISS + constraints”, “Ollama answer”).
- **Documentation/Knowledge_Graph.md** – **Neo4j schema and usage**
  - Node labels: Act, Section, Article, Case. Properties (act_id, section_id, article_number, full_text, case_id, year, judgment_text, etc.) from [01_constraints.cypher](neo4j/cypher/01_constraints.cypher) and [02_load_nodes.cypher](neo4j/cypher/02_load_nodes.cypher).
  - Relationships: IN_ACT (Section/Article → Act), CITES (Case → Section|Article) with optional `count`.
  - How the graph is built (Phase 2 Cypher) and queried in RAG ([neo4j_client_v3.py](phase4_rag/neo4j_client_v3.py): get_sections_by_numbers, get_articles_by_numbers, get_cases_citing_ids; Act-aware filtering).
  - One Mermaid diagram: entity-relationship style (nodes and relationship types).

---

## 4. Architecture images for PPTs

- **In-repo:** Each architecture MD will contain **Mermaid code blocks** (no spaces in node IDs; quoted labels for special chars). These render in GitHub, VS Code (with Mermaid extension), and many viewers.
- **Export to image:**
  - **Option A:** Add a small script `Documentation/export_architecture_images.py` (or `.ps1`/shell) that:
    - Writes 3 `.mmd` files (e.g. `architecture_overview.mmd`, `langgraph_workflow.mmd`, `knowledge_graph_schema.mmd`) with the same Mermaid source as in the MDs, and
    - Invokes **@mermaid-js/mermaid-cli** (`mmdc -i file.mmd -o file.png`) if available, or
    - Prints instructions: “Install @mermaid-js/mermaid-cli (npm install -g @mermaid-js/mermaid-cli) and run: mmdc -i Documentation/architecture_overview.mmd -o Documentation/images/architecture_overview.png” (and similarly for the other two).
  - **Option B:** Document in each MD: “To use in PPT: copy the Mermaid block to [Mermaid Live Editor](https://mermaid.live) and export as PNG/SVG.”
- Recommendation: **Option A** with mmdc if npm is acceptable; otherwise **Option B** in the docs. Create `Documentation/images/` for exported PNGs/SVGs.

---

## 5. File list summary


| #   | Path                                                  | Purpose                                                                                                            |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 1   | Documentation/Phase1_Ingestion.md                     | Phase 1: What, How, Why; datasets used; components; **Results** (pipeline/verify output).                          |
| 2   | Documentation/Phase2_Neo4j.md                         | Phase 2: What, How, Why; CSVs loaded; Cypher scripts; **Results** (smoke-test counts).                             |
| 3   | Documentation/Phase3_Embeddings.md                    | Phase 3: What, How, Why; datasets used; chunking, BGE, FAISS; **Results** (Baseline + Run 1/2/3 from results.txt). |
| 4   | Documentation/Phase4_RAG.md                           | Phase 4: What, How, Why; single narrative; **Results** (example queries and run output).                           |
| 5   | Documentation/Datasets.md                             | All datasets actually used; schema; head()/samples; script to refresh.                                             |
| 6   | Documentation/Architecture.md                         | End-to-end architecture; Mermaid; technologies.                                                                    |
| 7   | Documentation/LangGraph_Architecture.md               | LangGraph state, nodes, edges; Mermaid.                                                                            |
| 8   | Documentation/Knowledge_Graph.md                      | Neo4j schema, relationships, build & query; Mermaid.                                                               |
| 9   | Documentation/export_architecture_images.* (optional) | Script + .mmd files and/or instructions to produce PNG/SVG for PPTs.                                               |


---

## 6. Conventions

- **Tone:** Technical but clear; avoid jargon without definition. Content should be **report- and PPT-ready** (insightful, high value).
- **No duplication:** Phase 4 doc is the single place for RAG workflow and design; Phase 1–3 docs do not reiterate “Phase 4 changes.”
- **Datasets:** Only list datasets that are **read** in the codebase (legal_data_train or PDFs, Phase 1 CSVs, IndicLegalQA JSON); exclude anything that only “exists” in a folder.
- **Diagrams:** Mermaid only (no inline HTML/JS). Node IDs: camelCase or underscores; quoted labels for “(e.g. …)” or special characters. Store diagram source in MD and, for images, optionally in `.mmd` files for mmdc.
- **Results:** Include showable results in each phase doc (Phase 1 pipeline/verify output; Phase 2 smoke-test counts; Phase 3 baseline + multiple fine-tuning runs from results.txt; Phase 4 example queries and run output). Reference code, [phase3_embeddings/results.txt](phase3_embeddings/results.txt), terminal output, and agent chats as needed.

---

## 7. Optional script for dataset samples

- **Script:** `scripts/doc_dataset_samples.py` (or `Documentation/scripts/doc_dataset_samples.py`).
- **Behavior:** Load from paths in [config.py](config.py) and [phase3_embeddings/config.py](phase3_embeddings/config.py); for each existing file: print shape/schema and first 5 rows (or first 3 Q/A for JSON); optionally append to `Documentation/dataset_samples.txt`.
- **Doc:** In [Documentation/Datasets.md](Documentation/Datasets.md), include or reference this output and the command to re-run.

This plan gives you 4 phase docs, 1 dataset doc, 3 architecture docs, and optional image export and dataset-sample script, all aligned with the codebase and your “technical but easy to understand” and “only datasets actually used” requirements.