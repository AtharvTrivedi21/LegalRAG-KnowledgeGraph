# Graph-Constrained Legal RAG for Indian Law — Project Plan

**MTech Project | Phase 1–3**

---

## Overview

Build a **Graph-Constrained RAG** system for Indian law that combines:
- A **Knowledge Graph** (Neo4j) of cases, acts, sections, articles, and citation edges
- **Vector retrieval** (embeddings + FAISS) for semantic search
- **LangGraph** orchestration for multi-step RAG with graph constraints

Scope: Supreme Court judgments (2016–2025), Constitution of India, BNS, BNSS, BSA.

---

## From the Beginning: What We Did

### Setup (First)

1. **Virtual environment** (all deps in venv only)
   ```powershell
   cd c:\Users\ATHARV\LegalRAG
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Dependencies** (`requirements.txt`): `pandas`, `pdfplumber`, `matplotlib`, `cryptography<42`

3. **Config** (`config.py`): `BASE_PATH`, `OUTPUT_PATH`, `JUDGMENTS_SOURCE`, `SC_EXTRACTED_DIR`, `PDF_FILES`, `DATA_LIMIT`, `ACT_NAMES`

### Data Sources

- **Judgments**: `Datasets/SC_Judgements-16-25/<year>/*.pdf` (extracted) or `legal_data_train.csv` (fast fallback)
- **Acts**: `Constitution Of India.pdf`, `1_BNS.pdf`, `2 Bharatiya nagrik Suraksha sanhita.pdf`, `3 Bharatiya Sakshya Adhiniyam.pdf`

---

## Phase 1: Data Engineering + Legal NLP Exploration (Completed)

**Goal:** Extract, preprocess, and produce Neo4j-ready CSVs. No ML, no embeddings, no Neo4j.

### 1.1 Venv and Setup

- `python -m venv venv` → `venv\Scripts\activate` → `pip install -r requirements.txt`
- No global installs; all packages live in `venv/`

### 1.2 Load Judgments

**Source modes:**
- `JUDGMENTS_SOURCE = "pdf"`: Read from `Datasets/SC_Judgements-16-25/<year>/*.pdf` (slow; uses pdfplumber)
- `JUDGMENTS_SOURCE = "csv"`: Read from `legal_data_train.csv` (fast; recommended for quick KG prototype)

**Logic:**
- For PDFs: extract text per file, use folder name as `year`, filename stem as `case_id`
- For CSV: map `Text` → `judgment_text`, assign `case_id` as `case_{i}`, `year` = 0
- Output columns: `case_id`, `judgment_text`, `year`

### 1.3 Inspect Schema and Stats

- Shape, columns, dtypes, null counts
- Cases per year, average text length
- Plot: judgments per year (matplotlib bar chart)

### 1.4 PDF Extraction (Acts)

For each PDF (Constitution, BNS, BNSS, BSA):
- Extract text via `pdfplumber`
- Split into Articles (Constitution) or Sections (BNS, BNSS, BSA) using regex
- Build structured tables: `act_name`, `article_or_section_number`, `full_text`

**Regex patterns:**
- Article: `Article\s+(\d+(?:\(\d+\))?)`
- Section: `Section\s+(\d+(?:\(\d+\))?)`

### 1.5 Build Acts / Sections / Articles Tables

- `acts.csv`: `act_id`, `act_name`, `act_type`, `source_file`
- `articles.csv`: `article_id`, `act_id`, `article_number`, `full_text` (Constitution)
- `sections.csv`: `section_id`, `act_id`, `section_number`, `full_text` (BNS, BNSS, BSA)

IDs: `Constitution_Art_14`, `BNS_Sec_302`, etc.

### 1.6 Citation Edge Extraction

From `judgment_text`, regex extract:
- `Section X` → match to `sections.csv`
- `Article X` → match to `articles.csv`

Build `edges.csv`: `source_case_id`, `target_section_or_article`, `relation` (CITES)

### 1.7 Export and Plots

- Export: `cases.csv`, `sections.csv`, `articles.csv`, `acts.csv`, `edges.csv` to `phase1_output/`
- Plots: `judgments_per_year.png`, `top_cited_sections.png`, `top_cited_articles.png`

### 1.8 Files Created (Phase 1)

| File | Purpose |
|------|---------|
| `config.py` | Paths, source mode, file names |
| `src/judgments.py` | Load judgments (PDF or CSV) |
| `src/pdf_extractor.py` | Extract Articles/Sections from PDFs |
| `src/edges.py` | Citation extraction, edges table |
| `src/export.py` | Export CSVs, save plots |
| `src/run_pipeline.py` | Entry point |

**Run:** `python src/run_pipeline.py`

---

## Phase 2: Verify Outputs + Build Local Neo4j Knowledge Graph (Completed)

**Goal:** Turn Phase-1 CSVs into a Neo4j Knowledge Graph. No embeddings/FAISS/LangGraph.

### 2.1 Pre-import Verification

**Script:** `scripts/verify_phase1_output.py`

Checks:
- All 5 CSVs exist
- Required columns present
- Uniqueness / duplicates (warn)
- Foreign keys: `sections.act_id`, `articles.act_id` in acts
- Edge validity: sources in cases, targets in sections/articles

**Run:** `python scripts\verify_phase1_output.py --output-dir .\phase1_output`

### 2.2 Neo4j Desktop Setup

1. Install Neo4j Desktop (https://neo4j.com/download/)
2. Create project → Create DBMS (e.g. LegalRAG-KnowledgeGraph)
3. Start the DBMS
4. Open Query tool (Neo4j Browser)

### 2.3 Copy CSVs into Neo4j `import/` Directory

**Location:** `.Neo4jDesktop2` → `Data` → `dbmss` → `dbms-<id>` → create `import/` if missing

Copy into `import/`:
- `acts.csv`, `sections.csv`, `articles.csv`, `cases.csv`, `edges.csv`

**Helper script:**
```powershell
.\scripts\copy_phase1_csvs_to_neo4j_import.ps1 -ImportDir "C:\Users\ATHARV\.Neo4jDesktop2\Data\dbmss\dbms-<id>\import"
```

### 2.4 Run Cypher Scripts (Order Matters)

In Neo4j Query tool, run in order:

1. **01_constraints.cypher** — Uniqueness constraints for Case, Act, Section, Article; indexes for year, section_number, article_number
2. **02_load_nodes.cypher** — LOAD CSV: Act, Section, Article, Case nodes + IN_ACT relationships
3. **03_load_edges.cypher** — LOAD CSV: CITES relationships (aggregates duplicates into `r.count`)
4. **04_smoke_tests.cypher** — Validation queries (counts, top cited, act linkage)

**Note:** Neo4j 5.x uses `CALL { ... } IN TRANSACTIONS OF N ROWS` (not `USING PERIODIC COMMIT`).

### 2.5 Graph Schema (Result)

**Nodes:**
- `(:Case)` — case_id, judgment_text, year
- `(:Act)` — act_id, act_name, act_type, source_file
- `(:Section)` — section_id, act_id, section_number, full_text
- `(:Article)` — article_id, act_id, article_number, full_text

**Relationships:**
- `(:Case)-[:CITES]->(:Section|:Article)` (optional: `count` property)
- `(:Section)-[:IN_ACT]->(:Act)`, `(:Article)-[:IN_ACT]->(:Act)`

### 2.6 Files Created (Phase 2)

| File | Purpose |
|------|---------|
| `scripts/verify_phase1_output.py` | Validate Phase-1 CSVs |
| `scripts/copy_phase1_csvs_to_neo4j_import.ps1` | Copy CSVs to Neo4j import |
| `neo4j/cypher/01_constraints.cypher` | Constraints and indexes |
| `neo4j/cypher/02_load_nodes.cypher` | Node and IN_ACT import |
| `neo4j/cypher/03_load_edges.cypher` | CITES import |
| `neo4j/cypher/04_smoke_tests.cypher` | Smoke-test queries |
| `neo4j/README.md` | Phase-2 instructions |

---

## Phase 3: Graph-Constrained RAG (Planned)

**Goal:** Build the RAG pipeline that uses the KG to constrain retrieval, then retrieves text chunks via embeddings, and orchestrates via LangGraph.

### 3.1 Components

1. **Neo4j driver** — Python (`neo4j` package) to run Cypher queries
2. **Graph retrieval** — Traverse CITES, IN_ACT to find relevant cases, sections, articles
3. **Embeddings** — Sentence-transformers (e.g. `all-MiniLM-L6-v2` or legal-domain model)
4. **Vector store** — FAISS (or Chroma) for chunk embeddings
5. **Chunking** — Split `judgment_text`, `full_text` into chunks for embedding
6. **LangGraph** — Multi-step workflow: parse query → graph filter → vector retrieval → LLM generation

### 3.2 Hybrid Retrieval Flow

```
User query
    ↓
Parse / classify (e.g. "Section 302 IPC", "Article 14")
    ↓
Graph filter: Cypher to find cases citing those sections/articles
    ↓
Vector retrieval: Embed query, fetch top-k chunks from FAISS (optionally filtered by case_ids from graph)
    ↓
LLM: Combine context chunks + graph context → generate answer
```

### 3.3 Implementation Tasks (Phase 3)

| Task | Description |
|------|-------------|
| Add deps | `neo4j`, `sentence-transformers`, `faiss-cpu`, `langgraph`, `langchain` |
| Neo4j connector | Connect to local Neo4j, run graph queries |
| Chunking | Split cases/sections/articles into overlapping chunks |
| Embedding pipeline | Embed chunks, build FAISS index |
| Graph retrieval | Cypher queries for citation paths |
| LangGraph agent | Define nodes: query_router, graph_retriever, vector_retriever, LLM; wire into graph |
| API / CLI | Simple interface to ask questions |

### 3.4 Risks and Notes

- Large `cases.csv` → chunking and embedding may be memory-heavy; consider batching
- Legal-domain embeddings may improve accuracy over general models
- LangGraph state management: design schema for query, graph results, vector results, final answer

---

## Summary

| Phase | Status | Output |
|-------|--------|--------|
| 1 | Done | CSVs, plots in `phase1_output/` |
| 2 | Done | Neo4j KG with nodes and CITES edges |
| 3 | Planned | Graph-Constrained RAG with LangGraph |

---

## Quick Commands

```powershell
# Phase 1
cd c:\Users\ATHARV\LegalRAG
venv\Scripts\activate
python src/run_pipeline.py

# Verify
python scripts\verify_phase1_output.py --output-dir .\phase1_output

# Phase 2: Copy CSVs, then run 01 → 02 → 03 → 04 in Neo4j Query
```
