---
name: Phase-2 Neo4j KG Build
overview: After Phase-1 CSV generation, add a verification script and build the Legal Knowledge Graph in local Neo4j Desktop using LOAD CSV, with constraints/indexes and smoke-test Cypher queries. No embeddings/FAISS yet.
todos:
  - id: verify-script
    content: Create/run scripts/verify_phase1_output.py to validate Phase-1 CSV schemas, IDs, and referential integrity
    status: completed
  - id: neo4j-desktop-setup
    content: Set up Neo4j Desktop DBMS (Neo4j 5.x) and locate the import/ directory
    status: completed
  - id: copy-csvs
    content: Copy Phase-1 CSVs into Neo4j import/ directory for LOAD CSV
    status: completed
  - id: schema-constraints
    content: Create Neo4j constraints and indexes for Case/Act/Section/Article IDs and year
    status: completed
  - id: load-nodes
    content: LOAD CSV to create Act/Section/Article/Case nodes and IN_ACT relationships
    status: completed
  - id: load-edges
    content: LOAD CSV to create CITES/REFERS relationships from edges.csv
    status: completed
  - id: smoke-tests
    content: Run validation Cypher queries (counts, top cited, act linkage) and confirm expected results
    status: completed
isProject: false
---

# Phase-2: Verify Phase-1 Outputs + Build Local Neo4j Knowledge Graph (Desktop + LOAD CSV)

## Goal

Turn Phase‑1 outputs in `[c:\Users\ATHARV\LegalRAG\phase1_output\](c:\Users\ATHARV\LegalRAG\phase1_output\)` into a **local Neo4j** Knowledge Graph (KG) suitable for later Graph‑Constrained RAG work.

Scope here:

- Verify CSV integrity (no ML)
- Create graph schema (labels, relationships)
- Bulk import via **Cypher `LOAD CSV`**
- Run smoke-test queries to validate

Not in scope yet:

- Embeddings, FAISS, vector indexes, LangGraph orchestration

---

## Inputs and Outputs

### Inputs (from Phase‑1)

- `cases.csv`: `case_id, judgment_text, year`
- `acts.csv`: `act_id, act_name, act_type, source_file`
- `sections.csv`: `section_id, act_id, section_number, full_text`
- `articles.csv`: `article_id, act_id, article_number, full_text`
- `edges.csv`: `source_case_id, target_section_or_article, relation`

### Outputs (Phase‑2)

- A Neo4j database populated with:
  - `(:Case)` nodes
  - `(:Act)` nodes
  - `(:Section)` and `(:Article)` nodes
  - Relationships:
    - `(:Case)-[:CITES]->(:Section|:Article)` (from `edges.csv`)
    - `(:Section)-[:IN_ACT]->(:Act)` and `(:Article)-[:IN_ACT]->(:Act)`

---

## A. Pre-import Verification Script (Python)

Use `scripts/verify_phase1_output.py` (runs in your existing venv) to verify Phase‑1 outputs.

What it should check/report:

- **File presence**: ensure all 5 CSVs exist
- **Schema checks**: required columns present; no unexpected null-heavy ID columns
- **Uniqueness / duplicates (warn, not fail by default)**:
  - Phase‑1 can produce duplicate `section_id` / `article_id` rows due to PDF extraction layout.
  - Phase‑1 can produce duplicate rows in `edges.csv` because a case may mention the same target multiple times.
  - Your Neo4j import must therefore use `MERGE` (and optionally aggregate relationship counts).
- **Foreign key checks**:
  - every `sections.act_id` exists in `acts.act_id`
  - every `articles.act_id` exists in `acts.act_id`
- **Edge validity**:
  - `edges.source_case_id` exists in cases
  - `edges.target_section_or_article` exists in `sections.section_id` OR `articles.article_id`
  - `edges.relation` should be `CITES` (Phase‑1 currently outputs only `CITES`)
- **Basic stats report** (printed): counts of nodes/edges, top cited targets, year distribution
- **Exit codes**:
  - exits non‑zero on hard failures (missing columns, broken FK integrity)

Run command:

```powershell
cd c:\Users\ATHARV\LegalRAG
venv\Scripts\activate
python scripts\verify_phase1_output.py --output-dir .\phase1_output
```

---

## B. Neo4j Desktop Setup (Local)

1. Install Neo4j Desktop.
2. Create a project (e.g. `LegalRAG`).
3. Create a DBMS (Neo4j 5.x recommended).
4. Start the DBMS.

### Place CSVs where Neo4j can read them

For `LOAD CSV`, Neo4j reads from the **Neo4j import directory**.

- Copy these files into Neo4j’s `import/` folder:
  - `cases.csv`, `acts.csv`, `sections.csv`, `articles.csv`, `edges.csv`

(Neo4j Desktop shows the DBMS path; import folder is inside it.)

---

## C. Graph Schema (Constraints + Indexes)

Run these once (or use `IF NOT EXISTS`):

- **Uniqueness constraints**:
  - `Case(case_id)`
  - `Act(act_id)`
  - `Section(section_id)`
  - `Article(article_id)`
- **Indexes** (optional but helpful):
  - `Case(year)`
  - `Section(section_number)`
  - `Article(article_number)`

---

## D. Import Cypher (LOAD CSV)

Import order:

1. `Act`
2. `Section` and `Article` and link to `Act`
3. `Case`
4. `CITES` edges

Notes:

- Use `USING PERIODIC COMMIT`.
- Use `MERGE` on IDs.
- Keep long texts as node properties (`judgment_text`, `full_text`).
- Because `sections.csv` / `articles.csv` may contain duplicate IDs, ensure your node imports:
  - `MERGE` by `section_id` / `article_id`
  - set properties with care (e.g., prefer first-seen, or only set if property is null/empty)
- Because `edges.csv` contains many duplicate rows, you have two good options:
  - **Option A (simple)**: `MERGE` the relationship so duplicates don’t create parallel relationships.
  - **Option B (recommended)**: aggregate duplicates into a `count` property:
    - `MERGE (c)-[r:CITES]->(t) ON CREATE SET r.count = 1 ON MATCH SET r.count = r.count + 1`

---

## E. Smoke-test Queries (Validation)

After import:

- **Counts**:
  - `MATCH (c:Case) RETURN count(c)`
  - `MATCH (s:Section) RETURN count(s)`
  - `MATCH (a:Article) RETURN count(a)`
  - `MATCH ()-[r:CITES]->() RETURN count(r)`
- **Top cited**:
  - `MATCH (:Case)-[:CITES]->(t) RETURN labels(t), t.section_id, t.article_id, count(*) AS n ORDER BY n DESC LIMIT 20`
- **Act linkage**:
  - `MATCH (s:Section)-[:IN_ACT]->(act:Act) RETURN act.act_id, count(s) ORDER BY count(s) DESC`

---

## F. Handoff for Graph-Constrained RAG (later)

Once KG is correct, Phase‑3 can layer:

- Graph traversal constraints (Act/Section/Article subgraph filters)
- Hybrid retrieval: graph filter → text chunk retrieval (embeddings/FAISS)
- LangGraph orchestration

---

## Risks / Practical Notes

- `LOAD CSV` requires files in Neo4j `import/` dir (or enabling file URLs). Keeping in import/ is simplest.
- Importing very large `cases.csv` with full `judgment_text` may be slow and memory-heavy; it’s still workable locally but may require batching.
- If Phase‑1 re-runs, you’ll re-copy CSVs and either `MATCH (n) DETACH DELETE n` (dev only) or import into a fresh database.

