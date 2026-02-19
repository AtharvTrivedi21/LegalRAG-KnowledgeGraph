# Phase-2: Local Neo4j (Desktop) KG Build

This folder contains the Cypher scripts to import Phase-1 outputs into a **local Neo4j Desktop** database using `LOAD CSV`.

## Prerequisites

- Phase-1 outputs exist in `c:\Users\ATHARV\LegalRAG\phase1_output\`
- Neo4j Desktop installed
- A running Neo4j DBMS (Neo4j 5.x recommended)

## 1) Find Neo4j `import/` directory (Desktop)

In Neo4j Desktop:
- Open your DBMS → **Manage** → **Open Folder** (or similar)
- Locate the `import/` directory inside the DBMS directory

## 2) Copy CSVs into Neo4j `import/`

Copy these 5 files into Neo4j’s `import/` directory:
- `acts.csv`
- `sections.csv`
- `articles.csv`
- `cases.csv`
- `edges.csv`

All are produced by Phase-1 at:
`c:\Users\ATHARV\LegalRAG\phase1_output\`

### Optional: use the helper copy script (recommended)

From PowerShell:

```powershell
cd c:\Users\ATHARV\LegalRAG
.\scripts\copy_phase1_csvs_to_neo4j_import.ps1 -ImportDir "C:\path\to\neo4j\import"
```

## 3) Run Cypher scripts (order matters)

Open Neo4j Browser and run these scripts in order (copy/paste contents):

1. `cypher/01_constraints.cypher`
2. `cypher/02_load_nodes.cypher`
3. `cypher/03_load_edges.cypher`
4. `cypher/04_smoke_tests.cypher`

## Notes (important)

- `sections.csv` / `articles.csv` can contain duplicate IDs due to PDF parsing; the import uses `MERGE` and only fills missing text.
- `edges.csv` can contain many duplicates; `03_load_edges.cypher` aggregates them into relationship property `r.count`.

