# Phase 2 Neo4j v2 — Loading Phase 1 v2 data

Phase 1 v2 produces CSVs with PLAN_1 IDs (e.g. `BNS_2023`, `BNSS_2023`, `CONST_1950`, `BNS_s64`, `CONST_1950_Art21`). This doc describes how to load them into Neo4j using the v2 Cypher scripts.

## Prerequisites

- Neo4j 5.x (Desktop or Server).
- Phase 1 v2 run completed: `phase1_output_v2/` contains the CSVs.

## 1. Allow CSV import from file URLs

In Neo4j config (e.g. Neo4j Desktop → Manage → Settings, or `neo4j.conf`):

```conf
dbms.security.allow_csv_import_from_file_urls=true
```

Restart Neo4j after changing this.

## 2. Copy CSVs into Neo4j import directory

Copy all required CSVs from `phase1_output_v2/` into Neo4j’s `import/` directory.

**Neo4j Desktop:** right‑click the database → Open Folder → Import, then copy files there.

**Neo4j Server:** copy into the `import` directory configured in `neo4j.conf` (e.g. `<neo4j-home>/import/`).

Required files (same names as in `phase1_output_v2/`):

- `acts.csv`
- `parts.csv`
- `chapters.csv`
- `sections.csv`
- `articles.csv`
- `definitions.csv` (can be empty)
- `cases_sc.csv`
- `cases_iltur.csv`
- `case_cites_section.csv`
- `case_cites_article.csv`
- `section_references_section.csv`
- `section_defines_term.csv` (if present)
- `act_index.csv` (optional, for Phase 4 act index lookups)

Paths in the Cypher scripts use `file:///` and assume files are in the default `import/` folder (e.g. `file:///acts.csv`). If your import path differs, adjust the paths in `02_load_nodes_v2.cypher` and `03_load_edges_v2.cypher`.

## 3. Run Cypher scripts in order

Use Neo4j Browser or `cypher-shell`. Run in this order:

1. **01_constraints_v2.cypher** — constraints and indexes (Act, Part, Chapter, Section, Article, Definition, Case).
2. **02_load_nodes_v2.cypher** — load nodes from CSVs (acts, parts, chapters, sections, articles, definitions, cases).
3. **03_load_edges_v2.cypher** — load relationships (HAS_PART, HAS_CHAPTER, HAS_SECTION, HAS_ARTICLE, DEFINES_TERM, REFERENCES, CITES).

If you already have an older Phase 2 graph and want to replace it: clear the database (e.g. delete all nodes/relationships and drop old constraints), then run the three v2 scripts above.

## 4. Verify

Example checks in Neo4j Browser:

```cypher
MATCH (a:Act) RETURN a.act_id, a.act_name LIMIT 10;
MATCH (s:Section) RETURN s.section_id, s.act_id LIMIT 5;
MATCH (c:Case)-[r:CITES]->() RETURN type(r), count(r) LIMIT 1;
```

Phase 4 RAG uses `act_id` values `BNS_2023`, `BNSS_2023`, `BSA_2023`, `CONST_1950` (see `phase4_rag/query_parser_v3.py`). After loading v2 data, queries that resolve acts will use these IDs.

## Optional: smoke test script

You can add a script (e.g. `04_smoke_tests_v2.cypher`) that runs counts and sample queries to confirm node and relationship counts match expectations.
