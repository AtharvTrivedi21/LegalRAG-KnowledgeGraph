# Phase 2: Neo4j Knowledge Graph Load

## What

Phase 2 **loads the LegalRAG knowledge graph into Neo4j**. It does not read any datasets from the project folder directly; it assumes Phase 1 CSVs have been **copied into Neo4j’s `import/` directory**. The Cypher scripts create constraints and indexes, create nodes (Act, Section, Article, Case) with `IN_ACT` relationships, then create `CITES` relationships from cases to sections/articles. Smoke tests are provided to verify counts and topology.

## How

### Prerequisites

1. Phase 1 must have been run so that `phase1_output/` contains: `acts.csv`, `sections.csv`, `articles.csv`, `cases.csv`, `edges.csv`.
2. Copy these five files into Neo4j’s `import/` directory (e.g. via `scripts/copy_phase1_csvs_to_neo4j_import.ps1` or manually).
3. Run the Cypher scripts **in order** in Neo4j Browser or cypher-shell.

### Scripts (in order)

| Script | Purpose |
|--------|---------|
| `neo4j/cypher/01_constraints.cypher` | Creates unique constraints on `case_id`, `act_id`, `section_id`, `article_id`; optional indexes on `Case.year`, `Section.section_number`, `Article.article_number`. Run once per database. |
| `neo4j/cypher/02_load_nodes.cypher` | LOAD CSV from `file:///acts.csv`, `sections.csv`, `articles.csv`, `cases.csv`. Creates Act, Section, Article, Case nodes and `(Section)-[:IN_ACT]->(Act)`, `(Article)-[:IN_ACT]->(Act)`. Uses `IN TRANSACTIONS` for batching. |
| `neo4j/cypher/03_load_edges.cypher` | LOAD CSV from `file:///edges.csv`. Matches Case by `source_case_id`, Section/Article by `target_section_or_article`; creates `(Case)-[:CITES]->(Section|Article)`. Aggregates duplicate edges into relationship property `count`. |
| `neo4j/cypher/04_smoke_tests.cypher` | Read-only queries: node counts, relationship counts, top 20 cited targets, acts linkage distribution. Use to verify load and to obtain numbers for reports. |

### Datasets used (inputs)

The **only** inputs are the five CSVs in Neo4j’s `import/` directory, produced by Phase 1:

- `acts.csv` — act_id, act_name, act_type, source_file  
- `sections.csv` — section_id, act_id, section_number, full_text  
- `articles.csv` — article_id, act_id, article_number, full_text  
- `cases.csv` — case_id, judgment_text, year  
- `edges.csv` — source_case_id, target_section_or_article, relation  

No other datasets are read in Phase 2.

## Why

- **Constraints first:** Uniqueness on IDs prevents duplicate nodes and supports fast MERGE.
- **IN_ACT:** Sections and articles are explicitly linked to their Act for Act-aware querying in Phase 4.
- **CITES with count:** Duplicate citations (same case → same section) are merged and counted, useful for analytics and ranking.
- **Batched transactions:** `IN TRANSACTIONS OF N ROWS` keeps memory and lock usage under control for large CSVs.

## Results

After loading, run **04_smoke_tests.cypher** to get numbers suitable for reports or PPTs:

1. **Node counts:** `cases`, `acts`, `sections`, `articles`.
2. **Relationship counts:** `inActRels`, `citesRels`.
3. **Top 20 cited targets:** `labels`, `target_id`, `cites` (by sum of `count`).
4. **Acts linkage:** `nodeType`, `act_id`, `n` (how many sections/articles per act).

Example (run in Neo4j and paste/screenshot for docs):

```cypher
MATCH (c:Case) RETURN count(c) AS cases;
MATCH (a:Act) RETURN count(a) AS acts;
MATCH (s:Section) RETURN count(s) AS sections;
MATCH (ar:Article) RETURN count(ar) AS articles;
MATCH ()-[r:IN_ACT]->() RETURN count(r) AS inActRels;
MATCH ()-[r:CITES]->() RETURN count(r) AS citesRels;
```

**Note:** Run the smoke tests on your instance after each load to get your actual counts; use these numbers in reports and PPTs.
