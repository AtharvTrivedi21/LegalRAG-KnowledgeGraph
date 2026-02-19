# Phase 1: Data Ingestion Pipeline

## What

Phase 1 is the **data ingestion and preparation** stage of LegalRAG. It loads Supreme Court judgments and statutory text (Constitution, BNS, BNSS, BSA), extracts structured sections and articles, builds **citation edges** from cases to cited laws, and exports **Neo4j-ready CSVs** plus summary plots. All downstream phases (Neo4j load, chunking, RAG) depend on this output.

## How

### Data flow

1. **Load judgments** — From either CSV (`legal_data_train.csv`) or extracted PDFs (`SC_Judgements-16-25/<year>/*.pdf`), producing a DataFrame with `case_id`, `judgment_text`, `year`.
2. **Extract acts/sections/articles** — Parse Constitution and act PDFs with regex to produce `acts`, `sections`, and `articles` tables with stable IDs (e.g. `Constitution_Art_14`, `BNS_Sec_302`).
3. **Build citation edges** — Scan each judgment text for "Section X" and "Article X" mentions; match to existing section/article IDs; emit edges `(source_case_id, target_section_or_article, relation)`.
4. **Export** — Write five CSVs to `phase1_output/` and generate plots (judgments per year, top cited sections/articles).

### Components

| Component | File | Role |
|-----------|------|------|
| Entry point | `src/run_pipeline.py` | Orchestrates the four steps; reads `config.py`. |
| Judgments loader | `src/judgments.py` | Loads from CSV (column `Text`) or from extracted PDFs; maps to `case_id`, `judgment_text`, `year`. |
| PDF extractor | `src/pdf_extractor.py` | Uses pdfplumber + regex (`ARTICLE_PATTERN`, `SECTION_PATTERN`) to extract articles/sections and full text from act PDFs. |
| Edge builder | `src/edges.py` | `extract_citations_from_text()` finds Section/Article refs; `build_edges_df()` maps to valid section/article IDs and outputs CITES edges. |
| Export | `src/export.py` | Writes `cases.csv`, `sections.csv`, `articles.csv`, `acts.csv`, `edges.csv`; generates `judgments_per_year.png`, `top_cited_sections.png`, `top_cited_articles.png`. |
| Config | `config.py` | `BASE_PATH`, `OUTPUT_PATH` (`phase1_output`), `JUDGMENTS_SOURCE` (csv/pdf), `SC_EXTRACTED_DIR`, `PDF_FILES`, `ACT_NAMES`, `DATA_LIMIT`. |

### Datasets used

- **Judgments:** Either `BASE_PATH/legal_data_train.csv` (uses column `Text`; `JUDGMENTS_SOURCE=csv`) or `BASE_PATH/SC_EXTRACTED_DIR/<year>/*.pdf` (folder name = year, filename stem = case_id).
- **Acts:** PDFs under `BASE_PATH` from `config.PDF_FILES`: Constitution Of India.pdf, 1_BNS.pdf, 2 Bharatiya nagrik Suraksha sanhita.pdf, 3 Bharatiya Sakshya Adhiniyam.pdf.

### Output schema (Neo4j-ready)

- **cases.csv:** `case_id`, `judgment_text`, `year`
- **sections.csv:** `section_id`, `act_id`, `section_number`, `full_text`
- **articles.csv:** `article_id`, `act_id`, `article_number`, `full_text`
- **acts.csv:** `act_id`, `act_name`, `act_type`, `source_file`
- **edges.csv:** `source_case_id`, `target_section_or_article`, `relation` (CITES)

## Why

- **Dual judgment source:** CSV is fast for prototyping; PDFs support full fidelity when available.
- **Stable IDs:** Section/article IDs (e.g. `BNS_Sec_302`) allow unambiguous linking in the knowledge graph and retrieval.
- **Citation edges:** Cases citing sections/articles are the backbone of the graph used in Phase 4 for constrained retrieval.
- **Neo4j-ready CSVs:** Phase 2 loads these directly via LOAD CSV; no schema drift.

## Results

Running the pipeline produces console output that can be used in reports or PPTs:

```text
Phase-1 Legal RAG Data Pipeline
----------------------------------------
Base path: .../Datasets
Output path: .../phase1_output

Loading judgments...
  Loaded <N> cases
  Schema: ...
  Years: <min> - <max>

Extracting from PDFs...
  Acts: <n_acts>
  Sections: <n_sections>
  Articles: <n_articles>

Building citation edges...
  Edges: <n_edges>

Exporting CSVs and plots...
  Output written to: phase1_output

Done.
```

**How to capture:** Run from project root:

```bash
python src/run_pipeline.py
```

Use the printed counts (cases, acts, sections, articles, edges, year range) as your Phase 1 results. Optionally run validation and use its output as “Validation results”:

```bash
python scripts/verify_phase1_output.py --output-dir phase1_output
```

The verification script reports row counts per CSV, duplicate IDs (if any), edge target coverage, and expected plot files.
