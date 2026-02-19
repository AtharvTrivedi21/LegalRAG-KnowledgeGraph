# Graph-Constrained Legal RAG for Indian Law

Phase-1: Data Engineering + Legal NLP exploration (no ML, no embeddings, no Neo4j).

## Setup

### 1. Create virtual environment

```powershell
cd c:\Users\ATHARV\LegalRAG
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies (in venv only)

```powershell
pip install -r requirements.txt
```

### 3. Data requirements

Place the following files in the `Datasets/` folder:

- `SC_Judgements-16-25/` (already extracted) with subfolders `2016/ ... 2025/` containing `*.pdf`
- `Constitution Of India.pdf`
- `1_BNS.pdf` (Bharatiya Nyaya Sanhita)
- `2 Bharatiya nagrik Suraksha sanhita.pdf` (BNSS)
- `3 Bharatiya Sakshya Adhiniyam.pdf` (BSA)

If the extracted folder is missing, the pipeline falls back to `legal_data_train.csv` (if present).

## Run Phase-1 pipeline

```powershell
venv\Scripts\activate
python src/run_pipeline.py
```

Output is written to `phase1_output/`:

- `cases.csv` – case_id, judgment_text, year
- `sections.csv` – section_id, act_id, section_number, full_text
- `articles.csv` – article_id, act_id, article_number, full_text
- `acts.csv` – act_id, act_name, act_type, source_file
- `edges.csv` – source_case_id, target_section_or_article, relation (CITES/REFERS)
- Plots: `judgments_per_year.png`, `top_cited_sections.png`, `top_cited_articles.png`

## Configuration

Edit `config.py` to change:

- `BASE_PATH` – data directory (default: `Datasets/`)
- `OUTPUT_PATH` – output directory (default: `phase1_output/`)
- `SC_EXTRACTED_DIR` – extracted SC judgments folder name (default: `SC_Judgements-16-25`)
- `DATA_LIMIT` – set to e.g. `100` to process only the first 100 judgment PDFs (for quick testing)

Override `BASE_PATH` via environment variable:

```powershell
$env:LEGALRAG_DATA_PATH = "C:\path\to\data"
python src/run_pipeline.py
```

## Phase-2: Build the Knowledge Graph (Neo4j Desktop, local)

1. Run verification (recommended):

```powershell
venv\Scripts\activate
python scripts\verify_phase1_output.py --output-dir .\phase1_output
```

2. Follow instructions in `neo4j/README.md` to:
- Copy CSVs into Neo4j `import/`
- Run Cypher scripts in `neo4j/cypher/` to create constraints, import nodes/edges, and run smoke tests
