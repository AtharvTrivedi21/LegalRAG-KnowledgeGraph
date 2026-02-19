# Datasets Used in LegalRAG

This document lists **only** the datasets that are **actually read or consumed** by the codebase (Phase 1–4). Paths are relative to the project root unless stated otherwise. Schema and usage are described so you can reproduce or inspect data; sample rows can be refreshed with the script at the end.

---

## 1. legal_data_train.csv (Phase 1)

- **Path:** `Datasets/legal_data_train.csv` (or `BASE_PATH/legal_data_train.csv`; `BASE_PATH` defaults to `Datasets`, overridable via `LEGALRAG_DATA_PATH`).
- **When used:** When `config.JUDGMENTS_SOURCE = "csv"`. Also used as fallback if the extracted PDF folder is missing.
- **How read:** In `src/judgments.py`, `load_and_prepare_judgments(..., source="csv")` uses `pd.read_csv(csv_path, usecols=["Text"], ...)`. The code expects at least a **`Text`** column (judgment text). Optional: id and year columns are detected by name (e.g. case_id, year); if absent, `case_id` becomes `case_0`, `case_1`, ... and `year` is set to 0.
- **Schema (used in code):**
  - `Text` (required) — judgment text.
  - Other columns (optional): any column that matches ID_COLUMNS or YEAR_COLUMNS in `judgments.py` is used for `case_id` / `year`; otherwise synthetic IDs and year 0 are used.
- **Sample:** If the file exists, run the script below to print `head()` and shape. Otherwise document expected schema as above and note: “Run `python scripts/doc_dataset_samples.py` when the file is available to capture head().”

---

## 2. Supreme Court judgments (PDFs) (Phase 1)

- **Path pattern:** `BASE_PATH/SC_EXTRACTED_DIR/<year>/*.pdf`, e.g. `Datasets/SC_Judgements-16-25/2016/*.pdf`. `SC_EXTRACTED_DIR` is set in `config.py` (default `SC_Judgements-16-25`).
- **When used:** When `config.JUDGMENTS_SOURCE = "pdf"` and the extracted folder exists.
- **Structure:** One folder per year (folder name = 4-digit year). PDF filename stem is used as `case_id`; text is extracted with pdfplumber.
- **Schema (produced):** Each judgment becomes one row with `case_id`, `judgment_text`, `year` (from folder name).

---

## 3. Act PDFs (Phase 1)

- **Paths:** Under `BASE_PATH`, filenames from `config.PDF_FILES`:
  - `Constitution Of India.pdf`
  - `1_BNS.pdf`
  - `2 Bharatiya nagrik Suraksha sanhita.pdf`
  - `3 Bharatiya Sakshya Adhiniyam.pdf`
- **When used:** By `src/pdf_extractor.py` in `build_acts_sections_articles()`. Constitution is parsed for Articles; BNS, BNSS, BSA for Sections.
- **Output:** Feeds Phase 1 output tables (acts, sections, articles) and thus Phase 2 and Phase 3; not read as a “dataset” by Phase 2/3 directly.

---

## 4. Phase 1 output CSVs (Phase 2 and Phase 3)

Produced by Phase 1 in `phase1_output/`. Consumed by Phase 2 (after copy to Neo4j `import/`) and by Phase 3 (read directly from `phase1_output/`).

| File | Columns | Consumed by |
|------|---------|-------------|
| **acts.csv** | act_id, act_name, act_type, source_file | Phase 2 (02_load_nodes.cypher) |
| **sections.csv** | section_id, act_id, section_number, full_text | Phase 2, Phase 3 (chunk_corpus.py) |
| **articles.csv** | article_id, act_id, article_number, full_text | Phase 2, Phase 3 (chunk_corpus.py) |
| **cases.csv** | case_id, judgment_text, year | Phase 2, Phase 3 (chunk_corpus.py) |
| **edges.csv** | source_case_id, target_section_or_article, relation | Phase 2 (03_load_edges.cypher) |

- **Phase 3:** Reads only `cases.csv`, `sections.csv`, `articles.csv` from `phase3_embeddings/config.PHASE1_OUTPUT` (default `phase1_output`).
- **Sample:** Run `python scripts/doc_dataset_samples.py` to print shape and first 5 rows for each existing CSV under `phase1_output/`.

---

## 5. IndicLegalQA Dataset_10K_Revised.json (Phase 3)

- **Path:** `Datasets/IndicLegalQA Dataset_10K_Revised.json` (from `phase3_embeddings/config.INDIC_LEGAL_QA_PATH`).
- **When used:** By `phase3_embeddings/finetune_bge.py` for fine-tuning BGE and evaluation. Not used by Phase 1, 2, or 4.
- **Structure:** JSON list of objects. Each object has at least:
  - `question` — string (legal question).
  - `answer` — string (reference answer).
- **Usage:** Loaded with `json.load()`; pairs with non-empty question and answer are used; 80/20 train/eval split (TRAIN_EVAL_SPLIT, RANDOM_SEED in config).
- **Sample:** Run the script below to print first 3 question/answer pairs and total count.

---

## Refreshing samples (head and schema)

To regenerate shape, schema, and first rows for all datasets that exist, run from project root:

```bash
python scripts/doc_dataset_samples.py
```

Output can be pasted into this doc or saved to `Documentation/dataset_samples.txt`. The script reads from paths in `config.py` and `phase3_embeddings/config.py` and prints head() or first N items for each available file.
