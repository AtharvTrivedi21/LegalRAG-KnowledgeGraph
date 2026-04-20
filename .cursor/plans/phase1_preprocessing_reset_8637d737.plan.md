---
name: phase1_preprocessing_reset
overview: "Reset Phase 1 by rebuilding data preprocessing from first principles: robust PDF text extraction + structural parsing for statutes, correct download/normalization of IL-TUR datasets, and production of Neo4j-ready CSVs (with PLAN_1-style IDs) that are clean, deduplicated, and reference-resolvable."
todos:
  - id: statute-pdf-preprocess
    content: Implement robust statutory PDF preprocessing (PyMuPDF) + TOC/header/footer removal, save raw page JSONL.
    status: completed
  - id: statute-structure-parse
    content: Parse Acts/Constitution into hierarchical nodes (Act/Part/Chapter/Section/Article) with PLAN_1 IDs; emit CSVs + linkage edges.
    status: completed
  - id: statute-enrichment
    content: Extract definitions + intra/cross-act references + parse central act index PDFs; emit enrichment CSVs.
    status: completed
  - id: iltur-download-normalize
    content: Add IL-TUR downloader/normalizer with offline fallback; output cases_iltur.csv with stable IDs.
    status: completed
  - id: sc-pdf-normalize
    content: Normalize SC judgments PDFs into cases_sc.csv with year/case_id and extraction quality flags.
    status: completed
  - id: case-citation-extract
    content: Implement act-aware citation extraction for cases; produce resolved edges + unresolved report.
    status: completed
  - id: phase1-validation
    content: Add validation report + gates (counts, duplicates, min text length, citation resolution stats).
    status: completed
isProject: false
---

## Goals (Phase 1 only)

- **Correctly preprocess statutory PDFs** in `Datasets/` so we can reliably extract **Act → Part → Chapter → Section/Article** structure (not table-of-contents junk), with stable IDs.
- **Correctly obtain and preprocess the “legal_data_train” source** by downloading **IL-TUR** via HuggingFace `datasets`, and normalize it into a `cases` table compatible with our KG.
- Produce **Neo4j-ready artifacts** (CSV + validation reports) that Phase 2 can load to create a **fully connected KG**, including statute cross-references and case→statute citations.

## What’s broken today (why results are bad)

- Current `src/pdf_extractor.py` splits on literal `"Section X"` / `"Article X"` patterns, but your act PDFs mostly use **numeric headings (e.g., `1.` / `2.`) and TOC lists**, so Phase 1 is extracting **huge wrong spans** (TOC/contents) and creating **duplicate IDs**.
- Current citation extraction (`src/edges.py`) doesn’t resolve **act context** (“Section 302” becomes *all* acts that contain a 302), and duplicates occur because upstream extraction emits duplicate section rows.
- The CSV source in `config.py` points to `legal_data_train.csv`, but your dataset reality is different (you have `Datasets/legal_data.csv` today, but you want the **true IL-TUR download** instead).

## Directory + artifact conventions (new)

- Inputs (existing):
  - `Datasets/1_BNS.pdf`, `Datasets/2 Bharatiya nagrik Suraksha sanhita.pdf`, `Datasets/3 Bharatiya Sakshya Adhiniyam.pdf`, `Datasets/Constitution Of India.pdf`
  - Index PDFs: `Datasets/Albhabetical List of Central Acts.pdf`, `Datasets/Chronological List of Central Acts.pdf`
  - SC judgments: `Datasets/SC_Judgements-16-25/<year>/*.pdf`
- New intermediate outputs (Phase 1):
  - `phase1_output_v2/raw_pdf_text/*.jsonl` (per doc, per page: text + optional block metadata)
  - `phase1_output_v2/structured/*.jsonl` (act/judgment structured records)
  - `phase1_output_v2/*.csv` (Neo4j-ready)

## Canonical IDs (you selected PLAN_1 style)

- **Acts**: `BNS_2023`, `BNSS_2023`, `BSA_2023`, `CONST_1950` (+ later stubs like `IPC_1860`, `CrPC_1973`, `IEA_1872` if needed)
- **Sections**: `BNS_s64`, `BNSS_s22`, `BSA_s23`
- **Constitution articles**: `CONST_Art21`, `CONST_Art21A`
- **Chapters/Parts/Definitions**: `BNS_CH_1`, `BNS_PART_I`, `BNS_DEF_person`

## Phase 1 workplan

### A) Statutory PDF preprocessing (the “correctly preprocess PDFs” step)

- Create a new extractor using **PyMuPDF** (preferred) with a pdfplumber fallback.
- Preprocessing rules (must be deterministic and logged):
  - **Remove headers/footers** (detect repeated lines across many pages).
  - **Dehyphenate** line-break hyphens and normalize whitespace.
  - **Detect and skip TOC/contents zones** (pages containing `CONTENTS`, dotted leaders, dense section lists).
  - **Preserve page boundaries + provenance** (store `source_file`, `page_num`, `char_offsets`).
- Output: `phase1_output_v2/raw_pdf_text/{act_id}.jsonl`.

### B) Structural parsing into hierarchy (Act → Part → Chapter → Section/Article)

- Implement rule-based parsers per document type:
  - **BNS/BNSS/BSA**: detect `PART`, `CHAPTER`, and section starts like `^\d+[A-Z]?\.` (plus subclauses `(1)`, `(a)`, `(i)`), with page-merge handling.
  - **Constitution**: detect `PART` blocks and article starts (numeric headings + amendments like `21A`).
- Emit normalized tables:
  - `acts.csv`, `parts.csv`, `chapters.csv`, `sections.csv`, `articles.csv`
  - plus linkage edges: `act_part.csv`, `part_chapter.csv`, `chapter_section.csv`, `act_section.csv`, `act_article.csv` (depending on what’s present)
- Add a **verification stage** that checks expected approximate counts and flags anomalies:
  - BNS ~358 sections, BNSS ~531, BSA ~170, Constitution ~395 articles (tolerances allowed but large deviations fail).

### C) Enrichment for a “perfectly connected” statute KG

- **Definition extraction**: identify definition-heavy sections (“In this Act, unless…”) and extract term→definition pairs.
  - Output: `definitions.csv` + `section_defines_term.csv`.
- **Cross-reference extraction**:
  - Within-act: “section 64”, “sub-section (1) of section 64”.
  - Cross-act: “section 2 of the Bharatiya Nagarik Suraksha Sanhita…”, plus common old-law references (IPC/CrPC/IEA).
  - Output: `section_references_section.csv` with properties `context`, `reference_type`, `target_act_id`.
- **Central Acts index PDFs**:
  - Parse `Albhabetical...pdf` and `Chronological...pdf` into `act_index.csv` (title/year/act_number/category), for future expansion and act resolution.

### D) “legal_data_train” dataset (IL-TUR) — correct download + preprocessing

- Add a script that downloads **both** IL-TUR tasks (summarization + classification) using:
  - `from datasets import load_dataset; load_dataset("Exploration-Lab/IL-TUR", task_name, revision="script")`
- Because HF access may be blocked, include fallbacks:
  - **Primary**: try download; if blocked, print actionable instructions and allow a **manual drop-in** at `Datasets/iltur_raw/`.
  - **Secondary (temporary)**: allow using existing `Datasets/legal_data.csv` only as a stopgap, explicitly marked “legacy/unverified”.
- Normalize IL-TUR into:
  - `cases_iltur.csv` with `case_id`, `judgment_text`, `summary` (if available), `label` (if classification), `source="iltur"`, `year` (if derivable; else null).

### E) SC judgments preprocessing (PDFs) + case schema

- Triage `Datasets/SC_Judgements-16-25/<year>/*.pdf`:
  - detect text-extractable vs scanned; optionally queue OCR (kept optional).
- Normalize into `cases_sc.csv` with stable `case_id`, `year`, `full_text`, `source_file`, `source="sc_pdf"`.

### F) Case→statute citation extraction (so KG edges are correct)

- Replace the current naive `src/edges.py` logic with act-aware citation mining:
  - Support `section/s./u/s/sec./read with/r/w` and Constitution `article/art.`
  - Detect act mentions near the citation (BNS/BNSS/BSA/Constitution + aliases); map to canonical act_ids.
  - If act is **not** resolvable, keep the citation in `unresolved_case_cites.csv` rather than linking to every act.
- Outputs:
  - `case_cites_section.csv`, `case_cites_article.csv`, `unresolved_case_cites.csv`.

### G) Phase 1 validation gates (must pass before Phase 2 Neo4j)

- No duplicate primary IDs in: Acts/Parts/Chapters/Sections/Articles/Definitions.
- Section/article text must exceed a minimum length (to avoid TOC-only captures).
- Citation resolution rate reports by act and by source (SC vs IL-TUR).
- Produce a single `phase1_output_v2/validation_report.md` summarizing counts + failures.

## Phase 2 handoff (not implemented yet, but Phase 1 must enable it)

- Update `neo4j/cypher/`* to load:
  - hierarchy nodes + edges (`HAS_PART`, `HAS_CHAPTER`, `HAS_SECTION`, `DEFINES_TERM`, `REFERENCES`, etc.)
  - case→section/article citations with context/count
- Update Phase 4 query parsing/client to use new canonical act_ids (`BNS_2023`, etc.).

## Key repo files we will change/add in implementation

- Add new pipeline modules (new):
  - `phase1_preprocessing/pdf_text.py`, `phase1_preprocessing/act_parser.py`, `phase1_preprocessing/constitution_parser.py`
  - `phase1_preprocessing/definitions.py`, `phase1_preprocessing/citations.py`, `phase1_preprocessing/act_index_parser.py`
  - `phase1_preprocessing/iltur_download.py`, `phase1_preprocessing/sc_pdf_loader.py`
  - `phase1_preprocessing/run_phase1_v2.py`
- Update (existing):
  - `config.py` (new config section for Phase1 v2 paths)
  - potentially deprecate/replace `src/pdf_extractor.py` and `src/edges.py` usage in the main runner

## Implementation order (when you say “go implement Phase 1”)

- Build statutory PDF preprocessing + structural parsing first (this is the KG backbone).
- Then IL-TUR download + normalization.
- Then SC judgments normalization.
- Then case citation extraction + resolution.
- Then run validation gates and iterate until counts/quality look correct.

