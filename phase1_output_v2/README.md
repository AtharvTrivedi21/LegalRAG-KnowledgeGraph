# phase1_output_v2

Purpose
- Preprocessed legal documents, extracted sections, and CSV exports used to build the retrieval corpus and knowledge graph.

What’s here
- `raw_pdf_text/` — parsed JSONL of Acts (BNS_2023, BSA_2023, BNSS_2023, CONST_1950, etc.).
- CSVs: `articles.csv`, `chapters.csv`, `section_references_section.csv`, `act_section.csv`, etc.

How to use
- These CSVs serve as inputs for FAISS chunking and any Neo4j import pipelines.
- Validation artifacts: `validation_report.md`.

