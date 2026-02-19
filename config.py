"""
Configuration for Phase-1 Legal RAG data pipeline.
Paths, file names, and output directory.
"""
import os
from pathlib import Path

# Base path for input data (Datasets folder)
# Override via env var LEGALRAG_DATA_PATH if needed
BASE_PATH = Path(
    os.environ.get("LEGALRAG_DATA_PATH", "Datasets")
)

# Output directory for clean CSVs and plots
OUTPUT_PATH = Path("phase1_output")

# Judgment source mode:
# - "pdf": read extracted SC PDFs from SC_EXTRACTED_DIR/<year>/*.pdf (slower, most accurate)
# - "csv": use legal_data_train.csv (fastest; good for quick KG prototype)
JUDGMENTS_SOURCE = "csv"

# Supreme Court judgments folder (already extracted)
# Expected structure: BASE_PATH / SC_EXTRACTED_DIR / <year> / *.pdf
SC_EXTRACTED_DIR = "SC_Judgements-16-25"

# Acts PDFs (stored directly under BASE_PATH)
PDF_FILES = {
    "constitution": "Constitution Of India.pdf",
    "bns": "1_BNS.pdf",
    "bnss": "2 Bharatiya nagrik Suraksha sanhita.pdf",
    "bsa": "3 Bharatiya Sakshya Adhiniyam.pdf",
}

# Optional: limit number of judgment PDFs to process (None = all)
# Set to e.g. 100 for quick testing
DATA_LIMIT = None  # process all rows in legal_data_train.csv

# Act names (for sections/articles tables)
ACT_NAMES = {
    "constitution": "Constitution Of India",
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagrik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
}
