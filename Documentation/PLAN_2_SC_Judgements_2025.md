# Plan 2: SC Judgements 2025 → Neo4j Knowledge Graph
## 400 PDF Judgments — Preprocessing, Citation Extraction & Graph Import

---

## 0. Context & Goal

This plan covers the **dynamic case law layer** of the knowledge graph. We have ~400 Supreme Court of India judgments from 2025 (104 MB of PDFs). These are the **first full year of case law under BNS, BNSS, and BSA** (which came into force July 1, 2024), making this genuinely novel data.

**Input:** `sc_judgements_2025/` — directory of ~400 PDF files  
**Prerequisite:** Plan 1 must be completed first (the static legal corpus must exist in Neo4j before case law can be linked to it)  
**Output:** 
- `(:Case)` nodes for each judgment  
- `(:Case)-[:CITES]->(:Section)` edges linking judgments to specific law sections  
- `(:Case)-[:CITES_ACT]->(:Act)` edges for act-level citations  
- `(:Case)-[:CITES_CASE]->(:Case)` edges for case-to-case precedent citations  
- `(:Judge)` nodes linked via `(:Case)-[:DECIDED_BY]->(:Judge)`

---

## 1. Target Graph Schema (Case Law Layer)

### 1.1 New Node Labels

```
(:Case)
  case_id           String  PK  e.g. "SC_2025_001", or derived from case number
  case_number       String      "Civil Appeal No. 1234 of 2025"
  case_type         String      "Civil" | "Criminal" | "Constitutional" | "SLP"
  bench_type        String      "Division" | "Full" | "Constitution" | "Single"
  judgment_date     String      "2025-03-15"  (ISO format)
  court             String      "Supreme Court of India"
  outcome           String      "Allowed" | "Dismissed" | "Partly Allowed" | "Remanded"
  source_file       String      "SC_2025_001.pdf"
  coram             String      Raw text of judge names
  petitioner        String
  respondent        String
  summary           String      First 500 chars of headnote / ratio
  full_text         String      Full judgment text (can be large — consider storing separately)

(:Judge)
  judge_id          String  PK  e.g. "JUDGE_DY_CHANDRACHUD"
  name              String      "D.Y. Chandrachud"
  designation       String      "Chief Justice of India" | "Judge"

(:LegalPrinciple)
  principle_id      String  PK  e.g. "PRINCIPLE_SC_2025_001_1"
  text              String      Extracted ratio decidendi / legal holding
  case_id           String      FK → Case
```

### 1.2 New Relationship Types

```
(:Case)-[:CITES]->(:Section)
  props: {context: String, citation_type: "applied"|"distinguished"|"overruled"|"followed"|"referred"}

(:Case)-[:CITES_ACT]->(:Act)
  props: {context: String}

(:Case)-[:CITES_CASE]->(:Case)
  props: {citation_type: "followed"|"distinguished"|"overruled"|"referred", raw_citation: String}

(:Case)-[:DECIDED_BY]->(:Judge)
  props: {role: "author"|"concurring"|"dissenting"}

(:Case)-[:HAS_PRINCIPLE]->(:LegalPrinciple)

// NEW: Connect old-law citations to new-law equivalents via traversal
// (No new relationship needed — use existing EQUIVALENT_TO edges from Plan 1)
// Query: MATCH (c:Case)-[:CITES]->(old:Section)-[:EQUIVALENT_TO]->(new:Section)
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         INPUT: ~400 SC Judgment PDFs (104 MB)        │
│         sc_judgements_2025/*.pdf                     │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │   STAGE 1: TRIAGE     │
           │   classify_pdfs.py    │
           │   Scan + sort by type │
           │   Filter corrupt/empty│
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │   STAGE 2: EXTRACT    │
           │   pdf_to_text.py      │
           │   PyMuPDF per-page    │
           │   Detect header/body  │
           └───────────┬───────────┘
                       │
           ┌───────────▼────────────────────────────┐
           │         STAGE 3: PARSE                  │
           │         judgment_parser.py              │
           │  Extracts:                              │
           │  - Case metadata (number, date, parties)│
           │  - Coram (judge names)                  │
           │  - Headnote / Ratio decidendi           │
           │  - Body text sections                   │
           │  - Result/Order paragraph               │
           └───────────┬────────────────────────────┘
                       │
           ┌───────────▼────────────────────────────┐
           │       STAGE 4: CITATION EXTRACT         │
           │       citation_extractor.py             │
           │  Extracts:                              │
           │  - Section citations (BNS/BNSS/BSA/IPC) │
           │  - Article citations (Constitution)     │
           │  - Precedent case citations             │
           │  - Act-level citations                  │
           └───────────┬────────────────────────────┘
                       │
           ┌───────────▼────────────────────────────┐
           │       STAGE 5: RESOLVE & LINK           │
           │       citation_resolver.py              │
           │  Maps raw citations to section_ids      │
           │  in the existing Neo4j graph            │
           │  Flags unresolved citations             │
           └───────────┬────────────────────────────┘
                       │
           ┌───────────▼────────────────────────────┐
           │       STAGE 6: NEO4J IMPORT             │
           │       neo4j_case_importer.py            │
           │  MERGE Case nodes                       │
           │  MERGE Judge nodes                      │
           │  CREATE CITES relationships             │
           └────────────────────────────────────────┘
```

---

## 3. Project Structure

```
legal-kg/
├── judgements/
│   └── sc_2025/          # Raw PDF files (400 files, ~104 MB)
│       ├── SC_0001.pdf
│       ├── SC_0002.pdf
│       └── ...
├── pipeline_cases/
│   ├── classify_pdfs.py       # Stage 1: triage
│   ├── pdf_to_text.py         # Stage 2: extraction
│   ├── judgment_parser.py     # Stage 3: structural parsing
│   ├── citation_extractor.py  # Stage 4: citation mining
│   ├── citation_resolver.py   # Stage 5: resolve to graph IDs
│   ├── neo4j_case_importer.py # Stage 6: Neo4j import
│   └── config_cases.py        # Patterns, act map, judge list
├── output_cases/
│   ├── triage_report.csv      # Which PDFs are valid/corrupt
│   ├── cases_raw.jsonl        # One JSON per case (intermediate)
│   ├── cases.csv              # Final Case nodes
│   ├── judges.csv             # Judge nodes
│   ├── case_section_cites.csv # Case → Section edges
│   ├── case_act_cites.csv     # Case → Act edges
│   ├── case_case_cites.csv    # Case → Case edges (precedent)
│   └── unresolved_cites.csv   # Citations we couldn't map
├── cypher_cases/
│   ├── 01_import_cases.cypher
│   ├── 02_import_judges.cypher
│   ├── 03_import_citations.cypher
│   └── 04_verify_cases.cypher
└── run_cases_pipeline.py      # Master orchestration
```

---

## 4. Implementation — Stage by Stage

### Stage 1: Triage (`classify_pdfs.py`)

Before any parsing, classify all 400 PDFs:

```python
import fitz
import os
import csv
from pathlib import Path

def triage_pdfs(input_dir: str) -> list[dict]:
    """
    Scan all PDFs and classify them.
    Returns list of {file, status, page_count, text_extractable, detected_type}
    """
    results = []
    for pdf_file in sorted(Path(input_dir).glob("*.pdf")):
        result = {
            "file": pdf_file.name,
            "status": "ok",
            "page_count": 0,
            "text_extractable": False,
            "detected_type": "unknown",
            "first_100_chars": ""
        }
        try:
            doc = fitz.open(str(pdf_file))
            result["page_count"] = len(doc)

            # Check if text is extractable (not a scanned image PDF)
            first_page_text = doc[0].get_text("text").strip()
            result["text_extractable"] = len(first_page_text) > 50
            result["first_100_chars"] = first_page_text[:100]

            # Detect judgment type from first page
            text_lower = first_page_text.lower()
            if "civil appeal" in text_lower:
                result["detected_type"] = "Civil Appeal"
            elif "criminal appeal" in text_lower:
                result["detected_type"] = "Criminal Appeal"
            elif "special leave petition" in text_lower or "slp" in text_lower:
                result["detected_type"] = "SLP"
            elif "writ petition" in text_lower:
                result["detected_type"] = "Writ Petition"
            elif "transfer petition" in text_lower:
                result["detected_type"] = "Transfer Petition"
            else:
                result["detected_type"] = "Other"

        except Exception as e:
            result["status"] = f"error: {str(e)}"

        results.append(result)
        doc.close()

    return results
```

**Expected output:** ~395 valid PDFs, ~5 potentially corrupt or image-only

---

### Stage 2: PDF Text Extraction (`pdf_to_text.py`)

```python
import fitz
from pathlib import Path
import json

def extract_judgment_text(pdf_path: str) -> dict:
    """
    Extracts text from judgment PDF with page boundaries preserved.
    SC judgments have a consistent structure:
      - Page 1: Header (IN THE SUPREME COURT OF INDIA), Case number, Coram, Date
      - Pages 2-N: Body (facts, arguments, reasoning)
      - Last pages: Order/Directions
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "char_count": len(text)
        })

    full_text = "\n".join([p["text"] for p in pages])

    return {
        "file": Path(pdf_path).name,
        "page_count": len(pages),
        "total_chars": len(full_text),
        "pages": pages,
        "full_text": full_text
    }
```

---

### Stage 3: Judgment Parser (`judgment_parser.py`)

This is the most important stage. SC judgments have a consistent but non-trivial structure.

```python
import re
from datetime import datetime

# ─── SC Judgment Header Patterns ────────────────────────────────────────────

# "IN THE SUPREME COURT OF INDIA" (always on page 1)
SC_HEADER_RE = re.compile(r'IN THE SUPREME COURT OF INDIA', re.IGNORECASE)

# Case type + number: "CIVIL APPELLATE JURISDICTION\nCIVIL APPEAL NO. 1234 OF 2025"
CASE_NUMBER_RE = re.compile(
    r'(?:CIVIL|CRIMINAL|ORIGINAL|APPELLATE|WRIT)\s+(?:APPELLATE\s+)?JURISDICTION\s*\n'
    r'\s*(.+?(?:APPEAL|PETITION|SUIT)\s+NO[S]?\.\s*[\d\s,AND]+OF\s+\d{4})',
    re.IGNORECASE | re.DOTALL
)

# Alternative: just grab the case number line directly
CASE_NUM_SIMPLE_RE = re.compile(
    r'((?:CIVIL|CRIMINAL|TRANSFER|WRIT|SPECIAL LEAVE)\s+'
    r'(?:APPEAL|PETITION|APPLICATION)[S]?\s+NO[S]?\.\s*[\d\s,ANDTO]+OF\s+\d{4})',
    re.IGNORECASE
)

# Date: "JUDGMENT\nDate: 15.03.2025" or "Dated this 15th day of March, 2025"
DATE_RE = re.compile(
    r'(?:JUDGMENT\s+\n\s*)?'
    r'(?:Dated?\s+(?:this\s+)?\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'[,\s]+\d{4}'
    r'|\d{1,2}[./]\d{1,2}[./]\d{4})',
    re.IGNORECASE
)

# Coram: "CORAM:\nJUSTICE A.B. NAME\nJUSTICE C.D. NAME"
CORAM_RE = re.compile(
    r'(?:CORAM\s*:\s*\n|BEFORE\s*:\s*\n|HON\'BLE\s+)'
    r'((?:(?:HON\'BLE\s+)?(?:MR\.?\s+)?JUSTICE\s+[A-Z][A-Z\s\.\,]+\n?)+)',
    re.IGNORECASE
)

# Individual judge name extraction
JUDGE_NAME_RE = re.compile(
    r'(?:HON\'BLE\s+)?(?:MR\.?\s+)?(?:DR\.?\s+)?'
    r'JUSTICE\s+([A-Z][A-Z\s\.]+?)(?=\n|JUSTICE|$)',
    re.IGNORECASE
)

# Parties: "BETWEEN:\nPETITIONER_NAME ... Appellant(s)\nVERSUS\nRESPONDENT_NAME ... Respondent(s)"
PETITIONER_RE = re.compile(
    r'BETWEEN\s*:?\s*\n(.+?)(?:\s*\.{3,}|\s+Appellant)',
    re.IGNORECASE | re.DOTALL
)
RESPONDENT_RE = re.compile(
    r'VERSUS\s*\n(.+?)(?:\s*\.{3,}|\s+Respondent)',
    re.IGNORECASE | re.DOTALL
)

# Outcome detection (look in last 20% of document)
OUTCOME_PATTERNS = [
    (r'appeal\s+is\s+(?:hereby\s+)?allowed',            "Allowed"),
    (r'appeal\s+is\s+(?:hereby\s+)?dismissed',          "Dismissed"),
    (r'appeal\s+is\s+(?:hereby\s+)?partly\s+allowed',   "Partly Allowed"),
    (r'matter\s+is\s+(?:hereby\s+)?remanded',           "Remanded"),
    (r'petition\s+is\s+(?:hereby\s+)?allowed',          "Allowed"),
    (r'petition\s+is\s+(?:hereby\s+)?dismissed',        "Dismissed"),
    (r'writ\s+is\s+(?:hereby\s+)?issued',               "Writ Issued"),
    (r'acquittal\s+is\s+(?:hereby\s+)?maintained',      "Acquittal Maintained"),
    (r'conviction\s+is\s+(?:hereby\s+)?upheld',         "Conviction Upheld"),
]


def parse_judgment(extracted: dict) -> dict:
    """
    Parse structured judgment from extracted text dict (output of Stage 2).
    """
    text = extracted["full_text"]
    file_name = extracted["file"]

    # Generate case_id from filename
    case_id = f"SC_2025_{Path(file_name).stem}"

    # Extract case number
    case_number = None
    m = CASE_NUM_SIMPLE_RE.search(text[:1000])
    if m:
        case_number = re.sub(r'\s+', ' ', m.group(1)).strip()

    # Extract date
    judgment_date = None
    m = DATE_RE.search(text[:2000])
    if m:
        raw_date = m.group(0)
        judgment_date = normalize_date(raw_date)

    # Extract coram (judges)
    judges = []
    m = CORAM_RE.search(text[:2000])
    if m:
        coram_text = m.group(1)
        judges = [j.strip() for j in JUDGE_NAME_RE.findall(coram_text)]

    # Extract parties
    petitioner = None
    m = PETITIONER_RE.search(text[:3000])
    if m:
        petitioner = re.sub(r'\s+', ' ', m.group(1)).strip()[:200]

    respondent = None
    m = RESPONDENT_RE.search(text[:3000])
    if m:
        respondent = re.sub(r'\s+', ' ', m.group(1)).strip()[:200]

    # Detect outcome from last 20% of text
    tail = text[int(len(text) * 0.8):]
    outcome = "Unknown"
    for pattern, label in OUTCOME_PATTERNS:
        if re.search(pattern, tail, re.IGNORECASE):
            outcome = label
            break

    # Extract headnote/summary (text before "JUDGMENT" keyword)
    summary = ""
    judgment_marker = re.search(r'\bJUDGMENT\b', text, re.IGNORECASE)
    if judgment_marker and judgment_marker.start() > 100:
        summary = text[:judgment_marker.start()].strip()[-500:]

    # Determine case type
    case_type = "Unknown"
    if case_number:
        num_lower = case_number.lower()
        if "civil appeal" in num_lower:       case_type = "Civil Appeal"
        elif "criminal appeal" in num_lower:  case_type = "Criminal Appeal"
        elif "slp" in num_lower:              case_type = "SLP"
        elif "writ" in num_lower:             case_type = "Writ Petition"
        elif "transfer" in num_lower:         case_type = "Transfer Petition"

    return {
        "case_id": case_id,
        "case_number": case_number,
        "case_type": case_type,
        "judgment_date": judgment_date,
        "court": "Supreme Court of India",
        "petitioner": petitioner,
        "respondent": respondent,
        "outcome": outcome,
        "judges": judges,
        "coram_raw": ", ".join(judges),
        "summary": summary,
        "full_text": text,
        "source_file": file_name,
        "page_count": extracted["page_count"]
    }


def normalize_date(raw: str) -> str:
    """Convert messy date strings to ISO YYYY-MM-DD format."""
    raw = re.sub(r'\s+', ' ', raw).strip()
    # Try common formats
    formats = [
        "%d.%m.%Y", "%d/%m/%Y",
        "%dth %B, %Y", "%dst %B, %Y", "%dnd %B, %Y", "%drd %B, %Y",
        "%d %B, %Y", "%d %B %Y",
    ]
    raw_clean = re.sub(r'(?<=\d)(st|nd|rd|th)', '', raw, flags=re.IGNORECASE)
    for fmt in formats:
        try:
            return datetime.strptime(raw_clean.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # Return raw if parsing fails
```

---

### Stage 4: Citation Extractor (`citation_extractor.py`)

This is the most complex stage — extracting all citations from judgment body text.

```python
import re
from dataclasses import dataclass, field
from typing import Optional

# ─── Act canonical ID map ────────────────────────────────────────────────────
ACT_ID_MAP = {
    # New Sanhitas (primary target)
    r"Bharatiya Nyaya Sanhita|BNS":                        "BNS_2023",
    r"Bharatiya Nagarik Suraksha Sanhita|BNSS":            "BNSS_2023",
    r"Bharatiya Sakshya Adhiniyam|BSA":                    "BSA_2023",

    # Old laws (pre-2024 — still cited in 2025 cases as precedent)
    r"Indian Penal Code|IPC|I\.P\.C\.":                    "IPC_1860",
    r"Code of Criminal Procedure|Cr\.?P\.?C\.?|CrPC":     "CrPC_1973",
    r"Indian Evidence Act|Evidence Act":                    "IEA_1872",

    # Constitution
    r"Constitution of India|Constitution":                  "CONST_1950",

    # Other frequently cited acts
    r"Protection of Children from Sexual Offences|POCSO":  "POCSO_2012",
    r"Prevention of Corruption Act":                        "PCA_1988",
    r"Scheduled Castes and Scheduled Tribes.*Prevention":  "SCST_1989",
    r"Narcotic Drugs and Psychotropic Substances|NDPS":    "NDPS_1985",
    r"Negotiable Instruments Act":                          "NIA_1881",
    r"Income Tax Act":                                      "ITA_1961",
    r"Companies Act":                                       "CA_2013",
    r"Arbitration and Conciliation Act":                   "ACA_1996",
    r"Motor Vehicles Act":                                  "MVA_1988",
    r"Arms Act":                                            "ARMS_1959",
}

# ─── Section citation patterns ───────────────────────────────────────────────

# "Section 302 of the IPC"  /  "s. 64 of BNS"  /  "u/s 376"
SECTION_WITH_ACT_RE = re.compile(
    r'(?:section|s\.|sec\.|u/s|under\s+section)\s*'
    r'(\d+[A-Z]?(?:\([^\)]+\))*(?:\s*(?:read\s+with|r/?w\.?)\s*[\d()\s]+)?)'
    r'(?:\s+(?:of\s+)?(?:the\s+)?)?' +
    r'(' + '|'.join(ACT_ID_MAP.keys()) + r')',
    re.IGNORECASE
)

# Article references: "Article 21 of the Constitution" / "Art. 14"
ARTICLE_WITH_ACT_RE = re.compile(
    r'(?:article|art\.)\s*(\d+[A-Z]?(?:\([^\)]+\))*)'
    r'(?:\s+of\s+(?:the\s+)?Constitution(?:\s+of\s+India)?)?',
    re.IGNORECASE
)

# "Read with Section X" (compound citations)
READ_WITH_RE = re.compile(
    r'read\s+with\s+(?:section\s+)?(\d+[A-Z]?(?:\([^\)]+\))*)',
    re.IGNORECASE
)

# ─── Precedent case citation patterns ────────────────────────────────────────

# "(2024) 5 SCC 123"  /  "AIR 2024 SC 456"  /  "(2025) SCR 789"
SCC_RE = re.compile(
    r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)',
    re.IGNORECASE
)
AIR_SC_RE = re.compile(
    r'AIR\s+(\d{4})\s+SC\s+(\d+)',
    re.IGNORECASE
)
# Named case: "State of Maharashtra v. Manubhai Pragaji Vashi"
# Less reliable — use SCC/AIR citations as primary key

# ─── Extractor function ──────────────────────────────────────────────────────

def extract_all_citations(case_id: str, full_text: str) -> dict:
    """
    Main citation extraction function.
    Returns dict with:
      - section_cites:  [{case_id, act_id, section_number, context, raw}]
      - act_cites:      [{case_id, act_id, context}]
      - precedent_cites: [{case_id, citation_string, year, reporter, volume, page}]
    """
    section_cites = []
    act_cites = []
    precedent_cites = []

    seen_section_ids = set()
    seen_acts = set()

    # ── Section citations ──
    for match in SECTION_WITH_ACT_RE.finditer(full_text):
        section_num = clean_section_num(match.group(1))
        raw_act = match.group(2)
        act_id = resolve_act_name(raw_act)

        if not act_id:
            continue

        section_id = f"{act_id}_s{section_num}"
        context = get_context(full_text, match.start(), window=100)

        # Also extract "read with" sections
        rw_match = READ_WITH_RE.search(full_text[match.start():match.start()+150])
        read_with_sections = [clean_section_num(rw_match.group(1))] if rw_match else []

        if section_id not in seen_section_ids:
            seen_section_ids.add(section_id)
            section_cites.append({
                "case_id": case_id,
                "act_id": act_id,
                "section_number": section_num,
                "section_id": section_id,
                "read_with": read_with_sections,
                "context": context,
                "raw": match.group(0)
            })

        if act_id not in seen_acts:
            seen_acts.add(act_id)
            act_cites.append({"case_id": case_id, "act_id": act_id})

    # ── Constitution article citations ──
    for match in ARTICLE_WITH_ACT_RE.finditer(full_text):
        article_num = clean_section_num(match.group(1))
        section_id = f"CONST_1950_Art{article_num}"
        if section_id not in seen_section_ids:
            seen_section_ids.add(section_id)
            section_cites.append({
                "case_id": case_id,
                "act_id": "CONST_1950",
                "section_number": article_num,
                "section_id": section_id,
                "read_with": [],
                "context": get_context(full_text, match.start()),
                "raw": match.group(0)
            })

    # ── Precedent case citations ──
    for match in SCC_RE.finditer(full_text):
        year, volume, page = match.group(1), match.group(2), match.group(3)
        citation_str = f"({year}) {volume} SCC {page}"
        precedent_cites.append({
            "case_id": case_id,
            "citation_string": citation_str,
            "reporter": "SCC",
            "year": int(year),
            "volume": int(volume),
            "page": int(page),
            "raw": match.group(0)
        })

    for match in AIR_SC_RE.finditer(full_text):
        year, page = match.group(1), match.group(2)
        citation_str = f"AIR {year} SC {page}"
        precedent_cites.append({
            "case_id": case_id,
            "citation_string": citation_str,
            "reporter": "AIR",
            "year": int(year),
            "page": int(page),
            "raw": match.group(0)
        })

    return {
        "section_cites": section_cites,
        "act_cites": act_cites,
        "precedent_cites": precedent_cites
    }


def clean_section_num(raw: str) -> str:
    """Normalize section numbers: remove extra spaces, standardize brackets."""
    return re.sub(r'\s+', '', raw.strip()).upper()


def get_context(text: str, pos: int, window: int = 100) -> str:
    """Extract surrounding text context for a citation position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end].replace('\n', ' ').strip()


def resolve_act_name(raw_name: str) -> Optional[str]:
    """Map raw act name string to canonical act_id."""
    for pattern, act_id in ACT_ID_MAP.items():
        if re.search(pattern, raw_name, re.IGNORECASE):
            return act_id
    return None
```

---

### Stage 5: Citation Resolver (`citation_resolver.py`)

**Purpose:** Verify that extracted `section_id` values actually exist in the Neo4j graph. Flag those that don't.

```python
from neo4j import GraphDatabase
import pandas as pd

def resolve_citations(citations_df: pd.DataFrame, neo4j_driver) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split citations into resolved (section exists in graph) and unresolved.
    Returns (resolved_df, unresolved_df)
    """
    with neo4j_driver.session() as session:
        # Get all section_ids that exist in the graph
        result = session.run("MATCH (s:Section) RETURN s.section_id AS id")
        existing_ids = set(record["id"] for record in result)

    resolved = citations_df[citations_df["section_id"].isin(existing_ids)].copy()
    unresolved = citations_df[~citations_df["section_id"].isin(existing_ids)].copy()

    print(f"Resolved: {len(resolved)}/{len(citations_df)} ({len(resolved)/len(citations_df)*100:.1f}%)")
    print(f"Unresolved: {len(unresolved)} citations — check unresolved_cites.csv")

    # Analyze unresolved patterns
    if len(unresolved) > 0:
        print("\nMost common unresolved acts:")
        print(unresolved["act_id"].value_counts().head(10))

    return resolved, unresolved
```

**Expected resolution rate:**
- BNS/BNSS/BSA sections: ~90%+ if Plan 1 completed correctly
- IPC/CrPC/IEA: Stubs will exist from Plan 1's repeal relationships; add full sections if needed
- Constitution: Should be ~100% if Constitution.pdf was parsed in Plan 1

---

### Stage 6: Neo4j Case Importer (`neo4j_case_importer.py`)

```python
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

class CaseLawImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def import_cases(self, cases_df: pd.DataFrame):
        query = """
        UNWIND $rows AS row
        MERGE (c:Case {case_id: row.case_id})
        SET c.case_number   = row.case_number,
            c.case_type     = row.case_type,
            c.judgment_date = row.judgment_date,
            c.court         = row.court,
            c.petitioner    = row.petitioner,
            c.respondent    = row.respondent,
            c.outcome       = row.outcome,
            c.summary       = row.summary,
            c.source_file   = row.source_file
        """
        self._batch_write(query, cases_df)
        print(f"Imported {len(cases_df)} Case nodes")

    def import_judges(self, judges: list[dict]):
        """
        MERGE judges and link to cases.
        judges: [{case_id, judge_name, role}]
        """
        query = """
        UNWIND $rows AS row
        MERGE (j:Judge {judge_id: apoc.text.slug(row.judge_name)})
        ON CREATE SET j.name = row.judge_name
        WITH j, row
        MATCH (c:Case {case_id: row.case_id})
        MERGE (c)-[r:DECIDED_BY]->(j)
        SET r.role = row.role
        """
        with self.driver.session() as session:
            session.run(query, rows=judges)

    def import_section_citations(self, cites_df: pd.DataFrame):
        """
        Create CITES edges: Case → Section
        Only creates where Section already exists (MATCH not MERGE for Section).
        """
        query = """
        UNWIND $rows AS row
        MATCH (c:Case    {case_id:    row.case_id})
        MATCH (s:Section {section_id: row.section_id})
        MERGE (c)-[r:CITES]->(s)
        SET r.context       = row.context,
            r.citation_type = coalesce(row.citation_type, "referred")
        """
        self._batch_write(query, cites_df)
        print(f"Imported {len(cites_df)} CITES edges")

    def import_precedent_citations(self, prec_df: pd.DataFrame):
        """
        Create CITES_CASE edges between Case nodes.
        Creates stub nodes for cited cases not in our dataset.
        """
        query = """
        UNWIND $rows AS row
        MATCH (c:Case {case_id: row.case_id})
        MERGE (cited:Case {citation_string: row.citation_string})
        ON CREATE SET cited.is_stub = true,
                      cited.year = toInteger(row.year),
                      cited.reporter = row.reporter
        MERGE (c)-[r:CITES_CASE]->(cited)
        SET r.raw_citation = row.raw
        """
        self._batch_write(query, prec_df)
        print(f"Imported {len(prec_df)} CITES_CASE edges")

    def _batch_write(self, query: str, df: pd.DataFrame, batch_size: int = 200):
        rows = df.where(pd.notna(df), None).to_dict('records')
        with self.driver.session() as session:
            for i in tqdm(range(0, len(rows), batch_size)):
                session.run(query, rows=rows[i:i + batch_size])
```

---

## 5. Cypher Verification (`cypher_cases/04_verify_cases.cypher`)

```cypher
// ── Case node stats ──────────────────────────────────────────────────────────
MATCH (c:Case) WHERE c.is_stub IS NULL OR c.is_stub = false
RETURN
  count(c)                                    AS total_cases,
  count(c.judgment_date)                      AS cases_with_date,
  count(c.outcome)                            AS cases_with_outcome,
  size([c IN collect(c) WHERE c.outcome = "Allowed"])    AS allowed,
  size([c IN collect(c) WHERE c.outcome = "Dismissed"])  AS dismissed;

// ── Citation stats ───────────────────────────────────────────────────────────
MATCH (c:Case)-[:CITES]->(s:Section)
RETURN s.act_id AS act, count(*) AS total_citations
ORDER BY total_citations DESC;

// ── Expected output for 2025 BNS/BNSS/BSA citations ─────────────────────────
// BNS_2023 citations should be non-zero
// (if zero, citation extraction may have failed)

// ── Most cited sections ──────────────────────────────────────────────────────
MATCH (c:Case)-[:CITES]->(s:Section)
RETURN s.section_id, s.heading, count(c) AS cited_by_n_cases
ORDER BY cited_by_n_cases DESC
LIMIT 20;

// ── Cross-law connection: 2025 case cites BNS section equivalent to IPC section ──
MATCH (c:Case {court: "Supreme Court of India"})
  -[:CITES]->(new_s:Section {act_id: "BNS_2023"})
  -[:EQUIVALENT_TO]->(old_s:Section {act_id: "IPC_1860"})
RETURN c.case_id, new_s.section_id, old_s.section_id
LIMIT 10;
// This is the "graph value" query — shows new law being applied with old law context

// ── Judge activity ───────────────────────────────────────────────────────────
MATCH (c:Case)-[:DECIDED_BY]->(j:Judge)
RETURN j.name, count(c) AS cases_decided
ORDER BY cases_decided DESC
LIMIT 15;

// ── Case type distribution ───────────────────────────────────────────────────
MATCH (c:Case) WHERE c.is_stub IS NULL
RETURN c.case_type, count(c) AS count
ORDER BY count DESC;

// ── Precedent network: which older cases are most cited in 2025? ─────────────
MATCH (new:Case {court: "Supreme Court of India"})-[:CITES_CASE]->(old:Case)
WHERE old.is_stub = true
RETURN old.citation_string, count(new) AS cited_by
ORDER BY cited_by DESC
LIMIT 20;
```

---

## 6. Master Orchestration (`run_cases_pipeline.py`)

```python
from pathlib import Path
from pipeline_cases.classify_pdfs import triage_pdfs
from pipeline_cases.pdf_to_text import extract_judgment_text
from pipeline_cases.judgment_parser import parse_judgment
from pipeline_cases.citation_extractor import extract_all_citations
from pipeline_cases.citation_resolver import resolve_citations
from pipeline_cases.neo4j_case_importer import CaseLawImporter
import pandas as pd
import json
from tqdm import tqdm
import os

# Config
JUDGEMENTS_DIR = "./judgements/sc_2025"
OUTPUT_DIR = "./output_cases"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def run():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    pdf_files = sorted(Path(JUDGEMENTS_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    # ── Stage 1: Triage ──
    print("\n[Stage 1] Triaging PDFs...")
    triage = triage_pdfs(JUDGEMENTS_DIR)
    valid = [t for t in triage if t["status"] == "ok" and t["text_extractable"]]
    pd.DataFrame(triage).to_csv(f"{OUTPUT_DIR}/triage_report.csv", index=False)
    print(f"  Valid: {len(valid)}, Skipped: {len(triage) - len(valid)}")

    # ── Stages 2-4: Extract, Parse, Cite ──
    all_cases, all_section_cites, all_act_cites, all_precedent_cites, all_judges = [], [], [], [], []

    print("\n[Stages 2-4] Extracting, parsing, citing...")
    for entry in tqdm(valid):
        pdf_path = f"{JUDGEMENTS_DIR}/{entry['file']}"
        extracted = extract_judgment_text(pdf_path)
        parsed = parse_judgment(extracted)
        citations = extract_all_citations(parsed["case_id"], parsed["full_text"])

        all_cases.append({k: v for k, v in parsed.items() if k not in ("full_text", "judges")})
        all_section_cites.extend(citations["section_cites"])
        all_act_cites.extend(citations["act_cites"])
        all_precedent_cites.extend(citations["precedent_cites"])

        for judge in parsed.get("judges", []):
            all_judges.append({"case_id": parsed["case_id"], "judge_name": judge, "role": "author"})

    # Save intermediates
    cases_df = pd.DataFrame(all_cases)
    sec_cites_df = pd.DataFrame(all_section_cites)
    prec_cites_df = pd.DataFrame(all_precedent_cites)
    cases_df.to_csv(f"{OUTPUT_DIR}/cases.csv", index=False)
    sec_cites_df.to_csv(f"{OUTPUT_DIR}/case_section_cites_raw.csv", index=False)
    pd.DataFrame(all_judges).to_csv(f"{OUTPUT_DIR}/judges.csv", index=False)
    print(f"  Cases parsed: {len(cases_df)}")
    print(f"  Raw section citations: {len(sec_cites_df)}")

    # ── Stage 5: Resolve citations ──
    print("\n[Stage 5] Resolving citations against Neo4j...")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    resolved, unresolved = resolve_citations(sec_cites_df, driver)
    resolved.to_csv(f"{OUTPUT_DIR}/case_section_cites.csv", index=False)
    unresolved.to_csv(f"{OUTPUT_DIR}/unresolved_cites.csv", index=False)

    # ── Stage 6: Import to Neo4j ──
    print("\n[Stage 6] Importing to Neo4j...")
    importer = CaseLawImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    importer.import_cases(cases_df)
    importer.import_judges(all_judges)
    importer.import_section_citations(resolved)
    importer.import_precedent_citations(prec_cites_df)
    print("\nDone! Run cypher_cases/04_verify_cases.cypher to validate.")

if __name__ == "__main__":
    run()
```

---

## 7. Known Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Inconsistent PDF formatting | Different judgment templates across benches | Parse by detecting structural markers (CORAM, JUDGMENT, ORDER) not fixed positions |
| Judgment date in multiple formats | No standard: "15.03.2025", "15th March 2025", "March 15, 2025" | Multi-format `normalize_date()` with fallback to raw string |
| Multi-case PDFs | Some files contain 2-3 connected matters | Detect "W.P. (C) No." lists; split on case number boundaries |
| Scanned/image-only PDFs | Older cases scanned before digital | Detected in Stage 1 triage; flag for manual OCR (tesseract) |
| "Read with" compound citations | "s.302 r/w s.34 IPC" | Explicit `READ_WITH_RE` pattern; create multiple CITES edges |
| Partial section number matches | "section 3(2)(v)" — regex greedy matching | Test patterns on 20+ diverse samples before pipeline run |
| Judge name variations | "D.Y. Chandrachud" vs "Chandrachud, C.J." | Normalize using canonical judge list with known aliases |
| Pre-2024 case law cited | BNS era precedent not yet established | Old-law citations still create edges via EQUIVALENT_TO traversal |

---

## 8. Pre-run Checklist

Before running this pipeline:

- [ ] Plan 1 completed — BNS/BNSS/BSA/Constitution nodes exist in Neo4j
- [ ] Constraints and indexes created (run `cypher/01_constraints.cypher` from Plan 1)
- [ ] Neo4j connection tested and credentials in `.env`
- [ ] `pip install -r requirements.txt` done
- [ ] `sc_judgements_2025/` directory contains all 400 PDFs
- [ ] Test pipeline on 5 PDFs first: `python run_cases_pipeline.py --sample 5`

---

## 9. Expected Final Graph Metrics (after both plans)

```
Nodes:
  Act           ~  10  (BNS, BNSS, BSA, Const + stubs for IPC/CrPC/IEA + others)
  Chapter       ~ 120  (across all acts)
  Section       ~1500  (BNS 358 + BNSS 531 + BSA 170 + Constitution ~395 + others)
  Definition    ~  90  (from s.2 of each Sanhita)
  Case          ~ 400  (2025 SC judgments)
  Judge         ~  30  (active SC bench 2025)

Relationships:
  HAS_SECTION          ~1500
  HAS_CHAPTER          ~ 120
  CITES (case→section) ~3000–8000  (estimated 8-20 per judgment avg)
  CITES_CASE           ~5000–15000 (precedents cited heavily)
  EQUIVALENT_TO        ~ 700       (BNS↔IPC + BSA↔IEA mappings)
  CROSS_ACT_REFERENCES ~ 200       (within-Sanhita trio cross-refs)
  REPEALS              ~   3       (BNS→IPC, BNSS→CrPC, BSA→IEA)
  DECIDED_BY           ~ 800       (400 cases × avg 2 judges)
```
