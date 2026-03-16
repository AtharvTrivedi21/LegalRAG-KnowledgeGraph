# Plan 1: Static Legal Corpus → Neo4j Knowledge Graph
## Constitution of India, BNS, BNSS, BSA + Act Indices

---

## 0. Context & Goal

We are building a Neo4j knowledge graph for Indian law. This plan covers the **static, authoritative corpus** — the primary legal texts that form the schema backbone of the graph. Every other dataset (case law, etc.) will hang off this structure.

**Input files:**
- `1_BNS.pdf` — Bharatiya Nyaya Sanhita 2023 (Act 45/2023), penal code
- `2_Bharatiya_nagrik_Suraksha_sanhita.pdf` — Bharatiya Nagarik Suraksha Sanhita 2023 (Act 46/2023), procedure
- `3_Bharatiya_Sakshya_Adhiniyam.pdf` — Bharatiya Sakshya Adhiniyam 2023 (Act 47/2023), evidence
- `Constitution_Of_India.pdf` — Constitution of India 1950
- `Alphabetical_List_of_Central_Acts.pdf` — Index of ~891 central acts (alphabetical)
- `Chronological_List_of_Central_Acts.pdf` — Same acts sorted by year (1836–2025)

**Output:** A fully populated Neo4j database with nodes and relationships for every Act, Part, Chapter, Section, Definition, and cross-reference found in these documents.

---

## 1. Target Graph Schema

### 1.1 Node Labels & Properties

```
(:Act)
  act_id          String  PK  e.g. "BNS_2023", "CONST_1950"
  short_title     String      "Bharatiya Nyaya Sanhita"
  long_title      String      Full preamble title
  year            Integer     2023
  act_number      Integer     45
  act_type        String      "Penal" | "Procedure" | "Evidence" | "Constitutional" | "General"
  enforcement_date String     "2024-07-01"
  source_file     String      "1_BNS.pdf"

(:Part)
  part_id         String  PK  e.g. "BNS_PART_1"
  part_number     String      "I", "II", "III"
  part_title      String      "General Exceptions"
  act_id          String      FK → Act

(:Chapter)
  chapter_id      String  PK  e.g. "BNS_CH_3"
  chapter_number  String      "III"
  chapter_title   String      "Of Punishments"
  part_id         String      FK → Part (nullable)
  act_id          String      FK → Act

(:Section)
  section_id      String  PK  e.g. "BNS_s64", "CONST_Art21"
  section_number  String      "64", "21", "2(1)(a)"
  heading         String      "Punishment for rape"
  full_text       String      Complete section text including sub-sections
  has_proviso     Boolean
  has_exception   Boolean
  has_illustration Boolean
  act_id          String      FK → Act
  chapter_id      String      FK → Chapter

(:Definition)
  def_id          String  PK  e.g. "BNS_DEF_person"
  term            String      "person"
  defined_text    String      Full definition text
  act_id          String      FK → Act
  section_id      String      FK → Section where defined

(:Offence)
  offence_id      String  PK  e.g. "BNS_OFFENCE_s64"
  description     String      Short description
  punishment_text String      Full punishment clause text
  is_bailable     Boolean
  is_cognizable   Boolean
  section_id      String      FK → Section

(:ActIndex)
  index_id        String  PK  e.g. "IDX_IPC_1860"
  short_title     String      "Indian Penal Code"
  year            Integer     1860
  act_number      String      "45"
  category        String      "Criminal" | "Civil" | "Tax" etc.
```

### 1.2 Relationship Types

```
(Act)-[:HAS_PART]->(Part)
(Act)-[:HAS_CHAPTER]->(Chapter)          // for acts without Parts
(Act)-[:HAS_SECTION]->(Section)
(Part)-[:HAS_CHAPTER]->(Chapter)
(Chapter)-[:HAS_SECTION]->(Section)
(Act)-[:DEFINES]->(Definition)
(Section)-[:DEFINES_TERM]->(Definition)
(Section)-[:REFERENCES]->(Section)       // cross-section citations within same act
  props: {context: String, reference_type: "see_also"|"subject_to"|"notwithstanding"}
(Section)-[:CROSS_ACT_REFERENCES]->(Section)  // citations to other acts
  props: {target_act_id: String, context: String}
(Act)-[:REPEALS]->(Act)
  props: {section: String, effective_date: String}
(Act)-[:AMENDS]->(Act)
  props: {section: String}
(Act)-[:REPLACES]->(Act)                 // BNS replaces IPC, etc.
  props: {reason: String}
(Section)-[:EQUIVALENT_TO]->(Section)   // BNS s.X ≈ IPC s.Y
  props: {equivalence_type: "direct"|"modified"|"merged"|"split"}
(Section)-[:DESCRIBES_OFFENCE]->(Offence)
(Section)-[:USES_DEFINITION]->(Definition)
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    INPUT LAYER                       │
│  PDF files (Constitution, BNS, BNSS, BSA, indices)  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               EXTRACTION LAYER                       │
│  pdf_extractor.py — PyMuPDF / pdfplumber             │
│  Outputs: raw_text per page, detected headings       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               PARSING LAYER                          │
│  structure_parser.py — regex + rule-based            │
│  Segments text into Act/Chapter/Section/Subsection   │
│  Extracts: headings, section numbers, full text      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│             ENRICHMENT LAYER                         │
│  definition_extractor.py  — finds s.2 definitions   │
│  citation_extractor.py    — finds cross-references   │
│  equivalence_mapper.py    — maps BNS→IPC, etc.       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              SERIALIZATION LAYER                     │
│  Outputs structured JSON/CSV files:                  │
│  acts.csv, sections.csv, definitions.csv,            │
│  citations.csv, equivalences.csv                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               IMPORT LAYER                           │
│  neo4j_importer.py — MERGE via neo4j Python driver   │
│  OR: neo4j-admin bulk import for large datasets      │
└─────────────────────────────────────────────────────┘
```

---

## 3. Implementation Steps

### Step 3.1 — Project Setup

```
legal-kg/
├── corpus/                      # Raw PDF files (read-only)
│   ├── 1_BNS.pdf
│   ├── 2_BNSS.pdf
│   ├── 3_BSA.pdf
│   ├── Constitution_Of_India.pdf
│   ├── Alphabetical_List_of_Central_Acts.pdf
│   └── Chronological_List_of_Central_Acts.pdf
├── pipeline/
│   ├── pdf_extractor.py         # Step 1: PDF → raw text
│   ├── structure_parser.py      # Step 2: raw text → structured dict
│   ├── definition_extractor.py  # Step 3a: extract definitions
│   ├── citation_extractor.py    # Step 3b: extract cross-refs
│   ├── equivalence_mapper.py    # Step 3c: map to old laws
│   ├── neo4j_importer.py        # Step 4: load into Neo4j
│   └── config.py                # Act metadata, regex patterns
├── output/
│   ├── acts.csv
│   ├── chapters.csv
│   ├── sections.csv
│   ├── definitions.csv
│   ├── citations.csv
│   └── equivalences.csv
├── cypher/
│   ├── 01_constraints.cypher    # Indexes & uniqueness constraints
│   ├── 02_import_acts.cypher
│   ├── 03_import_sections.cypher
│   ├── 04_import_relationships.cypher
│   └── 05_verify.cypher
├── requirements.txt
└── run_pipeline.py              # Master orchestration script
```

**requirements.txt:**
```
pymupdf>=1.23.0       # fitz — best for structured PDF extraction
pdfplumber>=0.10.0    # fallback for complex table parsing
pandas>=2.0.0
neo4j>=5.0.0
tqdm
python-dotenv
```

---

### Step 3.2 — PDF Extraction (`pdf_extractor.py`)

**Purpose:** Convert PDFs to structured raw text, preserving heading hierarchy signals.

```python
import fitz  # PyMuPDF
import re
from pathlib import Path

ACT_CONFIG = {
    "BNS_2023": {
        "file": "1_BNS.pdf",
        "act_number": 45,
        "short_title": "Bharatiya Nyaya Sanhita",
        "section_prefix": "BNS",
        "article_type": "section",        # sections (not articles)
        "enforcement_date": "2024-07-01"
    },
    "BNSS_2023": {
        "file": "2_Bharatiya_nagrik_Suraksha_sanhita.pdf",
        "act_number": 46,
        "short_title": "Bharatiya Nagarik Suraksha Sanhita",
        "section_prefix": "BNSS",
        "article_type": "section",
        "enforcement_date": "2024-07-01"
    },
    "BSA_2023": {
        "file": "3_Bharatiya_Sakshya_Adhiniyam.pdf",
        "act_number": 47,
        "short_title": "Bharatiya Sakshya Adhiniyam",
        "section_prefix": "BSA",
        "article_type": "section",
        "enforcement_date": "2024-07-01"
    },
    "CONST_1950": {
        "file": "Constitution_Of_India.pdf",
        "act_number": None,
        "short_title": "Constitution of India",
        "section_prefix": "CONST",
        "article_type": "article",        # Articles, not sections
        "enforcement_date": "1950-01-26"
    }
}

def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Extract pages with font-size metadata to detect headings.
    Returns list of {page_num, text, blocks} dicts.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_text = page.get_text("text")
        pages.append({
            "page_num": page_num + 1,
            "text": page_text,
            "blocks": blocks
        })
    return pages

def detect_font_sizes(pages: list[dict]) -> dict:
    """
    Analyze font sizes across document to identify heading levels.
    Returns {font_size: heading_level} mapping.
    """
    size_counts = {}
    for page in pages:
        for block in page["blocks"]:
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span["size"])
                        size_counts[size] = size_counts.get(size, 0) + 1

    # Body text is the most common size; anything larger is a heading
    body_size = max(size_counts, key=size_counts.get)
    heading_sizes = sorted([s for s in size_counts if s > body_size], reverse=True)

    level_map = {}
    for i, size in enumerate(heading_sizes[:4]):  # max 4 heading levels
        level_map[size] = i + 1
    level_map[body_size] = 0  # body text
    return level_map
```

**Key considerations:**
- Indian legal PDFs often have multi-column layouts — use `page.get_text("blocks")` not `"text"` to preserve column order
- Section numbers may appear as bold text mid-paragraph, not as separate headings
- BSA has "Illustrations" in italic — detect and preserve separately
- Constitution has Schedule headers that look like Chapter headers

---

### Step 3.3 — Structure Parser (`structure_parser.py`)

**Purpose:** Convert raw extracted text into a hierarchical Python dict representing the Act's structure.

**Critical regex patterns:**

```python
# ── Section number patterns ────────────────────────────────────────────────
# BNS/BNSS/BSA: "1.", "2.", "34A.", "100."  (standalone at line start)
SECTION_NUM_RE = re.compile(
    r'^(\d+[A-Z]?)\.\s+([A-Z][^\n]+)',   # "64. Punishment for rape"
    re.MULTILINE
)

# Constitution: "Article 21", "Article 21A", "Article 356"
ARTICLE_NUM_RE = re.compile(
    r'^(\d+[A-Z]?)\.\s+([A-Z][^\n]+)',
    re.MULTILINE
)

# Sub-section: "(1)", "(2)", "(a)", "(i)"
SUBSECTION_RE = re.compile(r'^\((\d+|[a-z]|[ivxlcdm]+)\)\s+', re.MULTILINE)

# ── Chapter patterns ──────────────────────────────────────────────────────
CHAPTER_RE = re.compile(
    r'CHAPTER\s+([IVXLC\d]+)\s*\n\s*([A-Z][^\n]+)',
    re.MULTILINE | re.IGNORECASE
)

PART_RE = re.compile(
    r'PART\s+([IVXLC\d]+)\s*\n\s*([A-Z][^\n]+)',
    re.MULTILINE | re.IGNORECASE
)

# ── Definition detection ─────────────────────────────────────────────────
# "In this Act, unless the context otherwise requires,—"
# "(a) "person" means..."
DEFINITION_SECTION_RE = re.compile(
    r'[Ii]n this [Aa]ct.{0,60}(?:unless|—)',
)

# Individual defined term: "word" means / "expression" includes
DEFINED_TERM_RE = re.compile(
    r'"([^"]+)"\s+(?:means|includes|shall mean|shall include)\s+(.+?)(?=\n\([a-z]\)|$)',
    re.DOTALL
)

# ── Cross-reference patterns ─────────────────────────────────────────────
# Within-act: "section 64", "sub-section (1) of section 64"
INTRA_REF_RE = re.compile(
    r'(?:sub-section\s*\([^)]+\)\s+of\s+)?'
    r'section\s+(\d+[A-Z]?(?:\([^)]+\))*)',
    re.IGNORECASE
)

# Cross-act references (BNS/BNSS/BSA explicitly name each other)
CROSS_ACT_REF_RE = re.compile(
    r'(?:section|s\.)\s*(\d+[A-Z]?)\s+of\s+the\s+'
    r'(Bharatiya Nyaya Sanhita|Bharatiya Nagarik Suraksha Sanhita|'
    r'Bharatiya Sakshya Adhiniyam|Indian Penal Code|Code of Criminal Procedure|'
    r'Indian Evidence Act|Constitution of India)',
    re.IGNORECASE
)

# ── Repeal detection ─────────────────────────────────────────────────────
REPEAL_RE = re.compile(
    r'(?:hereby? )?repeal(?:ed|s)?\s+the\s+(.+?)\s*(?:\(Act|\d)',
    re.IGNORECASE
)
```

**Parser output structure:**
```python
{
    "act_id": "BNS_2023",
    "metadata": { ... },
    "parts": [
        {
            "part_id": "BNS_PART_1",
            "number": "I",
            "title": "Preliminary",
            "chapters": [
                {
                    "chapter_id": "BNS_CH_1",
                    "number": "I",
                    "title": "Definitions",
                    "sections": [
                        {
                            "section_id": "BNS_s1",
                            "number": "1",
                            "heading": "Short title, commencement and application",
                            "full_text": "...",
                            "subsections": [...],
                            "intra_references": ["BNS_s2", "BNS_s6"],
                            "cross_references": [
                                {"target_act": "BNSS_2023", "section": "2"}
                            ],
                            "illustrations": [...],
                            "has_proviso": True
                        }
                    ]
                }
            ]
        }
    ]
}
```

---

### Step 3.4 — Definition Extractor (`definition_extractor.py`)

**Purpose:** Extract all defined terms from Section 2 (Definitions) of each act.

```python
# BNS s.2 defines: "act", "animal", "child", "document", "electronic record",
#                  "gender", "harm", "injury", "judge", "local authority",
#                  "money", "movable property", "omission", "person",
#                  "public servant", "reason to believe", "valuable security", etc.

# BSA s.2 defines: "admission", "court", "document", "electronic record",
#                  "evidence", "fact", "fact in issue", "proved", etc.

def extract_definitions(section_text: str, act_id: str, section_id: str) -> list[dict]:
    definitions = []
    # Split on sub-clause markers: (a), (b), (c)...
    clauses = re.split(r'\n\s*\(([a-z])\)\s+', section_text)
    for clause in clauses:
        match = DEFINED_TERM_RE.search(clause)
        if match:
            definitions.append({
                "def_id": f"{act_id}_DEF_{match.group(1).lower().replace(' ', '_')}",
                "term": match.group(1),
                "defined_text": match.group(2).strip(),
                "act_id": act_id,
                "section_id": section_id
            })
    return definitions
```

---

### Step 3.5 — Act Index Parser (Alphabetical + Chronological PDFs)

**Purpose:** Build a catalogue of all 891 Central Acts for the `:ActIndex` nodes. These enable queries like "what acts existed before BNS was passed?" and support future expansion.

```python
def parse_act_index(pdf_path: str) -> list[dict]:
    """
    Parse the alphabetical/chronological list PDFs.
    Each row: Act Name | Year | Act Number | (sometimes: remarks)
    """
    import pdfplumber

    acts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if row and row[0] and not row[0].startswith('S.No'):
                        acts.append({
                            "short_title": row[1] if len(row) > 1 else row[0],
                            "year": extract_year(row),
                            "act_number": extract_act_number(row),
                            "category": classify_act(row[1] if len(row) > 1 else row[0])
                        })
    return acts

def classify_act(title: str) -> str:
    """Classify act into category based on keywords in title."""
    title_lower = title.lower()
    if any(w in title_lower for w in ['penal', 'criminal', 'punishment', 'offence']):
        return 'Criminal'
    elif any(w in title_lower for w in ['tax', 'income', 'customs', 'excise', 'gst']):
        return 'Tax'
    elif any(w in title_lower for w in ['civil', 'procedure', 'arbitration']):
        return 'Civil Procedure'
    elif any(w in title_lower for w in ['constitution', 'amendment']):
        return 'Constitutional'
    elif any(w in title_lower for w in ['company', 'corporation', 'trade']):
        return 'Commercial'
    else:
        return 'General'
```

---

### Step 3.6 — Equivalence Mapper (`equivalence_mapper.py`)

**Purpose:** Map BNS/BNSS/BSA sections to their IPC/CrPC/IEA equivalents. This is critical for case law that cites old laws — we can traverse `EQUIVALENT_TO` edges to connect old-law citations to new-law sections.

```python
# Hard-coded equivalence table (from official government comparison tables)
BNS_TO_IPC = {
    "BNS_s1": "IPC_s1",          # Short title
    "BNS_s2": "IPC_s2,IPC_s3",   # Definitions (merged)
    "BNS_s64": "IPC_s375,IPC_s376",  # Rape (consolidated)
    "BNS_s103": "IPC_s302",      # Murder
    "BNS_s304": "IPC_s499",      # Defamation
    # ... (full table has ~350 mappings)
}

BSA_TO_IEA = {
    "BSA_s1": "IEA_s1",
    "BSA_s2": "IEA_s3",          # Definitions expanded
    "BSA_s23": "IEA_s17",        # Admissions
    "BSA_s48": None,             # New section (no IEA equivalent)
    # ...
}

def generate_equivalences(bns_to_ipc: dict) -> list[dict]:
    edges = []
    for new_id, old_ids in bns_to_ipc.items():
        for old_id in old_ids.split(','):
            eq_type = "direct" if ',' not in old_ids else "merged"
            edges.append({
                "from_section": new_id.strip(),
                "to_section": old_id.strip(),
                "equivalence_type": eq_type
            })
    return edges
```

**Note:** Source the full equivalence tables from the official MHA (Ministry of Home Affairs) circulars published in 2023 when the new laws were notified. These are public documents.

---

### Step 3.7 — Neo4j Import (`neo4j_importer.py`)

```python
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

class LegalKGImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def setup_constraints(self):
        """Run first — creates indexes for fast MERGE operations."""
        constraints = [
            "CREATE CONSTRAINT act_id IF NOT EXISTS FOR (a:Act) REQUIRE a.act_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE",
            "CREATE CONSTRAINT def_id IF NOT EXISTS FOR (d:Definition) REQUIRE d.def_id IS UNIQUE",
            "CREATE INDEX section_number IF NOT EXISTS FOR (s:Section) ON (s.section_number)",
            "CREATE INDEX act_year IF NOT EXISTS FOR (a:Act) ON (a.year)",
        ]
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)

    def import_acts(self, acts_df: pd.DataFrame):
        query = """
        UNWIND $rows AS row
        MERGE (a:Act {act_id: row.act_id})
        SET a.short_title = row.short_title,
            a.year = toInteger(row.year),
            a.act_number = toInteger(row.act_number),
            a.act_type = row.act_type,
            a.enforcement_date = row.enforcement_date,
            a.source_file = row.source_file
        """
        self._batch_write(query, acts_df)

    def import_sections(self, sections_df: pd.DataFrame):
        query = """
        UNWIND $rows AS row
        MERGE (s:Section {section_id: row.section_id})
        SET s.section_number = row.section_number,
            s.heading = row.heading,
            s.full_text = row.full_text,
            s.act_id = row.act_id,
            s.chapter_id = row.chapter_id,
            s.has_proviso = toBoolean(row.has_proviso)
        WITH s, row
        MATCH (a:Act {act_id: row.act_id})
        MERGE (a)-[:HAS_SECTION]->(s)
        WITH s, row
        MATCH (c:Chapter {chapter_id: row.chapter_id})
        MERGE (c)-[:HAS_SECTION]->(s)
        """
        self._batch_write(query, sections_df)

    def import_cross_references(self, citations_df: pd.DataFrame):
        """
        Creates REFERENCES and CROSS_ACT_REFERENCES edges.
        Only creates edges where BOTH sections already exist.
        """
        query = """
        UNWIND $rows AS row
        MATCH (s1:Section {section_id: row.from_section_id})
        MATCH (s2:Section {section_id: row.to_section_id})
        MERGE (s1)-[r:CROSS_ACT_REFERENCES]->(s2)
        SET r.context = row.context,
            r.target_act_id = row.target_act_id
        """
        self._batch_write(query, citations_df)

    def import_repeal_relationships(self):
        """Hard-coded repeal relationships from the acts themselves."""
        repeals = [
            # BNS s.358 repeals IPC
            ("BNS_2023", "IPC_1860", "BNS_s358", "2024-07-01"),
            # BNSS s.531 repeals CrPC
            ("BNSS_2023", "CrPC_1973", "BNSS_s531", "2024-07-01"),
            # BSA s.170 repeals IEA
            ("BSA_2023", "IEA_1872", "BSA_s170", "2024-07-01"),
        ]
        query = """
        UNWIND $rows AS row
        MATCH (new:Act {act_id: row[0]})
        MERGE (old:Act {act_id: row[1]})  // creates stub for old acts
        ON CREATE SET old.short_title = row[1], old.is_repealed = true
        MERGE (new)-[r:REPEALS]->(old)
        SET r.section = row[2], r.effective_date = row[3]
        """
        with self.driver.session() as session:
            session.run(query, rows=repeals)

    def _batch_write(self, query: str, df: pd.DataFrame, batch_size: int = 500):
        rows = df.where(pd.notna(df), None).to_dict('records')
        with self.driver.session() as session:
            for i in tqdm(range(0, len(rows), batch_size)):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
```

---

### Step 3.8 — Cypher Setup (`cypher/01_constraints.cypher`)

Run this before any imports:

```cypher
// Uniqueness constraints (also create indexes automatically)
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Act)        REQUIRE a.act_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section)    REQUIRE s.section_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chapter)    REQUIRE c.chapter_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part)       REQUIRE p.part_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Definition) REQUIRE d.def_id IS UNIQUE;

// Additional indexes for frequent query patterns
CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.section_number);
CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.act_id);
CREATE INDEX IF NOT EXISTS FOR (a:Act)    ON (a.year);
CREATE INDEX IF NOT EXISTS FOR (a:Act)    ON (a.act_type);
CREATE FULLTEXT INDEX section_text IF NOT EXISTS FOR (s:Section) ON EACH [s.full_text, s.heading];
CREATE FULLTEXT INDEX def_text     IF NOT EXISTS FOR (d:Definition) ON EACH [d.term, d.defined_text];
```

---

### Step 3.9 — Verification Queries (`cypher/05_verify.cypher`)

Run after import to validate the graph:

```cypher
// ── Node counts ─────────────────────────────────────────────────────────
MATCH (a:Act)        RETURN "Acts" AS label, count(a) AS count
UNION ALL
MATCH (c:Chapter)    RETURN "Chapters", count(c)
UNION ALL
MATCH (s:Section)    RETURN "Sections", count(s)
UNION ALL
MATCH (d:Definition) RETURN "Definitions", count(d);

// ── Expected counts (approximate) ───────────────────────────────────────
// Acts:        4+ (BNS, BNSS, BSA, Constitution + stubs for IPC/CrPC/IEA)
// Sections:    BNS ~358, BNSS ~531, BSA ~170, Constitution ~395 ≈ 1450+
// Definitions: BNS s.2 ~20, BNSS s.2 ~50+, BSA s.2 ~15 ≈ 85+

// ── Cross-reference integrity check ─────────────────────────────────────
MATCH (s1:Section)-[r:CROSS_ACT_REFERENCES]->(s2:Section)
RETURN s1.act_id AS from_act, s2.act_id AS to_act, count(r) AS edge_count
ORDER BY edge_count DESC;
// Should show high counts for BSA→BNS (BSA explicitly references BNS s.64-78)

// ── Equivalence coverage ─────────────────────────────────────────────────
MATCH (s:Section {act_id: "BNS_2023"})
OPTIONAL MATCH (s)-[:EQUIVALENT_TO]->(old:Section)
RETURN
  count(s) AS total_bns_sections,
  count(old) AS sections_with_equivalence,
  count(s) - count(old) AS sections_without_equivalence;

// ── Sample path query (should return results) ─────────────────────────────
MATCH path = (bsa:Act {act_id:"BSA_2023"})
  -[:HAS_SECTION]->(s:Section)
  -[:CROSS_ACT_REFERENCES]->(s2:Section)
  <-[:HAS_SECTION]-(bns:Act {act_id:"BNS_2023"})
RETURN s.section_id, s2.section_id
LIMIT 10;
```

---

## 4. Known Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| OCR artifacts in PDFs | Scanned source documents | Use PyMuPDF's font-based extraction; flag sections with >5% non-ASCII chars |
| Multi-column layout | Indian Gazette format | Use block-based extraction, sort blocks by x-coordinate |
| Section numbers mid-paragraph | Bold inline numbers | Detect by font weight in PDF metadata, not just line-start regex |
| Sub-sections split across pages | Page breaks in PDF | Merge consecutive pages before section-splitting |
| Constitution numbering (1, 1A, 1B) | Amendment insertions | Handle `\d+[A-Z]?` pattern; maintain insertion order |
| Act index table format varies | Different PDF layouts | Use pdfplumber table extraction with column detection fallback |
| Definitions with cross-references | "X means Y as defined in section Z" | Parse definitions first, then resolve references in second pass |

---

## 5. Execution Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py --mode static_corpus

# OR run steps individually:
python pipeline/pdf_extractor.py       # outputs: output/raw_text/
python pipeline/structure_parser.py    # outputs: output/structured/
python pipeline/definition_extractor.py
python pipeline/citation_extractor.py
python pipeline/equivalence_mapper.py
python pipeline/neo4j_importer.py      # loads all output CSVs into Neo4j

# 3. Verify in Neo4j Browser
# Run: cypher/05_verify.cypher
```

---

## 6. Environment Variables (`.env`)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
CORPUS_DIR=./corpus
OUTPUT_DIR=./output
LOG_LEVEL=INFO
```
