---
name: Fix Neo4j Pipeline Gaps
overview: Fix the 5 identified gaps in the Phase 1 -> Phase 2 Neo4j pipeline across 4 phases, with a git commit after each phase.
todos:
  - id: phase-1
    content: "Phase 1: Fix definitions.py -- guard regex, curly quotes, definition verbs. Unit test. Commit."
    status: completed
  - id: phase-2
    content: "Phase 2: Fix act_parser.py (part-chapter linkage) + citations.py (Constitution article xrefs). Unit test. Commit."
    status: completed
  - id: phase-3
    content: "Phase 3: Update copy script + smoke tests. Re-run full pipeline. Verify all CSVs. Commit."
    status: completed
  - id: phase-4
    content: "Phase 4: Load into Neo4j, run smoke tests, verify all counts. Commit."
    status: completed
isProject: false
---

# Fix Neo4j Knowledge Graph Pipeline Gaps (4 Phases)

## Root Cause Analysis

### Bug 1: Definitions extraction produces 0 rows

Three compounding bugs in [definitions.py](phase1_preprocessing/definitions.py):

- **Guard regex mismatch.** `DEFINITION_SECTION_RE` matches `"In this Act"` but the actual text says `"In this Sanhita"` (BNS/BNSS). The guard returns `[]` for every section.
- **Quote character mismatch.** `QUOTED_TERM_RE` uses straight `"` but the PDF text uses Unicode curly quotes `\u201c` / `\u201d` (108 occurrences in sections.csv). No term will ever match.
- **Missing definition verbs.** Regex only handles `means|includes|shall mean|shall include` but BNS s.2 also uses `"act" denotes`.

### Bug 2: Part-Chapter linkage has only 18 rows (should be ~70+)

In [act_parser.py](phase1_preprocessing/act_parser.py), `parse_parts_chapters()` sets `part_id: None` for every chapter. The linking never happens. In [structure_export.py](phase1_preprocessing/structure_export.py), edges are only created when `c.get("part_id")` is truthy.

### Bug 3: Constitution article cross-references dropped

In [citations.py](phase1_preprocessing/citations.py), `CROSS_ACT_REF_RE` only matches `section X of the Constitution of India`. But the acts reference the Constitution with `article X of the Constitution` (5 instances found). Need an additional regex.

### Bug 4: Copy script outdated

[copy_phase1_csvs_to_neo4j_import.ps1](scripts/copy_phase1_csvs_to_neo4j_import.ps1) only copies 5 old v1 files instead of all 17 v2 files.

### Bug 5: Smoke tests don't cover v2 schema

[04_smoke_tests.cypher](neo4j/cypher/04_smoke_tests.cypher) doesn't verify Part, Chapter, Definition counts or HAS_PART, HAS_CHAPTER, DEFINES_TERM, REFERENCES edges.

---

## Phase 1: Fix Definitions Extraction

**Files changed:** [definitions.py](phase1_preprocessing/definitions.py)

**Step 1.1 -- Fix `DEFINITION_SECTION_RE`** to also match "Sanhita" and "Adhiniyam":

```python
DEFINITION_SECTION_RE = re.compile(
    r"[Ii]n this (?:[Aa]ct|[Ss]anhita|[Aa]dhiniyam).{0,80}(?:unless|\u2014|\u2013|—|––)",
)
```

**Step 1.2 -- Fix `QUOTED_TERM_RE`** to handle curly quotes and add "denotes":

```python
QUOTED_TERM_RE = re.compile(
    r'[\u201c\u201d""]([^""\u201c\u201d]+)[\u201c\u201d""]\s+'
    r'(?:means|includes|denotes|shall mean|shall include)\s+'
    r'(.+?)(?=\n\s*\([a-z0-9]+\)|\n\s*[\u201c""]\w|$)',
    re.DOTALL | re.IGNORECASE,
)
```

**Step 1.3 -- Fix `DEFINED_TERM_RE`** with same quote character and verb fixes.

**Step 1.4 -- Unit test:** Write a quick test script that reads BNS s.2 and BNSS s.2 text from `sections.csv`, calls `extract_definitions_from_section()`, and asserts results contain expected terms ("act", "animal", "child", "counterfeit", "person", etc.). Expect 15-30 from BNS, 30-60 from BNSS.

**Step 1.5 -- Commit:** `git add phase1_preprocessing/definitions.py && git commit`

---

## Phase 2: Fix Part-Chapter Linkage + Constitution Cross-References

**Files changed:** [act_parser.py](phase1_preprocessing/act_parser.py), [citations.py](phase1_preprocessing/citations.py)

**Step 2.1 -- Fix Part-Chapter linkage in `act_parser.py`.**
Add a `_assign_part_to_chapters()` function (similar to `_assign_chapter_part` for sections) that walks the full text and assigns each chapter to the PART that precedes it by text position. Call it from `parse_parts_chapters()` before returning. Set `chapter["part_id"]` accordingly.

**Step 2.2 -- Add Constitution article cross-references in `citations.py`.**
Add a new regex:

```python
CROSS_ACT_ARTICLE_RE = re.compile(
    r'(?:article|art\.)\s*(\d+[A-Z]?)\s+of\s+the\s+Constitution(?:\s+of\s+India)?',
    re.IGNORECASE,
)
```

Add extraction logic in `extract_references_from_section()` that generates cross-references with `to_section_id = "CONST_1950_ArtN"` and `reference_type = "cross_act"`, `target_act_id = "CONST_1950"`.

**Step 2.3 -- Unit test Part-Chapter:** Parse BNS raw text, assert every chapter has a non-None `part_id`. Count chapters and verify >= 20.

**Step 2.4 -- Unit test Constitution xrefs:** Run `extract_references_from_section()` on BNSS section text containing "article 356 of the Constitution". Assert output includes `CONST_1950_Art356` as a cross_act reference.

**Step 2.5 -- Commit:** `git add phase1_preprocessing/act_parser.py phase1_preprocessing/citations.py && git commit`

---

## Phase 3: Update Scripts, Re-run Pipeline, Verify CSVs

**Files changed:** [copy_phase1_csvs_to_neo4j_import.ps1](scripts/copy_phase1_csvs_to_neo4j_import.ps1), [04_smoke_tests_v2.cypher](neo4j/cypher/04_smoke_tests_v2.cypher) (new)

**Step 3.1 -- Rewrite copy script** to copy all 17 v2 CSV files. Default source dir to `phase1_output_v2`. File list:
`acts.csv`, `parts.csv`, `chapters.csv`, `sections.csv`, `articles.csv`, `definitions.csv`, `section_defines_term.csv`, `cases_sc_neo4j.csv`, `cases_iltur_neo4j.csv`, `act_part.csv`, `part_chapter.csv`, `chapter_section.csv`, `act_section.csv`, `act_article.csv`, `section_references_section.csv`, `case_cites_section.csv`, `case_cites_article.csv`

**Step 3.2 -- Create `04_smoke_tests_v2.cypher`** with queries to:

- Count all 7 node types
- Count all 8 relationship types
- Cross-act REFERENCES distribution (from_act -> to_act)
- Top 20 cited provisions
- Orphan node checks (Parts without HAS_PART incoming, Chapters without HAS_CHAPTER incoming)

**Step 3.3 -- Re-run the Phase 1 pipeline** (`structure_export.py`) to regenerate all CSVs with the fixes from Phases 1 and 2.

**Step 3.4 -- Verify CSV output counts:**


| CSV                                          | Was       | Expected After Fix                    |
| -------------------------------------------- | --------- | ------------------------------------- |
| `definitions.csv`                            | 0 rows    | 85+ rows                              |
| `section_defines_term.csv`                   | 0 rows    | 85+ rows                              |
| `part_chapter.csv`                           | 18 rows   | 70+ rows                              |
| `section_references_section.csv` (cross_act) | 44        | 49+ (added Constitution article refs) |
| All other CSVs                               | unchanged | same as before                        |


**Step 3.5 -- Commit:** `git add scripts/ neo4j/cypher/ phase1_output_v2/ phase1_preprocessing/ && git commit`

---

## Phase 4: Load into Neo4j and Final Verification

**Step 4.1 -- Copy CSVs** to Neo4j import directory (using updated script or manually).

**Step 4.2 -- Clear existing graph** (if re-loading on same DB):

```cypher
MATCH (n) DETACH DELETE n;
```

Then drop old constraints if needed.

**Step 4.3 -- Run Cypher scripts in order:**

1. `01_constraints_v2.cypher`
2. `02_load_nodes_v2.cypher`
3. `03_load_edges_v2.cypher`

**Step 4.4 -- Run `04_smoke_tests_v2.cypher`** and verify counts against expectations:


| Node/Edge    | Expected      |
| ------------ | ------------- |
| Acts         | 4             |
| Parts        | ~50           |
| Chapters     | ~72+          |
| Sections     | ~1,000+       |
| Articles     | ~395+         |
| Definitions  | 85+           |
| Cases        | ~90,000+      |
| HAS_PART     | ~50+          |
| HAS_CHAPTER  | ~72+ (was 18) |
| HAS_SECTION  | ~1,000+       |
| HAS_ARTICLE  | ~395+         |
| IN_ACT       | ~1,400+       |
| DEFINES_TERM | 85+ (was 0)   |
| REFERENCES   | ~15,800+      |
| CITES        | ~53,000+      |


**Step 4.5 -- Spot-check key paths** in Neo4j Browser:

```cypher
-- Definition path: Act -> Section -> Definition
MATCH (a:Act {act_id:"BNS_2023"})-[:HAS_SECTION]->(s:Section)-[:DEFINES_TERM]->(d:Definition)
RETURN s.section_id, d.term LIMIT 10;

-- Part-Chapter path
MATCH (a:Act {act_id:"BNS_2023"})-[:HAS_PART]->(p:Part)-[:HAS_CHAPTER]->(c:Chapter)
RETURN p.part_number, c.chapter_number, c.chapter_title LIMIT 10;

-- Constitution cross-ref path
MATCH (s:Section)-[r:REFERENCES {reference_type:"cross_act"}]->(ar:Article)
WHERE ar.act_id = "CONST_1950"
RETURN s.section_id, ar.article_id LIMIT 10;

-- Case citing path
MATCH (c:Case)-[:CITES]->(s:Section {act_id:"BNS_2023"})
RETURN c.case_id, s.section_id LIMIT 5;
```

**Step 4.6 -- Commit:** `git add . && git commit`