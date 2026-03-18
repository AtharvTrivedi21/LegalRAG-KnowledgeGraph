"""
Phase 4: Load all v2 CSVs into Neo4j and run smoke tests.

Requires:
  - Neo4j running on bolt://localhost:7687 (or NEO4J_URI env var)
  - .env file with NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  - All v2 CSVs in phase1_output_v2/ (run run_pipeline.py first)
  - neo4j Python driver: pip install neo4j

The script:
  1. Connects to Neo4j
  2. Clears the existing graph (DETACH DELETE)
  3. Runs 01_constraints_v2.cypher
  4. Loads all nodes and edges using inline Cypher with UNWIND (fast, no LOAD CSV)
  5. Runs smoke tests and prints counts
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load credentials from .env
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
OUTPUT_DIR = ROOT / "phase1_output_v2"

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed. Run: pip install neo4j")
    sys.exit(1)

csv.field_size_limit(10 * 1024 * 1024)


def read_csv(filename: str) -> list[dict]:
    path = OUTPUT_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_query(session, query: str, params: dict = None):
    result = session.run(query, params or {})
    return result.consume()


def batch_unwind(session, query: str, rows: list[dict], batch_size: int = 2000):
    """Run an UNWIND query in batches to avoid memory issues."""
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        result = session.run(query, {"rows": batch})
        summary = result.consume()
        total += summary.counters.nodes_created + summary.counters.relationships_created
    return total


def main():
    print(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to Neo4j: {e}")
        sys.exit(1)

    with driver.session() as session:
        # ================================================================
        # Step 1: Clear existing graph
        # ================================================================
        print("\n[1/5] Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")
        r = session.run("MATCH (n) RETURN count(n) AS cnt")
        remaining = r.single()["cnt"]
        print(f"  Nodes remaining after delete: {remaining}")

        # ================================================================
        # Step 2: Create constraints and indexes
        # ================================================================
        print("\n[2/5] Creating constraints and indexes...")
        constraints = [
            "CREATE CONSTRAINT act_id IF NOT EXISTS FOR (a:Act) REQUIRE a.act_id IS UNIQUE",
            "CREATE CONSTRAINT part_id IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT def_id IF NOT EXISTS FOR (d:Definition) REQUIRE d.def_id IS UNIQUE",
            "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
            "CREATE INDEX section_act_id IF NOT EXISTS FOR (s:Section) ON (s.act_id)",
            "CREATE INDEX article_act_id IF NOT EXISTS FOR (a:Article) ON (a.act_id)",
            "CREATE INDEX case_source IF NOT EXISTS FOR (c:Case) ON (c.source)",
        ]
        for stmt in constraints:
            try:
                session.run(stmt)
            except Exception as e:
                print(f"  Constraint/index warning: {e}")
        print("  Done.")

        # ================================================================
        # Step 3: Load nodes
        # ================================================================
        print("\n[3/5] Loading nodes...")

        # Acts
        rows = read_csv("acts.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (a:Act {act_id: row.act_id})
            SET a.short_title = row.short_title,
                a.year = toInteger(row.year),
                a.act_number = toInteger(row.act_number),
                a.act_type = row.act_type,
                a.source_file = row.source_file,
                a.enforcement_date = row.enforcement_date
        """, rows)
        print(f"  Acts: {len(rows)} rows loaded")

        # Parts
        rows = read_csv("parts.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (p:Part {part_id: row.part_id})
            SET p.part_number = row.part_number,
                p.part_title = row.part_title,
                p.act_id = row.act_id
        """, rows)
        print(f"  Parts: {len(rows)} rows loaded")

        # Chapters
        rows = read_csv("chapters.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (c:Chapter {chapter_id: row.chapter_id})
            SET c.chapter_number = row.chapter_number,
                c.chapter_title = row.chapter_title,
                c.act_id = row.act_id,
                c.part_id = row.part_id
        """, rows)
        print(f"  Chapters: {len(rows)} rows loaded")

        # Sections (in batches of 500 due to full_text size)
        rows = read_csv("sections.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (s:Section {section_id: row.section_id})
            SET s.act_id = row.act_id,
                s.chapter_id = row.chapter_id,
                s.section_number = row.section_number,
                s.heading = row.heading,
                s.full_text = row.full_text
        """, rows, batch_size=500)
        print(f"  Sections: {len(rows)} rows loaded")

        # Articles
        rows = read_csv("articles.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (a:Article {article_id: row.article_id})
            SET a.act_id = row.act_id,
                a.article_number = row.article_number,
                a.heading = row.heading,
                a.full_text = row.full_text
        """, rows, batch_size=500)
        print(f"  Articles: {len(rows)} rows loaded")

        # Definitions
        rows = read_csv("definitions.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (d:Definition {def_id: row.def_id})
            SET d.term = row.term,
                d.defined_text = row.defined_text,
                d.act_id = row.act_id,
                d.section_id = row.section_id
        """, rows)
        print(f"  Definitions: {len(rows)} rows loaded")

        # Cases (SC)
        rows = read_csv("cases_sc_neo4j.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (c:Case {case_id: row.case_id})
            SET c.year = row.year,
                c.source = row.source,
                c.source_file = row.source_file,
                c.judgment_text = row.judgment_text
        """, rows)
        print(f"  Cases (SC): {len(rows)} rows loaded")

        # Cases (IL-TUR)
        rows = read_csv("cases_iltur_neo4j.csv")
        n = batch_unwind(session, """
            UNWIND $rows AS row
            MERGE (c:Case {case_id: row.case_id})
            SET c.year = row.year,
                c.source = row.source,
                c.judgment_text = row.judgment_text
        """, rows)
        print(f"  Cases (IL-TUR): {len(rows)} rows loaded")

        # ================================================================
        # Step 4: Load edges
        # ================================================================
        print("\n[4/5] Loading edges...")

        # ACT -> HAS_PART -> Part
        rows = read_csv("act_part.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (a:Act {act_id: row.act_id})
            MATCH (p:Part {part_id: row.part_id})
            MERGE (a)-[:HAS_PART]->(p)
        """, rows)
        print(f"  HAS_PART: {len(rows)} edges")

        # Part -> HAS_CHAPTER -> Chapter
        rows = read_csv("part_chapter.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (p:Part {part_id: row.part_id})
            MATCH (c:Chapter {chapter_id: row.chapter_id})
            MERGE (p)-[:HAS_CHAPTER]->(c)
        """, rows)
        print(f"  HAS_CHAPTER: {len(rows)} edges")

        # Chapter -> HAS_SECTION -> Section
        rows = read_csv("chapter_section.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (c:Chapter {chapter_id: row.chapter_id})
            MATCH (s:Section {section_id: row.section_id})
            MERGE (c)-[:HAS_SECTION]->(s)
        """, rows)
        print(f"  HAS_SECTION: {len(rows)} edges")

        # Act -> IN_ACT (section)
        rows = read_csv("act_section.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (s:Section {section_id: row.section_id})
            MATCH (a:Act {act_id: row.act_id})
            MERGE (s)-[:IN_ACT]->(a)
        """, rows)
        print(f"  IN_ACT (Section): {len(rows)} edges")

        # Act -> HAS_ARTICLE -> Article
        rows = read_csv("act_article.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (a:Act {act_id: row.act_id})
            MATCH (ar:Article {article_id: row.article_id})
            MERGE (a)-[:HAS_ARTICLE]->(ar)
        """, rows)
        print(f"  HAS_ARTICLE: {len(rows)} edges")

        # Section -> DEFINES_TERM -> Definition
        rows = read_csv("section_defines_term.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (s:Section {section_id: row.section_id})
            MATCH (d:Definition {def_id: row.def_id})
            MERGE (s)-[:DEFINES_TERM]->(d)
        """, rows)
        print(f"  DEFINES_TERM: {len(rows)} edges")

        # Section -> REFERENCES -> Section/Article
        rows = read_csv("section_references_section.csv")
        # These can reference either Section or Article nodes
        ref_intra = [r for r in rows if r.get("reference_type") == "see_also"]
        ref_cross = [r for r in rows if r.get("reference_type") == "cross_act"]

        if ref_intra:
            batch_unwind(session, """
                UNWIND $rows AS row
                MATCH (s1:Section {section_id: row.from_section_id})
                MATCH (s2:Section {section_id: row.to_section_id})
                MERGE (s1)-[:REFERENCES {reference_type: row.reference_type, context: row.context}]->(s2)
            """, ref_intra)
        if ref_cross:
            # Cross-act refs may point to Section or Article
            batch_unwind(session, """
                UNWIND $rows AS row
                MATCH (s1:Section {section_id: row.from_section_id})
                OPTIONAL MATCH (s2:Section {section_id: row.to_section_id})
                OPTIONAL MATCH (a2:Article {article_id: row.to_section_id})
                WITH s1, s2, a2, row
                FOREACH (_ IN CASE WHEN s2 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (s1)-[:REFERENCES {reference_type: row.reference_type,
                                             target_act_id: row.target_act_id,
                                             context: row.context}]->(s2)
                )
                FOREACH (_ IN CASE WHEN a2 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (s1)-[:REFERENCES {reference_type: row.reference_type,
                                             target_act_id: row.target_act_id,
                                             context: row.context}]->(a2)
                )
            """, ref_cross)
        print(f"  REFERENCES: {len(rows)} edges ({len(ref_intra)} intra, {len(ref_cross)} cross_act)")

        # Case -> CITES -> Section
        rows = read_csv("case_cites_section.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (c:Case {case_id: row.case_id})
            MATCH (s:Section {section_id: row.section_id})
            MERGE (c)-[:CITES]->(s)
        """, rows, batch_size=1000)
        print(f"  CITES (Section): {len(rows)} edges")

        # Case -> CITES -> Article
        rows = read_csv("case_cites_article.csv")
        batch_unwind(session, """
            UNWIND $rows AS row
            MATCH (c:Case {case_id: row.case_id})
            MATCH (a:Article {article_id: row.article_id})
            MERGE (c)-[:CITES]->(a)
        """, rows, batch_size=1000)
        print(f"  CITES (Article): {len(rows)} edges")

        # ================================================================
        # Step 5: Smoke tests
        # ================================================================
        print("\n[5/5] Running smoke tests...")

        node_queries = [
            ("Acts",        "MATCH (n:Act) RETURN count(n) AS cnt"),
            ("Parts",       "MATCH (n:Part) RETURN count(n) AS cnt"),
            ("Chapters",    "MATCH (n:Chapter) RETURN count(n) AS cnt"),
            ("Sections",    "MATCH (n:Section) RETURN count(n) AS cnt"),
            ("Articles",    "MATCH (n:Article) RETURN count(n) AS cnt"),
            ("Definitions", "MATCH (n:Definition) RETURN count(n) AS cnt"),
            ("Cases",       "MATCH (n:Case) RETURN count(n) AS cnt"),
        ]
        rel_queries = [
            ("HAS_PART",    "MATCH ()-[r:HAS_PART]->() RETURN count(r) AS cnt"),
            ("HAS_CHAPTER", "MATCH ()-[r:HAS_CHAPTER]->() RETURN count(r) AS cnt"),
            ("HAS_SECTION", "MATCH ()-[r:HAS_SECTION]->() RETURN count(r) AS cnt"),
            ("HAS_ARTICLE", "MATCH ()-[r:HAS_ARTICLE]->() RETURN count(r) AS cnt"),
            ("IN_ACT",      "MATCH ()-[r:IN_ACT]->() RETURN count(r) AS cnt"),
            ("DEFINES_TERM","MATCH ()-[r:DEFINES_TERM]->() RETURN count(r) AS cnt"),
            ("REFERENCES",  "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS cnt"),
            ("CITES",       "MATCH ()-[r:CITES]->() RETURN count(r) AS cnt"),
        ]

        # Expected minimums
        min_counts = {
            "Acts": 4, "Parts": 24, "Chapters": 47, "Sections": 600,
            "Articles": 300, "Definitions": 1, "Cases": 10000,
            "HAS_PART": 24, "HAS_CHAPTER": 1, "HAS_SECTION": 600,
            "HAS_ARTICLE": 300, "IN_ACT": 600, "DEFINES_TERM": 1,
            "REFERENCES": 44, "CITES": 25000,
        }

        failures = []
        print(f"\n  {'Label':<16} {'Count':>10}  {'Min':>8}  Status")
        print("  " + "-" * 48)
        for label, query in node_queries + rel_queries:
            r = session.run(query)
            cnt = r.single()["cnt"]
            min_exp = min_counts.get(label, 0)
            status = "OK" if cnt >= min_exp else f"LOW (want >={min_exp})"
            if cnt < min_exp:
                failures.append(f"{label}: {cnt} < {min_exp}")
            print(f"  {label:<16} {cnt:>10}  {min_exp:>8}  {status}")

        # Spot checks
        print("\n  --- Spot checks ---")

        # BNS definitions path (via IN_ACT -- BNS has no direct Act-HAS_SECTION edges)
        r = session.run("""
            MATCH (s:Section)-[:IN_ACT]->(a:Act {act_id:'BNS_2023'})
            MATCH (s)-[:DEFINES_TERM]->(d:Definition)
            RETURN s.section_id, d.term LIMIT 5
        """)
        defs = r.data()
        if defs:
            print(f"  OK: BNS Definition path: {[(x['s.section_id'], x['d.term']) for x in defs]}")
        else:
            print("  WARN: BNS Definition path returned 0 rows")

        # BSA Part-Chapter path
        r = session.run("""
            MATCH (a:Act {act_id:'BSA_2023'})-[:HAS_PART]->(p:Part)-[:HAS_CHAPTER]->(c:Chapter)
            RETURN p.part_number, c.chapter_number, c.chapter_title LIMIT 5
        """)
        parts_chapters = r.data()
        if parts_chapters:
            print(f"  OK: BSA Part-Chapter path: {len(parts_chapters)} rows")
        else:
            print("  WARN: BSA Part-Chapter path returned 0 rows")
            failures.append("BSA Part-Chapter path: 0 rows")

        # Constitution cross-refs
        r = session.run("""
            MATCH (s:Section)-[r:REFERENCES {reference_type:'cross_act'}]->(ar:Article)
            WHERE ar.act_id = 'CONST_1950'
            RETURN s.section_id, ar.article_id LIMIT 5
        """)
        const_refs = r.data()
        if const_refs:
            print(f"  OK: Constitution cross-ref path: {[(x['s.section_id'], x['ar.article_id']) for x in const_refs]}")
        else:
            print("  WARN: Constitution cross-ref path returned 0 rows")

        # Case citing path
        r = session.run("""
            MATCH (c:Case)-[:CITES]->(s:Section {act_id:'BNS_2023'})
            RETURN c.case_id, s.section_id LIMIT 3
        """)
        case_cites = r.data()
        if case_cites:
            print(f"  OK: Case-cites-BNS path: {len(case_cites)} rows")
        else:
            print("  WARN: Case-cites-BNS path returned 0 rows")

        # Result
        print("\n=== SMOKE TEST RESULT ===")
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
            return 1
        else:
            print("  ALL SMOKE TESTS PASSED")
            return 0

    driver.close()


if __name__ == "__main__":
    sys.exit(main())
