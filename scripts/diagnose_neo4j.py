"""Diagnose Neo4j load issues."""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session() as session:
    # 1. Check Part nodes per act
    r = session.run("MATCH (p:Part) RETURN p.act_id AS act, count(p) AS cnt ORDER BY cnt DESC")
    print("=== Parts per act in graph ===")
    for row in r:
        print(f"  {row['act']}: {row['cnt']}")

    # 2. Check if Constitution parts are there
    r = session.run("MATCH (p:Part) WHERE p.act_id='CONST_1950' RETURN count(p) AS cnt")
    print(f"\nConstitution Parts in graph: {r.single()['cnt']} (CSV has 47)")

    # 3. Check duplicates in parts.csv
    import csv
    part_ids = []
    with open("phase1_output_v2/parts.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            part_ids.append(row["part_id"])
    from collections import Counter
    dupes = {k: v for k, v in Counter(part_ids).items() if v > 1}
    print(f"\nDuplicate part_ids in parts.csv: {len(dupes)}")
    for k, v in list(dupes.items())[:5]:
        print(f"  {k}: {v} times")
    print(f"Total parts.csv rows: {len(part_ids)}, unique: {len(set(part_ids))}")

    # 4. Check BNS Definition path - what's failing?
    r = session.run("MATCH (d:Definition) WHERE d.act_id='BNS_2023' RETURN d.def_id, d.term LIMIT 5")
    print("\n=== BNS Definitions in graph ===")
    for row in r:
        print(f"  {row['d.def_id']}: {row['d.term']}")

    r = session.run("""
        MATCH (s:Section)-[:DEFINES_TERM]->(d:Definition)
        WHERE d.act_id='BNS_2023'
        RETURN s.section_id, d.term LIMIT 5
    """)
    print("\n=== Section->DEFINES_TERM->Definition (BNS) ===")
    rows = r.data()
    if rows:
        for row in rows:
            print(f"  {row['s.section_id']} -> {row['d.term']}")
    else:
        print("  (none)")

    r = session.run("""
        MATCH (a:Act {act_id:'BNS_2023'})-[:HAS_SECTION]->(s:Section)
        RETURN count(s) AS cnt
    """)
    print(f"\nBNS sections via HAS_SECTION: {r.single()['cnt']}")

    r = session.run("""
        MATCH (s:Section {act_id:'BNS_2023'})-[:IN_ACT]->(a:Act {act_id:'BNS_2023'})
        RETURN count(s) AS cnt
    """)
    print(f"BNS sections via IN_ACT: {r.single()['cnt']}")

driver.close()
