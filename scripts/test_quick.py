"""Quick test: Neo4j + FAISS + Query parser + Neo4j client (no LLM)."""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = "[PASS]", "[FAIL]"
results = []

def check(label, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f"  ->  {detail}" if detail else ""))
    results.append((label, ok, detail))

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# 1. Neo4j
section("1. Neo4j")
try:
    from dotenv import load_dotenv; load_dotenv(Path(".env"))
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI","bolt://localhost:7687")
    d = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","neo4j")))
    d.verify_connectivity()
    check("Connection", True, uri)
    with d.session() as s:
        def cnt(c): return s.run(c).single()[0]
        for label, cypher, minimum in [
            ("Acts",        "MATCH (n:Act) RETURN count(n)",        4),
            ("Parts",       "MATCH (n:Part) RETURN count(n)",       24),
            ("Chapters",    "MATCH (n:Chapter) RETURN count(n)",    47),
            ("Sections",    "MATCH (n:Section) RETURN count(n)",    600),
            ("Articles",    "MATCH (n:Article) RETURN count(n)",    300),
            ("Definitions", "MATCH (n:Definition) RETURN count(n)", 1),
            ("Cases",       "MATCH (n:Case) RETURN count(n)",       10000),
            ("HAS_PART",    "MATCH ()-[r:HAS_PART]->() RETURN count(r)",     24),
            ("HAS_CHAPTER", "MATCH ()-[r:HAS_CHAPTER]->() RETURN count(r)",  1),
            ("HAS_SECTION", "MATCH ()-[r:HAS_SECTION]->() RETURN count(r)",  600),
            ("HAS_ARTICLE", "MATCH ()-[r:HAS_ARTICLE]->() RETURN count(r)",  300),
            ("IN_ACT",      "MATCH ()-[r:IN_ACT]->() RETURN count(r)",       600),
            ("DEFINES_TERM","MATCH ()-[r:DEFINES_TERM]->() RETURN count(r)", 1),
            ("REFERENCES",  "MATCH ()-[r:REFERENCES]->() RETURN count(r)",   44),
            ("CITES",       "MATCH ()-[r:CITES]->() RETURN count(r)",        25000),
        ]:
            v = cnt(cypher)
            check(label, v >= minimum, f"{v} (want >={minimum})")
        # Spot checks
        for label, cypher in [
            ("BNS Definition path",
             "MATCH (s:Section)-[:IN_ACT]->(a:Act {act_id:'BNS_2023'}) MATCH (s)-[:DEFINES_TERM]->(d:Definition) RETURN s.section_id, d.term LIMIT 2"),
            ("BSA Part->Chapter",
             "MATCH (p:Part {act_id:'BSA_2023'})-[:HAS_CHAPTER]->(c:Chapter) RETURN p.part_id, c.chapter_id LIMIT 2"),
            ("Constitution cross-ref",
             "MATCH (s:Section)-[:REFERENCES {reference_type:'cross_act'}]->(a:Article) WHERE a.act_id='CONST_1950' RETURN s.section_id, a.article_id LIMIT 2"),
            ("Case->CITES->BNS",
             "MATCH (c:Case)-[:CITES]->(s:Section {act_id:'BNS_2023'}) RETURN c.case_id, s.section_id LIMIT 2"),
        ]:
            rows = s.run(cypher).data()
            check(label, len(rows) > 0, str(rows))
    d.close()
except Exception as e:
    check("Neo4j exception", False, str(e))

# 2. FAISS
section("2. FAISS")
try:
    from phase4_rag.vector_retriever_v3 import retrieve_chunks
    for label, query in [
        ("murder/BNS",    "punishment for murder under BNS"),
        ("Article 21",    "fundamental rights Article 21 Constitution"),
        ("definition act","definition of act BNS section 2"),
    ]:
        t0 = time.time()
        res = retrieve_chunks(query, k=5)
        hits = res.get("chunks", [])
        top = hits[0].get("chunk_id","?") if hits else "none"
        check(f"Retrieval: {label}", len(hits) > 0,
              f"{len(hits)} chunks in {time.time()-t0:.2f}s  top={top}")
except Exception as e:
    check("FAISS exception", False, str(e))

# 3. Query Parser
section("3. Query Parser")
try:
    from phase4_rag.query_parser_v3 import parse_query
    for label, query, want_refs, want_sec, want_art in [
        ("Section+BNS",    "Section 103 of BNS",             True,  "BNS_2023",  None),
        ("Article+Const",  "Article 21 of the Constitution", True,  None,        "CONST_1950"),
        ("Natural lang",   "What is murder?",                False, None,        None),
        ("BNSS explicit",  "Section 200 BNSS",               True,  "BNSS_2023", None),
    ]:
        pq = parse_query(query)
        ok = (pq.has_explicit_refs == want_refs
              and (want_sec is None or pq.section_act_id == want_sec)
              and (want_art is None or pq.article_act_id == want_art))
        check(f"Parser: {label}", ok,
              f"refs={pq.has_explicit_refs} sec_act={pq.section_act_id} art_act={pq.article_act_id}")
except Exception as e:
    check("Parser exception", False, str(e))

# 4. Neo4j Client v3 (including Article fix)
section("4. Neo4j Client v3")
try:
    from phase4_rag.neo4j_client_v3 import get_sections_by_numbers, get_articles_by_numbers, get_cases_citing_ids
    secs = get_sections_by_numbers(["103"], act_id="BNS_2023")
    check("get_sections (BNS s.103)", len(secs) > 0,
          f"{len(secs)} result  id={secs[0].get('section_id','?') if secs else 'none'}")
    arts = get_articles_by_numbers(["21"], act_id="CONST_1950")
    check("get_articles (Art.21) -- FIXED", len(arts) > 0,
          f"{len(arts)} result  id={arts[0].get('article_id','?') if arts else 'none'}")
    if secs:
        cases = get_cases_citing_ids([secs[0]["section_id"]])
        check(f"get_cases_citing ({secs[0]['section_id']})", True, f"{len(cases)} cases")
except Exception as e:
    import traceback; traceback.print_exc()
    check("Client v3 exception", False, str(e))

# Summary
section("SUMMARY")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
for label, ok, detail in results:
    print(f"  {PASS if ok else FAIL} {label}")
print(f"\n  {passed}/{len(results)} passed  |  {failed} failed")
if failed == 0:
    print("\n  ALL CHECKS PASSED")
else:
    print("\n  FAILURES:")
    for label, ok, detail in results:
        if not ok:
            print(f"    - {label}: {detail}")
