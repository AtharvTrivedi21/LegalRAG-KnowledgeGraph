"""
LegalRAG System Test â€” runs entirely in the terminal with print output.
Tests each layer independently. LLM test uses a short 1-sentence prompt.
"""
from __future__ import annotations
import csv, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []

def check(label: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f"  ->  {detail}" if detail else ""))
    results.append((label, ok, detail))

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. NEO4J
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("1. Neo4j Graph Database")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"))
    from neo4j import GraphDatabase

    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    check("Connection", True, uri)

    with driver.session() as s:
        def cnt(cypher): return s.run(cypher).single()[0]

        acts     = cnt("MATCH (n:Act)        RETURN count(n)")
        parts    = cnt("MATCH (n:Part)       RETURN count(n)")
        chapters = cnt("MATCH (n:Chapter)    RETURN count(n)")
        sections = cnt("MATCH (n:Section)    RETURN count(n)")
        articles = cnt("MATCH (n:Article)    RETURN count(n)")
        defs     = cnt("MATCH (n:Definition) RETURN count(n)")
        cases    = cnt("MATCH (n:Case)       RETURN count(n)")

        check("Acts",        acts == 4,      f"{acts} (want 4)")
        check("Parts",       parts >= 24,    f"{parts} (want â‰¥24)")
        check("Chapters",    chapters >= 47, f"{chapters} (want â‰¥47)")
        check("Sections",    sections >= 600,f"{sections} (want â‰¥600)")
        check("Articles",    articles >= 300,f"{articles} (want â‰¥300)")
        check("Definitions", defs >= 1,      f"{defs} (want â‰¥1)")
        check("Cases",       cases >= 10000, f"{cases} (want â‰¥10 000)")

        has_part = cnt("MATCH ()-[r:HAS_PART]->()     RETURN count(r)")
        has_chap = cnt("MATCH ()-[r:HAS_CHAPTER]->()  RETURN count(r)")
        has_sec  = cnt("MATCH ()-[r:HAS_SECTION]->()  RETURN count(r)")
        has_art  = cnt("MATCH ()-[r:HAS_ARTICLE]->()  RETURN count(r)")
        in_act   = cnt("MATCH ()-[r:IN_ACT]->()       RETURN count(r)")
        def_term = cnt("MATCH ()-[r:DEFINES_TERM]->() RETURN count(r)")
        refs     = cnt("MATCH ()-[r:REFERENCES]->()   RETURN count(r)")
        cites    = cnt("MATCH ()-[r:CITES]->()        RETURN count(r)")

        check("HAS_PART",     has_part >= 24,  f"{has_part}")
        check("HAS_CHAPTER",  has_chap >= 1,   f"{has_chap} (BSA Part IV)")
        check("HAS_SECTION",  has_sec  >= 600, f"{has_sec}")
        check("HAS_ARTICLE",  has_art  >= 300, f"{has_art}")
        check("IN_ACT",       in_act   >= 600, f"{in_act}")
        check("DEFINES_TERM", def_term >= 1,   f"{def_term}")
        check("REFERENCES",   refs     >= 44,  f"{refs}")
        check("CITES",        cites    >= 25000,f"{cites}")

        # Spot-check paths
        bns_defs = s.run("""
            MATCH (s:Section)-[:IN_ACT]->(a:Act {act_id:'BNS_2023'})
            MATCH (s)-[:DEFINES_TERM]->(d:Definition)
            RETURN s.section_id, d.term LIMIT 3
        """).data()
        check("BNS Definition path",
              len(bns_defs) > 0,
              str([(r["s.section_id"], r["d.term"]) for r in bns_defs]))

        bsa_path = s.run("""
            MATCH (p:Part {act_id:'BSA_2023'})-[:HAS_CHAPTER]->(c:Chapter)
            RETURN p.part_id, c.chapter_id LIMIT 3
        """).data()
        check("BSA Part->Chapter path",
              len(bsa_path) > 0,
              str([(r["p.part_id"], r["c.chapter_id"]) for r in bsa_path]))

        const_refs = s.run("""
            MATCH (s:Section)-[:REFERENCES {reference_type:'cross_act'}]->(a:Article)
            WHERE a.act_id='CONST_1950'
            RETURN s.section_id, a.article_id LIMIT 3
        """).data()
        check("Constitution cross-ref path",
              len(const_refs) > 0,
              str([(r["s.section_id"], r["a.article_id"]) for r in const_refs]))

        case_cites = s.run("""
            MATCH (c:Case)-[:CITES]->(s:Section {act_id:'BNS_2023'})
            RETURN c.case_id, s.section_id LIMIT 2
        """).data()
        check("Case->CITES->BNS path",
              len(case_cites) > 0,
              str([(r["c.case_id"], r["s.section_id"]) for r in case_cites]))

    driver.close()

except Exception as e:
    check("Neo4j (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. FAISS / VECTOR RETRIEVAL
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("2. FAISS Vector Index")

try:
    meta_path = Path("phase3_embeddings/output/chunk_metadata.pkl")
    idx_path  = Path("phase3_embeddings/output/faiss.index")
    check("chunk_metadata.pkl exists", meta_path.exists(), str(meta_path))
    check("faiss.index exists",        idx_path.exists(),  str(idx_path))

    print("  Loading chunk metadata (may take a few seconds)...")
    t0 = time.time()
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"  Loaded {len(chunks):,} chunks in {time.time()-t0:.1f}s")
    check("Chunks loaded", len(chunks) > 0, f"{len(chunks):,} total")

    from collections import Counter
    by_type = Counter(ch.get("source_type", "?") for ch in chunks)
    check("Chunk types present",
          by_type.get("section", 0) > 0 and by_type.get("case", 0) > 0,
          f"section={by_type.get('section',0):,}  article={by_type.get('article',0):,}  case={by_type.get('case',0):,}")

    print("  Running retrieval queries...")
    from phase4_rag.vector_retriever_v3 import retrieve_chunks

    for query_label, query_text in [
        ("murder/BNS",       "punishment for murder under BNS"),
        ("Article 21",       "fundamental rights Article 21 Constitution"),
        ("definition of act","definition of act in BNS section 2"),
    ]:
        t0 = time.time()
        res = retrieve_chunks(query_text, k=5)
        elapsed = time.time() - t0
        hits = res.get("chunks", [])
        top_id = hits[0].get("chunk_id", "?") if hits else "none"
        check(f"Retrieval: {query_label}",
              len(hits) > 0,
              f"{len(hits)} chunks in {elapsed:.2f}s  top={top_id}")

except Exception as e:
    import traceback; traceback.print_exc()
    check("FAISS (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. QUERY PARSER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("3. Query Parser (V3)")

try:
    from phase4_rag.query_parser_v3 import parse_query

    cases = [
        ("Section ref + BNS",         "Section 103 of BNS",             True,  "BNS_2023",  None),
        ("Article ref + Constitution", "Article 21 of the Constitution", True,  None,        "CONST_1950"),
        ("Natural language",           "What is murder?",                False, None,        None),
        ("BNSS explicit",              "Section 200 BNSS",               True,  "BNSS_2023", None),
    ]
    for label, query, want_refs, want_sec_act, want_art_act in cases:
        pq  = parse_query(query)
        ok  = (pq.has_explicit_refs == want_refs
               and (want_sec_act is None or pq.section_act_id == want_sec_act)
               and (want_art_act is None or pq.article_act_id == want_art_act))
        check(f"Parser: {label}", ok,
              f"has_refs={pq.has_explicit_refs}  sec_act={pq.section_act_id}  art_act={pq.article_act_id}")

except Exception as e:
    check("Query parser (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. NEO4J CLIENT V3
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("4. Neo4j Client V3")

try:
    from phase4_rag.neo4j_client_v3 import (
        get_sections_by_numbers,
        get_articles_by_numbers,
        get_cases_citing_ids,
    )

    secs = get_sections_by_numbers(["103"], act_id="BNS_2023")
    check("get_sections_by_numbers (BNS s.103)",
          len(secs) > 0,
          f"{len(secs)} result(s)  id={secs[0].get('section_id','?') if secs else 'none'}")

    arts = get_articles_by_numbers(["21"], act_id="CONST_1950")
    check("get_articles_by_numbers (Art.21)",
          len(arts) > 0,
          f"{len(arts)} result(s)  id={arts[0].get('article_id','?') if arts else 'none'}")

    if secs:
        sid   = secs[0]["section_id"]
        cases = get_cases_citing_ids([sid])
        check(f"get_cases_citing_ids ({sid})", True,
              f"{len(cases)} cases cite this section")

except Exception as e:
    import traceback; traceback.print_exc()
    check("Neo4j client v3 (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. OLLAMA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("5. Ollama LLM (llama3:8b)")

try:
    import requests as _req
    r = _req.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    check("Ollama server reachable", True, "models: " + ", ".join(models))
    check("llama3:8b pulled", any("llama3" in m for m in models), str(models))

    from phase4_rag.llm_ollama import ChatMessage, chat_completion, OllamaError
    print("  Sending short prompt to LLM (may take 20-60s on CPU)...")
    t0 = time.time()
    try:
        resp = chat_completion(
            [ChatMessage(role="user",
                         content="In one sentence only: what is Section 103 of BNS about?")]
        )
        elapsed = time.time() - t0
        check("LLM response (short prompt)",
              bool(resp and len(resp) > 10),
              f"{elapsed:.1f}s  ->  {resp[:120]}")
    except OllamaError as e:
        check("LLM response", False, str(e)[:120])

except Exception as e:
    check("Ollama (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. FULL PIPELINE â€” graph-constrained query
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("6. Full Pipeline (Graph-Constrained Query)")

try:
    from phase4_rag.langgraph_workflow_v3 import build_app
    print("  Building LangGraph app...")
    app = build_app()

    query = "What does Section 103 of BNS say about murder?"
    print(f"  Query: {query}")
    t0 = time.time()
    state = app.invoke({"user_query": query})
    elapsed = time.time() - t0

    answer      = state.get("answer", "")
    chunks      = state.get("retrieved_chunks", [])
    g_meta      = state.get("graph_metadata") or {}
    constrained = state.get("graph_constraints") is not None
    secs_found  = len(g_meta.get("sections", []))
    arts_found  = len(g_meta.get("articles", []))
    cases_found = len(g_meta.get("cases", []))

    check("Graph constraints active",
          constrained,
          f"sections={secs_found}  articles={arts_found}  cases={cases_found}")
    check("FAISS chunks retrieved", len(chunks) > 0, f"{len(chunks)} chunks")
    check("Answer generated",       bool(answer and len(answer) > 50), f"{elapsed:.1f}s")
    check("Answer not an error",
          not any(w in answer.lower()[:100] for w in ("error", "unavailable", "cuda")),
          "")

    print(f"\n  --- Answer preview ({elapsed:.1f}s) ---")
    for line in answer[:600].splitlines():
        print(f"  {line}")

except Exception as e:
    import traceback; traceback.print_exc()
    check("Full pipeline (exception)", False, str(e))


# ----------------------------------------------------------------
# 7. NATURAL LANGUAGE QUERY - citations and BNS focus
# ----------------------------------------------------------------
section("7. Natural Language Query (Citation Fix Verification)")

try:
    from phase4_rag.langgraph_workflow_v3 import build_app as _build_app2
    print("  Building LangGraph app...")
    _app2 = _build_app2()

    nl_query = "Someone broke into my home and stole my property, also broke my windows and door."
    print(f"  Query: {nl_query}")
    t0 = time.time()
    nl_state = _app2.invoke({"user_query": nl_query})
    elapsed = time.time() - t0

    nl_answer  = nl_state.get("answer", "")
    nl_chunks  = nl_state.get("retrieved_chunks", [])
    nl_g_meta  = nl_state.get("graph_metadata") or {}
    nl_grouped = nl_state.get("grouped_sources") or {}
    nl_secs    = nl_g_meta.get("sections", [])

    sec_chunk_ids = list((nl_grouped.get("section") or {}).keys())
    check("Section chunks retrieved (natural language)",
          len(sec_chunk_ids) > 0,
          f"section source_ids: {sec_chunk_ids[:5]}")

    check("graph_metadata.sections populated (enrichment)",
          len(nl_secs) > 0,
          f"{len(nl_secs)} sections: {[s.get('section_id') for s in nl_secs[:3]]}")

    act_ids_found = list({s.get("act_id") for s in nl_secs if s.get("act_id")})
    # Accept any of the new Indian law codes (BNS/BNSS/BSA/Constitution) as valid
    new_codes = {"BNS_2023", "BNSS_2023", "BSA_2023", "CONST_1950"}
    check("New Indian law code cited (not IPC)",
          bool(new_codes & set(act_ids_found)),
          f"act_ids in metadata: {act_ids_found}")

    ipc_in_answer = "ipc" in nl_answer.lower() or "indian penal code" in nl_answer.lower()
    check("Answer does NOT reference IPC",
          not ipc_in_answer,
          "IPC found in answer" if ipc_in_answer else "clean")

    bns_in_answer = "bns" in nl_answer.lower() or "bharatiya nyaya" in nl_answer.lower()
    check("Answer references BNS",
          bns_in_answer,
          "BNS found" if bns_in_answer else "BNS not found in answer")

    check("Answer generated (natural language)", bool(nl_answer and len(nl_answer) > 50), f"{elapsed:.1f}s")

    print(f"\n  --- Natural language answer preview ({elapsed:.1f}s) ---")
    for line in nl_answer[:700].splitlines():
        print(f"  {line}")

except Exception as e:
    import traceback; traceback.print_exc()
    check("Natural language pipeline (exception)", False, str(e))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FINAL SUMMARY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section("FINAL SUMMARY")

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

for label, ok, detail in results:
    print(f"  {PASS if ok else FAIL} {label}")

print(f"\n  {passed}/{len(results)} checks passed  |  {failed} failed")
if failed == 0:
    print("\n  ALL SYSTEMS OPERATIONAL")
else:
    print(f"\n  ITEMS NEEDING ATTENTION:")
    for label, ok, detail in results:
        if not ok:
            print(f"    - {label}: {detail}")
