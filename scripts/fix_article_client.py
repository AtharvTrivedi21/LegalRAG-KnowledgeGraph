"""Check Article relationships and fix neo4j_client_v3.py if needed."""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
from neo4j import GraphDatabase

d = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j"))
)
with d.session() as s:
    r1 = s.run("MATCH (ar:Article)-[rel]->() RETURN type(rel) AS t, count(*) AS c").data()
    print("Article outgoing rels:", r1)
    r2 = s.run("MATCH ()-[rel]->(ar:Article) RETURN type(rel) AS t, count(*) AS c").data()
    print("Article incoming rels:", r2)
    r3 = s.run("MATCH (ar:Article) WHERE ar.act_id='CONST_1950' RETURN ar.article_number LIMIT 5").data()
    print("Sample article_numbers:", [x["ar.article_number"] for x in r3])
    # Try the correct query
    r4 = s.run("""
        MATCH (a:Act)-[:HAS_ARTICLE]->(ar:Article)
        WHERE ar.article_number IN ['21'] AND a.act_id = 'CONST_1950'
        RETURN ar.article_id, ar.article_number, a.act_id LIMIT 3
    """).data()
    print("HAS_ARTICLE lookup for Art.21:", r4)
d.close()
