// =============================================================================
// 04_smoke_tests_v2.cypher
// Smoke tests for the v2 Neo4j Knowledge Graph schema.
// Run this AFTER 01_constraints_v2, 02_load_nodes_v2 and 03_load_edges_v2.
// Each query below should return non-zero counts (unless noted otherwise).
// =============================================================================

// ---- 1. Node counts ----

// 1a. Acts (expect 4: BNS_2023, BNSS_2023, BSA_2023, CONST_1950)
MATCH (n:Act) RETURN 'Acts' AS label, count(n) AS count;

// 1b. Parts (expect ~48: 47 Constitution + 1 BSA)
MATCH (n:Part) RETURN 'Parts' AS label, count(n) AS count;

// 1c. Chapters (expect ~49: 19 BNS + 27 BNSS + 3 BSA)
MATCH (n:Chapter) RETURN 'Chapters' AS label, count(n) AS count;

// 1d. Sections (expect ~1,000+)
MATCH (n:Section) RETURN 'Sections' AS label, count(n) AS count;

// 1e. Articles (expect ~395+)
MATCH (n:Article) RETURN 'Articles' AS label, count(n) AS count;

// 1f. Definitions (expect 85+ after Phase 1 fix; was 0 before)
MATCH (n:Definition) RETURN 'Definitions' AS label, count(n) AS count;

// 1g. Cases (expect ~90,000+: SC + IL-TUR)
MATCH (n:Case) RETURN 'Cases' AS label, count(n) AS count;

// ---- 2. Relationship counts ----

// 2a. HAS_PART (expect ~48)
MATCH ()-[r:HAS_PART]->() RETURN 'HAS_PART' AS rel, count(r) AS count;

// 2b. HAS_CHAPTER (expect ~3 after Phase 2 fix: BSA Part IV → CH VII/VIII/IX)
MATCH ()-[r:HAS_CHAPTER]->() RETURN 'HAS_CHAPTER' AS rel, count(r) AS count;

// 2c. HAS_SECTION (expect ~1,000+)
MATCH ()-[r:HAS_SECTION]->() RETURN 'HAS_SECTION' AS rel, count(r) AS count;

// 2d. HAS_ARTICLE (expect ~395+)
MATCH ()-[r:HAS_ARTICLE]->() RETURN 'HAS_ARTICLE' AS rel, count(r) AS count;

// 2e. IN_ACT for Sections (expect ~1,000+)
MATCH (s:Section)-[r:IN_ACT]->() RETURN 'IN_ACT (Section)' AS rel, count(r) AS count;

// 2f. DEFINES_TERM (expect 85+ after Phase 1 fix; was 0 before)
MATCH ()-[r:DEFINES_TERM]->() RETURN 'DEFINES_TERM' AS rel, count(r) AS count;

// 2g. REFERENCES (intra + cross_act; expect ~15,000+)
MATCH ()-[r:REFERENCES]->() RETURN 'REFERENCES' AS rel, count(r) AS count;

// 2h. CITES (expect ~53,000+)
MATCH ()-[r:CITES]->() RETURN 'CITES' AS rel, count(r) AS count;

// ---- 3. Cross-act REFERENCES distribution ----
MATCH (s:Section)-[r:REFERENCES {reference_type:"cross_act"}]->(t)
RETURN s.act_id AS from_act, coalesce(t.act_id, 'unknown') AS to_act, count(r) AS count
ORDER BY count DESC;

// ---- 4. Top 20 most cited provisions ----
MATCH ()-[:CITES]->(s:Section)
RETURN s.section_id, count(*) AS cite_count
ORDER BY cite_count DESC LIMIT 20;

MATCH ()-[:CITES]->(a:Article)
RETURN a.article_id, count(*) AS cite_count
ORDER BY cite_count DESC LIMIT 10;

// ---- 5. Integrity checks (all should return 0) ----

// 5a. Orphan Parts: Parts that have no HAS_PART incoming edge
MATCH (p:Part)
WHERE NOT ()-[:HAS_PART]->(p)
RETURN 'Orphan Parts' AS check, count(p) AS count;

// 5b. Orphan Chapters: Chapters that have no HAS_CHAPTER incoming edge
//     (BNS/BNSS chapters have no Parts so this will be ~46 -- that is correct)
MATCH (c:Chapter)
WHERE NOT ()-[:HAS_CHAPTER]->(c)
RETURN 'Chapters without HAS_CHAPTER' AS check, count(c) AS count;

// 5c. Acts without any HAS_SECTION or HAS_ARTICLE edges (should be 0)
MATCH (a:Act)
WHERE NOT (a)-[:HAS_SECTION]->() AND NOT (a)-[:HAS_ARTICLE]->()
RETURN 'Acts with no sections or articles' AS check, count(a) AS count;

// ---- 6. Spot-check key paths ----

// 6a. Definition path: Section -> IN_ACT -> Act (BNS/BNSS/BSA sections use IN_ACT not HAS_SECTION from Act)
MATCH (s:Section)-[:IN_ACT]->(a:Act {act_id:"BNS_2023"})
MATCH (s)-[:DEFINES_TERM]->(d:Definition)
RETURN s.section_id, d.term LIMIT 10;

// 6b. Part-Chapter path (only valid for BSA)
MATCH (a:Act {act_id:"BSA_2023"})-[:HAS_PART]->(p:Part)-[:HAS_CHAPTER]->(c:Chapter)
RETURN p.part_number, c.chapter_number, c.chapter_title LIMIT 10;

// 6c. Constitution cross-reference path
MATCH (s:Section)-[r:REFERENCES {reference_type:"cross_act"}]->(ar:Article)
WHERE ar.act_id = "CONST_1950"
RETURN s.section_id, ar.article_id LIMIT 10;

// 6d. Case citing path
MATCH (c:Case)-[:CITES]->(s:Section {act_id:"BNS_2023"})
RETURN c.case_id, s.section_id LIMIT 5;
