// Phase-2 v2: Load relationships from phase1_output_v2 CSVs (Neo4j 5.x)
// Run 01_constraints_v2.cypher and 02_load_nodes_v2.cypher first.

// ---- Act -[:HAS_PART]-> Part ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///act_part.csv' AS row
  WITH row WHERE row.act_id IS NOT NULL AND row.part_id IS NOT NULL
  MATCH (a:Act {act_id: row.act_id})
  MATCH (p:Part {part_id: row.part_id})
  MERGE (a)-[:HAS_PART]->(p)
} IN TRANSACTIONS OF 1000 ROWS;

// ---- Part -[:HAS_CHAPTER]-> Chapter ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///part_chapter.csv' AS row
  WITH row WHERE row.part_id IS NOT NULL AND row.chapter_id IS NOT NULL
  MATCH (p:Part {part_id: row.part_id})
  MATCH (c:Chapter {chapter_id: row.chapter_id})
  MERGE (p)-[:HAS_CHAPTER]->(c)
} IN TRANSACTIONS OF 1000 ROWS;

// ---- Chapter -[:HAS_SECTION]-> Section ---- (and Act -[:HAS_SECTION]-> Section)
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///chapter_section.csv' AS row
  WITH row WHERE row.chapter_id IS NOT NULL AND row.section_id IS NOT NULL
  MATCH (ch:Chapter {chapter_id: row.chapter_id})
  MATCH (s:Section {section_id: row.section_id})
  MERGE (ch)-[:HAS_SECTION]->(s)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Act -[:HAS_SECTION]-> Section ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///act_section.csv' AS row
  WITH row WHERE row.act_id IS NOT NULL AND row.section_id IS NOT NULL
  MATCH (a:Act {act_id: row.act_id})
  MATCH (s:Section {section_id: row.section_id})
  MERGE (a)-[:HAS_SECTION]->(s)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Act -[:HAS_ARTICLE]-> Article ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///act_article.csv' AS row
  WITH row WHERE row.act_id IS NOT NULL AND row.article_id IS NOT NULL
  MATCH (a:Act {act_id: row.act_id})
  MATCH (ar:Article {article_id: row.article_id})
  MERGE (a)-[:HAS_ARTICLE]->(ar)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Section -[:IN_ACT]-> Act (for Phase 4 compatibility) ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///sections.csv' AS row
  WITH row WHERE row.section_id IS NOT NULL AND row.act_id IS NOT NULL
  MATCH (s:Section {section_id: row.section_id})
  MATCH (a:Act {act_id: row.act_id})
  MERGE (s)-[:IN_ACT]->(a)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Article -[:IN_ACT]-> Act (for Phase 4 compatibility) ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///articles.csv' AS row
  WITH row WHERE row.article_id IS NOT NULL AND row.act_id IS NOT NULL
  MATCH (ar:Article {article_id: row.article_id})
  MATCH (a:Act {act_id: row.act_id})
  MERGE (ar)-[:IN_ACT]->(a)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Section -[:DEFINES_TERM]-> Definition ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///section_defines_term.csv' AS row
  WITH row WHERE row.section_id IS NOT NULL AND row.def_id IS NOT NULL
  MATCH (s:Section {section_id: row.section_id})
  MATCH (d:Definition {def_id: row.def_id})
  MERGE (s)-[:DEFINES_TERM]->(d)
} IN TRANSACTIONS OF 500 ROWS;

// ---- Section -[:REFERENCES]-> Section or Article ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///section_references_section.csv' AS row
  WITH row WHERE row.from_section_id IS NOT NULL AND row.to_section_id IS NOT NULL
  MATCH (s1:Section {section_id: row.from_section_id})
  OPTIONAL MATCH (s2:Section {section_id: row.to_section_id})
  OPTIONAL MATCH (a2:Article {article_id: row.to_section_id})
  WITH row, s1, coalesce(s2, a2) AS target
  WHERE target IS NOT NULL
  MERGE (s1)-[r:REFERENCES]->(target)
  SET r.context = row.context, r.reference_type = row.reference_type
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Case -[:CITES]-> Section ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///case_cites_section.csv' AS row
  WITH row WHERE row.case_id IS NOT NULL AND row.section_id IS NOT NULL
  MATCH (c:Case {case_id: row.case_id})
  MATCH (s:Section {section_id: row.section_id})
  MERGE (c)-[r:CITES]->(s)
  SET r.context = row.context
} IN TRANSACTIONS OF 5000 ROWS;

// ---- Case -[:CITES]-> Article ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///case_cites_article.csv' AS row
  WITH row WHERE row.case_id IS NOT NULL AND row.article_id IS NOT NULL
  MATCH (c:Case {case_id: row.case_id})
  MATCH (ar:Article {article_id: row.article_id})
  MERGE (c)-[r:CITES]->(ar)
  SET r.context = row.context
} IN TRANSACTIONS OF 5000 ROWS;
