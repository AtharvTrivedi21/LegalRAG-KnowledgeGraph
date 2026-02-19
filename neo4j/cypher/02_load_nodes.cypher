// Phase-2: LOAD CSV node imports (Neo4j 5.x)
//
// IMPORTANT:
// - Copy Phase-1 CSVs into Neo4j's `import/` directory first:
//   acts.csv, sections.csv, articles.csv, cases.csv
// - Run 01_constraints.cypher before this file.

// ---- Acts ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///acts.csv' AS row
  WITH row
  WHERE row.act_id IS NOT NULL AND trim(row.act_id) <> ''
  MERGE (a:Act {act_id: row.act_id})
  SET a.act_name = row.act_name,
      a.act_type = row.act_type,
      a.source_file = row.source_file
} IN TRANSACTIONS OF 1000 ROWS;

// ---- Sections ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///sections.csv' AS row
  WITH row
  WHERE row.section_id IS NOT NULL AND trim(row.section_id) <> ''
  MERGE (s:Section {section_id: row.section_id})
  SET s.section_number = row.section_number,
      s.full_text = row.full_text
  WITH s, row
  MATCH (a:Act {act_id: row.act_id})
  MERGE (s)-[:IN_ACT]->(a)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Articles ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///articles.csv' AS row
  WITH row
  WHERE row.article_id IS NOT NULL AND trim(row.article_id) <> ''
  MERGE (ar:Article {article_id: row.article_id})
  SET ar.article_number = row.article_number,
      ar.full_text = row.full_text
  WITH ar, row
  MATCH (a:Act {act_id: row.act_id})
  MERGE (ar)-[:IN_ACT]->(a)
} IN TRANSACTIONS OF 2000 ROWS;

// ---- Cases ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///cases.csv' AS row
  WITH row
  WHERE row.case_id IS NOT NULL AND trim(row.case_id) <> ''
  MERGE (c:Case {case_id: row.case_id})
  SET c.year = toInteger(row.year),
      c.judgment_text = row.judgment_text
} IN TRANSACTIONS OF 200 ROWS;
