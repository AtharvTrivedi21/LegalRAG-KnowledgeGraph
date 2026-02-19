// Phase-2: LOAD CSV relationship imports (Neo4j 5.x)
//
// IMPORTANT:
// - Copy edges.csv into Neo4j's `import/` directory first.
// - Run 01_constraints.cypher and 02_load_nodes.cypher before this file.
//
// This script aggregates duplicate edges into relationship property `count`.

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
  WITH row
  WHERE row.source_case_id IS NOT NULL AND trim(row.source_case_id) <> ''
    AND row.target_section_or_article IS NOT NULL AND trim(row.target_section_or_article) <> ''
  MATCH (c:Case {case_id: row.source_case_id})
  WITH c, row
  OPTIONAL MATCH (s:Section {section_id: row.target_section_or_article})
  OPTIONAL MATCH (a:Article {article_id: row.target_section_or_article})
  WITH c, row, coalesce(s, a) AS target
  WHERE target IS NOT NULL
  MERGE (c)-[r:CITES]->(target)
  ON CREATE SET r.count = 1
  ON MATCH SET r.count = coalesce(r.count, 0) + 1
} IN TRANSACTIONS OF 2000 ROWS;
