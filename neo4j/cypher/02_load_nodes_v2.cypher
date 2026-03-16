// Phase-2 v2: Load nodes from phase1_output_v2 CSVs (Neo4j 5.x)
// Prereq: Copy phase1_output_v2/*.csv to Neo4j import/. Run 01_constraints_v2.cypher first.
// Note: acts.csv has short_title; we set act_name = short_title for Phase 4 compatibility.

// ---- Acts ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///acts.csv' AS row
  WITH row WHERE row.act_id IS NOT NULL AND trim(row.act_id) <> ''
  MERGE (a:Act {act_id: row.act_id})
  SET a.act_name = row.short_title,
      a.short_title = row.short_title,
      a.year = toIntegerOrNull(row.year),
      a.act_number = toIntegerOrNull(row.act_number),
      a.act_type = row.act_type,
      a.source_file = row.source_file,
      a.enforcement_date = row.enforcement_date
} IN TRANSACTIONS OF 500 ROWS;

// ---- Parts ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///parts.csv' AS row
  WITH row WHERE row.part_id IS NOT NULL AND trim(row.part_id) <> ''
  MERGE (p:Part {part_id: row.part_id})
  SET p.part_number = row.part_number,
      p.part_title = row.part_title,
      p.act_id = row.act_id
} IN TRANSACTIONS OF 500 ROWS;

// ---- Chapters ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///chapters.csv' AS row
  WITH row WHERE row.chapter_id IS NOT NULL AND trim(row.chapter_id) <> ''
  MERGE (c:Chapter {chapter_id: row.chapter_id})
  SET c.chapter_number = row.chapter_number,
      c.chapter_title = row.chapter_title,
      c.act_id = row.act_id,
      c.part_id = row.part_id
} IN TRANSACTIONS OF 500 ROWS;

// ---- Sections ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///sections.csv' AS row
  WITH row WHERE row.section_id IS NOT NULL AND trim(row.section_id) <> ''
  MERGE (s:Section {section_id: row.section_id})
  SET s.section_number = row.section_number,
      s.heading = row.heading,
      s.full_text = row.full_text,
      s.act_id = row.act_id,
      s.chapter_id = row.chapter_id
} IN TRANSACTIONS OF 1000 ROWS;

// ---- Articles ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///articles.csv' AS row
  WITH row WHERE row.article_id IS NOT NULL AND trim(row.article_id) <> ''
  MERGE (ar:Article {article_id: row.article_id})
  SET ar.article_number = row.article_number,
      ar.heading = row.heading,
      ar.full_text = row.full_text,
      ar.act_id = row.act_id
} IN TRANSACTIONS OF 1000 ROWS;

// ---- Definitions ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///definitions.csv' AS row
  WITH row WHERE row.def_id IS NOT NULL AND trim(row.def_id) <> ''
  MERGE (d:Definition {def_id: row.def_id})
  SET d.term = row.term,
      d.defined_text = row.defined_text,
      d.act_id = row.act_id,
      d.section_id = row.section_id
} IN TRANSACTIONS OF 500 ROWS;

// ---- Cases (from cases_sc.csv; use cases_sc_neo4j.csv if LOAD CSV fails on large full_text) ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///cases_sc_neo4j.csv' AS row
  WITH row WHERE row.case_id IS NOT NULL AND trim(row.case_id) <> ''
  MERGE (c:Case {case_id: row.case_id})
  SET c.year = toIntegerOrNull(row.year),
      c.source_file = row.source_file,
      c.source = row.source,
      c.judgment_text = row.judgment_text
} IN TRANSACTIONS OF 500 ROWS;

// ---- Cases (from cases_iltur.csv; use cases_iltur_neo4j.csv if LOAD CSV fails on large text) ----
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///cases_iltur_neo4j.csv' AS row
  WITH row WHERE row.case_id IS NOT NULL AND trim(row.case_id) <> ''
  MERGE (c:Case {case_id: row.case_id})
  SET c.year = toIntegerOrNull(row.year),
      c.source = row.source,
      c.judgment_text = row.judgment_text
} IN TRANSACTIONS OF 500 ROWS;
