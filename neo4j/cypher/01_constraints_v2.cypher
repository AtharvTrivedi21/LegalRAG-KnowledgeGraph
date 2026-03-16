// Phase-2 v2: Constraints + indexes for Phase 1 v2 schema (PLAN_1 IDs)
// Run once per database. Use with 02_load_nodes_v2.cypher and 03_load_edges_v2.cypher.
// Copy CSVs from phase1_output_v2/ to Neo4j import/ (see Documentation/Phase2_Neo4j_v2.md).
// Use same constraint names as original so v2 can replace old Phase 2 on a fresh DB.

CREATE CONSTRAINT act_act_id IF NOT EXISTS FOR (a:Act) REQUIRE a.act_id IS UNIQUE;
CREATE CONSTRAINT section_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE;
CREATE CONSTRAINT article_article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE;
CREATE CONSTRAINT case_case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE;
CREATE CONSTRAINT part_part_id IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE;
CREATE CONSTRAINT chapter_chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE;
CREATE CONSTRAINT definition_def_id IF NOT EXISTS FOR (d:Definition) REQUIRE d.def_id IS UNIQUE;

CREATE INDEX section_act_id IF NOT EXISTS FOR (s:Section) ON (s.act_id);
CREATE INDEX section_number IF NOT EXISTS FOR (s:Section) ON (s.section_number);
CREATE INDEX article_act_id IF NOT EXISTS FOR (a:Article) ON (a.act_id);
CREATE INDEX article_number IF NOT EXISTS FOR (a:Article) ON (a.article_number);
CREATE INDEX case_year IF NOT EXISTS FOR (c:Case) ON (c.year);
