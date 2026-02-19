// Phase-2: Constraints + indexes (Neo4j 5.x)
// Run once per database.

CREATE CONSTRAINT case_case_id IF NOT EXISTS
FOR (c:Case)
REQUIRE c.case_id IS UNIQUE;

CREATE CONSTRAINT act_act_id IF NOT EXISTS
FOR (a:Act)
REQUIRE a.act_id IS UNIQUE;

CREATE CONSTRAINT section_section_id IF NOT EXISTS
FOR (s:Section)
REQUIRE s.section_id IS UNIQUE;

CREATE CONSTRAINT article_article_id IF NOT EXISTS
FOR (a:Article)
REQUIRE a.article_id IS UNIQUE;

// Helpful indexes (optional)
CREATE INDEX case_year IF NOT EXISTS
FOR (c:Case)
ON (c.year);

CREATE INDEX section_number IF NOT EXISTS
FOR (s:Section)
ON (s.section_number);

CREATE INDEX article_number IF NOT EXISTS
FOR (a:Article)
ON (a.article_number);

