// Phase-2: Smoke tests (run after imports)

// Node counts
MATCH (c:Case) RETURN count(c) AS cases;
MATCH (a:Act) RETURN count(a) AS acts;
MATCH (s:Section) RETURN count(s) AS sections;
MATCH (ar:Article) RETURN count(ar) AS articles;

// Relationship counts
MATCH ()-[r:IN_ACT]->() RETURN count(r) AS inActRels;
MATCH ()-[r:CITES]->() RETURN count(r) AS citesRels;

// Top cited targets (by relationship count property)
MATCH (:Case)-[r:CITES]->(t)
RETURN labels(t) AS labels,
       coalesce(t.section_id, t.article_id) AS target_id,
       sum(coalesce(r.count, 1)) AS cites
ORDER BY cites DESC
LIMIT 20;

// Sanity: acts linkage distribution
MATCH (x)-[:IN_ACT]->(act:Act)
RETURN labels(x) AS nodeType, act.act_id AS act_id, count(*) AS n
ORDER BY n DESC;

