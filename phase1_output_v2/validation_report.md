# Phase 1 v2 Validation Report

## Counts
| File | Count |
|------|-------|
| acts.csv | 4 |
| articles.csv | 453 |
| chapters.csv | 49 |
| citation_resolution_rate_pct | 20.8 |
| definitions.csv | 0 |
| parts.csv | 48 |
| resolved_article_cites | 27399 |
| resolved_section_cites | 115891 |
| sections.csv | 741 |
| unresolved_cites | 546505 |

## Errors

parts.csv: 24 duplicate part_id values
chapters.csv: 2 duplicate chapter_id values
sections.csv: 7 duplicate section_id values

## Warnings

articles.csv: 4 rows with full_text length < 20
Citation resolution rate low: 20.8%
