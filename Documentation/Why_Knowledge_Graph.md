# Why a Knowledge Graph for Indian Legal RAG

## The Problem with Flat Vector Retrieval

Existing legal RAG systems (e.g. BNS Mitra) store document chunks in a flat FAISS vector index. This approach has fundamental limitations:

- **No structural awareness.** A section is just an anonymous blob of text, indistinguishable from its neighbours. The system has no concept of which Part, Chapter, or Act a provision belongs to.
- **Single-act scope.** Only one PDF (BNS) is ingested. Queries that span criminal procedure (BNSS) or evidence law (BSA) cannot be answered.
- **No cross-referencing.** There is no way to know that BNS Section 199 refers to BNSS Section 173, or that BSA Section 120 refers to BNS Section 64.
- **No case law.** No Supreme Court judgments, no way to ground answers in judicial interpretation.
- **No traceability.** The LLM produces section numbers, but there is no verifiable link back to the authoritative source text.

## Why a Knowledge Graph Solves This

A Neo4j property graph turns the legal corpus into a **structured, traversable, relational** data model instead of a bag of text chunks.

When the RAG pipeline receives a query like "What is Section 302 of BNS?", it does not just do vector similarity — it **walks the graph**:

1. Find the Section node for BNS s.302.
2. Retrieve its parent Chapter and Part for structural context.
3. Look up legal Definitions of terms used in that section.
4. Find other Sections that cross-reference it (within the same act or across acts).
5. Find Supreme Court Cases that cite it, with surrounding citation context.

Only then is the vector index queried — and it is **constrained** to retrieve chunks relevant to those graph-identified provisions. The graph decides *what matters*; FAISS handles *what's semantically similar*.

### Benefits

| Benefit | Description |
|---------|-------------|
| **Structured hierarchy** | Act → Part → Chapter → Section/Article mirrors how legislation is actually organized. |
| **Multi-hop reasoning** | Queries like "find cases citing sections in Chapter X of BNS" become graph traversals. |
| **Graph-constrained retrieval** | The graph identifies relevant provisions first; FAISS is filtered to those. |
| **Explicit relationships** | HAS_PART, HAS_CHAPTER, DEFINES_TERM, REFERENCES, CITES — each relationship has legal meaning. |
| **Traceable citations** | Every answer links back to specific section_id, article_id, and case_id values that can be verified against source material. |
| **Multi-act coverage** | Four acts + Constitution in one graph, interconnected by cross-references and case citations. |

---

## Graph Schema

### Node Types (7)

| Label | Key | Description |
|-------|-----|-------------|
| **Act** | `act_id` | Statute or Constitution (BNS_2023, BNSS_2023, BSA_2023, CONST_1950) |
| **Part** | `part_id` | Numbered Part of an act |
| **Chapter** | `chapter_id` | Chapter under a Part |
| **Section** | `section_id` | Section of BNS/BNSS/BSA (e.g. BNS_2023_s302) |
| **Article** | `article_id` | Article of the Constitution (e.g. CONST_1950_Art21) |
| **Definition** | `def_id` | Defined legal term extracted from definition sections |
| **Case** | `case_id` | Supreme Court or IL-TUR judgment |

### Relationship Types (8)

| Relationship | From → To | Meaning |
|-------------|-----------|---------|
| `HAS_PART` | Act → Part | Act contains Part |
| `HAS_CHAPTER` | Part → Chapter | Part contains Chapter |
| `HAS_SECTION` | Chapter → Section, Act → Section | Containment |
| `HAS_ARTICLE` | Act → Article | Act contains Article |
| `IN_ACT` | Section → Act, Article → Act | Reverse lookup for Phase 4 |
| `DEFINES_TERM` | Section → Definition | Section defines a legal term |
| `REFERENCES` | Section → Section \| Article | Cross-references within or across acts |
| `CITES` | Case → Section \| Article | Case cites a provision (with context) |

---

## How the Four Acts + Constitution Connect

### Cross-Act Section References (extracted from legislative text)

The three 2023 criminal law statutes explicitly reference each other in their text. These cross-references are extracted by regex from the full section text and stored as `REFERENCES` edges with `reference_type: cross_act`.

| From Act | To Act | Volume | Examples |
|----------|--------|--------|----------|
| **BNS → BNSS** | ~4 refs | BNS s.5 → BNSS s.474 (sentencing procedure), BNS s.199 → BNSS s.173 (FIR registration) |
| **BNSS → BNS** | ~30+ refs | BNSS s.218 → BNS s.64, BNSS s.220 → BNS s.85, BNSS s.346 → BNS s.71, etc. The procedure code heavily cross-references the penal code. |
| **BNSS → BSA** | ~4 refs | BNSS s.181 → BSA s.148 (witness contradiction), BNSS s.182 → BSA s.22 (confessions) |
| **BSA → BNS** | ~6 refs | BSA s.117 → BNS s.86 (cruelty definition), BSA s.118 → BNS s.80 (dowry death), BSA s.120 → BNS s.64 (rape) |

### Case Law as the Bridge to the Constitution

The Constitution of India predates the 2023 acts and does not reference their specific sections. Conversely, the 2023 acts reference the Constitution using "Article" rather than "section."

**Cases act as the connective tissue.** A Supreme Court judgment may cite Article 21 (right to life) and BNS Section 103 (murder) in the same opinion — creating a traversable path through the graph between the Constitution and the penal code.

```
BNS ←──REFERENCES──→ BNSS    (heavily connected, ~34 cross-refs)
        ↕                      ↕
       BSA ←──REFERENCES──→ BNSS  (moderately connected, ~4 refs)
       BSA ──REFERENCES──→ BNS   (moderately connected, ~6 refs)

Constitution ←──CITES──── Cases ────CITES──→ BNS / BNSS / BSA sections
```

### Within-Act References

Each act also has extensive **intra-act** references (e.g. BNS s.38 references BNS s.37 on private defence restrictions). These are stored as `REFERENCES` edges with `reference_type: see_also` and make up the bulk of the ~560+ total reference rows.

---

## Comparison: Flat RAG vs. Graph-Constrained RAG

| Dimension | BNS Mitra (Flat RAG) | Our System (Graph RAG) |
|-----------|---------------------|----------------------|
| **Knowledge base** | Single PDF (BNS only) | 4 Acts + Constitution + SC cases + IL-TUR |
| **Data representation** | Flat text chunks | Hierarchical graph (Act→Part→Chapter→Section) with definitions and cross-references |
| **Knowledge graph** | None | Neo4j with 7 node types and 8 relationship types |
| **Embeddings** | Off-the-shelf OllamaEmbeddings | BGE fine-tuned on IndicLegalQA (MRR@10 = 0.77) |
| **Retrieval** | Single-hop FAISS dense search | Graph-constrained hybrid retrieval with diversity controls |
| **Act disambiguation** | Not needed (single act) | Act-aware query parsing across BNS/BNSS/BSA/Constitution |
| **Case law** | Not included | SC judgments + IL-TUR with automated citation linking |
| **Answer format** | Free-form LLM output | Structured: Summary + Laws + Case Law + Recommendation with traceable citations |

---

## Known Gaps and Future Work

1. **Constitution article references from within acts.** The current cross-act regex in `citations.py` only captures `section X of the Constitution of India`. The acts actually reference the Constitution using "Article" — a pattern like `article X of the Constitution of India` needs to be added to capture these.

2. **REPLACES / EQUIVALENT_TO / REPEALS relationships.** Planned in `PLAN_1_Static_Legal_Corpus.md` but not yet implemented. BNS replaces IPC, BNSS replaces CrPC, BSA replaces IEA. Most existing case law cites old-law sections; without equivalence mappings, those citations cannot connect to the new-law provisions.

3. **Curated Constitution-to-Act mappings.** Constitutional fundamental rights articles (Part III, Articles 14–32) are directly relevant to criminal law provisions on arrest, bail, and personal liberty in BNSS. A curated mapping (not regex-extractable) would strengthen the graph.
