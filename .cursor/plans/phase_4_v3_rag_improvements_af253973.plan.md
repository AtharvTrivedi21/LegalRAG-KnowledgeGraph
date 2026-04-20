---
name: Phase 4 V3 RAG Improvements
overview: Implement Act-aware section disambiguation (filter by act_id when user mentions BNS/BNSS/BSA), surface parent Act names in all outputs and the Streamlit UI, and add a FAISS similarity confidence guard that skips generation when top score is below a threshold.
todos: []
isProject: false
---

# Phase 4 V3: Act-aware disambiguation, Act classification, and confidence guard

## Scope (local and modular)

- **Neo4j retriever**: [phase4_rag/neo4j_client.py](c:\Users\ATHARV\LegalRAG\phase4_rag\neo4j_client.py) — filter sections/articles by `act_id` when provided; return `act_name` for display.
- **Query parser**: [phase4_rag/query_parser.py](c:\Users\ATHARV\LegalRAG\phase4_rag\query_parser.py) — detect Act mentions (BNS, BNSS, BSA, Constitution) and pass act hints into the pipeline.
- **Workflow**: [phase4_rag/langgraph_workflow_v2.py](c:\Users\ATHARV\LegalRAG\phase4_rag\langgraph_workflow_v2.py) — pass act filters to Neo4j; apply confidence guard; include Act names in context and in a structured “Applicable Act” field for the answer/UI.
- **Vector retriever**: [phase4_rag/vector_retriever.py](c:\Users\ATHARV\LegalRAG\phase4_rag\vector_retriever.py) — return `top_faiss_similarity` (max score from the initial FAISS search) for the confidence guard.
- **Config**: [phase4_rag/config.py](c:\Users\ATHARV\LegalRAG\phase4_rag\config.py) — add `min_similarity_threshold` (e.g. 0.45).
- **UI**: [streamlit_app_v2.py](c:\Users\ATHARV\LegalRAG\streamlit_app_v2.py) — show “Applicable Act:  (act_id)” prominently and keep Acts as a separate field in cited references.

No changes to Phase 3, Neo4j load scripts, or other modules beyond the ones above.

---

## 1. Act-aware section disambiguation

**Problem:** Queries like “Section 302 of BNS” currently hit Neo4j by `section_number` only, so BNSS/BSA sections with the same number can be returned.

**1.1 Query parser — extract Act hints**

- In [query_parser.py](c:\Users\ATHARV\LegalRAG\phase4_rag\query_parser.py):
  - Add regex/patterns to detect explicit Act mentions, e.g. “of BNS”, “under BNSS”, “BSA”, “Bharatiya Nyaya Sanhita”, “Constitution of India”, “Article … Constitution”.
  - Normalize to canonical `act_id`s: `BNS`, `BNSS`, `BSA`, `Constitution` (align with [neo4j_client](c:\Users\ATHARV\LegalRAG\phase4_rag\neo4j_client.py) and existing `act_id` in DB).
  - Extend `ParsedQuery` with e.g. `section_act_id: Optional[str]` and `article_act_id: Optional[str]` (or a single `act_ids: List[str]` if one list is enough for both sections and articles). Use one field per “type” so that “Section 302 of BNS” sets only section_act_id, and “Article 14 Constitution” sets only article_act_id.
  - Keep existing behaviour when no Act is mentioned: act_id remains `None` and downstream uses section-only / article-only search (fallback).

**1.2 Neo4j client — filter by section_number AND act_id**

- In [neo4j_client.py](c:\Users\ATHARV\LegalRAG\phase4_rag\neo4j_client.py):
  - **Sections:** Change `get_sections_by_numbers(section_numbers, act_id=None)` (or similar). Query must:
    - If `act_id` is provided: `MATCH (s:Section)-[:IN_ACT]->(a:Act)` and add `WHERE s.section_number IN $nums AND a.act_id = $act_id` (or `a.act_id IN $act_ids` if multiple).
    - If `act_id` is not provided: keep current behaviour (match by `s.section_number IN $nums` only; join to Act only to return `act_id`/`act_name`).
    - Return dicts with at least: `section_id`, `section_number`, `act_id`, `act_name`, `full_text`. Get `act_name` from the matched `Act` node (e.g. `a.act_name AS act_name`).
  - **Articles:** Same idea for `get_articles_by_numbers(article_numbers, act_id=None)`: when `act_id` is provided, filter with `WHERE a.article_number IN $nums AND act_node.act_id = $act_id` (via `(ar:Article)-[:IN_ACT]->(act_node:Act)`), and return `article_id`, `article_number`, `act_id`, `act_name`, `full_text`.
  - Use the existing `Section`/`Article` → `IN_ACT` → `Act` relationship for both filtering and projecting `act_id`/`act_name` (schema in [02_load_nodes.cypher](c:\Users\ATHARV\LegalRAG\neo4j\cypher\02_load_nodes.cypher)).

**1.3 Workflow — pass Act hints into graph retriever**

- In [langgraph_workflow_v2.py](c:\Users\ATHARV\LegalRAG\phase4_rag\langgraph_workflow_v2.py):
  - In `node_graph_retriever`, read from `parsed_query` the new act fields (e.g. `section_act_id`, `article_act_id`).
  - Call `get_sections_by_numbers(section_numbers, act_id=section_act_id)` and `get_articles_by_numbers(article_numbers, act_id=article_act_id)`.
  - No change to how constraints/cases are built; only the section/article lists become act-filtered, so BNSS/BSA sections are excluded when the user asked for BNS.

Result: “Section 302 of BNS” yields only BNS section 302; “Section 302” without an Act keeps current behaviour (all acts with that number).

---

## 2. Explicit Act classification in outputs

**Goal:** Every retrieved section/article surfaces its parent Act name; the answer and UI show “Applicable Act: Bharatiya Nyaya Sanhita (BNS)” (or BNSS, BSA, Constitution of India) as a distinct field.

**2.1 Neo4j already returning act_id and act_name**

- After 1.2, section/article dicts from Neo4j include `act_id` and `act_name`. Ensure `act_name` is the full statutory name (e.g. “Bharatiya Nyaya Sanhita”) as stored in [Act nodes](c:\Users\ATHARV\LegalRAG\phase4_rag\neo4j_display_v2.py) (get_acts_by_ids returns `act_name`). If the DB stores short names, consider a small mapping in code (act_id → display name) for the four Acts: BNS, BNSS, BSA, Constitution.

**2.2 Answer formatter (workflow)**

- In [langgraph_workflow_v2.py](c:\Users\ATHARV\LegalRAG\phase4_rag\langgraph_workflow_v2.py):
  - **Graph context block:** In `_build_graph_context_block`, when emitting section/article lines, include the parent Act for each item, e.g. “Applicable Act: Bharatiya Nyaya Sanhita (BNS)” so the model sees it. Optionally add a single “Applicable Acts” line at the start summarizing distinct acts in the context.
  - **Structured “Applicable Act” in state:** Add to `WorkflowState` an optional field such as `applicable_acts: List[Dict]` (e.g. `[{act_id, act_name}]`) derived from graph_metadata sections/articles (and optionally from grouped_sources if you surface act_id from chunk metadata). Populate this in the graph retriever or answer generator from `graph_metadata` so the UI can show it without re-querying.
  - **System prompt:** Instruct the model to cite provisions with their Act where relevant (e.g. “Section 302, BNS (Bharatiya Nyaya Sanhita)”) and to not drop Act names from citations.

**2.3 Streamlit UI**

- In [streamlit_app_v2.py](c:\Users\ATHARV\LegalRAG\streamlit_app_v2.py):
  - **Applicable Act as separate field:** When displaying the answer, if `applicable_acts` (or the list of acts from graph_metadata) is non-empty, show a clear line above or below the answer, e.g. **Applicable Act(s):** “Bharatiya Nyaya Sanhita (BNS)”, “Bharatiya Nagarik Suraksha Sanhita (BNSS)”, etc. Treat these as official statutory sources.
  - **Cited references — Acts:** You already have “Cited references — Acts” using `get_acts_by_ids(act_ids)`. Keep it; optionally enrich display with the same “Applicable Act:  (act_id)” wording for consistency.
  - **Sections/Articles panels:** Continue showing act_id (and act_name when available) next to each section/article so every cited provision clearly shows its parent Act.

---

## 3. Lightweight confidence guard (FAISS similarity threshold)

**Goal:** If the top FAISS similarity is below a threshold (e.g. 0.45), return “Insufficient relevant statutory context found” and do not call the LLM.

**3.1 Config**

- In [config.py](c:\Users\ATHARV\LegalRAG\phase4_rag\config.py), under `RetrievalSettings`, add:
  - `min_similarity_threshold: float = 0.45`
  - Load from env e.g. `PHASE4_MIN_SIMILARITY_THRESHOLD` (default 0.45).

**3.2 Vector retriever**

- In [vector_retriever.py](c:\Users\ATHARV\LegalRAG\phase4_rag\vector_retriever.py):
  - In `retrieve_chunks`, after running the initial FAISS search (the `search(...)` call that produces `base_results`), compute `top_faiss_similarity = max(r["score"] for r in base_results)` (or 0.0 if no results).
  - Include `top_faiss_similarity` in the returned dict for every code path (unconstrained, constrained, fallback).

**3.3 Workflow**

- In [langgraph_workflow_v2.py](c:\Users\ATHARV\LegalRAG\phase4_rag\langgraph_workflow_v2.py):
  - **State:** Add optional `top_faiss_similarity: Optional[float]` and ensure the vector retriever node sets it from `result["top_faiss_similarity"]`.
  - **Answer generator:** At the start of `node_answer_generator`, if `vector_error` is set, keep current error message. Else if `top_faiss_similarity` is not None and `top_faiss_similarity < settings.retrieval.min_similarity_threshold`, set `state["answer"] = "Insufficient relevant statutory context found."` and return without calling the LLM. Otherwise proceed with existing generation and post-processing.

**Note:** When constraints are used, the “initial” FAISS search is still performed (e.g. `base_results = search(...)`); use that for `top_faiss_similarity` so the guard reflects semantic match quality, not the artificial 1.0 from must-include chunks.

---

## 4. Data flow summary

```mermaid
flowchart LR
  subgraph parse [Query Parser]
    Q[User query]
    PQ[ParsedQuery]
    Q --> PQ
  end
  subgraph graph [Graph Retriever]
    PQ --> G
    G[get_sections_by_numbers with act_id]
    G --> M[graph_metadata with act_name]
  end
  subgraph vector [Vector Retriever]
    V[retrieve_chunks]
    V --> T[top_faiss_similarity]
  end
  subgraph answer [Answer Generator]
    T --> Guard{top < threshold?}
    Guard -->|Yes| Msg["Insufficient relevant statutory context found"]
    Guard -->|No| Gen[LLM + Act-aware context]
    Gen --> Out[Answer + applicable_acts]
  end
  parse --> graph
  graph --> vector
  vector --> answer
```



---

## 5. File-level checklist


| File                                                                                                | Changes                                                                                                                                                                                          |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [phase4_rag/query_parser.py](c:\Users\ATHARV\LegalRAG\phase4_rag\query_parser.py)                   | Add Act detection (BNS/BNSS/BSA/Constitution); extend `ParsedQuery` with `section_act_id`, `article_act_id` (or equivalent).                                                                     |
| [phase4_rag/neo4j_client.py](c:\Users\ATHARV\LegalRAG\phase4_rag\neo4j_client.py)                   | `get_sections_by_numbers(nums, act_id=None)`, `get_articles_by_numbers(nums, act_id=None)`; filter by Act when provided; return `act_name` via `(Section/Article)-[:IN_ACT]->(Act)`.             |
| [phase4_rag/langgraph_workflow_v2.py](c:\Users\ATHARV\LegalRAG\phase4_rag\langgraph_workflow_v2.py) | Pass act hints to Neo4j; add `top_faiss_similarity` to state and confidence guard in answer node; add `applicable_acts` (or derive in UI); include Act names in graph context and system prompt. |
| [phase4_rag/vector_retriever.py](c:\Users\ATHARV\LegalRAG\phase4_rag\vector_retriever.py)           | Return `top_faiss_similarity` from `retrieve_chunks` (max score from initial FAISS run).                                                                                                         |
| [phase4_rag/config.py](c:\Users\ATHARV\LegalRAG\phase4_rag\config.py)                               | Add `min_similarity_threshold` to `RetrievalSettings` and env loading.                                                                                                                           |
| [streamlit_app_v2.py](c:\Users\ATHARV\LegalRAG\streamlit_app_v2.py)                                 | Show “Applicable Act(s): (act_id)” prominently; keep Acts as separate field; ensure sections/articles display act_name where available.                                                          |


---

## 6. Edge cases and fallbacks

- **Act not mentioned:** Parser leaves `section_act_id` / `article_act_id` as `None`; Neo4j uses section-only/article-only filter; current behaviour preserved.
- **Unknown Act string:** Parser can map only known aliases (BNS, BNSS, BSA, Constitution); otherwise treat as no act and use section-only search.
- **Constitution:** “Article 14 of Constitution” sets `article_act_id = "Constitution"`; `get_articles_by_numbers` filters by that act_id so only Constitution articles are returned.
- **Confidence guard with constraints:** Threshold is applied to the raw FAISS score of the initial search; must-include chunks (score 1.0) do not override this, so low semantic match still triggers “Insufficient relevant statutory context found” when appropriate.

