---
name: phase4-graph-constrained-legalrag
overview: "Design and implement Phase 4: a local, production-ready Graph-Constrained Legal RAG system that integrates Neo4j, FAISS, and a local LLM via Ollama using LangGraph and a Streamlit UI."
todos:
  - id: setup-config-and-package
    content: Create the phase4_rag package and central config for Neo4j, FAISS paths, and Ollama model settings.
    status: completed
  - id: implement-neo4j-client
    content: Implement Neo4j client wrapper with helper functions to resolve section/article IDs and cases citing them.
    status: completed
  - id: implement-query-parser
    content: Implement query parser to detect explicit section/article references from user queries.
    status: completed
  - id: implement-vector-retriever
    content: Wrap Phase 3 FAISS retriever and add graph-constrained filtering logic.
    status: completed
  - id: implement-llm-client-and-prompt
    content: Implement Ollama client and structured prompt construction for legal answers with citations.
    status: completed
  - id: build-langgraph-workflow
    content: Define LangGraph state, nodes, and edges for the end-to-end pipeline.
    status: completed
  - id: build-streamlit-ui
    content: Create a Streamlit UI that connects to the LangGraph app and displays answers, citations, and snippets.
    status: completed
  - id: add-error-handling-and-tests
    content: Add error handling for Neo4j, FAISS, Ollama and basic smoke tests for key query scenarios.
    status: completed
isProject: false
---

## Phase 4: Graph-Constrained LegalRAG – Implementation Plan

### 1. Overall architecture and technology choices

- **Language & runtime**: Use Python (same as previous phases) for all orchestration and backend components.
- **Core components**:
  - **Neo4j graph layer**: Local Neo4j Desktop DB accessed via official `neo4j` Python driver using `bolt://localhost:7687` with credentials from config/.env.
  - **Vector layer (Phase 3 reuse)**: Reuse existing FAISS index and metadata from `[phase3_embeddings/config.py](phase3_embeddings/config.py)` paths:
    - `phase3_embeddings/output/faiss.index`
    - `phase3_embeddings/output/chunk_metadata.pkl`
    - Fine-tuned model at `phase3_embeddings/models/bge-legal`.
  - **LLM layer**: Local Ollama instance exposing `llama3:8b` (or compatible) via HTTP on `http://localhost:11434`.
  - **Orchestration**: LangGraph (Python) to define a graph with four main nodes: `query_parser`, `graph_retriever`, `vector_retriever`, `answer_generator` plus simple control edges.
  - **UI**: Streamlit single-page app as the entrypoint (`streamlit_app.py` or similar) that talks to a lightweight application layer (not directly to FAISS/Neo4j where possible).
- **Data contracts (IDs & metadata)**:
  - Reuse existing canonical IDs:
    - Cases: `case_id` (Neo4j `(:Case {case_id})`, Phase 1 `cases.csv`, Phase 3 chunks with `source_type="case"` and `source_id == case_id`).
    - Sections: `section_id` (Neo4j `(:Section {section_id})`, Phase 1 `sections.csv`, Phase 3 chunks with `source_type="section"`).
    - Articles: `article_id` (Neo4j `(:Article {article_id})`, Phase 1 `articles.csv`, Phase 3 chunks with `source_type="article"`).
  - Chunk metadata contract (from Phase 3): each FAISS vector row corresponds to a dict with `{chunk_id, source_type, source_id, text}` plus `score` on retrieval.

### 2. Project structure for Phase 4

- **New package**: Create a `phase4_rag/` Python package to keep Phase 4 logic isolated but co-located, e.g.:
  - `[phase4_rag/__init__.py](phase4_rag/__init__.py)`
  - `[phase4_rag/config.py](phase4_rag/config.py)` – central config: Neo4j URI/creds, Ollama model name, top-k defaults, etc.
  - `[phase4_rag/neo4j_client.py](phase4_rag/neo4j_client.py)` – thin wrapper around `neo4j` Python driver with typed helper methods.
  - `[phase4_rag/vector_retriever.py](phase4_rag/vector_retriever.py)` – FAISS/embedding loader and graph-constrained retrieval helpers that reuse Phase 3 code.
  - `[phase4_rag/query_parser.py](phase4_rag/query_parser.py)` – regex/heuristic parser for explicit Section/Article references.
  - `[phase4_rag/langgraph_workflow.py](phase4_rag/langgraph_workflow.py)` – LangGraph nodes, state, and compilation of the graph.
  - `[phase4_rag/llm_ollama.py](phase4_rag/llm_ollama.py)` – simple client around Ollama HTTP API.
  - `[streamlit_app.py](streamlit_app.py)` – Streamlit UI as Phase 4 entrypoint.
- **Config & secrets**:
  - Use environment variables (via `os.getenv` or optional `python-dotenv`) in `phase4_rag/config.py` for:
    - `NEO4J_URI` (default `bolt://localhost:7687`).
    - `NEO4J_USER`, `NEO4J_PASSWORD`.
    - `OLLAMA_BASE_URL` (default `http://localhost:11434`).
    - `OLLAMA_MODEL` (default `llama3:8b`).
  - Keep defaults wired so the system works on a local dev machine without extra config if Neo4j and Ollama follow standard defaults.

### 3. Query parsing (`query_parser` node)

- **Node purpose**: Accept raw user query and detect explicit references to Sections/Articles and possibly Acts.
- **Implementation outline (in `phase4_rag/query_parser.py`)**:
  - Define simple, deterministic regexes based on Phase 1 patterns:
    - Articles: e.g. `r"Article\s+(\d+(?:[A-Z])?)"` → map to `article_id = f"Constitution_Art_{num}"` (assuming Constitution focus; configurable later).
    - Sections: e.g. `r"Section\s+(\d+(?:\(\d+\))?)"` → later used to match `Section.section_number` via Neo4j rather than constructing `section_id` strings directly.
  - Return a structured `ParsedQuery` object/dict with fields like:
    - `raw_query: str`
    - `article_numbers: list[str]` (raw numeric/string identifiers)
    - `section_numbers: list[str]`
    - `explicit_ids: dict` with keys `"article_ids"`, `"section_ids"` when mapping is unambiguous.
  - Also include a boolean flag `has_explicit_refs` for use in graph control flow.
- **LangGraph node behaviour**:
  - Input: base state with `{"user_query": str}`.
  - Output: add `parsed_query` into state and optionally `constraint_ids` (if directly mapping to known article/section IDs is feasible using regular rules).

### 4. Graph retrieval (`graph_retriever` node)

- **Neo4j client design (`phase4_rag/neo4j_client.py`)**:
  - Initialize a driver lazily with config values; expose context-managed sessions.
  - Provide helper methods such as:
    - `get_sections_by_numbers(section_numbers: list[str]) -> list[dict]` mapping section numbers to `Section` nodes (`section_id`, `section_number`, `act_id`, `full_text`).
    - `get_articles_by_numbers(article_numbers: list[str]) -> list[dict]` mapping to `Article` nodes.
    - `get_cases_citing_ids(target_ids: list[str]) -> list[dict]` returning `Case` nodes (and possibly edge metadata) that have `(:Case)-[:CITES]->(:Section|:Article)` where node IDs in `target_ids`.
    - `get_sections_and_articles_for_case_ids(case_ids: list[str]) -> list[dict]` if needed for reverse navigation.
  - Use parametrized Cypher queries for safety and performance, e.g.:
    - For articles by numbers:
      - `MATCH (a:Article) WHERE a.article_number IN $nums RETURN a.article_id AS article_id, a.article_number AS article_number, a.full_text AS full_text, a.act_id AS act_id`.
    - For sections by numbers:
      - `MATCH (s:Section) WHERE s.section_number IN $nums RETURN s.section_id AS section_id, s.section_number AS section_number, s.full_text AS full_text, s.act_id AS act_id`.
    - For cases citing ids:
      - `MATCH (c:Case)-[r:CITES]->(t) WHERE t.section_id IN $ids OR t.article_id IN $ids RETURN DISTINCT c.case_id AS case_id, c.year AS year`.
- **Graph constraint logic**:
  - From `parsed_query` and Neo4j results, build a set of constraint IDs:
    - `section_ids` and `article_ids` directly from `get_sections_by_numbers` / `get_articles_by_numbers`.
    - `case_ids` from `get_cases_citing_ids` using those IDs.
  - Consolidate them into a `GraphConstraints` structure, e.g.:
    - `allowed_case_ids: set[str]`
    - `allowed_section_ids: set[str]`
    - `allowed_article_ids: set[str]`.
- **Error handling in this node**:
  - If Neo4j connection fails (driver cannot connect):
    - Log/record in state: `graph_error = "neo4j_unavailable"` and `graph_constraints = None`.
    - Allow the pipeline to continue with *pure vector retrieval* (no constraints).
  - If no matching nodes found for parsed references:
    - Set `graph_constraints` to an empty constraint object but mark this fact in state (e.g., `graph_constraints.empty = True`) so vector retriever can decide whether to fall back to unconstrained search.
- **LangGraph node behaviour**:
  - Inputs: `parsed_query`, base `user_query`.
  - Outputs: `graph_constraints`, `graph_metadata` (e.g. original matched node details for UI display: matched `section_id`/`article_id`, titles, etc.), and any `graph_error`.

### 5. Vector retrieval (`vector_retriever` node)

- **Reuse Phase 3 retriever (`phase3_embeddings.retrieve`)**:
  - In `phase4_rag/vector_retriever.py`:
    - On module import or via an explicit `init_vector_store()` function, call:
      - `from phase3_embeddings.retrieve import load_index, search`.
      - Cache `index, metadata, model` at module or class level to avoid re-loading per request.
  - Define a core function `retrieve_chunks(query: str, k: int, constraints: GraphConstraints | None) -> list[dict]` with behaviour:
    - If `constraints` is `None` or explicitly marked as disabled → basic top-k semantic search via `search`.
    - If `constraints` has non-empty sets of allowed IDs → run `search` (e.g., with `k_base = max(k * 3, k)`) and filter results where:
      - For each result `r` with `source_type` and `source_id`:
        - If `source_type == "case"`: keep only if `source_id` in `allowed_case_ids` (or `allowed_section/article` if cross-constraint logic is desired).
        - If `source_type == "section"`: keep only if `source_id` in `allowed_section_ids`.
        - If `source_type == "article"`: keep only if `source_id` in `allowed_article_ids`.
      - Truncate to top-k post-filtering by original score.
    - If filtered results are empty (no chunks satisfy the constraint):
      - Option 1 (recommended): fall back to unconstrained search but annotate state: `used_fallback_unconstrained = True`.
      - Option 2: return empty and let answer generator handle “no relevant content” – but the fallback is better UX.
- **Outputs and structure**:
  - Standardize retrieved chunks into a list of dicts with at least:
    - `chunk_id`, `source_type`, `source_id`, `text`, `score`.
  - Also compute a `grouped_sources` summary for the UI and LLM prompt, e.g. aggregated by `source_type` and `source_id` (collapsing multiple chunks from the same case/section/article).
- **Error handling**:
  - If FAISS index or `chunk_metadata.pkl` missing or fail to load:
    - Expose an initialization error in the state (e.g. `vector_error = "index_not_available"`) and propagate to UI.
  - If model fails to load (e.g., missing fine-tuned model):
    - Optionally fall back to baseline `BGE_MODEL` (if easy to wire via Phase 3 code), else treat as a hard error.

### 6. Answer generation (`answer_generator` node) with Ollama

- **LLM client (`phase4_rag/llm_ollama.py`)**:
  - Implement a small client using `requests` to call Ollama’s HTTP API:
    - Endpoint: `POST {OLLAMA_BASE_URL}/api/chat` with body containing `model`, `messages`, and optional `stream` parameter.
  - Provide both streaming (for future) and non-streaming `generate_answer` function returning plain text.
  - Handle errors such as:
    - Connection error → raise a custom exception or return an error object consumed by LangGraph.
    - Model missing (`llama3:8b` not pulled) → parse error message and surface a clear hint to pull the model.
- **Prompt construction**:
  - In `phase4_rag/langgraph_workflow.py` or a helper module, construct a structured prompt template that includes:
    - **System / instructions block**:
      - Explain that the model is a legal assistant that must answer based **only** on provided context, cite sections/articles and cases, and say “I don’t know” if insufficient information.
    - **Context block** compiled from vector retrieval results:
      - For each grouped source, include:
        - `SourceType`: Case/Section/Article
        - `SourceID`: (e.g., `case_id`, `section_id`, or `article_id`)
        - `Score`: approximate relevance score
        - `Snippet(s)`: truncated chunk texts.
      - Optionally preface with graph-derived explanation like “These statutes/articles were explicitly referenced: …”.
    - **User query block**: the original user question.
  - Respect a token budget by limiting number and length of snippets (e.g., top 6–10 chunks, truncated to N characters).
- **LangGraph node behaviour**:
  - Input: `user_query`, `retrieved_chunks`, `graph_metadata`, and any error flags (`graph_error`, `vector_error`).
  - If there are no chunks (and no hard errors):
    - Ask the LLM to respond with a graceful “no information available” message.
  - If vector or LLM errors occurred:
    - Surface a concise error message in the state for the UI to render.

### 7. LangGraph workflow definition

- **State schema**:
  - Define a simple Pydantic-like or TypedDict state for LangGraph, e.g.:
    - `user_query: str`
    - `parsed_query: dict`
    - `graph_constraints: dict | None`
    - `graph_metadata: dict | None`
    - `retrieved_chunks: list[dict]`
    - `used_fallback_unconstrained: bool`
    - `graph_error: str | None`
    - `vector_error: str | None`
    - `answer: str | None`
- **Nodes**:
  - `query_parser` → adds `parsed_query`.
  - `graph_retriever` → adds `graph_constraints`, `graph_metadata`, `graph_error`.
  - `vector_retriever` → uses `graph_constraints` to retrieve chunks, sets `retrieved_chunks` and `used_fallback_unconstrained`, `vector_error`.
  - `answer_generator` → uses `user_query`, `retrieved_chunks` and `graph_metadata` to produce `answer` and maybe `answer_metadata`.
- **Edges**:
  - Linear flow for MVP: `query_parser` → `graph_retriever` → `vector_retriever` → `answer_generator`.
  - Use LangGraph’s conditional routing only if needed (e.g., skip `graph_retriever` when there are no explicit refs), but start with a simple sequential graph to reduce complexity.
- **Compilation**:
  - In `langgraph_workflow.py`, define a helper `build_app()` that returns a compiled LangGraph app object given configuration, to be reused in tests and the Streamlit app.

### 8. Streamlit UI design (`streamlit_app.py`)

- **Layout**:
  - Page title: "Phase 4 – Graph-Constrained LegalRAG".
  - Sidebar:
    - Runtime status indicators: Neo4j connection status, FAISS index status, Ollama model name.
    - Configuration controls (optional): `top_k` slider, a toggle for “Require graph constraint (no fallback) vs allow fallback”.
  - Main area:
    - Text input area for the legal query.
    - Button: "Run Graph-Constrained RAG".
    - After submission:
      - **Generated answer**: main markdown block.
      - **Cited sections/articles**: a table or list showing `section_id` / `article_id`, section number / article number, perhaps snippet of `full_text` from graph or from chunks.
      - **Source case_ids**: list/table grouped by case, with `case_id`, year, and snippet.
      - **Retrieved text snippets**: expandable accordions showing each chunk’s text along with `source_type`, `source_id`, and `score`.
      - Indicators if fallbacks occurred, e.g. "Graph constraints applied" vs "Fell back to unconstrained retrieval due to no matching graph results".
- **Integration with LangGraph**:
  - On each query:
    - Construct an initial state with `user_query`.
    - Run the LangGraph app synchronously to completion (simple `invoke`/`run` pattern).
    - Extract `answer`, `retrieved_chunks`, `graph_metadata`, and any error flags, and render accordingly.
- **Error and edge-case handling in UI**:
  - If `graph_error == "neo4j_unavailable"`: show a visible warning badge but still show unconstrained vector results if available.
  - If `vector_error` indicates index/model missing: show clear instructions (e.g., "Please run Phase 3 (chunking, fine-tuning, build_faiss) before using Phase 4.").
  - If LLM call fails: show concise error and suggest checking Ollama service and model availability.

### 9. Error handling and resilience across layers

- **Neo4j layer**:
  - Wrap driver initialization and session usage in try/except blocks.
  - Timeouts or connection errors should:
    - Not crash the app.
    - Be captured as `graph_error` flags and allow vector retrieval to proceed.
- **Vector layer**:
  - At startup (or first call), attempt to load FAISS index and metadata once.
  - If loading fails:
    - Record a persistent error flag that the UI can query before allowing user queries.
- **LLM layer**:
  - Detect connection refused / timeouts and respond with informative error messages.
  - Detect missing model (parse common Ollama error text) and suggest `ollama pull llama3:8b`.
- **LangGraph orchestration**:
  - Ensure each node defends against receiving partial state (e.g., `graph_constraints` may be `None` if the previous node errored).
  - Ensure state always contains enough information for UI to show something (even if just an error with raw user query).

### 10. Testing and validation

- **Local smoke tests (CLI)**:
  - Add a small CLI entry in `phase4_rag/langgraph_workflow.py` or a separate `scripts/test_phase4_pipeline.py` that:
    - Takes a query from the command line.
    - Runs the LangGraph pipeline.
    - Prints answer, cited ids, and number of retrieved chunks.
- **Integration tests (manual)**:
  - Validate key scenarios from the UI:
    - Query with explicit article reference (e.g., "Explain Article 14 of the Constitution").
      - Expect: graph finds `Constitution_Art_14`, vector chunks constrained to that article and cases citing it, answer cites correct IDs.
    - Query with explicit section reference.
    - Query without any explicit references (e.g., "What are the principles of natural justice?") – pipeline should run with unconstrained semantics.
    - Cases where Neo4j is stopped to confirm graceful degradation.
    - Cases where Ollama is stopped to check error display.
- **Performance considerations**:
  - Keep FAISS index in memory across requests.
  - Short-circuit or cap top-k and context length to preserve responsiveness for local usage.

### 11. Future extensibility hooks (not required now but planned for)

- **Reranking**: Add a reranking step (e.g., cross-encoder) after vector retrieval.
- **Conversation history**: Extend state to include previous turns and feed them into the LLM.
- **Advanced graph constraints**: e.g., limit retrieval to a specific Act, or to cases decided after/before a given year.
- **API mode**: Wrap LangGraph app in a FastAPI/Flask microservice for non-Streamlit uses.

This plan keeps everything local, leverages your existing Phase 3 artifacts and Neo4j schema, and clearly separates responsibilities across modules (Neo4j, vector retrieval, LangGraph workflow, LLM client, and UI), while providing explicit error-handling paths for Neo4j downtime, missing FAISS index/model, or Ollama/model issues.