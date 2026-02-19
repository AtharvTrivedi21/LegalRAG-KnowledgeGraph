# Phase 4: RAG Application (LangGraph + Neo4j + FAISS)

## What

Phase 4 is the **retrieval-augmented generation (RAG)** application. It orchestrates a **LangGraph workflow** that: parses the user query for section/article references (with optional Act disambiguation), queries **Neo4j** for matching sections/articles and citing cases, **rephrases** the query for semantic search, runs **FAISS retrieval** (optionally constrained to graph results), and generates an answer via **Ollama**. The Streamlit app (`streamlit_app_v3.py`) and test script (`test_three_queries.py`) invoke this workflow. Phase 4 does not read any datasets directly; it uses Phase 3 artifacts (FAISS index, chunk metadata, fine-tuned model) and the Neo4j graph populated from Phase 1 CSVs.

## How

### Workflow (linear)

1. **query_parser** — Parses `user_query` for article numbers, section numbers, and explicit act context (e.g. “Section 302 of BNS”). Outputs `parsed_query` (and optionally `section_act_id`, `article_act_id` for Act-aware disambiguation).
2. **graph_retriever** — Calls Neo4j: get sections by number (optionally filtered by act_id), get articles by number (optionally filtered by act_id), get cases citing those section/article IDs. Builds `graph_metadata` (sections, articles, cases) and `graph_constraints` (allowed_case_ids, allowed_section_ids, allowed_article_ids) for the vector retriever. Collects `applicable_acts` for display.
3. **query_rephrase** — Sends the user query to Ollama with a prompt to rewrite it as a formal legal query; stores result in `legal_query`. Used as the embedding query for retrieval.
4. **vector_retriever** — Embeds `legal_query` (or `user_query`), runs FAISS search. If `graph_constraints` are present, filters and diversifies results (e.g. min sections, min articles, constrained multiplier). Populates `retrieved_chunks` and `grouped_sources`; may set `used_fallback_unconstrained` if constrained retrieval failed.
5. **answer_generator** — Builds a prompt from graph context (sections/articles from Neo4j) and retrieval context (chunks from FAISS). Calls Ollama with a system prompt that enforces structured output (Summary, Applicable laws/provisions, Relevant case law, Recommendation). Strips internal labels, fixes markdown, and sets `answer_no_applicable_laws` / `answer_no_relevant_cases` based on answer content.

### Components

| Component | File | Role |
|-----------|------|------|
| Config | `phase4_rag/config_v3.py` | Neo4j (URI, user, password), Ollama (base_url, model, timeout), retrieval (top_k, constrained_multiplier, diversity_multiplier, min_sections_per_query, min_articles_per_query); paths to Phase 3 FAISS/metadata/model. |
| Query parser | `phase4_rag/query_parser_v3.py` | Extracts article/section numbers and optional act context; returns ParsedQuery (e.g. section_act_id, article_act_id for disambiguation). |
| Neo4j client | `phase4_rag/neo4j_client_v3.py` | get_sections_by_numbers(section_numbers, act_id=None), get_articles_by_numbers(article_numbers, act_id=None), get_cases_citing_ids(ids); Act-aware via IN_ACT. |
| Vector retriever | `phase4_rag/vector_retriever_v3.py` | Uses Phase 3 load_index/search; GraphConstraints; retrieve_chunks(), group_by_source(). |
| LangGraph workflow | `phase4_rag/langgraph_workflow_v3.py` | StateGraph(WorkflowState); nodes and edges as above; build_app() returns compiled graph. |
| LLM | `phase4_rag/llm_ollama.py` | ChatMessage, chat_completion(); talks to Ollama API. |
| UI | `streamlit_app_v3.py` | Streamlit app that invokes the V3 workflow and displays answer, applicable acts, sections, cases. |

### Datasets / inputs

- **No direct dataset paths.** Phase 4 reads from:
  - Phase 3: `FAISS_INDEX_PATH`, `CHUNK_METADATA_PATH`, `FINE_TUNED_MODEL_DIR` (from phase3_embeddings config).
  - Neo4j: graph built in Phase 2 from Phase 1 CSVs.
  - Ollama: model name from env/config (e.g. `llama3:8b`).

## Why

- **Graph-first then vector:** Resolving sections/articles and citing cases in Neo4j ensures the answer can cite specific provisions and case law; FAISS then adds semantically similar context.
- **Act-aware disambiguation:** Section numbers can repeat across BNS/BNSS/BSA; optional act_id in the query and in Neo4j queries reduces ambiguity.
- **Query rephrase:** Formal legal phrasing often improves retrieval quality vs. informal user text.
- **Structured answer:** Fixed headings (Summary, Applicable laws, Case law, Recommendation) make output consistent and report-ready.
- **Confidence signals:** answer_no_applicable_laws and answer_no_relevant_cases allow the UI to show when the model concluded no laws or no cases were relevant.

## Results

Phase 4 results are **demonstration outputs** from running the workflow on example queries.

### Example queries

The script `test_three_queries.py` runs three fixed queries (using the V2 workflow; the V3 app behaves similarly with Act-aware parsing and structured answer):

1. **"Explain Article 14 of the Constitution"**  
2. **"What does Section 302 of BNS say?"**  
3. **"Someone entered my home and stole my property"**  

### What a run produces

For each query, a run typically prints (or the Streamlit app shows):

- **Formal legal query** — If the rephrase step changed the query, a short version of `legal_query`.
- **Answer** — The model’s reply (Summary, Applicable laws/provisions, Relevant case law, Recommendation).
- **Retrieved** — Counts by `source_type` (e.g. `{'case': 5, 'section': 2, 'article': 1}`).

If retrieval or Ollama fails, an error message is shown instead (e.g. vector_error or Ollama error).

### How to capture for reports/PPT

1. From project root (with venv active and Neo4j/Ollama/Phase 3 set up):  
   `python test_three_queries.py`  
   Or run the Streamlit app:  
   `streamlit run streamlit_app_v3.py`
2. Copy the console output (or screenshot the Streamlit answer and retrieval mix) for each of the three queries.
3. Use these as “Phase 4 – RAG example results” in reports or slides. No need to invent answer text; the doc explains what the run shows and where to get live results.
