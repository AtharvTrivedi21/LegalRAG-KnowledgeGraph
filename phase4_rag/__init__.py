"""
Phase 4: Graph-Constrained LegalRAG package.

This package integrates:
- Neo4j (graph constraints),
- FAISS + fine-tuned BGE embeddings (Phase 3),
- A local LLM served via Ollama,
- LangGraph for orchestration.

The main public entrypoints are:
- `phase4_rag.langgraph_workflow.build_app` for the LangGraph workflow.
- `streamlit_app.py` at the repo root for the Streamlit UI.
"""

