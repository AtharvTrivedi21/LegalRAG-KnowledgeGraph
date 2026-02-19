from __future__ import annotations

import textwrap

import streamlit as st

from phase4_rag.config import settings
from phase4_rag.langgraph_workflow import build_app
from phase4_rag.neo4j_client import Neo4jUnavailableError, get_sections_by_numbers
from phase4_rag.vector_retriever import vector_store_status


@st.cache_resource(show_spinner=False)
def get_app():
    return build_app()


def _neo4j_status() -> str:
    try:
        # Lightweight probe: attempt a trivial query via a helper that
        # should be cheap if Neo4j is running.
        _ = get_sections_by_numbers([])
        return "connected"
    except Neo4jUnavailableError:
        return "unavailable"
    except Exception:
        return "error"


def main() -> None:
    st.set_page_config(page_title="Phase 4 – Graph-Constrained LegalRAG", layout="wide")

    st.title("Phase 4 – Graph-Constrained LegalRAG")
    st.write(
        "End-to-end local Graph-Constrained RAG over Neo4j, FAISS, and a local LLM via Ollama."
    )

    # Sidebar: runtime status and basic configuration
    with st.sidebar:
        st.header("Status")

        # Neo4j
        neo4j_status = _neo4j_status()
        if neo4j_status == "connected":
            st.success(f"Neo4j: connected ({settings.neo4j.uri})")
        elif neo4j_status == "unavailable":
            st.warning("Neo4j: unavailable (will fall back to unconstrained retrieval)")
        else:
            st.error("Neo4j: error (check connection and credentials)")

        # Vector store
        vs_status = vector_store_status()
        if vs_status["ok"]:
            st.success(f"FAISS index: loaded ({vs_status['index_size']} chunks)")
        else:
            st.error(f"FAISS index: not available\n\n{vs_status['error']}")

        # Ollama
        st.header("LLM")
        st.write(f"Model: `{settings.ollama.model}`")
        st.write(f"Ollama base URL: `{settings.ollama.base_url}`")

        st.header("Retrieval")
        top_k = st.slider(
            "Top-k chunks",
            min_value=3,
            max_value=20,
            value=settings.retrieval.top_k,
        )
        require_constraints = st.checkbox(
            "Require graph constraints (no fallback to unconstrained)",
            value=False,
            help="If enabled and no graph constraints are available, retrieval will still run "
            "but may return fewer strongly related results.",
        )

    # Main query area
    st.subheader("Query")
    query = st.text_area(
        "Enter a legal question (e.g., 'Explain Article 14 of the Constitution')",
        height=120,
    )

    if st.button("Run Graph-Constrained RAG", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Running LangGraph pipeline..."):
            app = get_app()
            # Pass top_k through state; vector node will respect it.
            initial_state = {"user_query": query, "top_k": top_k}
            state = app.invoke(initial_state)

        st.subheader("Answer")
        answer = state.get("answer") or "No answer generated."
        st.markdown(answer)

        graph_error = state.get("graph_error")
        vector_error = state.get("vector_error")
        used_fallback = state.get("used_fallback_unconstrained")

        cols = st.columns(3)
        with cols[0]:
            if graph_error:
                st.warning("Graph constraints: unavailable (Neo4j issue).")
            else:
                st.success("Graph constraints: applied when available.")
        with cols[1]:
            if vector_error:
                st.error("Vector retrieval: error (see answer for details).")
            else:
                st.success("Vector retrieval: OK.")
        with cols[2]:
            if used_fallback:
                st.info("Fallback: unconstrained retrieval used (no matching constrained hits).")
            else:
                st.write("Fallback: not used.")

        # Cited sections/articles and cases
        st.subheader("Cited sections / articles and cases")
        grouped_sources = state.get("grouped_sources") or {}

        # Sections & Articles
        sec_art_cols = st.columns(2)
        with sec_art_cols[0]:
            st.markdown("**Sections**")
            for sid, info in grouped_sources.get("section", {}).items():
                st.write(f"- `{sid}` (score ~ {info.get('max_score', 0.0):.3f})")
        with sec_art_cols[1]:
            st.markdown("**Articles**")
            for aid, info in grouped_sources.get("article", {}).items():
                st.write(f"- `{aid}` (score ~ {info.get('max_score', 0.0):.3f})")

        # Cases
        st.markdown("**Cases**")
        for cid, info in grouped_sources.get("case", {}).items():
            st.write(f"- `{cid}` (score ~ {info.get('max_score', 0.0):.3f})")

        # Retrieved snippets
        st.subheader("Retrieved text snippets")
        for stype, by_id in grouped_sources.items():
            if not by_id:
                continue
            st.markdown(f"**Source type: {stype}**")
            for sid, info in by_id.items():
                with st.expander(f"{stype.upper()} {sid} (score ~ {info.get('max_score', 0.0):.3f})"):
                    for idx, ch in enumerate(info.get("chunks", [])):
                        snippet = ch.get("text", "")
                        if len(snippet) > 1000:
                            snippet = snippet[:1000] + "..."
                        st.markdown(f"**Chunk {idx + 1}:**")
                        st.write(textwrap.fill(snippet, 100))


if __name__ == "__main__":
    main()

