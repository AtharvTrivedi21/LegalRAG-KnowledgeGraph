"""
Phase 4 V3: Streamlit UI with Act-aware disambiguation, Act classification, and confidence guard.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List

import streamlit as st

from phase4_rag.config_v3 import settings
from phase4_rag.langgraph_workflow_v3 import build_app
from phase4_rag.neo4j_client_v3 import Neo4jUnavailableError, get_sections_by_numbers
from phase4_rag.neo4j_display_v2 import get_acts_by_ids, get_case_details
from phase4_rag.vector_retriever_v3 import vector_store_status


@st.cache_resource(show_spinner=False)
def get_app():
    return build_app()


def _neo4j_status() -> str:
    try:
        _ = get_sections_by_numbers([])
        return "connected"
    except Neo4jUnavailableError:
        return "unavailable"
    except Exception:
        return "error"


def main() -> None:
    st.set_page_config(page_title="Phase 4 V3 – Graph-Constrained LegalRAG", layout="wide")

    st.title("Phase 4 V3 – Graph-Constrained LegalRAG")
    st.write(
        "Act-aware disambiguation, explicit Act classification, and confidence guard. "
        "Local RAG over Neo4j, FAISS, and Ollama."
    )

    with st.sidebar:
        st.header("Status")
        neo4j_status = _neo4j_status()
        if neo4j_status == "connected":
            st.success(f"Neo4j: connected ({settings.neo4j.uri})")
        elif neo4j_status == "unavailable":
            st.warning("Neo4j: unavailable (will fall back to unconstrained retrieval)")
        else:
            st.error("Neo4j: error (check connection and credentials)")

        vs_status = vector_store_status()
        if vs_status["ok"]:
            st.success(f"FAISS index: loaded ({vs_status['index_size']} chunks)")
        else:
            st.error(f"FAISS index: not available\n\n{vs_status['error']}")

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

    st.subheader("Query")
    query = st.text_area(
        "Enter a legal question or describe an incident (e.g. 'Section 302 of BNS' or 'Article 14 Constitution')",
        height=120,
    )

    if st.button("Run Graph-Constrained RAG (V3)", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Running LangGraph pipeline (V3)..."):
            app = get_app()
            initial_state = {"user_query": query, "top_k": top_k}
            state = app.invoke(initial_state)

        legal_query = state.get("legal_query")
        if legal_query and legal_query.strip() != (query or "").strip():
            with st.expander("Formal legal query (used for retrieval)"):
                st.caption("User input was rephrased into this query for vector search.")
                st.write(legal_query)

        # Applicable Act(s) — prominent display
        applicable_acts = state.get("applicable_acts") or []
        if applicable_acts:
            act_labels = [
                f"{a.get('act_name', '') or a.get('act_id', '')} ({a.get('act_id', '')})"
                for a in applicable_acts
                if a.get("act_id")
            ]
            if act_labels:
                st.subheader("Applicable Act(s)")
                for label in act_labels:
                    st.write(f"- **{label}**")
                st.divider()

        st.subheader("Answer")
        answer = state.get("answer") or "No answer generated."
        st.markdown(answer)

        graph_metadata = state.get("graph_metadata") or {}
        grouped_sources = state.get("grouped_sources") or {}
        no_applicable_laws = state.get("answer_no_applicable_laws") or False
        no_relevant_cases = state.get("answer_no_relevant_cases") or False

        with st.expander("Status (graph, vector, fallback)"):
            graph_error = state.get("graph_error")
            vector_error = state.get("vector_error")
            used_fallback = state.get("used_fallback_unconstrained")
            if graph_error:
                st.warning("Graph constraints: unavailable (Neo4j issue).")
            else:
                st.success("Graph constraints: applied when available.")
            if vector_error:
                st.error("Vector retrieval: error (see answer for details).")
            else:
                st.success("Vector retrieval: OK.")
            if used_fallback:
                st.info("Fallback: unconstrained retrieval used (no matching constrained hits).")
            else:
                st.write("Fallback: not used.")

        act_ids = []
        for sec in graph_metadata.get("sections") or []:
            aid = sec.get("act_id")
            if aid:
                act_ids.append(aid)
        for art in graph_metadata.get("articles") or []:
            aid = art.get("act_id")
            if aid:
                act_ids.append(aid)
        act_ids = list(dict.fromkeys(act_ids))
        acts_data = get_acts_by_ids(act_ids) if act_ids else []

        if not no_applicable_laws:
            with st.expander("Cited references — Acts"):
                if not act_ids:
                    st.caption("No acts cited.")
                elif not acts_data:
                    st.caption("Act IDs from context could not be resolved in the graph.")
                    for aid in act_ids:
                        st.write(f"- {aid}")
                else:
                    for a in acts_data:
                        act_name = a.get("act_name") or ""
                        act_id = a.get("act_id", "")
                        act_display = f"{act_name} ({act_id})" if act_name else act_id or "(no name)"
                        st.write(f"- **Applicable Act:** {act_display}")

        articles_seen: Dict[str, Dict[str, Any]] = {}
        for art in graph_metadata.get("articles") or []:
            aid = art.get("article_id")
            if aid and aid not in articles_seen:
                text = (art.get("full_text") or "")[:200]
                if len((art.get("full_text") or "")) > 200:
                    text += "..."
                articles_seen[aid] = {"article_number": art.get("article_number"), "act_id": art.get("act_id"), "act_name": art.get("act_name"), "snippet": text}
        for aid, info in (grouped_sources.get("article") or {}).items():
            if aid not in articles_seen:
                chunks = info.get("chunks", [])
                snippet = (chunks[0].get("text", "")[:200] + "...") if chunks and len(chunks[0].get("text", "")) > 200 else (chunks[0].get("text", "") if chunks else "")
                articles_seen[aid] = {"article_number": "-", "act_id": "-", "snippet": snippet}

        if not no_applicable_laws:
            with st.expander("Cited references — Articles"):
                if not articles_seen:
                    st.caption("No articles cited.")
                else:
                    for aid, info in articles_seen.items():
                        num = info.get("article_number") or "-"
                        act_id = info.get("act_id") or "-"
                        act_name = info.get("act_name") or ""
                        act_display = f"{act_name} ({act_id})" if act_name and act_id != "-" else act_id
                        st.write(f"- **{aid}** (Art. {num}, Act: {act_display})")
                        if info.get("snippet"):
                            st.caption(info["snippet"][:200] + ("..." if len(info.get("snippet", "")) > 200 else ""))

        sections_seen: Dict[str, Dict[str, Any]] = {}
        for sec in graph_metadata.get("sections") or []:
            sid = sec.get("section_id")
            if sid and sid not in sections_seen:
                text = (sec.get("full_text") or "")[:200]
                if len((sec.get("full_text") or "")) > 200:
                    text += "..."
                sections_seen[sid] = {"section_number": sec.get("section_number"), "act_id": sec.get("act_id"), "act_name": sec.get("act_name"), "snippet": text}
        for sid, info in (grouped_sources.get("section") or {}).items():
            if sid not in sections_seen:
                chunks = info.get("chunks", [])
                snippet = (chunks[0].get("text", "")[:200] + "...") if chunks and len(chunks[0].get("text", "")) > 200 else (chunks[0].get("text", "") if chunks else "")
                sections_seen[sid] = {"section_number": "-", "act_id": "-", "snippet": snippet}

        if not no_applicable_laws:
            with st.expander("Cited references — Sections"):
                if not sections_seen:
                    st.caption("No sections cited.")
                else:
                    for sid, info in sections_seen.items():
                        num = info.get("section_number") or "-"
                        act_id = info.get("act_id") or "-"
                        act_name = info.get("act_name") or ""
                        act_display = f"{act_name} ({act_id})" if act_name and act_id != "-" else act_id
                        st.write(f"- **{sid}** (Sec. {num}, Act: {act_display})")
                        if info.get("snippet"):
                            st.caption(info["snippet"][:200] + ("..." if len(info.get("snippet", "")) > 200 else ""))

        case_ids_from_graph = [c.get("case_id") for c in (graph_metadata.get("cases") or []) if c.get("case_id")]
        case_ids_from_retrieval = list((grouped_sources.get("case") or {}).keys())
        all_case_ids = list(dict.fromkeys(case_ids_from_graph + case_ids_from_retrieval))[:5]
        case_details = get_case_details(all_case_ids, snippet_len=300) if all_case_ids else []
        case_map = {c["case_id"]: c for c in case_details}
        _max_cases = 5

        if not no_relevant_cases:
            with st.expander("Cited references — Cases"):
                if not case_ids_from_graph and not case_ids_from_retrieval:
                    st.caption("No cases cited.")
                else:
                    shown = 0
                    for c in graph_metadata.get("cases") or []:
                        if shown >= _max_cases:
                            break
                        cid = c.get("case_id")
                        if cid:
                            det = case_map.get(cid, {})
                            year = det.get("year") if det.get("year") is not None else c.get("year")
                            year_str = f" (year: {year})" if year and year != 0 else ""
                            st.write(f"- **{cid}**{year_str}")
                            if det.get("snippet"):
                                st.caption(det["snippet"][:250] + ("..." if len(det["snippet"]) > 250 else ""))
                            shown += 1
                    for cid in case_ids_from_retrieval:
                        if shown >= _max_cases:
                            break
                        if cid not in [c.get("case_id") for c in (graph_metadata.get("cases") or [])]:
                            det = case_map.get(cid, {})
                            y = det.get("year")
                            year_str = f" (year: {y})" if y is not None and y != 0 else ""
                            st.write(f"- **{cid}**{year_str}")
                            if det.get("snippet"):
                                st.caption(det["snippet"][:250] + ("..." if len(det["snippet"]) > 250 else ""))
                            shown += 1

        with st.expander("Retrieved text snippets"):
            for stype, by_id in grouped_sources.items():
                if not by_id:
                    continue
                st.markdown(f"**{stype.upper()}**")
                for sid, info in by_id.items():
                    with st.expander(sid):
                        for idx, ch in enumerate(info.get("chunks", [])):
                            snippet = ch.get("text", "")
                            if len(snippet) > 1000:
                                snippet = snippet[:1000] + "..."
                            st.markdown(f"**Chunk {idx + 1}:**")
                            st.write(textwrap.fill(snippet, 100))


if __name__ == "__main__":
    main()
