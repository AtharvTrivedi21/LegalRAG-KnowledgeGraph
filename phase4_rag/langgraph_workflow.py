from __future__ import annotations

"""
LangGraph workflow for Phase 4: Graph-Constrained LegalRAG.

Nodes:
- query_parser     -> parse explicit section/article references
- graph_retriever  -> resolve references via Neo4j, build constraints
- vector_retriever -> FAISS semantic search with optional graph constraints
- answer_generator -> construct prompt and call Ollama to generate answer
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .config import settings
from .query_parser import parse_query, ParsedQuery
from .neo4j_client import (
    get_articles_by_numbers,
    get_sections_by_numbers,
    get_cases_citing_ids,
    Neo4jUnavailableError,
)
from .vector_retriever import (
    GraphConstraints,
    retrieve_chunks,
    group_by_source,
    vector_store_status,
)
from .llm_ollama import ChatMessage, chat_completion, OllamaError


class WorkflowState(TypedDict, total=False):
    user_query: str
    parsed_query: Dict[str, Any]
    graph_constraints: Optional[Dict[str, List[str]]]
    graph_metadata: Optional[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    grouped_sources: Dict[str, Dict[str, Any]]
    used_fallback_unconstrained: bool
    graph_error: Optional[str]
    vector_error: Optional[str]
    answer: Optional[str]


def _parsed_query_to_dict(pq: ParsedQuery) -> Dict[str, Any]:
    return {
        "raw_query": pq.raw_query,
        "article_numbers": pq.article_numbers,
        "section_numbers": pq.section_numbers,
        "explicit_ids": pq.explicit_ids,
        "has_explicit_refs": pq.has_explicit_refs,
    }


def node_query_parser(state: WorkflowState) -> WorkflowState:
    query = state.get("user_query", "") or ""
    parsed = parse_query(query)
    state["parsed_query"] = _parsed_query_to_dict(parsed)
    return state


def node_graph_retriever(state: WorkflowState) -> WorkflowState:
    parsed = state.get("parsed_query") or {}
    section_numbers = parsed.get("section_numbers") or []
    article_numbers = parsed.get("article_numbers") or []

    graph_metadata: Dict[str, Any] = {
        "sections": [],
        "articles": [],
        "cases": [],
    }

    try:
        sections = get_sections_by_numbers(section_numbers)
        articles = get_articles_by_numbers(article_numbers)
        graph_metadata["sections"] = sections
        graph_metadata["articles"] = articles

        # Collect IDs for constraint building
        section_ids = [s["section_id"] for s in sections]
        article_ids = [a["article_id"] for a in articles]

        cases = get_cases_citing_ids([*section_ids, *article_ids])
        graph_metadata["cases"] = cases
        case_ids = [c["case_id"] for c in cases]

        constraints = GraphConstraints(
            allowed_case_ids=case_ids,
            allowed_section_ids=section_ids,
            allowed_article_ids=article_ids,
        )

        if constraints.is_empty:
            state["graph_constraints"] = None
        else:
            state["graph_constraints"] = asdict(constraints)

        state["graph_metadata"] = graph_metadata
        state["graph_error"] = None
        return state
    except Neo4jUnavailableError as exc:
        # Degrade gracefully – no graph constraints, but record the issue.
        state["graph_constraints"] = None
        state["graph_metadata"] = None
        state["graph_error"] = f"neo4j_unavailable: {exc}"
        return state


def node_vector_retriever(state: WorkflowState) -> WorkflowState:
    query = state.get("user_query", "") or ""
    constraints_dict = state.get("graph_constraints")

    constraints: Optional[GraphConstraints] = None
    if constraints_dict:
        constraints = GraphConstraints(
            allowed_case_ids=constraints_dict.get("allowed_case_ids", []),
            allowed_section_ids=constraints_dict.get("allowed_section_ids", []),
            allowed_article_ids=constraints_dict.get("allowed_article_ids", []),
        )

    result = retrieve_chunks(query, k=settings.retrieval.top_k, constraints=constraints)

    error = result.get("error")
    if error:
        state["retrieved_chunks"] = []
        state["grouped_sources"] = {}
        state["vector_error"] = error
        state["used_fallback_unconstrained"] = bool(result.get("used_fallback_unconstrained"))
        return state

    chunks = result.get("chunks", [])
    state["retrieved_chunks"] = chunks
    state["grouped_sources"] = group_by_source(chunks)
    state["vector_error"] = None
    state["used_fallback_unconstrained"] = bool(result.get("used_fallback_unconstrained"))
    return state


def _build_system_prompt() -> str:
    return (
        "You are a legal research assistant for Indian law. "
        "Answer the user's question using ONLY the provided context (cases, sections, constitutional articles). "
        "Cite the relevant section_id, article_id, or case_id in your answer. "
        "Only say you cannot answer if the context is clearly irrelevant or empty; otherwise give a clear, factual answer. "
        "Do not fabricate citations or legal provisions not present in the context.\n"
    )


def _build_graph_context_block(graph_metadata: Optional[Dict[str, Any]], max_chars_per_item: int = 2000) -> str:
    """
    Build context from Neo4j article/section full_text when the user asked for a
    specific article/section. This ensures the model sees the actual law text even
    if FAISS had no chunk for that article.
    """
    if not graph_metadata:
        return ""
    lines: List[str] = []
    for art in graph_metadata.get("articles") or []:
        aid = art.get("article_id", "?")
        text = (art.get("full_text") or "").strip()
        if text:
            if len(text) > max_chars_per_item:
                text = text[:max_chars_per_item] + "..."
            lines.append(f"[ARTICLE FROM KNOWLEDGE GRAPH] {aid}")
            lines.append(text)
            lines.append("")
    for sec in graph_metadata.get("sections") or []:
        sid = sec.get("section_id", "?")
        text = (sec.get("full_text") or "").strip()
        if text:
            if len(text) > max_chars_per_item:
                text = text[:max_chars_per_item] + "..."
            lines.append(f"[SECTION FROM KNOWLEDGE GRAPH] {sid}")
            lines.append(text)
            lines.append("")
    return "\n".join(lines).strip() if lines else ""


def _build_context_block(grouped_sources: Dict[str, Dict[str, Any]], max_snippets: int = 10) -> str:
    lines: List[str] = []
    count = 0
    for stype, by_id in grouped_sources.items():
        for sid, info in by_id.items():
            if count >= max_snippets:
                break
            header = f"[{stype.upper()}] {sid} (max_score={info.get('max_score', 0.0):.3f})"
            lines.append(header)
            for ch in info.get("chunks", [])[:1]:
                text = ch.get("text", "")
                if len(text) > 600:
                    text = text[:600] + "..."
                lines.append(text)
                lines.append("")  # blank line
                count += 1
                if count >= max_snippets:
                    break
        if count >= max_snippets:
            break
    return "\n".join(lines) if lines else "No legal context was retrieved."


def node_answer_generator(state: WorkflowState) -> WorkflowState:
    user_query = state.get("user_query", "") or ""
    grouped_sources = state.get("grouped_sources") or {}
    graph_metadata = state.get("graph_metadata")

    # Handle hard failures earlier in the pipeline.
    if state.get("vector_error"):
        state["answer"] = (
            "Vector retrieval is not available right now:\n"
            f"{state['vector_error']}\n"
            "Please ensure Phase 3 artifacts are built and try again."
        )
        return state

    system_prompt = _build_system_prompt()
    graph_context = _build_graph_context_block(graph_metadata)
    retrieval_context = _build_context_block(grouped_sources)
    context_block = (
        (graph_context + "\n\n" + retrieval_context).strip()
        if graph_context
        else retrieval_context
    )

    composed_user = (
        "User question:\n"
        f"{user_query}\n\n"
        "Relevant legal context (cases, sections, articles):\n"
        f"{context_block}\n\n"
        "Using ONLY the context above, provide a concise, factual answer. "
        "Cite the section_id, article_id, or case_id you relied on."
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=composed_user),
    ]

    try:
        answer = chat_completion(messages)
        state["answer"] = answer
        return state
    except OllamaError as exc:
        state["answer"] = (
            "The local LLM (Ollama) is not available or returned an error:\n"
            f"{exc}\n"
            "Please ensure Ollama is running and the model is pulled "
            f"(e.g. `ollama pull {settings.ollama.model}`)."
        )
        return state


def build_app():
    """
    Build and return the LangGraph app for Phase 4.
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("query_parser", node_query_parser)
    graph.add_node("graph_retriever", node_graph_retriever)
    graph.add_node("vector_retriever", node_vector_retriever)
    graph.add_node("answer_generator", node_answer_generator)

    graph.set_entry_point("query_parser")
    graph.add_edge("query_parser", "graph_retriever")
    graph.add_edge("graph_retriever", "vector_retriever")
    graph.add_edge("vector_retriever", "answer_generator")
    graph.add_edge("answer_generator", END)

    return graph.compile()


if __name__ == "__main__":
    # Simple CLI smoke test:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m phase4_rag.langgraph_workflow \"<query>\"")
        sys.exit(1)

    query = sys.argv[1]
    app = build_app()
    final_state = app.invoke({"user_query": query})

    print("Answer:\n")
    print(final_state.get("answer", ""))
    print("\n---\n")
    print("Retrieved chunks:", len(final_state.get("retrieved_chunks", [])))
    print("Graph error:", final_state.get("graph_error"))
    print("Vector error:", final_state.get("vector_error"))

