"""
Phase 4 V3: LangGraph workflow with Act-aware disambiguation, Act classification, and confidence guard.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .config_v3 import settings
from .query_parser_v3 import parse_query, ParsedQuery
from .neo4j_client_v3 import (
    get_articles_by_numbers,
    get_sections_by_numbers,
    get_sections_by_ids,
    get_articles_by_ids,
    get_cases_citing_ids,
    Neo4jUnavailableError,
)
from .vector_retriever_v3 import (
    GraphConstraints,
    retrieve_chunks,
    group_by_source,
)
from .llm_ollama import ChatMessage, chat_completion, OllamaError


class WorkflowState(TypedDict, total=False):
    user_query: str
    legal_query: Optional[str]
    parsed_query: Dict[str, Any]
    graph_constraints: Optional[Dict[str, List[str]]]
    graph_metadata: Optional[Dict[str, Any]]
    applicable_acts: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    grouped_sources: Dict[str, Dict[str, Any]]
    used_fallback_unconstrained: bool
    top_faiss_similarity: Optional[float]
    graph_error: Optional[str]
    vector_error: Optional[str]
    answer: Optional[str]
    answer_no_applicable_laws: bool
    answer_no_relevant_cases: bool
    explicit_constraints_from_query: bool
    derived_constraints_from_retrieval: bool


def _parsed_query_to_dict(pq: ParsedQuery) -> Dict[str, Any]:
    return {
        "raw_query": pq.raw_query,
        "article_numbers": pq.article_numbers,
        "section_numbers": pq.section_numbers,
        "explicit_ids": pq.explicit_ids,
        "has_explicit_refs": pq.has_explicit_refs,
        "section_act_id": getattr(pq, "section_act_id", None),
        "article_act_id": getattr(pq, "article_act_id", None),
    }


def _collect_applicable_acts(sections: List[Dict], articles: List[Dict]) -> List[Dict[str, Any]]:
    seen: Dict[str, str] = {}
    for s in sections:
        aid = s.get("act_id")
        aname = s.get("act_name") or aid
        if aid and aid not in seen:
            seen[aid] = aname
    for a in articles:
        aid = a.get("act_id")
        aname = a.get("act_name") or aid
        if aid and aid not in seen:
            seen[aid] = aname
    return [{"act_id": k, "act_name": v} for k, v in seen.items()]


def node_query_parser(state: WorkflowState) -> WorkflowState:
    query = state.get("user_query", "") or ""
    parsed = parse_query(query)
    state["parsed_query"] = _parsed_query_to_dict(parsed)
    return state


def node_graph_retriever(state: WorkflowState) -> WorkflowState:
    parsed = state.get("parsed_query") or {}
    section_numbers = parsed.get("section_numbers") or []
    article_numbers = parsed.get("article_numbers") or []
    section_act_id = parsed.get("section_act_id")
    article_act_id = parsed.get("article_act_id")

    graph_metadata: Dict[str, Any] = {"sections": [], "articles": [], "cases": []}

    try:
        sections = get_sections_by_numbers(section_numbers, act_id=section_act_id)
        articles = get_articles_by_numbers(article_numbers, act_id=article_act_id)
        graph_metadata["sections"] = sections
        graph_metadata["articles"] = articles

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
        state["explicit_constraints_from_query"] = bool(state.get("graph_constraints"))
        state["derived_constraints_from_retrieval"] = False

        state["graph_metadata"] = graph_metadata
        state["applicable_acts"] = _collect_applicable_acts(sections, articles)
        state["graph_error"] = None
        return state
    except Neo4jUnavailableError as exc:
        state["graph_constraints"] = None
        state["graph_metadata"] = None
        state["applicable_acts"] = []
        state["graph_error"] = f"neo4j_unavailable: {exc}"
        state["explicit_constraints_from_query"] = False
        state["derived_constraints_from_retrieval"] = False
        return state


REPHRASE_PROMPT = (
    "You are an expert in Indian criminal law. "
    "IMPORTANT: The Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and Indian Evidence Act (IEA) "
    "have been REPEALED and replaced by the following NEW laws effective 2024: "
    "Bharatiya Nyaya Sanhita 2023 (BNS) replaces IPC, "
    "Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS) replaces CrPC, "
    "Bharatiya Sakshya Adhiniyam 2023 (BSA) replaces IEA. "
    "Rewrite the user's informal question or incident description into a concise, formal legal query "
    "using the NEW codes (BNS/BNSS/BSA) and appropriate legal terminology. "
    "Do NOT reference IPC, CrPC, or IEA sections. Do not answer the question, only rewrite it. "
    "Keep any explicit references to Article or Section numbers unchanged.\n\n"
    "User input:\n{user_query}\n\nFormal legal query:"
)


def node_query_rephrase(state: WorkflowState) -> WorkflowState:
    user_query = (state.get("user_query") or "").strip()
    if not user_query:
        state["legal_query"] = ""
        return state
    try:
        messages = [ChatMessage(role="user", content=REPHRASE_PROMPT.format(user_query=user_query))]
        legal = chat_completion(messages).strip()
        state["legal_query"] = legal if legal else user_query
    except OllamaError:
        state["legal_query"] = user_query
    return state


def _enrich_graph_metadata_from_chunks(
    state: WorkflowState, grouped_sources: Dict[str, Dict[str, Any]]
) -> None:
    """
    For natural language queries (no explicit Section/Article refs), graph_metadata
    will have empty sections and articles lists. After FAISS retrieval, look up the
    retrieved section/article source_ids in Neo4j to populate graph_metadata.
    This ensures the citation panels in the UI are filled for ALL query types.
    """
    graph_metadata = state.get("graph_metadata") or {"sections": [], "articles": [], "cases": []}

    # Only enrich when graph retriever found nothing (natural language query)
    already_has_sections = bool(graph_metadata.get("sections"))
    already_has_articles = bool(graph_metadata.get("articles"))
    if already_has_sections and already_has_articles:
        return

    try:
        if not already_has_sections:
            section_ids = list((grouped_sources.get("section") or {}).keys())
            if section_ids:
                sections = get_sections_by_ids(section_ids)
                graph_metadata["sections"] = sections

        if not already_has_articles:
            article_ids = list((grouped_sources.get("article") or {}).keys())
            if article_ids:
                articles = get_articles_by_ids(article_ids)
                graph_metadata["articles"] = articles

        state["graph_metadata"] = graph_metadata

        # Also update applicable_acts from enriched metadata
        if not state.get("applicable_acts"):
            state["applicable_acts"] = _collect_applicable_acts(
                graph_metadata.get("sections", []),
                graph_metadata.get("articles", []),
            )
    except Neo4jUnavailableError:
        pass  # Non-fatal: citations won't show but answer still works


def node_vector_retriever(state: WorkflowState) -> WorkflowState:
    query = (state.get("legal_query") or state.get("user_query", "") or "").strip()
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
    state["top_faiss_similarity"] = result.get("top_faiss_similarity")
    if error:
        state["retrieved_chunks"] = []
        state["grouped_sources"] = {}
        state["vector_error"] = error
        state["used_fallback_unconstrained"] = bool(result.get("used_fallback_unconstrained"))
        return state

    chunks = result.get("chunks", [])
    grouped = group_by_source(chunks)
    state["retrieved_chunks"] = chunks
    state["grouped_sources"] = grouped
    state["vector_error"] = None
    state["used_fallback_unconstrained"] = bool(result.get("used_fallback_unconstrained"))

    # Enrich graph_metadata from FAISS results for natural language queries
    _enrich_graph_metadata_from_chunks(state, grouped)

    # If no explicit graph constraints from parser, derive constraints from top retrieved
    # section/article source_ids and run a second constrained retrieval pass.
    if not constraints:
        section_items = sorted(
            (grouped.get("section") or {}).items(),
            key=lambda kv: kv[1].get("max_score", 0.0),
            reverse=True,
        )
        article_items = sorted(
            (grouped.get("article") or {}).items(),
            key=lambda kv: kv[1].get("max_score", 0.0),
            reverse=True,
        )
        top_section_ids = [sid for sid, _ in section_items[:3]]
        top_article_ids = [aid for aid, _ in article_items[:1]]

        if top_section_ids or top_article_ids:
            derived_constraints = GraphConstraints(
                allowed_case_ids=[],
                allowed_section_ids=top_section_ids,
                allowed_article_ids=top_article_ids,
            )
            rerun = retrieve_chunks(query, k=settings.retrieval.top_k, constraints=derived_constraints)
            if not rerun.get("error") and rerun.get("chunks"):
                rerun_chunks = rerun.get("chunks", [])
                rerun_grouped = group_by_source(rerun_chunks)
                state["retrieved_chunks"] = rerun_chunks
                state["grouped_sources"] = rerun_grouped
                state["graph_constraints"] = asdict(derived_constraints)
                state["derived_constraints_from_retrieval"] = True
                state["used_fallback_unconstrained"] = bool(rerun.get("used_fallback_unconstrained"))
                state["top_faiss_similarity"] = rerun.get("top_faiss_similarity", state.get("top_faiss_similarity"))

                # Refresh graph metadata/cases using derived IDs so case panel reflects GraphRAG path.
                _enrich_graph_metadata_from_chunks(state, rerun_grouped)
                try:
                    derived_ids = top_section_ids + top_article_ids
                    cases = get_cases_citing_ids(derived_ids) if derived_ids else []
                    graph_metadata = state.get("graph_metadata") or {"sections": [], "articles": [], "cases": []}
                    graph_metadata["cases"] = cases
                    state["graph_metadata"] = graph_metadata
                except Neo4jUnavailableError:
                    pass

    return state


def _build_system_prompt_v3() -> str:
    return (
        "You are a legal research assistant for Indian law. "
        "Your knowledge base contains ONLY the following statutes: "
        "Bharatiya Nyaya Sanhita 2023 (BNS_2023), "
        "Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS_2023), "
        "Bharatiya Sakshya Adhiniyam 2023 (BSA_2023), and "
        "the Constitution of India (CONST_1950). "
        "The Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), and Indian Evidence Act (IEA) "
        "are REPEALED. Do NOT cite IPC, CrPC, or IEA sections — they no longer apply. "
        "Answer the user's question using ONLY the provided context (cases, sections, constitutional articles). "
        "Write in clear, professional legal language suitable for a report screenshot. "
        "Be concise and avoid repetition. "
        "Cite the relevant section_id, article_id, or case_id in your answer. "
        "Always include the parent Act name when citing sections or articles "
        "(e.g. 'Section 330, BNS_2023 (Bharatiya Nyaya Sanhita)' or 'Article 14, CONST_1950'). "
        "Under 'Applicable laws / provisions', list only the MOST RELEVANT laws for the incident. "
        "Prefer 3 to 6 law bullets; do not dump long irrelevant lists. "
        "Each bullet must include the law and one-line relevance. "
        "Do not include constitutional articles unless the user query is constitutional in nature or the context strongly requires it. "
        "For criminal incident queries, prefer substantive BNS provisions first; include BNSS/BSA only when directly needed. "
        "Do not say 'none stated' or 'none explicitly stated' if the context contains any SECTION or ARTICLE text"
        "—identify and cite the applicable ones. "
        "Only if the context truly contains no sections or articles may you state that no applicable laws were provided. "
        "Do not fabricate citations or legal provisions not present in the context. "
        "Structure your answer with these markdown headings on separate lines only: "
        "## Summary, ## Applicable laws / provisions, ## Relevant case law (if any), ## Recommendation / next steps. "
        "Under each heading, use short bullet points. "
        "In 'Applicable laws / provisions', each bullet must be one law line in this format: "
        "'- Section <number>, <ACT_ID> (<Act Name>): <one-line relevance>' or "
        "'- Article <number>, <ACT_ID> (<Act Name>): <one-line relevance>'. "
        "For criminal incident questions, prioritize BNS_2023 provisions first; include BNSS/BSA only when directly necessary. "
        "If many candidate laws are available, choose the ones that directly map to the incident facts (act, intent, harm, procedure). "
        "Do not include or repeat internal labels like [ARTICLE FROM KNOWLEDGE GRAPH] or [SECTION FROM KNOWLEDGE GRAPH] in your answer; "
        "cite sources by article_id, section_id, or case_id with their Act (e.g. BNS_2023_s330, CONST_1950_Art14).\n"
    )


def _format_act_display(act_id: Optional[str], act_name: Optional[str]) -> str:
    if act_id and act_name:
        return f"{act_name} ({act_id})"
    return act_id or act_name or ""


def _build_graph_context_block(graph_metadata: Optional[Dict[str, Any]], max_chars_per_item: int = 2000) -> str:
    if not graph_metadata:
        return ""
    lines: List[str] = []
    for art in graph_metadata.get("articles") or []:
        aid = art.get("article_id", "?")
        act_display = _format_act_display(art.get("act_id"), art.get("act_name"))
        text = (art.get("full_text") or "").strip()
        if text:
            if len(text) > max_chars_per_item:
                text = text[:max_chars_per_item] + "..."
            lines.append(f"[ARTICLE FROM KNOWLEDGE GRAPH] {aid}")
            if act_display:
                lines.append(f"Applicable Act: {act_display}")
            lines.append(text)
            lines.append("")
    for sec in graph_metadata.get("sections") or []:
        sid = sec.get("section_id", "?")
        act_display = _format_act_display(sec.get("act_id"), sec.get("act_name"))
        text = (sec.get("full_text") or "").strip()
        if text:
            if len(text) > max_chars_per_item:
                text = text[:max_chars_per_item] + "..."
            lines.append(f"[SECTION FROM KNOWLEDGE GRAPH] {sid}")
            if act_display:
                lines.append(f"Applicable Act: {act_display}")
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
            # Include act_id in the header so the LLM knows which Act each chunk belongs to
            chunks = info.get("chunks", [])
            act_id = chunks[0].get("act_id", "") if chunks else ""
            act_suffix = f" Act: {act_id}" if act_id else ""
            header = f"[{stype.upper()}] {sid}{act_suffix} (score={info.get('max_score', 0.0):.3f})"
            lines.append(header)
            for ch in chunks[:1]:
                text = ch.get("text", "")
                if len(text) > 600:
                    text = text[:600] + "..."
                lines.append(text)
                lines.append("")
                count += 1
                if count >= max_snippets:
                    break
        if count >= max_snippets:
            break
    return "\n".join(lines) if lines else "No legal context was retrieved."


def _strip_internal_labels(text: str) -> str:
    # Remove internal tags but preserve newlines/markdown structure.
    text = text.replace("[ARTICLE FROM KNOWLEDGE GRAPH]", "").replace("[SECTION FROM KNOWLEDGE GRAPH]", "")
    # Normalize trailing spaces per line without flattening paragraph breaks.
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_markdown_layout(text: str) -> str:
    # Ensure headings exist on their own lines.
    heading_names = [
        "Summary",
        "Applicable laws / provisions",
        "Relevant case law (if any)",
        "Recommendation / next steps",
    ]
    for h in heading_names:
        text = re.sub(rf"(?i)\s*##\s*{re.escape(h)}\s*", f"\n## {h}\n", text)
        text = re.sub(rf"(?i)(^|\n)\s*{re.escape(h)}\s*[:\-]?\s*", f"\n## {h}\n", text)

    # If first heading is missing but answer starts with prose, prefix summary.
    if not re.search(r"(?m)^##\s+Summary\s*$", text):
        text = "## Summary\n" + text.strip()

    # Put law lines on bullets if model emitted inline separators.
    text = re.sub(r"\s+\-\s+Section\s+", r"\n- Section ", text)
    text = re.sub(r"\s+\-\s+Article\s+", r"\n- Article ", text)

    # Keep markdown compact but readable.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _answer_indicates_no_relevant(answer: str) -> tuple[bool, bool]:
    import re as _re
    parts = _re.split(r"##\s+", answer, flags=_re.IGNORECASE)
    laws_content = ""
    cases_content = ""
    for p in parts:
        p_lower = p.lower()
        if p_lower.startswith("applicable laws"):
            laws_content = p[:600]
        elif p_lower.startswith("relevant case law"):
            cases_content = p[:600]
    none_phrase = _re.compile(r"\b(none found|no relevant|do not relate|do not address|provided cases? do not)\b", _re.I)
    no_laws = bool(laws_content and none_phrase.search(laws_content))
    no_cases = bool(cases_content and none_phrase.search(cases_content))
    return (no_laws, no_cases)


def _remove_empty_sections(text: str) -> str:
    import re as _re
    blocks = _re.split(r"(##\s+[^\n#]+)", text)
    none_phrase = _re.compile(r"\b(none found|no relevant|do not relate|do not address|provided cases? do not)\b", _re.I)
    result = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if _re.match(r"##\s+", block):
            header = block
            content = blocks[i + 1] if i + 1 < len(blocks) else ""
            if none_phrase.search(content):
                i += 2
                continue
            result.append(header + content)
            i += 2
        else:
            result.append(block)
            i += 1
    out = "".join(result).strip()
    out = _re.sub(r"\n{3,}", "\n\n", out)
    return out


def node_answer_generator(state: WorkflowState) -> WorkflowState:
    grouped_sources = state.get("grouped_sources") or {}
    graph_metadata = state.get("graph_metadata")

    if state.get("vector_error"):
        state["answer"] = (
            "Vector retrieval is not available right now:\n"
            f"{state['vector_error']}\n"
            "Please ensure Phase 3 artifacts are built and try again."
        )
        return state

    user_query = state.get("user_query", "") or ""
    system_prompt = _build_system_prompt_v3()
    graph_context = _build_graph_context_block(graph_metadata)
    retrieval_context = _build_context_block(grouped_sources)
    context_block = (graph_context + "\n\n" + retrieval_context).strip() if graph_context else retrieval_context

    composed_user = (
        "User question:\n"
        f"{user_query}\n\n"
        "Relevant legal context (cases, sections, articles):\n"
        f"{context_block}\n\n"
        "Reply using ONLY the context above, in the structured format with the four headings. "
        "Cite the section_id, article_id, or case_id you relied on."
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=composed_user),
    ]

    try:
        answer = chat_completion(messages)
        answer = _strip_internal_labels(answer)
        no_laws, no_cases = _answer_indicates_no_relevant(answer)
        state["answer_no_applicable_laws"] = no_laws
        state["answer_no_relevant_cases"] = no_cases
        answer = _fix_markdown_layout(answer)
        answer = _remove_empty_sections(answer)
        state["answer"] = answer
        return state
    except OllamaError as exc:
        state["answer"] = (
            "The local LLM (Ollama) is not available or returned an error:\n"
            f"{exc}\n"
            f"Please ensure Ollama is running and the model is pulled (e.g. `ollama pull {settings.ollama.model}`)."
        )
        return state


def build_app():
    """Build and return the Phase 4 V3 LangGraph app."""
    graph = StateGraph(WorkflowState)

    graph.add_node("query_parser", node_query_parser)
    graph.add_node("graph_retriever", node_graph_retriever)
    graph.add_node("query_rephrase", node_query_rephrase)
    graph.add_node("vector_retriever", node_vector_retriever)
    graph.add_node("answer_generator", node_answer_generator)

    graph.set_entry_point("query_parser")
    graph.add_edge("query_parser", "graph_retriever")
    graph.add_edge("graph_retriever", "query_rephrase")
    graph.add_edge("query_rephrase", "vector_retriever")
    graph.add_edge("vector_retriever", "answer_generator")
    graph.add_edge("answer_generator", END)

    return graph.compile()
