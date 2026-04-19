"""
Comprehensive metric computation for the BNS-Only RAG Comparison.

Five dimensions:
  1. Retrieval   - hit rate, MRR (based on retrieved chunks vs gold sections)
  2. Accuracy    - section precision/recall/F1, correct act cited
  3. Hallucination - IPC references, fabricated sections, grounding score
  4. Speed       - rephrase/retrieval/generation/total latency
  5. Answer Quality - offense hit, completeness, relevance, safety disclaimer
"""
import re
from typing import Dict, List, Set, Any

# All valid BNS_2023 section numbers (1-358, not all exist but used for fabrication check)
# We load the actual set from the test case gold + known BNS range
_BNS_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)
_IPC_MARKERS = re.compile(
    r"\b(?:ipc|indian penal code|crpc|code of criminal procedure|iea|indian evidence act)\b",
    re.IGNORECASE,
)
_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with",
    "is", "are", "was", "were", "be", "as", "that", "this", "it", "its", "at",
    "from", "about", "into", "than", "then", "so", "if", "but", "also", "such",
    "can", "could", "should", "would", "may", "might", "will", "shall", "have",
    "has", "had", "not", "no", "do", "does", "did", "their", "there", "which",
    "been", "being", "over", "under", "between", "within", "more", "most",
}

# BNS 2023 has sections 1-358 (with some gaps). We treat any number in this range as valid.
_BNS_MAX_SECTION = 358


def _extract_sections(text: str) -> Set[str]:
    """Extract normalized section numbers from text (e.g. 'Section 303' -> '303')."""
    return {m.upper() for m in _BNS_SECTION_RE.findall(text)}


def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenization with stopword removal."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {t for t in text.split() if t and t not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _overlap_ratio(source: Set[str], target: Set[str]) -> float:
    """Fraction of source words present in target."""
    if not source:
        return 0.0
    return len(source & target) / len(source)


_SOURCE_ID_SECTION_RE = re.compile(r"_s(\d+[A-Za-z]?)$")


def _section_num_from_source_id(sid: str) -> str:
    """Extract section number from source_id like 'BNS_2023_s303' -> '303'."""
    m = _SOURCE_ID_SECTION_RE.search(sid or "")
    return m.group(1).upper() if m else ""


def _is_valid_bns_section(num_str: str) -> bool:
    """Return True if the section number is plausibly within BNS range (1-358)."""
    try:
        n = int(re.sub(r"[A-Za-z]", "", num_str))
        return 1 <= n <= _BNS_MAX_SECTION
    except ValueError:
        return False


def compute_metrics(case: Dict, result: Dict) -> Dict[str, Any]:
    """
    Compute all metrics for a single (case, result) pair.

    Args:
        case: test case dict with expected_bns_sections, offense_keywords, key_issues
        result: adapter output dict with answer, context_text, cited_sections, timings

    Returns:
        Flat dict of all metric values.
    """
    answer: str = result.get("answer", "")
    context: str = result.get("context_text", "")
    timings: Dict = result.get("timings", {})
    cited_sections: List[str] = result.get("cited_sections", [])

    gold_sections: Set[str] = {s.upper() for s in case.get("expected_bns_sections", [])}
    cited_set: Set[str] = set(cited_sections)

    # --- 1. Accuracy ---
    if cited_set and gold_sections:
        section_precision = len(cited_set & gold_sections) / len(cited_set)
        section_recall = len(cited_set & gold_sections) / len(gold_sections)
    elif not cited_set:
        section_precision = 0.0
        section_recall = 0.0
    else:
        section_precision = 0.0
        section_recall = 0.0

    if section_precision + section_recall > 0:
        section_f1 = 2 * section_precision * section_recall / (section_precision + section_recall)
    else:
        section_f1 = 0.0

    # --- 1b. Retrieval metrics (Hit Rate, MRR) ---
    retrieved_chunks: List[Dict] = result.get("retrieved_chunks", [])
    retrieved_section_nums = []
    for chunk in retrieved_chunks:
        snum = _section_num_from_source_id(chunk.get("source_id", ""))
        if snum:
            retrieved_section_nums.append(snum)

    # Hit Rate: 1 if ANY gold section appears in retrieved chunks
    hit_rate = int(bool(gold_sections & set(retrieved_section_nums)))

    # MRR: 1/rank of the first gold section found in retrieval order
    mrr = 0.0
    for rank, snum in enumerate(retrieved_section_nums, start=1):
        if snum in gold_sections:
            mrr = 1.0 / rank
            break

    # correct_act_cited: answer mentions BNS but NOT IPC/CrPC/IEA
    answer_lower = answer.lower()
    mentions_bns = bool(re.search(r"\bbns\b|\bbharatiya nyaya sanhita\b", answer_lower))
    mentions_ipc = bool(_IPC_MARKERS.search(answer))
    correct_act_cited = int(mentions_bns and not mentions_ipc)

    # --- 2. Hallucination ---
    ipc_reference_count = len(_IPC_MARKERS.findall(answer))

    # Fabricated sections: cited but not in BNS range (1-358)
    fabricated = {s for s in cited_set if not _is_valid_bns_section(s)}
    fabricated_section_count = len(fabricated)

    # Grounding: fraction of cited sections that appear in retrieved context
    context_sections = _extract_sections(context)
    if cited_set:
        grounding_score = len(cited_set & context_sections) / len(cited_set)
    else:
        grounding_score = 1.0  # no citations = no hallucination w.r.t. this metric
    hallucination_flag = int(grounding_score < 0.5)

    # --- 3. Speed ---
    rephrase_latency_sec = float(timings.get("rephrase_sec", 0.0))
    retrieval_latency_sec = float(timings.get("retrieval_sec", 0.0))
    generation_latency_sec = float(timings.get("generation_sec", 0.0))
    total_latency_sec = float(timings.get("total_sec", 0.0))

    # --- 4. Answer Quality ---
    offense_keywords = case.get("offense_keywords", [])
    key_issues = case.get("key_issues", [])
    description = case.get("description", "")

    # offense_category_hit: 1 if answer mentions at least one offense keyword
    offense_category_hit = int(
        any(kw.lower() in answer_lower for kw in offense_keywords)
    )

    # offense_keyword_coverage: fraction of offense keywords covered
    answer_words = _tokenize(answer)
    if offense_keywords:
        covered = sum(
            1 for kw in offense_keywords
            if _tokenize(kw) & answer_words
        )
        offense_keyword_coverage = covered / len(offense_keywords)
    else:
        offense_keyword_coverage = 0.0

    # completeness_score: fraction of key_issue content words in answer
    if key_issues:
        key_words = _tokenize(" ".join(key_issues))
        completeness_score = _overlap_ratio(key_words, answer_words)
    else:
        completeness_score = 0.0

    # key_issue_coverage: fraction of individual key issues at least partially covered
    if key_issues:
        covered_issues = sum(
            1 for issue in key_issues
            if _tokenize(issue) & answer_words
        )
        key_issue_coverage = covered_issues / len(key_issues)
    else:
        key_issue_coverage = 0.0

    # answer_relevance_score: Jaccard between description and answer
    desc_words = _tokenize(description)
    answer_relevance_score = _jaccard(desc_words, answer_words)

    # context_relevance_score: Jaccard between answer and context
    context_words = _tokenize(context)
    context_relevance_score = _jaccard(answer_words, context_words)

    # answer_length_words
    answer_length_words = len(answer.split())

    # has_safety_disclaimer
    safety_phrases = [
        "consult a lawyer", "consult an advocate", "consult a legal",
        "legal professional", "legal expert", "qualified lawyer", "human lawyer",
    ]
    has_safety_disclaimer = int(any(p in answer_lower for p in safety_phrases))

    return {
        # Retrieval
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        # Accuracy
        "section_precision": round(section_precision, 4),
        "section_recall": round(section_recall, 4),
        "section_f1": round(section_f1, 4),
        "correct_act_cited": correct_act_cited,
        # Hallucination
        "ipc_reference_count": ipc_reference_count,
        "fabricated_section_count": fabricated_section_count,
        "grounding_score": round(grounding_score, 4),
        "hallucination_flag": hallucination_flag,
        # Speed
        "rephrase_latency_sec": round(rephrase_latency_sec, 3),
        "retrieval_latency_sec": round(retrieval_latency_sec, 3),
        "generation_latency_sec": round(generation_latency_sec, 3),
        "total_latency_sec": round(total_latency_sec, 3),
        # Answer Quality
        "offense_category_hit": offense_category_hit,
        "offense_keyword_coverage": round(offense_keyword_coverage, 4),
        "completeness_score": round(completeness_score, 4),
        "key_issue_coverage": round(key_issue_coverage, 4),
        "answer_relevance_score": round(answer_relevance_score, 4),
        "context_relevance_score": round(context_relevance_score, 4),
        "answer_length_words": answer_length_words,
        "has_safety_disclaimer": has_safety_disclaimer,
    }


def summarize_metrics(rows: List[Dict]) -> Dict[str, float]:
    """
    Compute per-system averages across all test cases.
    rows: list of metric dicts (all from the same system).
    """
    if not rows:
        return {}
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summary = {}
    for k in numeric_keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        summary[f"avg_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return summary
