from dotenv import load_dotenv
load_dotenv()

import time
import re
from typing import List, Dict, Set

from rag_chain import answer_query
from config import ACTIVE_CONFIG


TEST_CASES: List[Dict] = [
    {"id": 1, "description": "Someone stole my mobile phone from my pocket in a crowded bus."},
    {"id": 2, "description": "A man entered my house at night by breaking the door and took my jewelry."},
    {"id": 3, "description": "My neighbor keeps threatening to beat me if I use the common parking area."},
    {"id": 4, "description": "Two people started fighting on the street and one hit the other with a stick causing injuries."},
    {"id": 5, "description": "A person sent me abusive and vulgar messages repeatedly on WhatsApp."},
    {"id": 6, "description": "Someone forged my signature on a cheque and withdrew money from my account."},
    {"id": 7, "description": "My landlord locked me out of my rented room without any notice and kept my belongings inside."},
    {"id": 8, "description": "A group of people damaged my shop during a protest by throwing stones and breaking the glass."},
    {"id": 9, "description": "A man followed me continuously on my way home and tried to touch me inappropriately."},
    {"id": 10, "description": "Someone hacked my social media account and posted offensive content using my name."},
]

# --- Simple helpers for text processing --- #

SECTION_REGEX = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)

STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with",
    "is", "are", "was", "were", "be", "as", "that", "this", "it", "its", "at",
    "from", "about", "into", "than", "then", "so", "if", "but", "also", "such",
    "can", "could", "should", "would", "may", "might", "will", "shall", "have",
    "has", "had", "not", "no", "do", "does", "did", "their", "there", "which",
    "been", "being", "over", "under", "between", "within", "more", "most",
}


def tokenize_content(text: str) -> Set[str]:
    """Very simple tokenization + stopword removal."""
    text = text.lower()
    # keep letters, digits and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return set(tokens)


def extract_sections(text: str) -> Set[str]:
    """Extract normalized section identifiers from text (e.g., '118', '63A')."""
    sections = set()
    for match in SECTION_REGEX.findall(text):
        sections.add(match.upper())
    return sections


def compute_overlap_ratio(source_words: Set[str], target_words: Set[str]) -> float:
    """
    Fraction of source_words that are present in target_words.
    If source_words is empty, returns 0.0.
    """
    if not source_words:
        return 0.0
    return len(source_words & target_words) / len(source_words)


def has_safety_disclaimer(text: str) -> int:
    """
    Heuristic: check if answer recommends consulting a lawyer or legal expert.
    """
    lower = text.lower()
    phrases = [
        "consult a lawyer",
        "consult an advocate",
        "consult a legal",
        "legal professional",
        "legal expert",
        "qualified lawyer",
        "human lawyer",
    ]
    return int(any(p in lower for p in phrases))


# --- Metric computation --- #


def compute_metrics(case: Dict, result: Dict, total_latency_sec: float) -> Dict:
    """
    Compute 10 evaluation metrics for a single run.
    """
    answer_text: str = result["answer"]
    docs = result["docs"]
    timings = result.get("timings", {})

    retrieval_time = float(timings.get("retrieval_time_sec", 0.0))
    generation_time = float(timings.get("generation_time_sec", 0.0))

    # base texts
    context_text = "\n\n".join(d.page_content for d in docs)
    rephrased_query = result.get("rephrased_query", "")

    # lengths
    answer_len_words = len(answer_text.split())

    # sections
    context_sections = extract_sections(context_text)
    answer_sections = extract_sections(answer_text)

    context_section_count = len(context_sections)
    answer_section_count = len(answer_sections)

    grounded_section_fraction = 1.0
    if answer_section_count > 0:
        grounded_section_fraction = (
            len(answer_sections & context_sections) / answer_section_count
        )

    # word-level grounding & completeness
    context_words = tokenize_content(context_text)
    answer_words = tokenize_content(answer_text)
    query_words = tokenize_content(rephrased_query)

    context_overlap_ratio = compute_overlap_ratio(answer_words, context_words)
    query_answer_overlap_ratio = compute_overlap_ratio(query_words, answer_words)

    safety_flag = has_safety_disclaimer(answer_text)

    metrics = {
        # 1–3: latency / performance
        "total_latency_sec": float(total_latency_sec),
        "retrieval_time_sec": retrieval_time,
        "generation_time_sec": generation_time,

        # 4–5: verbosity & basic answer size
        "answer_len_words": answer_len_words,

        # 6–7: section usage & grounding
        "context_section_count": context_section_count,
        "answer_section_count": answer_section_count,
        "grounded_section_fraction": grounded_section_fraction,

        # 8–9: faithfulness & completeness proxies
        "context_overlap_ratio": context_overlap_ratio,
        "query_answer_overlap_ratio": query_answer_overlap_ratio,

        # 10: safety proxy
        "has_safety_disclaimer": safety_flag,
    }

    return metrics


# --- Single-case runner --- #


def run_single_case(case: Dict) -> Dict:
    """
    One evaluation run:
    - calls your full RAG pipeline (answer_query)
    - measures total latency
    - computes 10 metrics
    - returns everything locally (no LangSmith)
    """
    start_total = time.time()
    result = answer_query(case["description"])
    total_latency = time.time() - start_total

    metrics = compute_metrics(case, result, total_latency)

    return {
        "case": case,
        "result": result,
        "metrics": metrics,
        "config_name": ACTIVE_CONFIG.config_name,
    }


def main():
    print(f"[EVAL] Running {len(TEST_CASES)} test cases with config: {ACTIVE_CONFIG.config_name}")

    all_outputs = []

    for case in TEST_CASES:
        print(f"\n[EVAL] Test case {case['id']}: {case['description']}")
        out = run_single_case(case)
        all_outputs.append(out)
        m = out["metrics"]

        print(
            f"  total_latency={m['total_latency_sec']:.2f}s "
            f"(retrieval={m['retrieval_time_sec']:.2f}s, "
            f"generation={m['generation_time_sec']:.2f}s)"
        )
        print(
            f"  answer_len={m['answer_len_words']} words, "
            f"context_sections={m['context_section_count']}, "
            f"answer_sections={m['answer_section_count']}, "
            f"grounded_section_fraction={m['grounded_section_fraction']:.2f}"
        )
        print(
            f"  context_overlap={m['context_overlap_ratio']:.2f}, "
            f"query_answer_overlap={m['query_answer_overlap_ratio']:.2f}, "
            f"safety_disclaimer={bool(m['has_safety_disclaimer'])}"
        )

    print("\n[EVAL] Done. Metrics computed locally for all test cases.")


if __name__ == "__main__":
    main()