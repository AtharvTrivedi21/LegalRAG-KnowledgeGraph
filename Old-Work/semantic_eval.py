from dotenv import load_dotenv
load_dotenv()

import time
import re
import csv
import os
from typing import List, Dict, Set

from rag_chain import answer_query
from new_config import ACTIVE_CONFIG

# === 1. Test cases (same as before, with updated #9) ===

TEST_CASES: List[Dict] = [
    {"id": 1, "description": "Someone stole my mobile phone from my pocket in a crowded bus."},
    {"id": 2, "description": "A man entered my house at night by breaking the door and took my jewelry."},
    {"id": 3, "description": "My neighbor keeps threatening to beat me if I use the common parking area."},
    {"id": 4, "description": "Two people started fighting on the street and one hit the other with a stick causing injuries."},
    {"id": 5, "description": "A person sent me abusive and vulgar messages repeatedly on WhatsApp."},
    {"id": 6, "description": "Someone forged my signature on a cheque and withdrew money from my account."},
    {"id": 7, "description": "My landlord locked me out of my rented room without any notice and kept my belongings inside."},
    {"id": 8, "description": "A group of people damaged my shop during a protest by throwing stones and breaking the glass."},
    {
        "id": 9,
        "description": "A stranger keeps calling and sending messages threatening to burn my shop if I do not pay him money."
    },
    {"id": 10, "description": "Someone hacked my social media account and posted offensive content using my name."},
]

# === 2. Gold annotations for the 5 semantic metrics ===
# These are heuristic "gold" labels for evaluation (not official legal advice).

EVAL_GOLD: Dict[int, Dict] = {
    1: {
        "offense_category": "theft",
        "offense_keywords": ["theft", "stealing", "stolen", "pickpocket", "pick-pocket"],
        "key_issues": [
            "movable property",
            "dishonest intention",
            "without consent",
            "taking away",
            "property of another person",
        ],
    },
    2: {
        "offense_category": "housebreaking and theft",
        "offense_keywords": [
            "housebreaking",
            "house-breaking",
            "trespass",
            "theft",
            "burglary",
            "robbery",
        ],
        "key_issues": [
            "entry into dwelling house",
            "at night",
            "without permission",
            "stealing property",
            "dishonest intention",
        ],
    },
    3: {
        "offense_category": "criminal intimidation",
        "offense_keywords": [
            "criminal intimidation",
            "threatening",
            "threats",
            "fear of injury",
            "intimidate",
        ],
        "key_issues": [
            "threats to cause harm",
            "create fear",
            "intent to force or prevent an act",
            "wrongful intimidation",
        ],
    },
    4: {
        "offense_category": "assault and causing hurt",
        "offense_keywords": [
            "assault",
            "hurt",
            "grievous hurt",
            "physical attack",
            "beating",
            "injury",
        ],
        "key_issues": [
            "physical assault",
            "use of weapon or stick",
            "causing bodily injury",
            "intention or knowledge to cause hurt",
        ],
    },
    5: {
        "offense_category": "harassment and obscene messages",
        "offense_keywords": [
            "harassment",
            "obscene messages",
            "abusive language",
            "insult",
            "defamation",
            "cyber bullying",
        ],
        "key_issues": [
            "repeated messages",
            "abusive or vulgar content",
            "intent to insult or annoy",
            "mental harassment",
        ],
    },
    6: {
        "offense_category": "forgery and cheque fraud",
        "offense_keywords": [
            "forgery",
            "fraud",
            "fake signature",
            "cheque",
            "cheque fraud",
            "dishonestly",
        ],
        "key_issues": [
            "forged signature",
            "cheque used without authority",
            "dishonest withdrawal of money",
            "false document",
        ],
    },
    7: {
        "offense_category": "unlawful eviction / wrongful confinement of property",
        "offense_keywords": [
            "illegal eviction",
            "unlawful eviction",
            "wrongful confinement",
            "locking out",
            "landlord dispute",
        ],
        "key_issues": [
            "landlord locked tenant out",
            "no notice",
            "personal belongings inside",
            "depriving access to property",
        ],
    },
    8: {
        "offense_category": "mischief and property damage",
        "offense_keywords": [
            "mischief",
            "damage to property",
            "vandalism",
            "rioting",
            "destruction",
        ],
        "key_issues": [
            "damage to shop",
            "throwing stones",
            "breaking glass",
            "intentional destruction of property",
        ],
    },
    9: {
        "offense_category": "extortion / criminal intimidation",
        "offense_keywords": [
            "extortion",
            "criminal intimidation",
            "threatening",
            "threat to burn",
            "demanding money",
        ],
        "key_issues": [
            "threat to burn shop",
            "demanding money",
            "creating fear of injury to property",
            "forcing payment",
        ],
    },
    10: {
        "offense_category": "cybercrime / hacking and defamation",
        "offense_keywords": [
            "hacking",
            "unauthorized access",
            "cybercrime",
            "defamation",
            "impersonation",
            "social media account",
        ],
        "key_issues": [
            "account accessed without permission",
            "posting offensive content",
            "damage to reputation",
            "misuse of identity",
        ],
    },
}

# === 3. Text helpers ===

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
    """Simple tokenization + stopword removal."""
    text = text.lower()
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
    """Fraction of source_words that are present in target_words."""
    if not source_words:
        return 0.0
    return len(source_words & target_words) / len(source_words)

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)

# === 4. Semantic metrics per case ===

def offense_category_hit(answer_text: str, gold: Dict) -> int:
    """1 if answer mentions at least one offense keyword, else 0."""
    lower = answer_text.lower()
    for kw in gold["offense_keywords"]:
        if kw.lower() in lower:
            return 1
    return 0

def retrieval_offense_hit(context_text: str, gold: Dict) -> int:
    """1 if retrieved context mentions at least one offense keyword, else 0."""
    lower = context_text.lower()
    for kw in gold["offense_keywords"]:
        if kw.lower() in lower:
            return 1
    return 0

def completeness_score(answer_text: str, gold: Dict) -> float:
    """
    0–1 score: fraction of key_issues whose content words
    appear in the answer.
    """
    key_issue_words = tokenize_content(" ".join(gold["key_issues"]))
    answer_words = tokenize_content(answer_text)
    return compute_overlap_ratio(key_issue_words, answer_words)

def answer_relevance_score(description: str, answer_text: str) -> float:
    """
    Jaccard similarity between content words in description and answer.
    Proxy for topical relevance.
    """
    desc_words = tokenize_content(description)
    answer_words = tokenize_content(answer_text)
    return jaccard_similarity(desc_words, answer_words)

def hallucination_grounding_score(answer_text: str, context_text: str) -> float:
    """
    0–1: fraction of sections cited in the answer that also appear in the context.
    1.0 = all cited sections are grounded in retrieved docs.
    0.0 = none are.
    """
    answer_sections = extract_sections(answer_text)
    context_sections = extract_sections(context_text)

    if not answer_sections:
        # no sections cited, we can't compute a grounding ratio
        return 1.0  # treat as fully grounded w.r.t. this metric
    grounded = len(answer_sections & context_sections)
    return grounded / len(answer_sections)


def hallucination_flag(answer_text: str, context_text: str, threshold: float = 0.5) -> int:
    """
    Binary hallucination flag derived from the grounding score.
    """
    grounding = hallucination_grounding_score(answer_text, context_text)
    # If no sections, grounding_score is 1.0, so this will return 0 (no hallucination)
    return int(grounding < threshold)


# === 5. Run one case and compute metrics ===

def run_single_case(case: Dict) -> Dict:
    """
    Run full RAG pipeline and compute 5 semantic metrics.
    """
    case_id = case["id"]
    description = case["description"]
    gold = EVAL_GOLD[case_id]

    start = time.time()
    result = answer_query(description)
    total_latency = time.time() - start

    answer_text: str = result["answer"]
    docs = result["docs"]

    context_text = "\n\n".join(d.page_content for d in docs)

        # 5 semantic metrics (+ new granular ones)
    m = {}
    m["offense_category_hit"] = offense_category_hit(answer_text, gold)
    m["retrieval_offense_hit"] = retrieval_offense_hit(context_text, gold)

    # NEW: richer offense metrics
    m["offense_keyword_coverage"] = offense_keyword_coverage(answer_text, gold)
    m["retrieval_offense_keyword_coverage"] = retrieval_offense_keyword_coverage(context_text, gold)

    # Existing completeness
    m["completeness_score"] = completeness_score(answer_text, gold)

    # NEW: issue-level completeness
    m["key_issue_coverage"] = key_issue_coverage(answer_text, gold)

    # Existing relevance: description vs answer
    m["answer_relevance_score"] = answer_relevance_score(description, answer_text)

    # NEW: context vs answer relevance
    m["context_relevance_score"] = context_relevance_score(answer_text, context_text)

    # NEW: hallucination grounding + existing flag
    m["hallucination_grounding_score"] = hallucination_grounding_score(answer_text, context_text)
    m["hallucination_flag"] = hallucination_flag(answer_text, context_text)

    # latency as before
    m["total_latency_sec"] = float(total_latency)


    # # 5 semantic metrics
    # m = {}
    # m["offense_category_hit"] = offense_category_hit(answer_text, gold)
    # m["retrieval_offense_hit"] = retrieval_offense_hit(context_text, gold)
    # m["completeness_score"] = completeness_score(answer_text, gold)
    # m["answer_relevance_score"] = answer_relevance_score(description, answer_text)
    # m["hallucination_flag"] = hallucination_flag(answer_text, context_text)

    # # We can also keep latency here just for reference
    # m["total_latency_sec"] = float(total_latency)

    return {
        "case": case,
        "metrics": m,
        "config_name": ACTIVE_CONFIG.config_name,
        "llm_model_name": ACTIVE_CONFIG.llm_model_name,
        "embedding_model_name": ACTIVE_CONFIG.embedding_model_name,
    }

# === 6. CSV saving ===

def save_semantic_results(outputs: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    fieldnames = [
        "config_name",
        "llm_model_name",
        "embedding_model_name",
        "case_id",
        "case_description",
        # semantic metrics:
        "offense_category_hit",
        "retrieval_offense_hit",
        "offense_keyword_coverage",              # NEW
        "retrieval_offense_keyword_coverage",    # NEW
        "completeness_score",
        "key_issue_coverage",                    # NEW
        "answer_relevance_score",
        "context_relevance_score",               # NEW
        "hallucination_flag",
        "hallucination_grounding_score",         # NEW
        # optional latency:
        "total_latency_sec",
    ]


    file_exists = os.path.exists(csv_path)
    write_header = not file_exists or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for out in outputs:
            case = out["case"]
            m = out["metrics"]

            row = {
                "config_name": out["config_name"],
                "llm_model_name": out["llm_model_name"],
                "embedding_model_name": out["embedding_model_name"],
                "case_id": case["id"],
                "case_description": case["description"],
                "offense_category_hit": m["offense_category_hit"],
                "retrieval_offense_hit": m["retrieval_offense_hit"],
                "offense_keyword_coverage": m["offense_keyword_coverage"],
                "retrieval_offense_keyword_coverage": m["retrieval_offense_keyword_coverage"],
                "completeness_score": m["completeness_score"],
                "key_issue_coverage": m["key_issue_coverage"],
                "answer_relevance_score": m["answer_relevance_score"],
                "context_relevance_score": m["context_relevance_score"],
                "hallucination_flag": m["hallucination_flag"],
                "hallucination_grounding_score": m["hallucination_grounding_score"],
                "total_latency_sec": m["total_latency_sec"],
            }
            writer.writerow(row)

def offense_keyword_coverage(answer_text: str, gold: Dict) -> float:
    """
    0–1: fraction of offense_keywords whose content words appear in the answer.
    More informative than a single 0/1 hit.
    """
    answer_words = tokenize_content(answer_text)
    if not gold.get("offense_keywords"):
        return 0.0

    covered = 0
    total = 0
    for phrase in gold["offense_keywords"]:
        phrase_words = tokenize_content(phrase)
        if not phrase_words:
            continue
        total += 1
        if phrase_words & answer_words:
            covered += 1

    if total == 0:
        return 0.0
    return covered / total


def retrieval_offense_keyword_coverage(context_text: str, gold: Dict) -> float:
    """
    0–1: same idea but for retrieved context instead of the answer.
    """
    context_words = tokenize_content(context_text)
    if not gold.get("offense_keywords"):
        return 0.0

    covered = 0
    total = 0
    for phrase in gold["offense_keywords"]:
        phrase_words = tokenize_content(phrase)
        if not phrase_words:
            continue
        total += 1
        if phrase_words & context_words:
            covered += 1

    if total == 0:
        return 0.0
    return covered / total

def key_issue_coverage(answer_text: str, gold: Dict) -> float:
    """
    0–1: fraction of key_issues that are at least partially covered in the answer.
    An issue is 'covered' if any content word from that issue appears in the answer.
    """
    answer_words = tokenize_content(answer_text)
    issues = gold.get("key_issues", [])
    if not issues:
        return 0.0

    covered = 0
    total = 0
    for issue in issues:
        issue_words = tokenize_content(issue)
        if not issue_words:
            continue
        total += 1
        if issue_words & answer_words:
            covered += 1

    if total == 0:
        return 0.0
    return covered / total

def context_relevance_score(answer_text: str, context_text: str) -> float:
    """
    0–1 Jaccard similarity between answer and retrieved context content words.
    Proxy for: is the answer grounded in the retrieved text?
    """
    answer_words = tokenize_content(answer_text)
    context_words = tokenize_content(context_text)
    return jaccard_similarity(answer_words, context_words)


# === 7. Main ===

def main():
    print(f"[RAG] Active config: {ACTIVE_CONFIG.config_name}")
    print(f"[RAG] LLM: {ACTIVE_CONFIG.llm_model_name}")
    print(f"[RAG] Embeddings: {ACTIVE_CONFIG.embedding_model_name}")
    print(f"[SEMANTIC EVAL] Running {len(TEST_CASES)} test cases with config: {ACTIVE_CONFIG.config_name}")

    outputs: List[Dict] = []

    for case in TEST_CASES:
        print(f"\n[CASE {case['id']}] {case['description']}")
        out = run_single_case(case)
        outputs.append(out)
        m = out["metrics"]

        print(
            f"  offense_hit={m['offense_category_hit']}, "
            f"offense_cov={m['offense_keyword_coverage']:.2f}, "
            f"key_issue_cov={m['key_issue_coverage']:.2f}, "
            f"ans_rel={m['answer_relevance_score']:.2f}, "
            f"ctx_rel={m['context_relevance_score']:.2f}, "
            f"halluc_flag={m['hallucination_flag']}, "
            f"halluc_ground={m['hallucination_grounding_score']:.2f}"
        )


    csv_path = "semantic_results.csv"
    save_semantic_results(outputs, csv_path)
    print(f"\n[SEMANTIC EVAL] Done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()