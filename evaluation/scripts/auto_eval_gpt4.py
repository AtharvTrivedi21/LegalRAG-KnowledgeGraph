import json
import re
import csv
from statistics import mean

INPUT = "evaluation/results/system3_raw_results.jsonl"
OUTPUT = "evaluation/results/gpt4_eval_results.csv"

def extract_sections_from_text(text):
    # Find patterns like [BNS Section 123] or BNS_2023_s123 or "Section 123" and return as strings
    nums = set(re.findall(r'\[BNS Section (\d+)\]', text))
    nums |= set(re.findall(r'BNS_2023_s(\d+)', text))
    nums |= set(re.findall(r'\bSection\s+(\d+)\b', text))
    nums |= set(re.findall(r'\bArticle\s+(\d+)\b', text))
    return set(nums)

def score_case(query, context_sections, answer, answer_cited_sections):
    # Extract sections mentioned in query
    query_secs = set(re.findall(r'Article\s+(\d+)', query)) | set(re.findall(r'Section\s+(\d+)', query))

    # Extract requested topics from query (keywords)
    topic_keywords = ['burden of proof','admissibility','circumstantial','compensation','restitution',
                      'report','fir','investigation','procedure','punishment','file a complaint',
                      'rights','obligations','evidence','identity','prove','reporting','procedure','penalty']
    q_lower = query.lower()
    requested_topics = set(k for k in topic_keywords if k in q_lower)

    # Context relevance: measure how well context covers query sections or topics
    if not context_sections:
        context_relevance = 0.0
    else:
        if query_secs:
            overlap = query_secs & context_sections
            frac = len(overlap) / len(query_secs)
            if frac == 1.0:
                context_relevance = 1.0
            elif frac >= 0.5:
                context_relevance = 0.8
            elif frac > 0.0:
                context_relevance = 0.5
            else:
                # No section overlap; check topical overlap by scanning context text heuristics done upstream
                context_relevance = 0.4
        else:
            # No explicit sections in query — check if context contains topic keywords
            # (caller may provide context text to extract topics; here treat presence of any section as partial)
            context_relevance = 0.8 if len(context_sections) >= 2 else 0.5

    # Groundedness: check that sections cited in the answer exist in context; penalize IPC mentions
    answer_lower = (answer or "").lower()
    mentions_ipc = 'ipc' in answer_lower or 'indian penal code' in answer_lower
    # Extract numeric citations from answer like [BNS Section 123], Section 123, Article 123, or numeric in brackets
    cited_nums = set(re.findall(r'\[BNS Section (\d+)\]', answer)) | set(re.findall(r'Section\s+(\d+)', answer)) | set(re.findall(r'Article\s+(\d+)', answer))
    # If explicit answer_cited_sections provided, prefer it
    cited = set(str(x) for x in (answer_cited_sections or [])) or cited_nums
    if not cited:
        # No explicit citations found: if answer makes legal claims and context exists, give medium score
        groundedness = 0.6 if context_sections and len(answer.strip())>80 else 0.2
    else:
        total = len(cited)
        found = sum(1 for s in cited if s in context_sections)
        frac = found / total if total>0 else 0.0
        if mentions_ipc:
            groundedness = 0.3
        elif frac == 1.0:
            groundedness = 1.0
        elif frac >= 0.6:
            groundedness = 0.8
        elif frac > 0.0:
            groundedness = 0.5
        else:
            groundedness = 0.1

    # Answer relevance: check whether answer addresses requested topics and gives procedural guidance
    hits = 0
    a_lower = answer_lower
    for t in requested_topics:
        if t in a_lower:
            hits += 1
    # Also consider presence of procedural verbs or recommendation phrases
    procedural_phrases = ['report to','file a complaint','you should','recommend','seek','entitled to','may be entitled','cooperate with the investigation','approach the']
    proc_hits = sum(1 for p in procedural_phrases if p in a_lower)
    # Score logic
    if hits >= max(1, len(requested_topics)//2) and proc_hits >= 1:
        answer_relevance = 1.0
    elif hits >= 1 and (proc_hits >= 1 or len(answer.strip())>200):
        answer_relevance = 0.8
    elif len(answer.strip())>80:
        answer_relevance = 0.5
    else:
        answer_relevance = 0.1

    return round(answer_relevance,1), round(context_relevance,1), round(groundedness,1)

def make_justification(case_id, answer_relevance, context_relevance, groundedness, query, context_sections, answer, answer_cited_sections):
    # Build one-sentence justification summarizing key reasons (concise)
    reasons = []
    if answer_relevance>=1.0:
        reasons.append("Answer directly addresses query topics and procedures")
    elif answer_relevance>=0.8:
        reasons.append("Answer mostly addresses the query with relevant guidance")
    elif answer_relevance>=0.5:
        reasons.append("Answer is partially relevant but lacks depth")
    else:
        reasons.append("Answer is brief or off-topic")

    if context_relevance>=1.0:
        reasons.append("retrieved context contains the requested BNS sections")
    elif context_relevance>=0.7:
        reasons.append("most retrieved sections are relevant to the query")
    elif context_relevance>=0.4:
        reasons.append("context is mixed; some sections are tangential")
    else:
        reasons.append("context lacks the sections requested")

    if groundedness>=1.0:
        reasons.append("citations in the answer are present in the retrieved context")
    elif groundedness>=0.7:
        reasons.append("most citations are grounded in the context")
    elif groundedness>=0.4:
        reasons.append("some citations are not found in the context")
    else:
        reasons.append("answer contains unsupported or fabricated claims")

    return "; ".join(reasons) + "."

def main():
    rows = []
    with open(INPUT, 'r', encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case_id = obj.get("case_id")
            res = obj.get("result", {})
            # In some files the nested structure is result.result
            if 'result' in res and isinstance(res['result'], dict):
                res = res['result']
            query = res.get("rephrased_query","") or ""
            answer = res.get("answer","") or ""
            # prefer explicit cited_sections if present, else extract from context_text
            answer_cited_sections = res.get("cited_sections") or []
            # context sections: try res.cited_sections (retrieved), else extract from context_text
            context_sections = set()
            if isinstance(res.get("cited_sections"), list):
                context_sections = set(str(x) for x in res.get("cited_sections"))
            else:
                context_text = res.get("context_text","") or ""
                context_sections = extract_sections_from_text(context_text)

            ar, cr, gr = score_case(query, context_sections, answer, answer_cited_sections)
            avg = round(mean([ar,cr,gr]),1)
            justification = make_justification(case_id, ar, cr, gr, query, context_sections, answer, answer_cited_sections)
            rows.append({
                "case_id": case_id,
                "answer_relevance": f"{ar:.1f}",
                "context_relevance": f"{cr:.1f}",
                "groundedness": f"{gr:.1f}",
                "avg_relevance": f"{avg:.1f}",
                "justification": justification
            })

    # Sort rows by case_id
    rows.sort(key=lambda r: int(r["case_id"]))
    # Write CSV
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["case_id","answer_relevance","context_relevance","groundedness","avg_relevance","justification"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == "__main__":
    main()

