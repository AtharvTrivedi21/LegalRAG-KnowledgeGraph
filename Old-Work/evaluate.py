import json
from typing import List, Dict

from tqdm import tqdm

from rag_chain import answer_query
from config import ACTIVE_CONFIG


# Example structure of test_cases.json:
# [
#   {
#     "id": 1,
#     "description": "Someone stole my mobile phone from my pocket in a bus.",
#     "expected_sections": ["theft-related-section-number-here"]
#   },
#   ...
# ]


def load_test_cases(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_sections_from_answer(answer: str) -> List[str]:
    """
    TODO: Implement a proper extraction based on how the model cites sections.
    For now, this is a placeholder stub.
    """
    # You can later implement regex like r"Section\s+(\d+)" etc.
    return []


def evaluate(test_cases: List[Dict]):
    total = len(test_cases)
    correct = 0

    for case in tqdm(test_cases, desc="Evaluating cases"):
        desc = case["description"]
        expected = set(case.get("expected_sections", []))

        result = answer_query(desc)
        answer = result["answer"]

        predicted = set(extract_sections_from_answer(answer))

        # naive: “correct” if any overlap
        if expected and (predicted & expected):
            correct += 1

    accuracy = correct / total if total else 0.0
    print(f"Config: {ACTIVE_CONFIG.config_name}")
    print(f"Total cases: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")


def main():
    test_cases_path = "data/test_cases.json"
    test_cases = load_test_cases(test_cases_path)
    evaluate(test_cases)


if __name__ == "__main__":
    main()