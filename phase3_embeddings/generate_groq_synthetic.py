"""
Generate synthetic BNS retrieval training pairs using Groq (Llama 3.1 8B).

For each crime-relevant BNS section (section_number >= 45), calls Groq to
generate 3 realistic citizen incident descriptions in plain informal English.
Pairs each generated query with the real BNS section text as the positive.

Writes incrementally (crash-safe resume) to:
  phase3_embeddings/bns_groq_synthetic_pairs.jsonl

Usage:
  python -m phase3_embeddings.generate_groq_synthetic

Set GROQ_API_KEY env var before running.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent

# Load .env so GROQ_API_KEY is available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
SECTIONS_CSV = ROOT / "phase1_output_v2" / "sections.csv"
OUTPUT_JSONL = ROOT / "phase3_embeddings" / "bns_groq_synthetic_pairs.jsonl"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_RPM_LIMIT = 30  # free tier

_groq_call_times: list = []


def _groq_rate_limit():
    now = time.time()
    while _groq_call_times and _groq_call_times[0] < now - 60.0:
        _groq_call_times.pop(0)
    if len(_groq_call_times) >= GROQ_RPM_LIMIT:
        wait = _groq_call_times[0] + 60.0 - now + 0.5
        if wait > 0:
            print(f"  [rate-limit] waiting {wait:.1f}s...")
            time.sleep(wait)
    _groq_call_times.append(time.time())


def _groq_call(prompt: str, api_key: str, max_retries: int = 5) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        _groq_rate_limit()
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            if resp.status_code == 429:
                wait = 10.0
                try:
                    m = re.search(r"try again in (\d+\.?\d*)s", resp.json().get("error", {}).get("message", ""))
                    if m:
                        wait = float(m.group(1)) + 1.0
                except Exception:
                    pass
                print(f"  [429] rate limited, waiting {wait:.1f}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq error {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            print(f"  [network error] {e}, retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Groq call failed after {max_retries} attempts")


PROMPT_TEMPLATE = """\
You are helping build a legal information retrieval training dataset for Indian criminal law.

BNS Section {sec_num} - {heading}

Write exactly 3 realistic incident descriptions that a citizen would report to the police or search for online to find this BNS section.

Rules:
- Use informal, plain English (not legal terminology)
- Each description: 1-3 sentences, specific and vivid
- Vary the scenario, victim, location, and circumstances across the 3 descriptions
- Do NOT mention section numbers or law names
- Each description must clearly map to this specific offense

Output format - exactly 3 lines, one description per line, no numbering, no bullets:
<description 1>
<description 2>
<description 3>"""


def _parse_descriptions(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    lines = [re.sub(r"^[\d\.\-\*\)]+\s*", "", l) for l in lines]
    lines = [l for l in lines if len(l) > 20]
    return lines[:3]


def _load_completed_sections() -> set:
    completed = set()
    if OUTPUT_JSONL.exists():
        with OUTPUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    completed.add(obj.get("meta", {}).get("section_number"))
                except Exception:
                    pass
    return completed


def main():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Run: set GROQ_API_KEY=your_key")
        sys.exit(1)

    df = pd.read_csv(SECTIONS_CSV)
    bns = df[df["act_id"] == "BNS_2023"].drop_duplicates("section_id")
    crime = bns[bns["section_number"].astype(int) >= 45].copy()
    crime = crime.sort_values("section_number", key=lambda x: x.astype(int))

    completed = _load_completed_sections()
    remaining = [
        row for _, row in crime.iterrows()
        if str(int(row["section_number"])) not in completed
    ]

    print(f"Total crime sections: {len(crime)}")
    print(f"Already done: {len(completed)}, Remaining: {len(remaining)}")
    print(f"Output: {OUTPUT_JSONL}\n")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, row in enumerate(remaining):
        sec_num = int(row["section_number"])
        heading = str(row.get("heading", "")).strip()[:120]
        full_text = str(row.get("full_text", "")).strip()
        if not full_text:
            continue

        prompt = PROMPT_TEMPLATE.format(sec_num=sec_num, heading=heading)

        print(f"[{i+1}/{len(remaining)}] Section {sec_num}: {heading[:60]}...")
        try:
            response = _groq_call(prompt, api_key)
            descriptions = _parse_descriptions(response)
        except Exception as e:
            print(f"  ERROR: {e} — skipping")
            continue

        if not descriptions:
            print(f"  WARNING: no valid descriptions parsed, skipping")
            continue

        positive = f"Section {sec_num} - {heading}\n{full_text}"
        with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
            for desc in descriptions:
                record = {
                    "query": desc,
                    "positive": positive,
                    "meta": {
                        "section_number": str(sec_num),
                        "heading": heading,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(f"  -> {len(descriptions)} queries written (total so far: {written})")

    total = sum(1 for _ in OUTPUT_JSONL.open(encoding="utf-8") if _.strip())
    print(f"\nDone. Total pairs in file: {total}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
