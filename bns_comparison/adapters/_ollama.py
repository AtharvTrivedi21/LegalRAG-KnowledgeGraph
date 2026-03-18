"""
Shared Ollama HTTP helpers for all three adapters.
Uses direct HTTP calls — no langchain dependency required.
"""
import re
from typing import List, Set

import requests

from bns_comparison.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT

_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)


def ollama_chat(messages: List[dict], model: str = OLLAMA_LLM_MODEL) -> str:
    """Call Ollama /api/chat and return response text."""
    payload = {"model": model, "messages": messages, "stream": False}
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama chat error {resp.status_code}: {resp.text}")
    data = resp.json()
    content = (data.get("message") or {}).get("content") or ""
    return content.strip()


def extract_section_numbers(text: str) -> List[str]:
    """Extract BNS section numbers from answer text (e.g. 'Section 303' -> '303')."""
    return list({m.upper() for m in _SECTION_RE.findall(text)})
