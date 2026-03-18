"""
Shared Ollama HTTP helpers for all three adapters.
Uses direct HTTP calls — no langchain dependency required.
"""
import re
import time
from typing import List

import requests

from bns_comparison.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT

_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)

_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds between retries on GPU OOM


def ollama_chat(messages: List[dict], model: str = OLLAMA_LLM_MODEL) -> str:
    """Call Ollama /api/chat and return response text. Retries on GPU OOM errors."""
    payload = {"model": model, "messages": messages, "stream": False}
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = (data.get("message") or {}).get("content") or ""
                return content.strip()
            err_text = resp.text
            # Retry on GPU OOM or runner crash
            if resp.status_code == 500 and (
                "cudaMalloc" in err_text or "out of memory" in err_text
                or "exit status" in err_text
            ):
                last_err = f"Ollama GPU OOM (attempt {attempt+1}): {err_text}"
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)
                    continue
            raise RuntimeError(f"Ollama chat error {resp.status_code}: {err_text}")
        except requests.RequestException as exc:
            last_err = str(exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
    raise RuntimeError(f"Ollama chat failed after {_MAX_RETRIES} attempts: {last_err}")


def extract_section_numbers(text: str) -> List[str]:
    """Extract BNS section numbers from answer text (e.g. 'Section 303' -> '303')."""
    return list({m.upper() for m in _SECTION_RE.findall(text)})
