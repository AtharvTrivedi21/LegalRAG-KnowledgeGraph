"""
Shared LLM helpers for all adapters.
Supports two backends:
  - Local Ollama (default)
  - Groq Cloud (set GROQ_API_KEY env var to enable)

Uses direct HTTP calls — no langchain dependency required.
"""
import os
import re
import time
from typing import List

import requests

from bns_comparison.config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_MODEL, GROQ_RPM_LIMIT,
)

_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)

_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds between retries on GPU OOM

# Rate limiter state for Groq free tier (30 req/min)
_groq_call_times: List[float] = []


def _groq_rate_limit():
    """Enforce Groq free tier rate limit (30 req/min) by sleeping if needed."""
    now = time.time()
    window = 60.0
    # Remove calls older than 60s
    while _groq_call_times and _groq_call_times[0] < now - window:
        _groq_call_times.pop(0)
    if len(_groq_call_times) >= GROQ_RPM_LIMIT:
        wait = _groq_call_times[0] + window - now + 0.5
        if wait > 0:
            print(f"    [Groq] Rate limit reached, waiting {wait:.1f}s...")
            time.sleep(wait)
    _groq_call_times.append(time.time())


def _groq_chat(messages: List[dict], model: str = GROQ_MODEL) -> str:
    """Call Groq Cloud API (OpenAI-compatible) and return response text."""
    _groq_rate_limit()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()

            err_text = resp.text
            # Rate limit (429) — wait and retry
            if resp.status_code == 429:
                retry_after = 10
                try:
                    err_json = resp.json()
                    msg = err_json.get("error", {}).get("message", "")
                    import re as _re
                    m = _re.search(r"try again in (\d+\.?\d*)s", msg)
                    if m:
                        retry_after = float(m.group(1)) + 1
                except Exception:
                    pass
                last_err = f"Groq rate limited (attempt {attempt+1})"
                print(f"    [Groq] Rate limited, waiting {retry_after:.1f}s...")
                time.sleep(retry_after)
                continue

            raise RuntimeError(f"Groq API error {resp.status_code}: {err_text[:300]}")
        except requests.RequestException as exc:
            last_err = str(exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(5)

    raise RuntimeError(f"Groq chat failed after {_MAX_RETRIES} attempts: {last_err}")


def _ollama_chat(messages: List[dict], model: str = OLLAMA_LLM_MODEL) -> str:
    """Call local Ollama /api/chat and return response text."""
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


# Auto-select backend based on environment
_USE_GROQ = bool(GROQ_API_KEY)

if _USE_GROQ:
    print("[LLM Backend] Using Groq Cloud (llama3-8b-8192)")
else:
    print("[LLM Backend] Using local Ollama (llama3:8b)")


def ollama_chat(messages: List[dict], model: str = OLLAMA_LLM_MODEL) -> str:
    """Route to Groq or local Ollama based on GROQ_API_KEY env var."""
    if _USE_GROQ:
        return _groq_chat(messages)
    return _ollama_chat(messages, model)


def extract_section_numbers(text: str) -> List[str]:
    """Extract BNS section numbers from answer text (e.g. 'Section 303' -> '303')."""
    return list({m.upper() for m in _SECTION_RE.findall(text)})
