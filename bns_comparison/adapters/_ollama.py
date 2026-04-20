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
from typing import Dict, List

import requests

from bns_comparison.config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_TIMEOUT,
    GROQ_API_KEY, GROQ_API_KEYS, GROQ_MODEL, GROQ_RPM_LIMIT,
)

_SECTION_RE = re.compile(r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)", re.IGNORECASE)

_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds between retries on GPU OOM

# Rate limiter state for Groq free tier (30 req/min) per key
_groq_call_times_by_key: Dict[str, List[float]] = {}
_groq_key_cooldown_until: Dict[str, float] = {}
_groq_key_idx = 0


def _groq_rate_limit(key: str):
    """Enforce Groq free tier rate limit (30 req/min) per key by sleeping if needed."""
    now = time.time()
    window = 60.0
    call_times = _groq_call_times_by_key.setdefault(key, [])
    # Remove calls older than 60s
    while call_times and call_times[0] < now - window:
        call_times.pop(0)
    if len(call_times) >= GROQ_RPM_LIMIT:
        wait = call_times[0] + window - now + 0.5
        if wait > 0:
            print(f"    [Groq] Rate limit reached, waiting {wait:.1f}s...")
            time.sleep(wait)
    call_times.append(time.time())


def _next_key(keys: List[str]) -> str:
    """Pick next available key based on cooldown; wait if all cooling down."""
    global _groq_key_idx
    now = time.time()
    n = len(keys)
    for i in range(n):
        idx = (_groq_key_idx + i) % n
        key = keys[idx]
        if _groq_key_cooldown_until.get(key, 0.0) <= now:
            _groq_key_idx = idx
            return key

    # All keys are cooling down: wait until earliest key unlocks.
    earliest = min(_groq_key_cooldown_until.get(k, now) for k in keys)
    wait = max(earliest - now, 0.5)
    print(f"    [Groq] All keys cooling down, waiting {wait:.1f}s...")
    time.sleep(wait)
    # after wait, return current index key
    return keys[_groq_key_idx % n]


def _groq_chat(messages: List[dict], model: str = GROQ_MODEL) -> str:
    """Call Groq Cloud API (OpenAI-compatible) and return response text."""
    global _groq_key_idx
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    keys = GROQ_API_KEYS or ([GROQ_API_KEY] if GROQ_API_KEY else [])
    if not keys:
        raise RuntimeError("Groq requested but no API key configured")

    last_err = None
    total_attempts = max(_MAX_RETRIES, len(keys) * 8)
    for attempt in range(total_attempts):
        key = _next_key(keys)
        _groq_rate_limit(key)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
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
                _groq_key_cooldown_until[key] = time.time() + retry_after
                _groq_key_idx = (_groq_key_idx + 1) % len(keys)
                if len(keys) > 1:
                    print(f"    [Groq] Rate limited; switching key ({_groq_key_idx+1}/{len(keys)})")
                else:
                    print(f"    [Groq] Rate limited, waiting {retry_after:.1f}s...")
                    time.sleep(retry_after)
                continue

            if resp.status_code in (401, 403):
                last_err = f"Groq auth error {resp.status_code} (attempt {attempt+1})"
                _groq_key_idx = (_groq_key_idx + 1) % len(keys)
                if len(keys) > 1:
                    print(f"    [Groq] Auth error; switching key ({_groq_key_idx+1}/{len(keys)})")
                    continue
                raise RuntimeError(f"Groq API auth error {resp.status_code}: {err_text[:300]}")

            raise RuntimeError(f"Groq API error {resp.status_code}: {err_text[:300]}")
        except requests.RequestException as exc:
            last_err = str(exc)
            if attempt < total_attempts - 1:
                time.sleep(5)

    raise RuntimeError(f"Groq chat failed after {total_attempts} attempts: {last_err}")


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
_USE_GROQ = bool(GROQ_API_KEYS or GROQ_API_KEY)

if _USE_GROQ:
    n_keys = len(GROQ_API_KEYS) if GROQ_API_KEYS else 1
    print(f"[LLM Backend] Using Groq Cloud ({n_keys} key(s), {GROQ_MODEL})")
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
