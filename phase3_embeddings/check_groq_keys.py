"""
Check health of multiple Groq API keys.

It can discover keys from:
1) .env variables: GROQ_API_KEY_1..GROQ_API_KEY_10, GROQ_API_KEY
2) Commented legacy lines in .env such as:
   # GROQ_API_KEY=gsk_xxx  #API KEY 1
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import requests


ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"


def _mask(key: str) -> str:
    if len(key) < 10:
        return "***"
    return f"{key[:8]}...{key[-6:]}"


def _load_keys() -> List[Tuple[str, str]]:
    # Prefer explicit env vars first
    pairs: List[Tuple[str, str]] = []
    for i in range(1, 11):
        name = f"GROQ_API_KEY_{i}"
        val = os.environ.get(name, "").strip()
        if val:
            pairs.append((name, val))
    single = os.environ.get("GROQ_API_KEY", "").strip()
    if single:
        pairs.append(("GROQ_API_KEY", single))

    # Also parse .env lines (including commented legacy format)
    if ENV_PATH.exists():
        text = ENV_PATH.read_text(encoding="utf-8", errors="ignore")

        # Active style: GROQ_API_KEY_1=...
        for m in re.finditer(r"^\s*(GROQ_API_KEY(?:_\d+)?)\s*=\s*([^\s#]+)", text, flags=re.MULTILINE):
            name, key = m.group(1), m.group(2).strip()
            pairs.append((name, key))

        # Commented legacy: # GROQ_API_KEY=... #API KEY N
        for m in re.finditer(r"^\s*#\s*GROQ_API_KEY\s*=\s*([^\s#]+)\s*(?:#\s*API KEY\s*(\d+))?", text, flags=re.MULTILINE):
            key = m.group(1).strip()
            idx = m.group(2)
            name = f"LEGACY_COMMENT_KEY_{idx}" if idx else "LEGACY_COMMENT_KEY"
            pairs.append((name, key))

    # Deduplicate by key value, keep first seen name
    seen = set()
    dedup = []
    for name, key in pairs:
        if key in seen:
            continue
        seen.add(key)
        dedup.append((name, key))
    return dedup


def _check_key(name: str, key: str) -> Tuple[bool, str]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"network_error: {e}"

    if resp.status_code == 200:
        return True, "ok"
    if resp.status_code == 429:
        return True, "rate_limited_but_valid"
    if resp.status_code == 401:
        return False, "unauthorized"
    return False, f"http_{resp.status_code}"


def main():
    keys = _load_keys()
    if not keys:
        print("No Groq keys found in environment or .env")
        raise SystemExit(1)

    print(f"Found {len(keys)} candidate keys")
    all_ok = True
    for i, (name, key) in enumerate(keys, start=1):
        ok, detail = _check_key(name, key)
        status = "PASS" if ok else "FAIL"
        print(f"[{i}] {name:22} {_mask(key):22} -> {status} ({detail})")
        all_ok = all_ok and ok

    if all_ok:
        print("ALL_KEYS_RESPONDING=YES")
    else:
        print("ALL_KEYS_RESPONDING=NO")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

