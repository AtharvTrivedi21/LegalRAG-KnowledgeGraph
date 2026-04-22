from __future__ import annotations

"""
Ollama LLM client for Phase 4.

This module provides a small wrapper around the local Ollama HTTP API
for non-streaming chat completions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os

import requests

from .config import settings


class OllamaError(RuntimeError):
    """Raised when the Ollama backend returns an error or is unreachable."""


@dataclass
class ChatMessage:
    role: str
    content: str


def _get_last_groq_key() -> str:
    preferred = os.getenv("GROQ_API_KEY_3", "").strip()
    if preferred:
        return preferred
    candidates = [
        os.getenv("GROQ_API_KEY_1", "").strip(),
        os.getenv("GROQ_API_KEY_2", "").strip(),
        os.getenv("GROQ_API_KEY_3", "").strip(),
        os.getenv("GROQ_API_KEY_4", "").strip(),
        os.getenv("GROQ_API_KEY_5", "").strip(),
    ]
    candidates = [k for k in candidates if k]
    if candidates:
        return candidates[0]
    return os.getenv("GROQ_API_KEY", "").strip()


def _groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()


def _build_url(path: str) -> str:
    base = settings.ollama.base_url.rstrip("/")
    return f"{base}{path}"


def _groq_chat_completion(messages: List[ChatMessage]) -> str:
    key = _get_last_groq_key()
    if not key:
        raise OllamaError("Groq key not configured. Set GROQ_API_KEY_N in .env.")
    payload: Dict[str, Any] = {
        "model": _groq_model(),
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise OllamaError(f"Failed to reach Groq endpoint: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"Groq error {resp.status_code}: {resp.text}")
    data = resp.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not content:
        raise OllamaError("Groq responded without message content.")
    return content.strip()


def chat_completion(messages: List[ChatMessage]) -> str:
    """
    Call Ollama's /api/chat endpoint and return the final response text.

    This uses non-streaming mode for simplicity.
    """
    # If any Groq key is configured, try Groq first (last configured key),
    # but gracefully fall back to local Ollama on provider-side failures
    # (e.g. organization_restricted, auth/rate issues, temporary outages).
    if _get_last_groq_key():
        try:
            return _groq_chat_completion(messages)
        except OllamaError as groq_err:
            print(f"[LLM] Groq unavailable ({groq_err}); falling back to local Ollama.")

    payload: Dict[str, Any] = {
        "model": settings.ollama.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
    }
    url = _build_url("/api/chat")
    # Enforce minimum 180s so Streamlit/IDE env (e.g. OLLAMA_TIMEOUT=60) never causes read timeout
    timeout = max(settings.ollama.request_timeout, 180)

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaError(f"Failed to reach Ollama at {url}: {exc}") from exc

    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()
    # Expected schema: {"message": {"role": "...", "content": "..."}, ...}
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise OllamaError("Ollama responded without message content.")
    return content

