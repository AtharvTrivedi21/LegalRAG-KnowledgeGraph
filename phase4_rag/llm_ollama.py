from __future__ import annotations

"""
Ollama LLM client for Phase 4.

This module provides a small wrapper around the local Ollama HTTP API
for non-streaming chat completions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import requests

from .config import settings


class OllamaError(RuntimeError):
    """Raised when the Ollama backend returns an error or is unreachable."""


@dataclass
class ChatMessage:
    role: str
    content: str


def _build_url(path: str) -> str:
    base = settings.ollama.base_url.rstrip("/")
    return f"{base}{path}"


def chat_completion(messages: List[ChatMessage]) -> str:
    """
    Call Ollama's /api/chat endpoint and return the final response text.

    This uses non-streaming mode for simplicity.
    """
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

