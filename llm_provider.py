"""llm_provider.py - one place that decides which provider serves a model name.

  LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free      -> OpenRouter (OPENROUTER_API_KEY)
  AGENT_MODEL=groq:openai/gpt-oss-120b                  -> Groq       (GROQ_API_KEY)   (prefix "groq:")
"""
from __future__ import annotations

import os

GROQ_URL = "https://api.groq.com/openai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def resolve(name: str) -> dict | None:
    """-> {"api_key", "base_url", "model"} for a role's model name, or None when the key is missing."""
    name = (name or "").strip()
    if name.lower().startswith("groq:"):
        key, url, name = os.getenv("GROQ_API_KEY"), os.getenv("GROQ_BASE_URL", GROQ_URL), name[5:].strip()
    else:
        key, url = os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_BASE_URL", OPENROUTER_URL)
    return {"api_key": key, "base_url": url, "model": name} if key and name else None