"""agents/llm.py - one chat model per role. A role's model can live on OpenRouter (default) or Groq (prefix "groq:").
PLANNER_MODEL, AGENT_MODEL (qa + analyst tool calling), WRITER_MODEL (final answers); unset roles use LLM_MODEL.
Timeouts and retries are set here."""
from __future__ import annotations

import os

import llm_provider

_ROLE_ENV = {"planner": "PLANNER_MODEL", "agent": "AGENT_MODEL", "writer": "WRITER_MODEL"}


def model_name(role: str) -> str:
    return (os.getenv(_ROLE_ENV.get(role, "")) or os.getenv("LLM_MODEL") or "").strip()


def get_chat(role: str = "agent", max_tokens: int | None = None):
    from langchain_openai import ChatOpenAI
    cfg = llm_provider.resolve(model_name(role))
    if not cfg:
        raise RuntimeError(f"No API key or model configured for role {role!r} ({model_name(role)!r})")
    return ChatOpenAI(
        model=cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
        max_tokens=max_tokens or int(os.getenv("AGENT_MAX_TOKENS", "3500")),
        temperature=0.1,
        timeout=float(os.getenv("AGENT_TIMEOUT", "90")),
        max_retries=int(os.getenv("AGENT_LLM_RETRIES", "2")),
    )