"""Anthropic model factory.

DeepAgents accepts any LangChain chat model object, so we build a
langchain_anthropic.ChatAnthropic pointed directly at the Anthropic API.
"""
from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic


def make_model(model: str | None = None, *, temperature: float = 0.0) -> ChatAnthropic:
    """Build a ChatAnthropic client.

    model: Anthropic model id, e.g. "claude-sonnet-4-6". Falls back to
    HARNESS_MODEL env var, then to "claude-haiku-4-5".
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")

    slug = model or os.environ.get("HARNESS_MODEL", "claude-haiku-4-5")

    # Cap output tokens; DeepAgents otherwise requests very large budgets.
    max_tokens = int(os.environ.get("HARNESS_MAX_TOKENS", "4096"))

    # Retry transient upstream errors (Anthropic 429/5xx).
    max_retries = int(os.environ.get("HARNESS_MAX_RETRIES", "5"))

    # Note: temperature=0.0 with certain Anthropic models may require API workarounds.
    return ChatAnthropic(
        model=slug,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


def subagent_model() -> ChatAnthropic:
    """Cheaper/faster model for role subagents; defaults to claude-haiku-4-5."""
    slug = os.environ.get("HARNESS_SUBAGENT_MODEL", "claude-haiku-4-5")
    return make_model(slug)
