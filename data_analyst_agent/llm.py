"""LLM client factories -- the only place ChatAnthropic gets constructed.

Kept separate from the nodes so the graph and critic modules stay importable
(and unit-testable) without requiring ANTHROPIC_API_KEY to be set.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from data_analyst_agent.config import CHEAP_MODEL, STRONG_MODEL


@lru_cache(maxsize=None)
def get_strong_llm() -> ChatAnthropic:
    """Query generation and the critic's intent-match check: correctness-
    critical steps that stay on the strong model per the design doc."""
    return ChatAnthropic(model=STRONG_MODEL, temperature=0, max_tokens=4096)


@lru_cache(maxsize=None)
def get_cheap_llm() -> ChatAnthropic:
    """Intent parsing and narrative composition: cheap-model territory,
    with routing that escalates to the strong model on ambiguity/uncertainty
    left as a documented extension point (see graph.py)."""
    return ChatAnthropic(model=CHEAP_MODEL, temperature=0, max_tokens=2048)
