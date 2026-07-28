"""Configuration: model routing, retry budget, LangSmith tracing.

Tiered-model routing per the design doc: cheap model for intent parsing and
charting, strong model for query generation and the critic/verification
pass (correctness-critical steps don't get cheaped out).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

TODAY = os.environ.get("DAA_TODAY", "2026-07-28")
"""Fixed 'today' for resolving relative dates ('last quarter', 'last 30
days') against the warehouse's synthetic data range, which ends 2026-07-27."""

STRONG_MODEL = os.environ.get("DAA_STRONG_MODEL", "claude-opus-5")
CHEAP_MODEL = os.environ.get("DAA_CHEAP_MODEL", "claude-haiku-4-5")

MAX_QUERY_ATTEMPTS = int(os.environ.get("DAA_MAX_QUERY_ATTEMPTS", "3"))
SQL_TIMEOUT_S = float(os.environ.get("DAA_SQL_TIMEOUT_S", "10"))

# Reconciliation / plausibility thresholds used by the critic.
CONTROL_TOTAL_TOLERANCE = float(os.environ.get("DAA_CONTROL_TOTAL_TOLERANCE", "0.15"))

# LangSmith tracing is opt-in via env vars (LANGSMITH_TRACING=true,
# LANGSMITH_API_KEY=..., LANGSMITH_PROJECT=...) -- nothing to do here beyond
# loading .env; langsmith/langchain read these directly from the environment.
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "data-analyst-agent")
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
