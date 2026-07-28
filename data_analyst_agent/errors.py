"""Typed failure modes the graph can catch and route on (retry vs. give up)."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for errors the orchestrator knows how to recover from."""


class GovernanceError(AgentError):
    """SQL failed the read-only/single-statement/no-raw-table static check."""


class QueryTimeoutError(AgentError):
    """Query exceeded its execution time budget and was cancelled."""


class SQLExecutionError(AgentError):
    """The warehouse rejected the SQL (e.g. hallucinated column/table)."""
