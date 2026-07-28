"""Shared data shapes: the planner's Intent, the final Answer, and the
LangGraph state that threads through the orchestrator."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field

from data_analyst_agent.db import RunSQLResult
from data_analyst_agent.governance import User


class Intent(BaseModel):
    """Structured output of the planner: what the question is actually
    asking, resolved to concrete metric/grain/filters/timeframe."""

    metric: str = Field(description="The metric being asked about, e.g. 'revenue', 'order_count'.")
    regions: list[str] | None = Field(
        default=None, description="Region filter, e.g. ['EU']. None means no region filter."
    )
    start_date: str | None = Field(default=None, description="Inclusive ISO date filter, or null.")
    end_date: str | None = Field(default=None, description="Exclusive ISO date filter, or null.")
    comparison: str | None = Field(
        default=None,
        description="What the metric should be compared against, if anything, e.g. "
        "'prior period' or 'same period last year'. Null if no comparison was asked for.",
    )
    breakdown_dimension: str | None = Field(
        default=None,
        description="Dimension to group/break down by if the question asks for one, "
        "e.g. 'region'. Null if the question wants a single number.",
    )
    is_ambiguous: bool = Field(
        description="True if the question is genuinely ambiguous about metric definition, "
        "timeframe, or scope in a way that would change the answer."
    )
    proposed_default: str | None = Field(
        default=None,
        description="If ambiguous, a concrete default interpretation to propose to the user, "
        "e.g. 'net revenue, last 30 days, all regions you can access'.",
    )
    clarifying_question: str | None = Field(
        default=None, description="If ambiguous, the question to ask the user."
    )


class CriticResult(BaseModel):
    passed: bool
    checks_run: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list, description="Caveats to surface in the final answer.")
    feedback: str | None = Field(
        default=None, description="If failed, actionable feedback fed back into query generation."
    )


class ChartSpec(BaseModel):
    kind: str  # "bar" | "line" | "none"
    x: str | None = None
    y: str | None = None
    path: str | None = Field(default=None, description="Filesystem path to a rendered PNG, if any.")


class Answer(BaseModel):
    headline: str
    explanation: str
    sql: str | None = None
    chart: ChartSpec | None = None
    caveats: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class AgentState(TypedDict, total=False):
    question: str
    user: User

    intent: Optional[Intent]
    schema_context: str

    sql: Optional[str]
    attempt: int
    max_attempts: int

    governance_ok: Optional[bool]
    governance_reason: Optional[str]

    sql_result: Optional[RunSQLResult]
    sql_error: Optional[str]

    critic: Optional[CriticResult]
    retry_feedback: list[str]

    control_total: Optional[float]

    answer: Optional[Answer]

    trace_notes: list[str]
    _internal: dict[str, Any]
