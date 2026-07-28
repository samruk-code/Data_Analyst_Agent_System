"""Execution node: run the (already governance-checked) SQL read-only under
the user's row-level-security scope. A hallucinated column/table surfaces
here as a typed error and gets fed back into query generation as feedback,
rather than crashing the graph."""

from __future__ import annotations

from data_analyst_agent.config import MAX_QUERY_ATTEMPTS, SQL_TIMEOUT_S
from data_analyst_agent.db import run_sql
from data_analyst_agent.errors import AgentError
from data_analyst_agent.state import AgentState


def execute_sql(state: AgentState) -> dict:
    try:
        result = run_sql(state["sql"], state["user"], timeout_s=SQL_TIMEOUT_S)
        return {"sql_result": result, "sql_error": None}
    except AgentError as exc:
        feedback = list(state.get("retry_feedback") or []) + [
            f"The warehouse rejected the SQL: {exc}"
        ]
        return {"sql_result": None, "sql_error": str(exc), "retry_feedback": feedback}


def route_after_execution(state: AgentState) -> str:
    if state.get("sql_error") is None:
        return "critique"
    if state.get("attempt", 0) < state.get("max_attempts", MAX_QUERY_ATTEMPTS):
        return "retry"
    return "give_up"
