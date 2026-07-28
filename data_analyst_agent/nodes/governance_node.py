"""Governance gate node: static safety check before anything touches the
warehouse. Pure wrapper around governance.check_and_prepare -- no LLM calls."""

from __future__ import annotations

from data_analyst_agent.config import MAX_QUERY_ATTEMPTS
from data_analyst_agent.governance import check_and_prepare
from data_analyst_agent.state import AgentState


def governance_gate(state: AgentState) -> dict:
    result = check_and_prepare(state["sql"])
    if result.ok:
        return {"governance_ok": True, "governance_reason": None, "sql": result.sql}
    feedback = list(state.get("retry_feedback") or []) + [
        f"Governance rejected the SQL: {result.reason}"
    ]
    return {"governance_ok": False, "governance_reason": result.reason, "retry_feedback": feedback}


def route_after_governance(state: AgentState) -> str:
    if state.get("governance_ok"):
        return "execute"
    if state.get("attempt", 0) < state.get("max_attempts", MAX_QUERY_ATTEMPTS):
        return "retry"
    return "give_up"
