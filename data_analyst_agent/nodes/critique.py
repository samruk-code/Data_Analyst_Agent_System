"""Critic node: verification pass between "ran" and "right"."""

from __future__ import annotations

from data_analyst_agent.config import MAX_QUERY_ATTEMPTS
from data_analyst_agent.critic import critique
from data_analyst_agent.state import AgentState


def critic_node(state: AgentState) -> dict:
    result = critique(
        question=state["question"],
        intent=state["intent"],
        sql=state["sql"],
        result=state["sql_result"],
        user=state["user"],
    )
    if not result.passed:
        feedback = list(state.get("retry_feedback") or []) + [result.feedback]
        return {"critic": result, "retry_feedback": feedback}
    return {"critic": result}


def route_after_critique(state: AgentState) -> str:
    critic = state["critic"]
    if critic.passed:
        return "analyze"
    if state.get("attempt", 0) < state.get("max_attempts", MAX_QUERY_ATTEMPTS):
        return "retry"
    # Out of retries: proceed to an answer anyway, but the unresolved critic
    # feedback becomes a loud caveat rather than a silently shipped number.
    return "analyze"
