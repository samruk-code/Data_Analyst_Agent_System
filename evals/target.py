"""The system under evaluation, wrapped as a plain ``dict -> dict`` function
-- the shape ``langsmith.evaluate`` (and our local runner) both expect."""

from __future__ import annotations

from data_analyst_agent.config import MAX_QUERY_ATTEMPTS
from data_analyst_agent.governance import get_user
from data_analyst_agent.graph import get_graph


def target(inputs: dict) -> dict:
    user = get_user(inputs["user_id"])
    initial_state = {
        "question": inputs["question"],
        "user": user,
        "max_attempts": MAX_QUERY_ATTEMPTS,
        "attempt": 0,
        "retry_feedback": [],
        "trace_notes": [],
        "_internal": {},
    }
    final_state = get_graph().invoke(initial_state)
    answer = final_state["answer"]
    result = final_state.get("sql_result")
    critic = final_state.get("critic")
    return {
        "headline": answer.headline,
        "explanation": answer.explanation,
        "needs_clarification": answer.needs_clarification,
        "sql": answer.sql,
        "caveats": answer.caveats,
        "columns": result.columns if result else [],
        "rows": result.rows if result else [],
        "critic_passed": critic.passed if critic else None,
        "attempts": final_state.get("attempt", 0),
    }
