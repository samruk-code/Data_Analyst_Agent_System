"""Compose the final answer: layered for both audiences.

Headline in plain language, a short computation explanation, a chart when
useful, and the exact SQL shown (collapsible in a real UI) so a skeptical
user can verify. Showing the query is non-negotiable -- it converts "trust
the black box" into "verify the work."
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from data_analyst_agent.llm import get_cheap_llm
from data_analyst_agent.state import AgentState, Answer

_SYSTEM_PROMPT = """Write a headline and a short explanation for a non-technical
business user, based ONLY on the facts provided. Never state a number, filter, or
comparison that isn't directly supported by the SQL and result summary given to you.

- headline: one sentence, plain language, states the key number(s) and, if a
  comparison is given, the direction/magnitude of change.
- explanation: 2-4 sentences in plain English describing how the number was
  computed (metric definition, filters applied, timeframe) -- not the SQL syntax
  itself, but what it means. If flags/caveats are present, do not repeat them here;
  they're shown separately.
"""


class _Narrative(BaseModel):
    headline: str = Field(description="One sentence, plain language, includes the key number(s).")
    explanation: str = Field(description="2-4 sentences of plain-English computation explanation.")


def _failure_answer(state: AgentState, caveats: list[str]) -> Answer:
    reason = state.get("sql_error") or state.get("governance_reason") or "Unknown failure."
    attempt = state.get("attempt", 0)
    return Answer(
        headline="I couldn't produce a verified answer to this question.",
        explanation=(
            f"After {attempt} attempt(s), the query still failed: {reason}. Rather than "
            f"show a number I can't stand behind, I'm stopping here -- happy to try again "
            f"with a narrower or differently scoped question."
        ),
        sql=state.get("sql"),
        caveats=caveats,
    )


def compose_answer(state: AgentState) -> dict:
    critic = state.get("critic")
    internal = state.get("_internal") or {}
    result = state.get("sql_result")

    caveats: list[str] = []
    if critic:
        caveats.extend(critic.flags)
        if not critic.passed:
            caveats.append(
                f"Verification did not fully pass after {state.get('attempt', 0)} "
                f"attempt(s): {critic.feedback}"
            )

    if result is None:
        return {"answer": _failure_answer(state, caveats)}

    comparison_text = internal.get("comparison_text")
    chart = internal.get("chart")

    llm = get_cheap_llm().with_structured_output(_Narrative)
    narrative: _Narrative = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            SystemMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"Intent: {state['intent'].model_dump_json()}\n"
                    f"SQL:\n{state['sql']}\n"
                    f"Result summary:\n{result.summary}\n"
                    f"Comparison: {comparison_text or 'none computed'}"
                )
            ),
        ]
    )

    headline = narrative.headline
    if comparison_text and comparison_text not in headline:
        headline = f"{headline} {comparison_text}"

    answer = Answer(
        headline=headline,
        explanation=narrative.explanation,
        sql=state["sql"],
        chart=chart,
        caveats=caveats,
    )
    return {"answer": answer}
