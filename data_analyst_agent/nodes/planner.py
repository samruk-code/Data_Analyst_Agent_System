"""Planner: interpret the NL question -> structured Intent, or clarify.

Per the design doc: clarify rather than guess when the question is
genuinely ambiguous (which "revenue"? which date grain?), but don't
interrogate the user on a crisply specified question. Runs on the cheap
model -- intent parsing is not the correctness-critical step; query
generation and the critic are.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from data_analyst_agent.config import TODAY
from data_analyst_agent.llm import get_cheap_llm
from data_analyst_agent.semantic_layer import METRICS
from data_analyst_agent.state import AgentState, Answer, Intent

_METRIC_NAMES = ", ".join(METRICS.keys())

_SYSTEM_PROMPT = f"""You are the planning step of a data analyst agent over a retail
sales warehouse. Interpret the user's natural-language question into a structured
Intent.

Today's date is {TODAY}. The warehouse contains data from 2025-10-01 through
2026-07-27. Resolve relative timeframes ("last quarter", "last 30 days", "this year")
into concrete start_date/end_date (end_date exclusive) using today's date.

Known governed metrics: {_METRIC_NAMES}. If the question asks for something outside
these (e.g. a metric with no governed definition), still fill in `metric` with your
best concrete label (e.g. "units_sold") -- downstream steps fall back to
schema-grounded SQL when there's no semantic-layer entry for it.

Known dimension for breakdowns: "region" (values: US, EU, APAC).

Mark `is_ambiguous=true` ONLY when the question's meaning would genuinely change
based on an unstated choice -- which metric definition, which timeframe, which scope
-- not just because it lacks optional detail. Examples:
  - "How are sales doing?" -> ambiguous: "sales" could mean revenue, units, or order
    count, and "doing" has no timeframe or comparison. Propose a concrete default
    (e.g. "net revenue, last 30 days vs. the prior 30 days, all regions you can
    access") and ask a short clarifying question offering that default.
  - "What was net revenue in the EU last quarter?" -> NOT ambiguous. Metric, region,
    and timeframe are all specified. Just resolve it.
  - "How many orders came from each region last month?" -> NOT ambiguous.
If ambiguous, set proposed_default and clarifying_question. If not ambiguous, leave
those null.
"""


def plan_intent(state: AgentState) -> dict:
    llm = get_cheap_llm().with_structured_output(Intent)
    intent: Intent = llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), SystemMessage(content=f"Question: {state['question']}")]
    )
    return {"intent": intent, "attempt": 0, "retry_feedback": [], "trace_notes": [f"intent: {intent.model_dump_json()}"]}


def route_after_planning(state: AgentState) -> str:
    intent = state["intent"]
    return "clarify" if intent.is_ambiguous else "ground"


def clarify(state: AgentState) -> dict:
    intent = state["intent"]
    headline = intent.clarifying_question or "Could you clarify what you're asking for?"
    explanation = (
        f"Your question could reasonably mean a few things. If you don't want to "
        f"specify, I'd default to: {intent.proposed_default}"
    )
    return {
        "answer": Answer(
            headline=headline,
            explanation=explanation,
            needs_clarification=True,
            caveats=[],
        )
    }
