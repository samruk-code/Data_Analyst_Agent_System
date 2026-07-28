"""Query generation: compose from the semantic layer + schema-grounded SQL.

Stays on the strong model -- per the design doc, complex multi-join query
generation is one of the two correctness-critical steps (the other is the
critic) that don't get cheaped out under tiered-model routing.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from data_analyst_agent.llm import get_strong_llm
from data_analyst_agent.state import AgentState

_SYSTEM_PROMPT = """You write DuckDB SQL for a read-only data analyst agent. Ground
every query in the governed metric definitions and validated join paths provided
below -- never invent a join or redefine a metric's semantics. Query only the named
views (orders, order_items, refunds, customers, regions, agg_daily_revenue); these
are already scoped to the caller's row-level permissions and PII policy, so you do
not need to add your own permission filters.

Rules:
- SELECT/WITH only. One statement. No semicolon-separated statements.
- When a governed metric definition is provided for the question's metric, follow
  its exact expression and notes -- especially any warning about pre-aggregating a
  one-to-many table before joining it.
- Prefer COUNT(DISTINCT order_id) over COUNT(*) whenever a one-to-many join
  (order_items, refunds) is present, and never SUM(orders.total_amount) after such a
  join without first collapsing the joined table to one row per order_id.
- Apply the intent's region/date filters directly as WHERE clauses.
- Return concrete, executable SQL -- no placeholders, no comments explaining what
  you would do.

{schema_context}
"""


class GeneratedSQL(BaseModel):
    sql: str = Field(description="A single, executable, read-only DuckDB SQL statement.")
    rationale: str = Field(description="One sentence on how the SQL satisfies the intent.")


def generate_sql(state: AgentState) -> dict:
    llm = get_strong_llm().with_structured_output(GeneratedSQL)
    intent = state["intent"]

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(schema_context=state["schema_context"])),
        SystemMessage(
            content=f"Original question: {state['question']}\nStructured intent: {intent.model_dump_json()}"
        ),
    ]
    feedback = state.get("retry_feedback") or []
    if feedback:
        joined = "\n".join(f"- {f}" for f in feedback)
        messages.append(
            SystemMessage(
                content=f"Your previous attempt(s) had issues. Fix them in this attempt:\n{joined}"
            )
        )

    generated: GeneratedSQL = llm.invoke(messages)
    attempt = state.get("attempt", 0) + 1
    return {"sql": generated.sql, "attempt": attempt}
