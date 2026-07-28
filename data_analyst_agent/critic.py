"""The critic: "ran" is not "right".

Execution merely means the SQL didn't error. This module is the separate
verification pass the design doc insists on: cheap deterministic sanity
checks first (row counts, control-total reconciliation, order-of-magnitude
plausibility), then an LLM check that the SQL actually answers the asked
question. Guards the gap between "ran" and "right" -- the same reason a
careful analyst re-checks a surprising result before sending it.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from data_analyst_agent.config import CONTROL_TOTAL_TOLERANCE
from data_analyst_agent.db import RunSQLResult, run_sql
from data_analyst_agent.errors import AgentError
from data_analyst_agent.governance import User
from data_analyst_agent.llm import get_strong_llm
from data_analyst_agent.state import CriticResult, Intent


def _numeric_target_column(result: RunSQLResult) -> str | None:
    df = result.to_dataframe()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        return None
    named = [c for c in numeric_cols if "revenue" in c.lower()]
    return named[0] if named else numeric_cols[0]


def control_total_revenue(
    user: User,
    regions: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> float | None:
    """Independently computed reference number from the materialized
    agg_daily_revenue table -- built from orders/refunds directly, never
    through order_items, so it can't inherit a fan-out bug of its own.
    Reused by the critic (reconciliation) and by analyze() (period
    comparisons)."""
    where = []
    if regions:
        region_list = ", ".join(f"'{r}'" for r in regions)
        where.append(f"region IN ({region_list})")
    if start_date:
        where.append(f"date >= DATE '{start_date}'")
    if end_date:
        where.append(f"date < DATE '{end_date}'")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT SUM(net_revenue) AS control_total FROM agg_daily_revenue {where_sql}"
    try:
        result = run_sql(sql, user)
    except AgentError:
        return None
    if result.row_count == 0 or not result.rows or result.rows[0][0] is None:
        return None
    return float(result.rows[0][0])


def _control_total_revenue(user: User, intent: Intent) -> float | None:
    return control_total_revenue(user, intent.regions, intent.start_date, intent.end_date)


def run_deterministic_checks(
    intent: Intent, result: RunSQLResult, user: User
) -> tuple[list[str], list[str], str | None]:
    """Returns (checks_run, flags, blocking_feedback). A non-None
    blocking_feedback means the critic fails and the query should be retried
    with that feedback appended to context."""
    checks: list[str] = []
    flags: list[str] = []

    # 1. Row count sanity.
    checks.append("row_count")
    if result.row_count == 0:
        flags.append("Query returned no rows -- verify the filters (date range, region) are correct.")

    # 2. Null-rate sanity on the sample.
    checks.append("null_rate")
    df = result.to_dataframe()
    if not df.empty:
        for col in df.columns:
            null_frac = df[col].isna().mean()
            if null_frac > 0.5:
                flags.append(f"Column '{col}' is more than half null in the result sample.")

    # 3. Control-total reconciliation + order-of-magnitude, for revenue metrics.
    if intent.metric.lower() == "revenue":
        checks.append("control_total_reconciliation")
        target_col = _numeric_target_column(result)
        control_total = _control_total_revenue(user, intent)
        if target_col is not None and control_total is not None and not df.empty:
            computed_total = float(df[target_col].sum())
            denom = max(abs(control_total), 1.0)
            rel_diff = abs(computed_total - control_total) / denom
            if rel_diff > CONTROL_TOTAL_TOLERANCE:
                feedback = (
                    f"Computed {target_col}={computed_total:,.2f} diverges from the "
                    f"agg_daily_revenue control total={control_total:,.2f} by "
                    f"{rel_diff:.0%}, which exceeds the {CONTROL_TOTAL_TOLERANCE:.0%} "
                    f"tolerance. This is the signature of a join fan-out (e.g. joining "
                    f"orders to order_items or to un-aggregated refunds before summing "
                    f"total_amount) or a wrong/missing filter. Re-derive the metric using "
                    f"the semantic layer's 'revenue' definition exactly, pre-aggregating "
                    f"refunds to one row per order_id before joining."
                )
                return checks, flags, feedback
            elif rel_diff > CONTROL_TOTAL_TOLERANCE / 3:
                flags.append(
                    f"Computed total is {rel_diff:.0%} off the control total "
                    f"({computed_total:,.2f} vs {control_total:,.2f}) -- within tolerance "
                    f"but worth a second look."
                )

    return checks, flags, None


class _MatchesIntentVerdict(BaseModel):
    matches: bool = Field(description="True if the SQL actually answers the asked question.")
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


def llm_matches_intent(question: str, intent: Intent, sql: str) -> _MatchesIntentVerdict:
    """LLM check: does this SQL actually answer the asked question? Catches
    silent filter errors and metric-semantics mistakes a passing execution
    won't surface on its own. Stays on the strong model -- verification is
    correctness-critical, same as query generation."""
    llm = get_strong_llm().with_structured_output(_MatchesIntentVerdict)
    prompt = (
        "You are a rigorous SQL reviewer. Given the user's original question, the "
        "structured intent extracted from it, and the SQL a query generator wrote, "
        "decide whether the SQL actually answers the question as asked -- right metric "
        "definition, right filters, right grain, right comparison. Be skeptical: a query "
        "that runs without error can still answer the wrong question.\n\n"
        f"Question: {question}\n"
        f"Intent: {intent.model_dump_json(indent=2)}\n"
        f"SQL:\n{sql}"
    )
    return llm.invoke([SystemMessage(content=prompt)])


def critique(question: str, intent: Intent, sql: str, result: RunSQLResult, user: User) -> CriticResult:
    checks, flags, blocking_feedback = run_deterministic_checks(intent, result, user)
    if blocking_feedback:
        return CriticResult(passed=False, checks_run=checks, flags=flags, feedback=blocking_feedback)

    verdict = llm_matches_intent(question, intent, sql)
    checks.append("llm_matches_intent")
    if not verdict.matches:
        return CriticResult(
            passed=False,
            checks_run=checks,
            flags=flags,
            feedback=f"The SQL does not answer the question as asked: {verdict.reasoning}",
        )

    return CriticResult(passed=True, checks_run=checks, flags=flags, feedback=None)
