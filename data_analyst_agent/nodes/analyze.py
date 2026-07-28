"""Analyze: Python operates on the SMALL result set the SQL returned.

Per the design doc's SQL/Python split: the warehouse does the heavy lifting
(filtering, joining, aggregating), Python only reshapes/charts/computes
simple stats on the already-small result. No raw big data ever leaves the
warehouse.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_analyst_agent.critic import control_total_revenue
from data_analyst_agent.state import AgentState, ChartSpec

CHART_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


def _prior_period(start_date: str, end_date: str) -> tuple[str, str]:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    duration = end - start
    prior_end = start
    prior_start = start - duration
    return prior_start.isoformat(), prior_end.isoformat()


def _maybe_chart(state: AgentState) -> ChartSpec | None:
    intent = state["intent"]
    result = state.get("sql_result")
    if result is None or result.row_count < 2 or not intent.breakdown_dimension:
        return None
    df = result.to_dataframe()
    dim_col = next((c for c in df.columns if c.lower() == intent.breakdown_dimension.lower()), None)
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if dim_col is None or not numeric_cols:
        return None
    y_col = numeric_cols[0]

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"{uuid.uuid4().hex}.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[dim_col].astype(str), df[y_col])
    ax.set_xlabel(dim_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {dim_col}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return ChartSpec(kind="bar", x=dim_col, y=y_col, path=str(path))


def _maybe_comparison(state: AgentState) -> str | None:
    intent = state["intent"]
    result = state.get("sql_result")
    if (
        result is None
        or not intent.comparison
        or intent.metric.lower() != "revenue"
        or not intent.start_date
        or not intent.end_date
    ):
        return None

    df = result.to_dataframe()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if not numeric_cols:
        return None
    current_total = float(df[numeric_cols[0]].sum())

    prior_start, prior_end = _prior_period(intent.start_date, intent.end_date)
    prior_total = control_total_revenue(state["user"], intent.regions, prior_start, prior_end)
    if prior_total is None or prior_total == 0:
        return None

    pct_change = (current_total - prior_total) / abs(prior_total)
    direction = "up" if pct_change >= 0 else "down"
    return (
        f"That's {direction} {abs(pct_change):.1%} versus the prior period "
        f"({prior_start} to {prior_end}), which was {prior_total:,.2f}."
    )


def analyze(state: AgentState) -> dict:
    chart = _maybe_chart(state)
    comparison_text = _maybe_comparison(state)
    return {"_internal": {**(state.get("_internal") or {}), "chart": chart, "comparison_text": comparison_text}}
