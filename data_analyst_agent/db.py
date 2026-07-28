"""The one and only path to the warehouse: read-only, capped, sandboxed SQL.

Mirrors the ``RunSQLInput`` / ``RunSQLResult`` tool shape from the design
doc. Two things are enforced by the tool itself, not the prompt:

- read-only (governance.check_and_prepare rejects anything but SELECT/CTE)
- execution under the requesting user's permissions (governance.secured_connection)

so no prompt-injected instruction can make this mutate data or see rows the
user isn't entitled to.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
from pydantic import BaseModel, Field

from data_analyst_agent.errors import GovernanceError, QueryTimeoutError, SQLExecutionError
from data_analyst_agent.governance import User, check_and_prepare, secured_connection

DEFAULT_TIMEOUT_S = 10.0
SAMPLE_ROWS_IN_SUMMARY = 20


class RunSQLInput(BaseModel):
    sql: str
    timeout_s: float = DEFAULT_TIMEOUT_S


class RunSQLResult(BaseModel):
    columns: list[str]
    rows: list[list] = Field(description="Result rows, capped to a sample if large.")
    row_count: int
    truncated: bool
    summary: str = Field(description="Plain-language description of the result, for the model.")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


def _summarize(df: pd.DataFrame, truncated: bool) -> str:
    if df.empty:
        return "Query returned 0 rows."
    parts = [f"Query returned {len(df)} row(s) and {len(df.columns)} column(s): {list(df.columns)}."]
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols[:5]:
        parts.append(
            f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, sum={df[col].sum():.2f}"
        )
    if truncated:
        parts.append(f"  (sample truncated to first {SAMPLE_ROWS_IN_SUMMARY} rows)")
    return "\n".join(parts)


def run_sql(sql: str, user: User, timeout_s: float = DEFAULT_TIMEOUT_S) -> RunSQLResult:
    """Execute ``sql`` read-only, under ``user``'s row-level-security scope,
    with a row cap and a wall-clock timeout. Raises a typed AgentError on any
    failure so the caller (the graph) can decide whether to retry."""
    gov = check_and_prepare(sql)
    if not gov.ok:
        raise GovernanceError(gov.reason or "Query rejected by governance policy.")

    con = secured_connection(user)
    outcome: dict = {}

    def _worker() -> None:
        try:
            outcome["df"] = con.execute(gov.sql).fetchdf()
        except Exception as exc:  # noqa: BLE001 - re-raised typed on the main thread
            outcome["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    start = time.monotonic()
    worker.start()
    worker.join(timeout_s)

    if worker.is_alive():
        try:
            con.interrupt()
        except Exception:  # noqa: BLE001 - best-effort cancellation
            pass
        worker.join(2.0)
        con.close()
        raise QueryTimeoutError(
            f"Query exceeded the {timeout_s:.0f}s execution budget and was cancelled."
        )

    con.close()
    elapsed = time.monotonic() - start

    if "error" in outcome:
        raise SQLExecutionError(str(outcome["error"]))

    df: pd.DataFrame = outcome["df"]
    row_count = len(df)
    truncated = row_count > SAMPLE_ROWS_IN_SUMMARY
    sample = df.head(SAMPLE_ROWS_IN_SUMMARY)

    return RunSQLResult(
        columns=list(df.columns),
        rows=sample.astype(object).where(pd.notnull(sample), None).values.tolist(),
        row_count=row_count,
        truncated=truncated,
        summary=_summarize(df, truncated) + f"\n  (executed in {elapsed:.2f}s)",
    )
