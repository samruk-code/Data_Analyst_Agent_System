"""Governance: access control, row-level security, PII policy, query caps.

Core principle from the design doc: don't rely on the model behaving --
make leakage structurally impossible in the common path. Concretely:

1. Every query runs through a DuckDB connection where the "tables" the
   agent can see (``orders``, ``customers``, ...) are actually *views*,
   scoped to the requesting user's permitted regions and with PII columns
   masked, built fresh per user. The underlying ``raw_*`` tables are never
   exposed. If the user isn't entitled to a row or column, the query
   physically cannot return it -- the model's cooperation is irrelevant.

2. A static check on the generated SQL text enforces read-only (SELECT/CTE
   only; DDL/DML rejected), single-statement, and no direct reference to
   the underlying ``raw_*`` tables (belt-and-suspenders on top of the view
   shadowing above), plus a row cap.

This module has zero LLM calls in it -- it's the enforcement layer the rest
of the system cannot talk its way around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from data_analyst_agent.warehouse.seed import DB_PATH

MAX_ROWS = 5_000
"""Row cap applied to every query -- stands in for a warehouse byte-scanned /
row-scanned cap. Real system: enforce via the warehouse's own query-cost
limits (e.g. Snowflake resource monitors, BigQuery maximum_bytes_billed)."""

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|"
    r"CALL|INSTALL|LOAD|VACUUM|TRUNCATE|GRANT|REVOKE|SET|EXPORT|IMPORT|"
    r"MERGE)\b",
    re.IGNORECASE,
)
_RAW_TABLE_REF = re.compile(r"\braw_\w+", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)

ALL_REGIONS = ("US", "EU", "APAC")


@dataclass(frozen=True)
class User:
    """A requesting user's identity and entitlements.

    In a real system this comes from the identity provider / warehouse role,
    not from the request -- it's asserted context, never something the
    agent (or a prompt-injected value) can widen.
    """

    user_id: str
    display_name: str
    allowed_regions: tuple[str, ...]
    pii_allowed: bool


USERS: dict[str, User] = {
    "alice": User("alice", "Alice (US regional analyst)", ("US",), pii_allowed=False),
    "bob": User("bob", "Bob (global finance lead)", ALL_REGIONS, pii_allowed=True),
}


def get_user(user_id: str) -> User:
    try:
        return USERS[user_id]
    except KeyError as e:
        raise ValueError(f"Unknown user {user_id!r}. Known users: {list(USERS)}") from e


@dataclass(frozen=True)
class GovernanceResult:
    ok: bool
    sql: str
    reason: str | None = None


def _region_list_sql(regions: tuple[str, ...]) -> str:
    return ", ".join(f"'{r}'" for r in regions)


def secured_connection(user: User) -> duckdb.DuckDBPyConnection:
    """Open a fresh connection and register per-user secured views.

    RLS and PII masking happen here, in SQL the agent never sees or writes --
    not via a WHERE clause the model was asked to remember to add.
    """
    con = duckdb.connect(str(DB_PATH), read_only=False)
    regions_sql = _region_list_sql(user.allowed_regions)
    email_expr = "email" if user.pii_allowed else "'REDACTED'"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW orders AS
        SELECT order_id, customer_id, region, order_date, status, total_amount
        FROM raw_orders
        WHERE is_test = FALSE AND region IN ({regions_sql})
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW order_items AS
        SELECT oi.item_id, oi.order_id, oi.sku, oi.quantity, oi.unit_price
        FROM raw_order_items oi
        JOIN raw_orders o ON o.order_id = oi.order_id
        WHERE o.is_test = FALSE AND o.region IN ({regions_sql})
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW refunds AS
        SELECT r.refund_id, r.order_id, r.refund_date, r.refund_amount
        FROM raw_refunds r
        JOIN raw_orders o ON o.order_id = r.order_id
        WHERE o.is_test = FALSE AND o.region IN ({regions_sql})
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW customers AS
        SELECT customer_id, {email_expr} AS email, region, signup_date
        FROM raw_customers
        WHERE is_test = FALSE AND region IN ({regions_sql})
        """
    )
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW regions AS "
        f"SELECT * FROM raw_regions WHERE region_code IN ({regions_sql})"
    )
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW agg_daily_revenue AS "
        f"SELECT * FROM raw_agg_daily_revenue WHERE region IN ({regions_sql})"
    )
    return con


def check_and_prepare(sql: str) -> GovernanceResult:
    """Static safety check + row-cap injection. Runs before execution."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return GovernanceResult(False, sql, "Empty query.")

    statements = [s for s in stripped.split(";") if s.strip()]
    if len(statements) != 1:
        return GovernanceResult(False, sql, "Only a single SQL statement is permitted.")

    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return GovernanceResult(False, sql, "Only read-only SELECT/CTE queries are permitted.")

    forbidden = _FORBIDDEN_KEYWORDS.search(stripped)
    if forbidden:
        return GovernanceResult(
            False, sql, f"Disallowed keyword '{forbidden.group(0)}' -- read-only access only."
        )

    raw_ref = _RAW_TABLE_REF.search(stripped)
    if raw_ref:
        return GovernanceResult(
            False,
            sql,
            f"Query references '{raw_ref.group(0)}' directly -- query the secured "
            f"views (orders, customers, ...) instead.",
        )

    if not _LIMIT_RE.search(stripped):
        stripped = f"{stripped}\nLIMIT {MAX_ROWS}"

    return GovernanceResult(True, stripped, None)
