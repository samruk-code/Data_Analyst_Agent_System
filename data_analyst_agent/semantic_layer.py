"""Semantic layer: governed metric definitions + verified query exemplars.

Per the design doc, this is the spine of the system: where coverage exists,
the agent composes from these pre-validated definitions instead of
reinventing business logic in raw SQL. It encodes things a model can't infer
on its own -- "revenue" nets refunds, the correct grain, which of several
similar tables/columns is canonical -- as reusable snippets, not prose the
model has to remember.

This demo's semantic layer is deliberately partial (metric coverage is
"incomplete", per the design brief's stated assumption): `revenue` and
`order_count` are defined; anything else falls back to schema-grounded SQL
generation via the catalog in ``warehouse/schema_catalog.py``, with more
verification and more willingness to hedge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDef:
    name: str
    description: str
    sql_expression: str
    required_tables: tuple[str, ...]
    notes: str


METRICS: dict[str, MetricDef] = {
    "revenue": MetricDef(
        name="revenue",
        description=(
            "Net revenue: gross order totals minus refunds. Always computed "
            "at the orders grain (never via order_items)."
        ),
        sql_expression=(
            "SUM(o.total_amount) - COALESCE(SUM(refund_totals.refund_total), 0)"
        ),
        required_tables=("orders",),
        notes=(
            "Join refunds only after pre-aggregating them to one row per "
            "order_id (SUM(refund_amount) GROUP BY order_id), then LEFT JOIN "
            "that to orders. Joining raw refunds rows directly to orders "
            "before aggregating will fan out orders with >1 refund. "
            "Reconcile against agg_daily_revenue.net_revenue as a control total."
        ),
    ),
    "order_count": MetricDef(
        name="order_count",
        description="Number of distinct orders.",
        sql_expression="COUNT(DISTINCT o.order_id)",
        required_tables=("orders",),
        notes=(
            "Use COUNT(DISTINCT order_id), not COUNT(*), whenever the query "
            "joins orders to a one-to-many table (order_items, refunds) -- "
            "COUNT(*) after such a join counts line items/refunds, not orders."
        ),
    ),
}


FEW_SHOT_EXEMPLARS: list[dict[str, str]] = [
    {
        "question": "What was net revenue in the EU last quarter?",
        "sql": (
            "SELECT ROUND(SUM(o.total_amount) - COALESCE(SUM(rt.refund_total), 0), 2) "
            "AS net_revenue\n"
            "FROM orders o\n"
            "LEFT JOIN (\n"
            "    SELECT order_id, SUM(refund_amount) AS refund_total\n"
            "    FROM refunds GROUP BY order_id\n"
            ") rt ON rt.order_id = o.order_id\n"
            "WHERE o.region = 'EU'\n"
            "  AND o.order_date >= DATE '2026-01-01' AND o.order_date < DATE '2026-04-01'"
        ),
        "why": (
            "Refunds are pre-aggregated to one row per order before the join, so "
            "an order with two partial refunds doesn't fan out the orders side."
        ),
    },
    {
        "question": "How many orders came from each region last month?",
        "sql": (
            "SELECT region, COUNT(DISTINCT order_id) AS order_count\n"
            "FROM orders\n"
            "WHERE order_date >= DATE '2026-06-01' AND order_date < DATE '2026-07-01'\n"
            "GROUP BY region\n"
            "ORDER BY order_count DESC"
        ),
        "why": "No join needed, so a plain COUNT(*) would also be correct here -- "
        "COUNT(DISTINCT order_id) is used defensively/for consistency.",
    },
    {
        "question": "What's the total quantity sold per SKU this year?",
        "sql": (
            "SELECT sku, SUM(quantity) AS units_sold\n"
            "FROM order_items oi\n"
            "JOIN orders o ON o.order_id = oi.order_id\n"
            "WHERE o.order_date >= DATE '2026-01-01'\n"
            "GROUP BY sku\n"
            "ORDER BY units_sold DESC"
        ),
        "why": (
            "Item-level question, so order_items is the right grain. This never "
            "sums orders.total_amount, so no fan-out risk."
        ),
    },
]


def render_semantic_layer() -> str:
    lines = ["GOVERNED METRIC DEFINITIONS (compose from these; do not redefine them):"]
    for m in METRICS.values():
        lines.append(f"- {m.name}: {m.description}")
        lines.append(f"    expression: {m.sql_expression}")
        lines.append(f"    requires: {', '.join(m.required_tables)}")
        lines.append(f"    notes: {m.notes}")
    lines.append("")
    lines.append("VERIFIED QUESTION -> SQL EXEMPLARS FOR THIS WAREHOUSE:")
    for ex in FEW_SHOT_EXEMPLARS:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL:\n{ex['sql']}")
        lines.append(f"(why this shape: {ex['why']})")
        lines.append("")
    return "\n".join(lines)


def metric_coverage(metric_name: str) -> MetricDef | None:
    return METRICS.get(metric_name.lower().strip())
