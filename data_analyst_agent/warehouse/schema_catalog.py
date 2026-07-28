"""Schema catalog: table/column descriptions + sample values.

This is the "schema RAG" fallback described in the design doc: when a
question isn't fully covered by the semantic layer, the query generator is
grounded in real, described tables and sample values instead of raw column
names alone (knowing ``status`` contains ``'shipped'``, ``'returned'`` is
what lets the model filter correctly).

The catalog describes the *secured views* the agent actually queries
(``orders``, ``customers``, ...), not the ``raw_`` tables underneath --
those are never shown to the model. For a real warehouse with thousands of
tables this would be a vector-indexed retrieval step; here the whole
catalog is small enough to put in context directly, so retrieval is just
"return everything."
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnDoc:
    name: str
    type: str
    description: str
    sample_values: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableDoc:
    name: str
    description: str
    columns: list[ColumnDoc]
    grain: str


CATALOG: list[TableDoc] = [
    TableDoc(
        name="orders",
        description=(
            "One row per order. This is the canonical order-level fact table -- "
            "use this for revenue and order-count metrics. Already excludes "
            "internal test accounts and is filtered to the caller's permitted regions."
        ),
        grain="One row per order_id.",
        columns=[
            ColumnDoc("order_id", "INTEGER", "Primary key."),
            ColumnDoc("customer_id", "INTEGER", "Foreign key to customers.customer_id."),
            ColumnDoc("region", "VARCHAR", "Order region.", ["US", "EU", "APAC"]),
            ColumnDoc("order_date", "DATE", "Date the order was placed."),
            ColumnDoc(
                "status",
                "VARCHAR",
                "Order fulfillment status.",
                ["completed", "shipped", "returned"],
            ),
            ColumnDoc(
                "total_amount",
                "DECIMAL",
                "Order-level total in USD, gross of refunds. Sum this directly for "
                "revenue -- do NOT sum it after joining to order_items, that "
                "fans the total out once per line item and inflates the sum.",
            ),
        ],
    ),
    TableDoc(
        name="order_items",
        description=(
            "One row per line item within an order (many rows per order). Use "
            "this only for item/SKU-level questions (quantity sold, revenue by "
            "SKU from unit_price * quantity). Never join this to orders and "
            "then SUM(orders.total_amount) -- that double/triple counts the "
            "order total once per item."
        ),
        grain="One row per item_id; many rows per order_id.",
        columns=[
            ColumnDoc("item_id", "INTEGER", "Primary key."),
            ColumnDoc("order_id", "INTEGER", "Foreign key to orders.order_id."),
            ColumnDoc("sku", "VARCHAR", "Product SKU.", ["SKU-0001", "SKU-0042"]),
            ColumnDoc("quantity", "INTEGER", "Units of this SKU in this line item."),
            ColumnDoc("unit_price", "DECIMAL", "Price per unit in USD for this line item."),
        ],
    ),
    TableDoc(
        name="refunds",
        description=(
            "One row per refund. An order can have zero, one, or more partial "
            "refunds. Net revenue = orders.total_amount minus the SUM of "
            "matching refunds.refund_amount, grouped by order_id before "
            "subtracting (an order with 2 partial refunds must not be double "
            "counted as 2 orders)."
        ),
        grain="One row per refund_id; zero or more rows per order_id.",
        columns=[
            ColumnDoc("refund_id", "INTEGER", "Primary key."),
            ColumnDoc("order_id", "INTEGER", "Foreign key to orders.order_id."),
            ColumnDoc("refund_date", "DATE", "Date the refund was issued."),
            ColumnDoc("refund_amount", "DECIMAL", "Refund amount in USD."),
        ],
    ),
    TableDoc(
        name="customers",
        description=(
            "One row per customer. Already excludes internal test accounts and "
            "is filtered to the caller's permitted regions. The `email` column "
            "is PII: it is masked to 'REDACTED' unless the caller is authorized "
            "to see it -- this is enforced by the view, not by the query."
        ),
        grain="One row per customer_id.",
        columns=[
            ColumnDoc("customer_id", "INTEGER", "Primary key."),
            ColumnDoc(
                "email",
                "VARCHAR",
                "Customer email (PII). May read as 'REDACTED' depending on caller's access.",
            ),
            ColumnDoc("region", "VARCHAR", "Customer home region.", ["US", "EU", "APAC"]),
            ColumnDoc("signup_date", "DATE", "Date the customer signed up."),
        ],
    ),
    TableDoc(
        name="regions",
        description="Small dimension table mapping region codes to display names.",
        grain="One row per region_code.",
        columns=[
            ColumnDoc("region_code", "VARCHAR", "Primary key.", ["US", "EU", "APAC"]),
            ColumnDoc("region_name", "VARCHAR", "Human-readable region name."),
        ],
    ),
    TableDoc(
        name="agg_daily_revenue",
        description=(
            "Materialized, pre-aggregated daily net revenue by region, computed "
            "independently from orders/refunds (not through order_items). Use "
            "this as a fast path for revenue-over-time questions, and as the "
            "control total to reconcile any ad-hoc revenue computation against."
        ),
        grain="One row per (date, region).",
        columns=[
            ColumnDoc("date", "DATE", "Calendar date."),
            ColumnDoc("region", "VARCHAR", "Region.", ["US", "EU", "APAC"]),
            ColumnDoc("gross_revenue", "DECIMAL", "Sum of order totals before refunds."),
            ColumnDoc("refund_amount", "DECIMAL", "Sum of refunds issued that day's orders."),
            ColumnDoc("net_revenue", "DECIMAL", "gross_revenue - refund_amount. The governed 'revenue' number."),
        ],
    ),
]

JOIN_GRAPH: list[str] = [
    "orders.order_id = order_items.order_id  (one orders row -> many order_items rows)",
    "orders.order_id = refunds.order_id      (one orders row -> zero or more refunds rows)",
    "orders.customer_id = customers.customer_id",
    "orders.region = regions.region_code",
    "customers.region = regions.region_code",
]


def render_catalog(tables: list[str] | None = None) -> str:
    """Render the catalog (optionally filtered to a subset of tables) as text
    for the LLM prompt. In a warehouse with thousands of tables this is where
    you'd do embedding-based retrieval of the relevant slice instead of
    returning everything."""
    docs = CATALOG if tables is None else [t for t in CATALOG if t.name in tables]
    lines: list[str] = []
    for t in docs:
        lines.append(f"TABLE {t.name}  (grain: {t.grain})")
        lines.append(f"  {t.description}")
        for c in t.columns:
            sample = f" e.g. {c.sample_values}" if c.sample_values else ""
            lines.append(f"  - {c.name} ({c.type}): {c.description}{sample}")
        lines.append("")
    lines.append("VALIDATED JOIN PATHS (do not invent other joins):")
    for j in JOIN_GRAPH:
        lines.append(f"  - {j}")
    return "\n".join(lines)
