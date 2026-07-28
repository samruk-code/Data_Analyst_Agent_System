"""Synthetic "warehouse" for the data analyst agent demo.

Stands in for the "single cloud warehouse, billions of rows in the big fact
tables" scenario from the design doc. Real scale is out of reach for a local
demo, so instead we reproduce the *shapes* that make data-analyst agents fail:

- A one-to-many join (orders -> order_items) that inflates SUM(total_amount)
  if you join to line items and forget to re-aggregate first (the "fan-out"
  failure mode).
- A refunds table that must be netted out of gross revenue to get the
  governed "revenue" metric right.
- A materialized daily aggregate (``raw_agg_daily_revenue``) that acts as a
  control total for reconciliation, computed independently from the detail
  tables.
- PII (customer email) and a region column used for row-level security.
- A deliberate `is_test` flag that must be filtered out of every metric.

Tables are prefixed ``raw_`` and are never exposed to the agent directly --
:mod:`data_analyst_agent.governance` builds per-user secured views on top of
them named without the prefix (``orders``, ``customers``, ...), so row-level
security and PII masking are enforced by the database engine, not by prompt
instructions.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "warehouse.duckdb"

REGIONS = [
    ("US", "United States"),
    ("EU", "Europe"),
    ("APAC", "Asia Pacific"),
]

N_CUSTOMERS = 4_000
N_ORDERS = 20_000
START_DATE = dt.date(2025, 10, 1)
END_DATE = dt.date(2026, 7, 27)
SKUS = [f"SKU-{i:04d}" for i in range(1, 121)]


def _generate(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    region_codes = [r[0] for r in REGIONS]
    regions_df = pd.DataFrame(REGIONS, columns=["region_code", "region_name"])

    customer_region = rng.choice(region_codes, size=N_CUSTOMERS, p=[0.5, 0.3, 0.2])
    signup_offsets = rng.integers(0, (END_DATE - START_DATE).days, size=N_CUSTOMERS)
    customers_df = pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUSTOMERS + 1),
            "email": [f"user{i}@example.com" for i in range(1, N_CUSTOMERS + 1)],
            "region": customer_region,
            "signup_date": [START_DATE + dt.timedelta(days=int(o)) for o in signup_offsets],
            # ~2% of accounts are internal test accounts and must be excluded
            # from every business metric.
            "is_test": rng.random(N_CUSTOMERS) < 0.02,
        }
    )

    order_customer_idx = rng.integers(0, N_CUSTOMERS, size=N_ORDERS)
    order_customer = customers_df["customer_id"].to_numpy()[order_customer_idx]
    order_region = customers_df["region"].to_numpy()[order_customer_idx]
    order_is_test = customers_df["is_test"].to_numpy()[order_customer_idx]

    day_offsets = rng.integers(0, (END_DATE - START_DATE).days, size=N_ORDERS)
    order_dates = [START_DATE + dt.timedelta(days=int(o)) for o in day_offsets]

    # APAC had a real, deliberate slowdown in March 2026 (fewer orders) so
    # that "why did revenue drop" questions have a genuine, decomposable
    # answer localized to one region/month rather than noise.
    keep_mask = np.ones(N_ORDERS, dtype=bool)
    for i, (d, r) in enumerate(zip(order_dates, order_region)):
        if r == "APAC" and d.year == 2026 and d.month == 3 and rng.random() < 0.55:
            keep_mask[i] = False

    order_customer = order_customer[keep_mask]
    order_region = order_region[keep_mask]
    order_is_test = order_is_test[keep_mask]
    order_dates = [d for d, k in zip(order_dates, keep_mask) if k]
    n_orders = len(order_dates)

    # Order-level total: this is the grain-correct number. Summing it after
    # joining to order_items (one-to-many) is the classic fan-out bug.
    order_totals = np.round(rng.gamma(shape=3.0, scale=18.0, size=n_orders) + 5, 2)
    statuses = rng.choice(
        ["completed", "completed", "completed", "shipped", "returned"], size=n_orders
    )

    orders_df = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "customer_id": order_customer,
            "region": order_region,
            "order_date": order_dates,
            "status": statuses,
            "total_amount": order_totals,
            "is_test": order_is_test,
        }
    )

    # Order items: 1-5 line items per order, unit prices that roughly
    # reconcile to total_amount but are NOT required to sum exactly --
    # real warehouses are messy this way, and it's irrelevant to the trap
    # (the trap is fan-out on a re-aggregated order.total_amount, not a
    # mismatch between item totals and order totals).
    item_rows = []
    item_id = 1
    n_items_per_order = rng.integers(1, 6, size=n_orders)
    for oid, n_items in zip(orders_df["order_id"], n_items_per_order):
        for _ in range(int(n_items)):
            item_rows.append(
                {
                    "item_id": item_id,
                    "order_id": int(oid),
                    "sku": rng.choice(SKUS),
                    "quantity": int(rng.integers(1, 4)),
                    "unit_price": round(float(rng.gamma(2.0, 8.0) + 2), 2),
                }
            )
            item_id += 1
    order_items_df = pd.DataFrame(item_rows)

    # ~6% of non-test orders get a partial or full refund.
    refundable = orders_df[~orders_df["is_test"]].sample(frac=0.06, random_state=seed)
    refund_rows = []
    for i, row in enumerate(refundable.itertuples(), start=1):
        frac = rng.choice([1.0, 0.5, 0.25])
        refund_amount = round(row.total_amount * frac, 2)
        refund_date = row.order_date + dt.timedelta(days=int(rng.integers(1, 14)))
        refund_rows.append(
            {
                "refund_id": i,
                "order_id": row.order_id,
                "refund_date": refund_date,
                "refund_amount": refund_amount,
            }
        )
    refunds_df = pd.DataFrame(refund_rows)

    return {
        "raw_regions": regions_df,
        "raw_customers": customers_df,
        "raw_orders": orders_df,
        "raw_order_items": order_items_df,
        "raw_refunds": refunds_df,
    }


def build_warehouse(db_path: Path = DB_PATH, seed: int = 7, force: bool = False) -> Path:
    """Create (or rebuild) the DuckDB warehouse file with synthetic data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not force:
        return db_path
    if db_path.exists():
        db_path.unlink()

    tables = _generate(seed=seed)
    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            con.register("_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")

        # Materialized, independently-computed control total. This is what
        # the critic reconciles ad-hoc revenue queries against -- it is
        # built directly from raw_orders/raw_refunds, never through
        # raw_order_items, so it can't inherit the fan-out bug itself.
        con.execute(
            """
            CREATE TABLE raw_agg_daily_revenue AS
            SELECT
                o.order_date AS date,
                o.region AS region,
                ROUND(SUM(o.total_amount), 2) AS gross_revenue,
                ROUND(COALESCE(SUM(r.refund_total), 0), 2) AS refund_amount,
                ROUND(SUM(o.total_amount) - COALESCE(SUM(r.refund_total), 0), 2) AS net_revenue
            FROM raw_orders o
            LEFT JOIN (
                SELECT order_id, SUM(refund_amount) AS refund_total
                FROM raw_refunds
                GROUP BY order_id
            ) r ON r.order_id = o.order_id
            WHERE o.is_test = FALSE
            GROUP BY 1, 2
            """
        )

        con.execute("ALTER TABLE raw_orders ADD PRIMARY KEY (order_id)")
    finally:
        con.close()
    return db_path


if __name__ == "__main__":
    path = build_warehouse(force=True)
    print(f"Warehouse built at {path}")
