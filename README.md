# Data Analyst Agent

A natural-language data analyst agent built on **LangGraph** (orchestrator loop),
**LangChain** (`langchain-anthropic` for model calls, structured outputs), and
**LangSmith** (tracing + evals), implementing the system design in `text.txt`
end to end over a synthetic DuckDB "warehouse."

A business user asks a question in plain language; the agent grounds it in a
semantic layer and schema catalog, generates SQL, runs it read-only under the
user's own permissions, verifies the result with a critic before trusting it,
and returns a plain-English answer with the exact SQL shown for audit.

## Why a DuckDB warehouse instead of Snowflake/BigQuery

The design doc assumes a real cloud warehouse with billions of rows. This repo
reproduces the *shapes* that make data-analyst agents fail — not the scale —
in a single embedded DuckDB file so the whole system runs locally with no
infrastructure:

- `orders` (one row per order) / `order_items` (many rows per order) — the
  classic **fan-out trap**: `SUM(orders.total_amount)` after joining to
  `order_items` silently ~3x's the real number.
- `refunds` — must be pre-aggregated to one row per order before netting
  against `orders.total_amount`, or a **metric-semantics** bug creeps in.
- `agg_daily_revenue` — a materialized, independently-computed **control
  total** the critic reconciles ad-hoc revenue queries against.
- Row-level security (region) and PII (`customers.email`) enforced by
  per-user DuckDB views, not by asking the model to add a `WHERE` clause.

Run `uv run python -m data_analyst_agent.warehouse.seed` to (re)build
`data/warehouse.duckdb` (created automatically on first CLI run).

## Architecture -> code map

| Design doc concept | Module |
|---|---|
| Orchestrator / agent loop | `data_analyst_agent/graph.py` (LangGraph `StateGraph`) |
| Planner (intent, ambiguity) | `nodes/planner.py` |
| Semantic layer + schema RAG | `semantic_layer.py`, `warehouse/schema_catalog.py`, `nodes/grounding.py` |
| Query Gen + Run | `nodes/query_gen.py`, `db.py` (`RunSQLInput`/`RunSQLResult`) |
| Governance gate (RLS, PII, read-only, cost caps) | `governance.py`, `nodes/governance_node.py` |
| Analyzer / Critic | `critic.py`, `nodes/critique.py` |
| Working memory / retry-with-feedback | `state.py` (`AgentState.retry_feedback`), the `generate_sql <-> governance/execute/critic` loop in `graph.py` |
| Analysis (Python on the small result set) | `nodes/analyze.py` |
| Answer composition (headline + explanation + chart + SQL) | `nodes/compose.py` |
| Evals (execution accuracy, clarify-not-guess, governance red-team) | `evals/` |

The orchestrator loop mirrors Appendix A of the design doc almost 1:1:
`plan -> (clarify | ground) -> generate_sql -> governance_gate -> execute_sql
-> critic -> (retry generate_sql | analyze) -> compose_answer`, bounded by
`DAA_MAX_QUERY_ATTEMPTS` (default 3).

### Governance is structural, not a prompt

`governance.secured_connection(user)` creates fresh, per-request DuckDB views
(`orders`, `customers`, ...) scoped to the user's `allowed_regions` and with
`email` masked unless `pii_allowed`. The agent is only ever handed these view
names — the underlying `raw_*` tables are never in its schema context, and
`governance.check_and_prepare` additionally rejects any SQL that references
`raw_*` directly, or anything other than a single read-only `SELECT`/`WITH`
statement. **The model's cooperation is irrelevant**: even a SQL query that
tries to `SELECT * FROM customers` for a region-restricted user simply cannot
return rows it isn't entitled to, and PII columns read back as `'REDACTED'`
regardless of what the query asks for. `evals/dataset.py` includes red-team
probes that verify this against a model that actively tries to select
forbidden columns/regions.

### The critic catches the fan-out bug

`critic.run_deterministic_checks` reconciles any `revenue`-metric result
against `agg_daily_revenue` (computed independently, never through
`order_items`). A fan-out bug inflates the number ~3x, which blows well past
the reconciliation tolerance (`DAA_CONTROL_TOTAL_TOLERANCE`, default 15%) —
the critic fails, feeds a specific diagnosis back to `generate_sql`, and the
next attempt corrects it.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY (required); LANGSMITH_* optional
```

Requires an Anthropic API key (`ANTHROPIC_API_KEY`) for the LLM calls — query
generation and the critic's intent-match check run on the strong model
(`claude-opus-5` by default); intent parsing and narrative composition run on
a cheaper model (`claude-haiku-4-5`), per the design doc's tiered-model
routing (correctness-critical steps don't get cheaped out).

**Troubleshooting:** if `uv run data-analyst-agent ...` fails with
`ModuleNotFoundError: No module named 'data_analyst_agent'` right after a
fresh clone, `uv sync` cached a stale/empty build of the local project. Fix
with a clean reinstall:

```bash
rm -rf .venv
uv sync --reinstall
```

(`uv run python -m data_analyst_agent.cli ...` also works as a workaround
without touching the venv, since it doesn't depend on the installed console
script.)

## Run it

```bash
uv run data-analyst-agent "What was net revenue in the US region?"
uv run data-analyst-agent --user alice "List customer emails"   # RLS + PII demo
uv run data-analyst-agent "How are sales doing?"                # ambiguity -> clarify
uv run data-analyst-agent   # interactive mode
```

Two demo users (`governance.py`) exercise governance-by-construction:

- `bob` — global finance lead: all regions, PII allowed.
- `alice` — US regional analyst: US only, no PII (`email` reads as `REDACTED`).

## Evals

```bash
uv run python -m evals.run_evals            # uses LangSmith if LANGSMITH_API_KEY is set
uv run python -m evals.run_evals --local    # force local mode, no upload
```

`evals/dataset.py` covers the design doc's three eval layers:

1. **Execution accuracy** — agent's numeric result vs. an independently
   written gold SQL query, run through the same governed views.
2. **Clarification behavior** — clarify on genuinely ambiguous questions
   ("How are sales doing?"), answer directly on crisp ones.
3. **Governance red-team** — a hard pass/fail gate, not a graded score: must
   be 100%, always.

With `LANGSMITH_TRACING=true` set, every graph node invocation (planner,
query generation, governance, execution, critic, analysis, composition) is
captured as a trace span in LangSmith automatically via LangChain's
instrumentation — no extra code needed beyond the env vars in `.env`.

## Known simplifications vs. the full design

- **Scale**: thousands of rows, not billions — DuckDB, not Snowflake/BigQuery.
  Governance and reconciliation logic is scale-independent; the byte-scanned
  cost caps and partition-filter requirements from the design doc are not
  implemented (a row cap and a wall-clock query timeout stand in for them —
  see `governance.MAX_ROWS` and `db.DEFAULT_TIMEOUT_S`).
- **Schema retrieval**: the whole catalog (6 tables) is handed to the model
  directly rather than embedding-retrieved, since it fits comfortably in
  context. A larger warehouse would need real vector retrieval over table/
  column descriptions in `warehouse/schema_catalog.py`.
- **Prompt-injection-via-data**: not separately demonstrated, but the same
  containment argument applies — the agent is read-only and permission-capped,
  so even a successful injection from a free-text column couldn't mutate data
  or exceed the user's access.
