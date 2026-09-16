# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An NL-to-SQL "data analyst agent": a LangGraph orchestrator that turns a natural-language
business question into governed, read-only DuckDB SQL, executes it under per-user row-level
security, verifies the result with a separate critic pass, and composes a plain-language answer
that shows its work (the exact SQL that ran).

Uses `uv` for dependency management (`uv.lock` present, Python >=3.13 pinned via `.python-version`).

## Commands

```bash
# Run the agent (builds the synthetic warehouse on first run)
uv run data-analyst-agent "What was net revenue in the EU last quarter?"
uv run data-analyst-agent --user alice "List customer emails"   # run as a different demo user
uv run data-analyst-agent --rebuild-warehouse                    # force-regenerate the warehouse
uv run data-analyst-agent                                        # interactive REPL mode

# Eval suite (execution accuracy, clarify-not-guess, governance red-team)
uv run python -m evals.run_evals            # uses LangSmith if LANGSMITH_API_KEY is set
uv run python -m evals.run_evals --local    # force local run, prints a table, no upload

# Rebuild the warehouse directly (also removes/regenerates data/warehouse.duckdb)
uv run python -m data_analyst_agent.warehouse.seed
```

There is no separate lint/test/typecheck command configured in `pyproject.toml` — the eval suite
(`evals/run_evals.py`) is the correctness check for this project, and its governance red-team
gate (`governance_redteam` evaluator) must always score 100%; a single leak is a hard failure
(`_run_local` raises `SystemExit(1)` if any governance example scores 0).

Config is env-driven via `.env` (loaded through `python-dotenv` in `config.py`): `DAA_TODAY`,
`DAA_STRONG_MODEL`, `DAA_CHEAP_MODEL`, `DAA_MAX_QUERY_ATTEMPTS`, `DAA_SQL_TIMEOUT_S`,
`DAA_CONTROL_TOTAL_TOLERANCE`, plus `ANTHROPIC_API_KEY` and the optional `LANGSMITH_*` tracing
vars.

## Architecture

### The graph (`graph.py`)

A `StateGraph` over `AgentState` (`state.py`) implementing "loop, not one-shot": generate SQL →
look at what happened → decide retry/give-up/proceed, bounded by `MAX_QUERY_ATTEMPTS`.

```
plan_intent → [clarify → END] | [ground → generate_sql → governance_gate
  → (retry generate_sql | execute_sql) → (retry generate_sql | critic)
  → (retry generate_sql | analyze) → compose_answer → END]
```

Every node transition is a LangSmith trace span for free when tracing is enabled. When adding a
new node, wire both the node function and, if it can fail/retry, a `route_after_*` conditional
edge function — the existing nodes (`governance_node.py`, `execution.py`, `critique.py`) are the
templates.

### Governance is structural, not prompted (`governance.py`)

This is the load-bearing design decision in the codebase: access control and PII masking are
enforced by SQL views the model never sees, not by prompt instructions the model could ignore or
be injected around.

- `secured_connection(user)` opens a DuckDB connection and creates per-user **temp views**
  (`orders`, `customers`, `order_items`, `refunds`, `regions`, `agg_daily_revenue`) scoped to
  `user.allowed_regions`, with `customers.email` masked to `'REDACTED'` unless `user.pii_allowed`.
  The underlying `raw_*` tables are never exposed to the agent.
- `check_and_prepare(sql)` is a second, independent static check (belt-and-suspenders) run before
  execution: single statement, `SELECT`/`WITH` only, no forbidden DDL/DML keywords, no direct
  `raw_*` table reference, and injects a `LIMIT MAX_ROWS` if missing.
- This module has zero LLM calls — it's the enforcement layer nothing upstream can talk its way
  around. Any change here is security-relevant; re-run `uv run python -m evals.run_evals --local`
  and confirm the `governance_redteam` gate still passes 100%.
- Two demo users encode the two axes of entitlement: `alice` (US-only, no PII) and `bob` (all
  regions, PII allowed) — see `USERS` in `governance.py`.

### Semantic layer + schema catalog (`semantic_layer.py`, `warehouse/schema_catalog.py`)

Two-tier grounding for SQL generation, assembled by `nodes/grounding.py`:

1. **Semantic layer** (`METRICS` in `semantic_layer.py`) — governed, pre-validated metric
   definitions (currently `revenue`, `order_count`) the model should compose from rather than
   reinvent. Deliberately incomplete: this is the intended coverage gap the design leans on.
2. **Schema catalog** (`CATALOG`/`JOIN_GRAPH` in `schema_catalog.py`) — fallback for anything
   outside semantic-layer coverage: described columns with sample values and a validated join
   graph, so the model isn't inventing joins or guessing at categorical values.

Both describe only the secured views, never the `raw_*` tables.

### The synthetic warehouse (`warehouse/seed.py`)

Generates a DuckDB file (`data/warehouse.duckdb`, gitignored) reproducing the specific failure
shapes data-analyst agents fall into, on purpose:

- **Fan-out trap**: `orders` → `order_items` is one-to-many; summing `orders.total_amount` after
  that join without re-aggregating first inflates revenue. `raw_agg_daily_revenue` is a
  materialized control total built independently (straight from `raw_orders`/`raw_refunds`,
  never through `raw_order_items`) precisely so it can't inherit the same bug, and the critic
  reconciles against it.
- **Refund netting**: net revenue = orders total minus refunds, aggregated to one row per
  `order_id` *before* joining (an order with multiple partial refunds must not fan out orders).
- **`is_test` flag**: ~2% of accounts/orders are internal test data that must be excluded from
  every metric; the secured views filter it out so the model doesn't have to remember to.
- **A real, localized anomaly**: APAC orders are deliberately suppressed in March 2026, so
  "why did revenue drop" questions have a genuine, decomposable answer.

Regenerating the warehouse (`--rebuild-warehouse` or `warehouse/seed.py` directly) uses a fixed
seed (`seed=7`), so output is deterministic unless the generation logic itself changes.

### The critic: "ran" is not "right" (`critic.py`)

A separate verification pass between execution and answering. Deterministic checks run first
(cheap, no LLM): row-count sanity, null-rate sanity, and — for revenue metrics — reconciliation
against the `agg_daily_revenue` control total within `CONTROL_TOTAL_TOLERANCE`. A relative
difference beyond tolerance is treated as the signature of a fan-out bug or bad filter and blocks
with actionable feedback fed back into `generate_sql` on retry. Only after deterministic checks
pass does an LLM check (`llm_matches_intent`, strong model) verify the SQL actually answers the
question as asked — a query can execute cleanly and still answer the wrong question.

### Tiered model routing (`config.py`, `llm.py`)

Correctness-critical steps (`generate_sql`, `critic`'s `llm_matches_intent`) use
`get_strong_llm()`; everything else (`plan_intent`, `compose_answer`'s narrative) uses
`get_cheap_llm()`. When adding a new LLM-calling node, decide which tier it belongs to using this
same criterion — is a mistake here directly a wrong-number risk, or is it downstream narration.

### SQL/Python split (`nodes/analyze.py`)

The warehouse does all filtering/joining/aggregating; Python only reshapes, charts, or computes
simple derived stats (period-over-period comparison, a bar chart when a `breakdown_dimension` was
requested) on the already-small result set `execute_sql` returned. No raw big-data table scan
logic belongs in `analyze.py` or `compose.py`.

### Typed errors drive retry routing (`errors.py`)

`GovernanceError`, `QueryTimeoutError`, `SQLExecutionError` all subclass `AgentError`. Nodes catch
`AgentError` and turn it into `retry_feedback` text fed back into the next `generate_sql` call
rather than letting the graph crash — this is how the model "sees" its own SQL errors and
timeouts. A new failure mode that should be retryable-with-feedback should be a new `AgentError`
subclass, not a bare exception.

### Evals (`evals/`)

`evals/dataset.py` encodes three example kinds matching the failure modes above: `kind:
execution_accuracy` (agent's numeric result must match a gold SQL query within
`EXECUTION_ACCURACY_TOLERANCE`), `kind: clarification`/`no_clarification` (must clarify on
genuinely ambiguous questions, must not nag on crisp ones), and `kind: governance_redteam_region`/
`governance_redteam_pii` (must fail closed regardless of what SQL the model generates — scored as
a pass/fail gate, not a graded metric). `evals/target.py` wraps the compiled graph as a plain
`dict -> dict` function so it runs identically under LangSmith's `evaluate()` or the local runner.
