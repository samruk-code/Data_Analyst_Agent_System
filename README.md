# Data Analyst Agent

An NL-to-SQL agent over a DuckDB sales warehouse that treats "the LLM wrote correct, safe SQL" as
something to verify, not assume.

Most NL-to-SQL demos stop at "the query ran without error." This one is built around a different
question: **would you let this answer go to a VP without a human checking it first?** That means
access control the model can't override even if it wanted to, a verification pass that catches
queries which execute cleanly but answer the wrong question, and an eval suite that treats a
governance leak as a hard failure, not a metric to improve over time.

## Why this design

Three failure modes make NL-to-SQL agents unsafe or unreliable in practice, and this project
builds a specific defense for each:

- **A cooperative model is not an access-control layer.** Even a well-behaved LLM will
  occasionally forget a `WHERE region = ...` filter, or a prompt injection could try to talk it
  out of one. So access control isn't a filter the model is asked to remember — it's enforced by
  SQL views the model queries against, built fresh per request from the requesting user's
  entitlements. If a user isn't allowed to see EU rows or raw email addresses, those values are
  never in the result set the model could return, regardless of what SQL it writes. See
  [`governance.py`](data_analyst_agent/governance.py).
- **"The query ran" is not "the query is right."** The classic silent-failure mode in this domain
  is a one-to-many join (orders → line items, orders → refunds) that fans out a `SUM()` and
  quietly inflates a number, or a metric definition (net vs. gross revenue) applied inconsistently
  across queries. A critic pass reconciles every revenue query against an independently computed
  control total and blocks + retries with specific feedback when the numbers don't reconcile,
  before a human ever sees the answer. See [`critic.py`](data_analyst_agent/critic.py).
- **Ambiguity should be resolved by asking, not guessing.** "How are sales doing?" has no single
  correct SQL query — the agent detects when a question's meaning would genuinely change based on
  an unstated choice (which metric, which timeframe, which scope) and asks, rather than picking an
  interpretation silently. It does *not* nag on questions that are already crisp.

## Architecture

A [LangGraph](https://github.com/langchain-ai/langgraph) state machine implementing "loop, not
one-shot": generate SQL, look at what happened, decide retry / give up / proceed — bounded by a
max attempt budget, with every node transition available as a LangSmith trace span.

```
question
   │
   ▼
plan_intent ──ambiguous──▶ clarify ──▶ (ask the user, stop)
   │ crisp
   ▼
ground (semantic layer + schema catalog)
   │
   ▼
generate_sql ◀────────────────────────────┐ (retry with feedback)
   │                                       │
   ▼                                       │
governance_gate ──rejected────────────────┤
   │ ok                                    │
   ▼                                       │
execute_sql ──warehouse error──────────────┤
   │ ok                                    │
   ▼                                       │
critic ──fails reconciliation/intent check─┘
   │ passes
   ▼
analyze (chart, period comparison — Python over the small result set only)
   │
   ▼
compose_answer (headline + plain-English explanation + the exact SQL, shown, not hidden)
```

Every retry loop feeds the *specific* failure back into `generate_sql` as context — a hallucinated
column, a governance rejection, a control-total mismatch — rather than just asking the model to
"try again."

### Key components

| Component | Role |
|---|---|
| [`governance.py`](data_analyst_agent/governance.py) | Per-user secured DuckDB views (row-level security + PII masking) and a static SQL safety check. Zero LLM calls — this is the enforcement layer, not a prompt. |
| [`semantic_layer.py`](data_analyst_agent/semantic_layer.py) | Governed metric definitions (`revenue`, `order_count`) the agent composes from instead of reinventing business logic in raw SQL. Deliberately incomplete — coverage gaps fall back to schema-grounded generation. |
| [`warehouse/schema_catalog.py`](data_analyst_agent/warehouse/schema_catalog.py) | Described tables/columns with sample values and a validated join graph, for the questions the semantic layer doesn't cover. |
| [`critic.py`](data_analyst_agent/critic.py) | Deterministic checks (row count, null rate, control-total reconciliation) plus an LLM check that the SQL actually answers the question asked. |
| [`warehouse/seed.py`](data_analyst_agent/warehouse/seed.py) | A synthetic warehouse engineered to reproduce real failure shapes: a fan-out-prone join, refunds that must be netted and pre-aggregated, an `is_test` flag every metric must exclude, and a materialized control total built independently of the detail tables. |
| [`config.py`](data_analyst_agent/config.py) / [`llm.py`](data_analyst_agent/llm.py) | Tiered model routing — a strong model for query generation and critique (correctness-critical), a cheap model for intent parsing and narrative composition. |
| [`evals/`](evals/) | Execution accuracy, clarify-vs-guess behavior, and a governance red-team suite that must score 100% — a single PII or region leak fails the run. |

## Quickstart

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/). Set `ANTHROPIC_API_KEY` in a `.env`
file (see `.env.example` if present, or just export it).

```bash
# First run builds the synthetic warehouse automatically
uv run data-analyst-agent "What was net revenue in the EU last quarter?"

# Run as a different demo user to see row-level security / PII masking in action
uv run data-analyst-agent --user alice "List customer emails"

# Interactive mode
uv run data-analyst-agent
```

Example output:

```
  Net revenue in the EU last quarter was $412,318.06.

  This is calculated as the sum of order totals for EU orders placed between
  2026-01-01 and 2026-04-01, minus refunds issued against those orders.

  SQL (auditable -- this is exactly what ran):
  SELECT ROUND(SUM(o.total_amount) - COALESCE(SUM(rt.refund_total), 0), 2) AS net_revenue
  FROM orders o
  LEFT JOIN (
      SELECT order_id, SUM(refund_amount) AS refund_total
      FROM refunds GROUP BY order_id
  ) rt ON rt.order_id = o.order_id
  WHERE o.region = 'EU'
    AND o.order_date >= DATE '2026-01-01' AND o.order_date < DATE '2026-04-01'
```

Two demo users are wired up to make the governance boundary visible: `bob` (global finance lead,
all regions, PII allowed) and `alice` (US-only regional analyst, no PII access). Asking `alice` for
EU revenue or customer emails demonstrates the enforcement — the underlying rows are structurally
absent from what she can query, not filtered after the fact.

## Evals

```bash
uv run python -m evals.run_evals            # uploads to LangSmith if LANGSMITH_API_KEY is set
uv run python -m evals.run_evals --local    # local run, prints a results table
```

The eval dataset ([`evals/dataset.py`](evals/dataset.py)) encodes the failure modes above directly:
execution accuracy against a gold query for questions with the classic fan-out trap, clarify-vs-guess
behavior on ambiguous vs. crisp questions, and governance red-team probes (asking a
region-restricted, PII-restricted user for exactly the data they shouldn't get). The governance
red-team gate is pass/fail, not a graded score — any leak fails the run.

## Configuration

Environment variables (loaded from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. |
| `DAA_STRONG_MODEL` | `claude-opus-5` | Model for query generation and critique. |
| `DAA_CHEAP_MODEL` | `claude-haiku-4-5` | Model for intent parsing and narrative composition. |
| `DAA_TODAY` | `2026-07-28` | Fixed "today" for resolving relative timeframes against the warehouse's synthetic date range. |
| `DAA_MAX_QUERY_ATTEMPTS` | `3` | Retry budget for the generate → govern → execute → critique loop. |
| `DAA_SQL_TIMEOUT_S` | `10` | Wall-clock execution budget per query. |
| `DAA_CONTROL_TOTAL_TOLERANCE` | `0.15` | Relative-difference threshold for the critic's revenue reconciliation check. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | — | Optional tracing/eval upload. |

## Known limitations

- No CI pipeline yet — the eval suite isn't run automatically on push.
- No unit tests for the pure/deterministic logic (governance checks, critic math) independent of
  the eval suite, which requires an API key and hits real models.
- The warehouse is synthetic and single-file (DuckDB); it demonstrates the governance and
  reliability patterns rather than warehouse-scale performance.
- LLM calls have no retry/backoff around transient API errors.

## License

No license file is currently included; treat this as source-available for reference rather than
under an open license until one is added.
