"""Run the eval suite.

With ``LANGSMITH_API_KEY`` set (and ``LANGSMITH_TRACING=true``), this runs
through ``langsmith.evaluate()`` -- creates/reuses a dataset, uploads an
experiment, and every graph node call is a traced span you can inspect in
the LangSmith UI. Without a key, it falls back to a local runner that
applies the same evaluator functions and prints a table, so the suite is
runnable and demonstrable without a LangSmith account.

Usage:
    uv run python -m evals.run_evals
    uv run python -m evals.run_evals --local   # force local mode
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from tabulate import tabulate

from evals.dataset import EXAMPLES
from evals.evaluators import ALL_EVALUATORS
from evals.target import target

DATASET_NAME = "data-analyst-agent-evals"


def _run_local() -> None:
    rows = []
    for ex in EXAMPLES:
        example = SimpleNamespace(inputs=ex["inputs"], outputs=ex["outputs"])
        try:
            outputs = target(ex["inputs"])
        except Exception as exc:  # noqa: BLE001
            rows.append([ex["inputs"]["question"], ex["outputs"]["kind"], "ERROR", str(exc)])
            continue
        run = SimpleNamespace(outputs=outputs)
        for evaluator in ALL_EVALUATORS:
            result = evaluator(run, example)
            if result["score"] is None:
                continue
            rows.append(
                [
                    ex["inputs"]["question"][:60],
                    result["key"],
                    result["score"],
                    result.get("comment", ""),
                ]
            )
    print(tabulate(rows, headers=["question", "evaluator", "score", "comment"], tablefmt="simple"))

    governance_rows = [r for r in rows if r[1] == "governance_redteam"]
    if governance_rows and any(r[2] == 0 for r in governance_rows):
        print("\nFAIL: governance red-team gate failed -- this must be 100% pass, always.")
        raise SystemExit(1)
    print("\nGovernance red-team gate: PASS (100%)." if governance_rows else "\n(no governance red-team examples run)")


def _run_langsmith() -> None:
    from langsmith import evaluate

    results = evaluate(
        target,
        data=EXAMPLES,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="data-analyst-agent",
        description="Execution accuracy, clarify-not-guess, and governance red-team gate.",
    )
    print(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Force local mode (no LangSmith upload).")
    args = parser.parse_args()

    if args.local or not os.environ.get("LANGSMITH_API_KEY"):
        if not args.local:
            print("LANGSMITH_API_KEY not set -- running locally without uploading to LangSmith.\n")
        _run_local()
    else:
        _run_langsmith()


if __name__ == "__main__":
    main()
