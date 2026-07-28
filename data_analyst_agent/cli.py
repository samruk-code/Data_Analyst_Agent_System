"""CLI: ask the data analyst agent a question.

    uv run data-analyst-agent "What was net revenue in the EU last quarter?"
    uv run data-analyst-agent --user alice "List customer emails"
    uv run data-analyst-agent   # interactive mode
"""

from __future__ import annotations

import argparse

from data_analyst_agent.governance import USERS, get_user
from data_analyst_agent.graph import run_analysis
from data_analyst_agent.state import Answer
from data_analyst_agent.warehouse.seed import build_warehouse


def _print_answer(answer: Answer) -> None:
    print()
    print(f"  {answer.headline}")
    print()
    if not answer.needs_clarification:
        print(answer.explanation)
        if answer.chart and answer.chart.path:
            print(f"\n[chart saved to {answer.chart.path}]")
        if answer.caveats:
            print("\nCaveats:")
            for c in answer.caveats:
                print(f"  - {c}")
        if answer.sql:
            print("\nSQL (auditable -- this is exactly what ran):")
            print(answer.sql)
    else:
        print(answer.explanation)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Data analyst agent over a synthetic sales warehouse.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument(
        "--user", default="bob", choices=list(USERS), help="Which demo user to run as (governs RLS/PII)."
    )
    parser.add_argument("--rebuild-warehouse", action="store_true", help="Rebuild the synthetic warehouse.")
    args = parser.parse_args()

    build_warehouse(force=args.rebuild_warehouse)
    user = get_user(args.user)

    if args.question:
        _print_answer(run_analysis(args.question, user))
        return

    print(f"Data analyst agent -- running as {user.display_name}. Ctrl-C or empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            _print_answer(run_analysis(question, user))
        except Exception as exc:  # noqa: BLE001 - CLI top-level guard
            print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
