"""CLI entry point for the code debug agent."""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()


DEFAULT_TASK = (
    "Fix the failing tests in the demo workspace. "
    "The multiply function in calculator.py has a bug."
)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Code Debug AI Agent — vertical slice demo")
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help="Task for the agent to perform",
    )
    args = parser.parse_args()
    asyncio.run(run(args.task))


async def run(task: str) -> None:
    from code_debug_agent.graph import run_workflow

    print("=" * 60)
    print("Code Debug AI Agent")
    print("Flow: User → Supervisor → Coding (MCP) → Testing (MCP)")
    print("=" * 60)
    print(f"\nTask: {task}\n")

    try:
        result = await run_workflow(task)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Supervisor Plan ---")
    print(result.get("plan", "(none)"))

    print("\n--- Coding Result ---")
    print(result.get("coding_result", "(none)"))

    print("\n--- Testing Result ---")
    print(result.get("testing_result", "(none)"))

    print("\n--- Summary ---")
    if result.get("tests_passed"):
        print("Status: SUCCESS — all tests passed")
    else:
        retries = result.get("retry_count", 0)
        print(f"Status: INCOMPLETE — tests did not pass after {retries} retries")

    sys.exit(0 if result.get("tests_passed") else 1)


if __name__ == "__main__":
    cli()
