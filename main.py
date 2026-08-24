import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from typing import Any
from app.state import create_initial_state
from app.graph.workflow import create_graph
from app.config import WORKSPACE_ROOT, MODEL_NAME
from app.agents.research_agent import extract_target_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def resolve_run_config(argv: list[str] | None = None) -> tuple[str, str, str | None]:
    """Parse CLI args into (user_request, repository_path, target_file).

    A path to an existing file is treated as the buggy file to analyze/fix/review.
    Remaining non-file arguments become the goal text.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    file_paths: list[Path] = []
    texts: list[str] = []

    for item in args:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        if candidate.is_file():
            file_paths.append(candidate.resolve())
        else:
            texts.append(item)

    if file_paths:
        target = file_paths[0]
        repo_path = str(target.parent)
        target_file = target.name
        user_request = " ".join(texts).strip()
        if not user_request:
            user_request = (
                f"Analyze, debug, modify, and review '{target_file}'. "
                "Find the bugs, apply a correct fix, then verify."
            )
        elif target_file not in user_request:
            user_request = f"{user_request} (target file: {target_file})"
        return user_request, repo_path, target_file

    user_request = " ".join(texts).strip()
    if not user_request:
        user_request = "Fix the multiply bug in calculator.py and verify tests pass"

    repo_path = str(WORKSPACE_ROOT)
    target_file = extract_target_file(user_request)
    return user_request, repo_path, target_file


def main():
    print("=" * 50)
    print("       AI SOFTWARE ENGINEERING TEAM")
    print("=" * 50)

    user_request, repo_path, target_file = resolve_run_config()
    os.environ["WORKSPACE_ROOT"] = repo_path

    print(f"Target Repository: {repo_path}")
    if target_file:
        print(f"Target File: {target_file}")
    print(f"Goal: {user_request}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 50)

    graph = create_graph()
    initial_state = create_initial_state(user_request, repo_path, target_file=target_file)

    print("Initializing multi-agent graph execution...")
    print("Flow: research (analyze) -> coding (debug/modify) -> testing -> reviewer")
    try:
        final_state: dict[str, Any] = dict(initial_state)
        events = graph.stream(initial_state)
        for event in events:
            for node_name, state_update in event.items():
                print(f"\n>>> Node [{node_name}] Executed.")
                for key, value in state_update.items():
                    if key in ("messages", "errors"):
                        final_state[key] = final_state.get(key, []) + value
                    else:
                        final_state[key] = value

                if "next_agent" in state_update:
                    print(f"    Next Routing Decision: {state_update['next_agent']}")
                if "current_task" in state_update:
                    print(f"    Current Active Task: {state_update['current_task']}")
                if "completed_tasks" in state_update:
                    print(f"    Completed Tasks: {state_update['completed_tasks']}")
                if "research_results" in state_update:
                    print(f"    Research Result: {state_update['research_results']}")
                if "code_changes" in state_update:
                    print(f"    Code Changes: {state_update['code_changes']}")
                if "test_results" in state_update:
                    print(f"    Test Results: {state_update['test_results']}")
                if "review_results" in state_update:
                    print(f"    Review Results: {state_update['review_results']}")

        task_success = final_state.get("task_success", False)
        print("\n" + "=" * 50)
        if task_success:
            print("Graph run successfully completed.")
            print("=" * 50)
            status_code = 0
        else:
            print("TASK FAILED.")
            errors = final_state.get("errors", [])
            if errors:
                print("Errors encountered:")
                for err in errors:
                    print(f" - {err}")
            print("=" * 50)
            status_code = 1

        from app.utils.logging import tracer
        print("\n" + tracer.generate_report())
        sys.exit(status_code)
    except Exception:
        logger.exception("An error occurred during graph execution:")
        sys.exit(1)


if __name__ == "__main__":
    main()
