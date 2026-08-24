import operator
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State representing the current execution context of the AI Software Engineering graph."""

    user_request: str
    repository_path: str

    target_file: Optional[str]
    research_files: List[str]
    relevant_files: List[str]
    coding_attempts: int
    task_success: bool

    plan: Dict[str, Any]
    current_task: str
    completed_tasks: List[str]
    iteration_count: int

    research_results: str
    code_changes: str
    test_results: str
    review_results: str

    next_agent: str

    messages: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    _a2a_execution: bool


def create_initial_state(
    user_request: str,
    repository_path: str,
    target_file: Optional[str] = None,
) -> AgentState:
    """Helper to initialize a clean, fully-populated AgentState."""
    return {
        "user_request": user_request,
        "repository_path": repository_path,
        "target_file": target_file,
        "research_files": [],
        "relevant_files": [],
        "coding_attempts": 0,
        "task_success": False,
        "plan": {},
        "current_task": "",
        "completed_tasks": [],
        "iteration_count": 0,
        "research_results": "",
        "code_changes": "",
        "test_results": "",
        "review_results": "",
        "next_agent": "",
        "messages": [],
        "errors": [],
        "_a2a_execution": False,
    }
