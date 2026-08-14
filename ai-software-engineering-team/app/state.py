import operator
from typing import TypedDict, List, Dict, Any, Annotated

class AgentState(TypedDict):
    """
    State representing the current execution context of the AI Software Engineering graph.
    """
    # Inputs & Global Settings
    user_request: str
    repository_path: str
    
    # Task planning and iteration tracking
    plan: Dict[str, Any]  # Dictionary detailing the planned tasks
    current_task: str
    completed_tasks: List[str]
    iteration_count: int
    
    # Subagent outputs
    research_results: str
    code_changes: str
    test_results: str
    review_results: str
    
    # State routing & error handling
    next_agent: str
    
    # Lists that accumulate during the execution path
    messages: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]


def create_initial_state(user_request: str, repository_path: str) -> AgentState:
    """Helper to initialize a clean AgentState."""
    return {
        "user_request": user_request,
        "repository_path": repository_path,
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
        "errors": []
    }
