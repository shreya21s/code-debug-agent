import logging
from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.graph.routing import route_next_node

logger = logging.getLogger(__name__)

def get_supervisor_node():
    from app.agents.supervisor import supervisor_node
    return supervisor_node


def get_research_node():
    from app.agents.research_agent import research_node
    return research_node


def get_coding_node():
    from app.agents.coding_agent import coding_node
    return coding_node


def get_testing_node():
    from app.agents.testing_agent import testing_node
    return testing_node


def get_reviewer_node():
    from app.agents.reviewer_agent import reviewer_node
    return reviewer_node


def final_response_node(state: AgentState) -> dict:
    """End node that outputs the final synthesized response."""
    logger.info("Executing final_response node.")

    task_success = bool(state.get("task_success", False))
    iteration_val = state.get("iteration_count")
    iteration = int(iteration_val) if iteration_val is not None else 0
    errors = list(state.get("errors") or [])

    is_successful = task_success or (
        len(errors) == 0 and state.get("next_agent") in ("complete", "final_response", "")
    )

    errors_to_add = []
    if iteration >= 10 and not is_successful:
        err_msg = "Iteration limit (10) reached without successful completion."
        errors_to_add.append(err_msg)
        errors.extend(errors_to_add)

    if is_successful:
        content = f"TASK SUCCESS. Final execution complete. Iterations run: {iteration}."
    else:
        content = "TASK FAILED. Errors encountered:\n" + "\n".join(f"- {err}" for err in errors)

    res = {
        "task_success": is_successful,
        "messages": [{
            "role": "assistant",
            "content": content,
        }],
    }
    if errors_to_add:
        res["errors"] = errors_to_add
    return res


def create_graph():
    """Constructs and compiles the multi-agent LangGraph workflow."""
    # pyrefly: ignore [bad-specialization]
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor_agent", get_supervisor_node())
    workflow.add_node("research_agent", get_research_node())
    workflow.add_node("coding_agent", get_coding_node())
    workflow.add_node("testing_agent", get_testing_node())
    workflow.add_node("reviewer_agent", get_reviewer_node())
    workflow.add_node("final_response", final_response_node)

    workflow.add_edge(START, "supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        route_next_node,
        {
            "research_agent": "research_agent",
            "coding_agent": "coding_agent",
            "testing_agent": "testing_agent",
            "reviewer_agent": "reviewer_agent",
            "supervisor_agent": "supervisor_agent",
            "final_response": "final_response",
        },
    )

    workflow.add_edge("research_agent", "supervisor_agent")
    workflow.add_edge("coding_agent", "supervisor_agent")
    workflow.add_edge("testing_agent", "supervisor_agent")
    workflow.add_edge("reviewer_agent", "supervisor_agent")
    workflow.add_edge("final_response", END)

    return workflow.compile()
