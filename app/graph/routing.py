import logging
from typing import Literal
from app.state import AgentState

logger = logging.getLogger(__name__)

# Max iteration safety threshold
MAX_ITERATIONS = 10

def route_next_node(state: AgentState) -> Literal["research_agent", "coding_agent", "testing_agent", "reviewer_agent", "supervisor_agent", "final_response"]:
    """
    Determines which agent to route to based on state.next_agent and iteration_count.
    """
    iteration = state.get("iteration_count", 0)
    next_agent = state.get("next_agent", "")
    
    # 1. Prevent infinite loops
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Iteration limit ({MAX_ITERATIONS}) reached. Routing directly to final_response.")
        # Modify state to include error (in LangGraph we do this in the node, but we can also log/warn here)
        return "final_response"
        
    # 2. Check routing direction
    if next_agent == "research":
        return "research_agent"
    elif next_agent == "coding":
        return "coding_agent"
    elif next_agent == "testing":
        return "testing_agent"
    elif next_agent == "reviewer":
        return "reviewer_agent"
    elif next_agent == "supervisor":
        return "supervisor_agent"
    elif next_agent in ("complete", "final_response"):
        return "final_response"
        
    # Fallback to supervisor to figure out next steps
    logger.info(f"Unknown next_agent '{next_agent}'. Defaulting back to supervisor_agent.")
    return "supervisor_agent"
