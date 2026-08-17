import logging
from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.graph.routing import route_next_node

logger = logging.getLogger(__name__)

# Lazy load nodes/agents to avoid circular imports later on
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
    return {
        "messages": [{
            "role": "assistant",
            "content": f"Final execution complete. Iterations run: {state.get('iteration_count', 0)}"
        }]
    }

def create_graph():
    """Constructs and compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("supervisor_agent", get_supervisor_node())
    workflow.add_node("research_agent", get_research_node())
    workflow.add_node("coding_agent", get_coding_node())
    workflow.add_node("testing_agent", get_testing_node())
    workflow.add_node("reviewer_agent", get_reviewer_node())
    workflow.add_node("final_response", final_response_node)
    
    # Define Core Routing & Edges
    workflow.add_edge(START, "supervisor_agent")
    
    # The Supervisor node uses conditional routing
    workflow.add_conditional_edges(
        "supervisor_agent",
        route_next_node,
        {
            "research_agent": "research_agent",
            "coding_agent": "coding_agent",
            "testing_agent": "testing_agent",
            "reviewer_agent": "reviewer_agent",
            "supervisor_agent": "supervisor_agent",
            "final_response": "final_response"
        }
    )
    
    # Subagents always return to the Supervisor after completing their execution
    workflow.add_edge("research_agent", "supervisor_agent")
    workflow.add_edge("coding_agent", "supervisor_agent")
    workflow.add_edge("testing_agent", "supervisor_agent")
    workflow.add_edge("reviewer_agent", "supervisor_agent")
    
    # Final response goes to END
    workflow.add_edge("final_response", END)
    
    return workflow.compile()
