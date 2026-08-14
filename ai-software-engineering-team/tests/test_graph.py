import pytest
from app.state import create_initial_state, AgentState
from app.graph.routing import route_next_node, MAX_ITERATIONS
from app.graph.workflow import create_graph

def test_initial_state():
    """Test initial state creation helper."""
    state = create_initial_state("Verify logins", "/path/to/repo")
    assert state["user_request"] == "Verify logins"
    assert state["repository_path"] == "/path/to/repo"
    assert state["iteration_count"] == 0
    assert len(state["completed_tasks"]) == 0
    assert len(state["errors"]) == 0

def test_routing_logic():
    """Test routing based on state values."""
    # Test fallback
    state = create_initial_state("Verify logins", "/path/to/repo")
    assert route_next_node(state) == "supervisor_agent"
    
    # Test explicit routes
    state["next_agent"] = "research"
    assert route_next_node(state) == "research_agent"
    
    state["next_agent"] = "coding"
    assert route_next_node(state) == "coding_agent"
    
    state["next_agent"] = "testing"
    assert route_next_node(state) == "testing_agent"
    
    state["next_agent"] = "reviewer"
    assert route_next_node(state) == "reviewer_agent"
    
    state["next_agent"] = "complete"
    assert route_next_node(state) == "final_response"

def test_routing_loop_prevention():
    """Test that routing to final_response occurs if max iterations exceeded."""
    state = create_initial_state("Verify logins", "/path/to/repo")
    state["iteration_count"] = MAX_ITERATIONS
    state["next_agent"] = "research"  # Should be ignored due to loop prevention
    assert route_next_node(state) == "final_response"

def test_full_graph_execution():
    """Runs the compiled graph locally in mock mode and verifies it reaches final_response."""
    graph = create_graph()
    initial_state = create_initial_state("Verify login fails", "/path/to/repo")
    
    result = graph.invoke(initial_state)
    
    # Check that it executed through all agents and finished
    assert result["iteration_count"] > 0
    assert "T1" in result["completed_tasks"]
    assert "T2" in result["completed_tasks"]
    assert "T3" in result["completed_tasks"]
    assert "T4" in result["completed_tasks"]
    assert result["next_agent"] == "complete"
    
    # Check that mock logs and results accumulated in state
    assert "Research Diagnosis" in result["research_results"]
    assert "calculator.py" in result["code_changes"]
    assert "Test Success" in result["test_results"]
    assert "Approved" in result["review_results"]
    
    # Check messages accumulated
    assert len(result["messages"]) == 5 # 4 subagents + 1 final response message
