import os
from pathlib import Path
import pytest

from app.state import create_initial_state
from app.agents.research_agent import research_node
from app.agents.coding_agent import coding_node
from app.agents.testing_agent import run_tests_subprocess
from app.agents.reviewer_agent import reviewer_node

def test_research_agent_node(tmp_path):
    """Test that Research node executes and returns expected fields."""
    repo = tmp_path / "mock_repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')", encoding="utf-8")
    
    state = create_initial_state("Find printing bugs", str(repo))
    state["current_task"] = "Analyze printing flow"
    state["plan"] = {"current_task_id": "T1"}
    
    res = research_node(state)
    
    assert "research_results" in res
    assert "T1" in res["completed_tasks"]
    assert len(res["messages"]) == 1
    assert res["messages"][0]["role"] == "researcher"
    assert res["next_agent"] == "supervisor"

def test_coding_agent_path_traversal(tmp_path):
    """Verify path traversal prevention in Coding Agent."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    
    state = create_initial_state("Fix bug", str(repo))
    state["current_task"] = "Edit file"
    state["plan"] = {"current_task_id": "T2"}
    
    # We will temporarily mock get_llm inside coding_agent to return an out-of-bounds path
    from app.agents import coding_agent
    from app.agents.coding_agent import CodeChange
    from langchain_core.runnables import Runnable
    
    class MockLLM(Runnable):
        def invoke(self, input, config=None, **kwargs):
            return CodeChange(
                reasoning="Attempting path traversal",
                file_path="../outside_file.py",  # Traversal path!
                new_content="print('hack')"
            )
    
    original_get_llm = coding_agent.get_llm
    coding_agent.get_llm = lambda: MockLLM()
    
    try:
        res = coding_node(state)
        # Check that error is captured and traversal blocked
        assert len(res.get("errors", [])) > 0
        assert "Path traversal attempt blocked" in res["errors"][0]
        assert res["next_agent"] == "supervisor"
    finally:
        coding_agent.get_llm = original_get_llm

def test_coding_agent_node(tmp_path):
    """Verify Coding Agent writes correct file changes inside workspace."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    
    state = create_initial_state("Fix calculator multiply", str(repo))
    state["current_task"] = "Apply multiplication fix"
    state["plan"] = {"current_task_id": "T2"}
    
    original_root = os.getenv("WORKSPACE_ROOT")
    try:
        res = coding_node(state)
        
        assert "code_changes" in res
        assert "T2" in res["completed_tasks"]
        assert res["next_agent"] == "supervisor"
        
        # Check that calculator.py was written inside sandbox folder
        calc_file = repo / "calculator.py"
        assert calc_file.exists()
        assert "Fixed multiply bug" in calc_file.read_text()
    finally:
        if original_root:
            os.environ["WORKSPACE_ROOT"] = original_root
        else:
            os.environ.pop("WORKSPACE_ROOT", None)

def test_testing_agent_node(tmp_path):
    """Test Testing Agent node execution."""
    from app.agents.testing_agent import testing_node
    repo = tmp_path / "sandbox"
    repo.mkdir()
    
    # Write a passing pytest file
    (repo / "test_dummy.py").write_text("def test_dummy():\n    assert True\n", encoding="utf-8")
    
    state = create_initial_state("Run pytest", str(repo))
    state["current_task"] = "Verify test suites"
    state["plan"] = {"current_task_id": "T3"}
    
    res = testing_node(state)
    
    assert "test_results" in res
    assert "T3" in res["completed_tasks"]
    assert "Test Success: True" in res["test_results"]
    assert res["next_agent"] == "supervisor"

def test_reviewer_agent_node():
    """Verify Reviewer node approves passing tests and rejects failing tests."""
    state = create_initial_state("Review", "/path")
    state["plan"] = {"current_task_id": "T4"}
    
    # 1. Case where tests passed
    state["test_results"] = "Test Success: True\nSummary: Passed all 3 tests"
    res1 = reviewer_node(state)
    assert "Approved: True" in res1["review_results"]
    assert len(res1.get("errors", [])) == 0
    
    # 2. Case where tests failed
    state["test_results"] = "Test Success: False\nSummary: Failed 1 test"
    res2 = reviewer_node(state)
    assert "Approved: False" in res2["review_results"]
    assert len(res2.get("errors", [])) > 0
    assert "Reviewer rejected" in res2["errors"][0]

def test_a2a_client_server_integration():
    """Verify A2A client-server HTTP communication and task delegation."""
    import time
    import threading
    import uvicorn
    from app.a2a.server import app as a2a_app
    import app.a2a.client as client
    
    # Custom Uvicorn server runner that runs in a background thread
    class TestServer(uvicorn.Server):
        def install_signal_handlers(self):
            pass
            
    config = uvicorn.Config(a2a_app, host="127.0.0.1", port=8009, log_level="warning")
    server = TestServer(config)
    
    # Start server in thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    # Wait briefly for server to boot
    time.sleep(0.5)
    
    original_research_port = client.A2A_RESEARCH_PORT
    original_reviewer_port = client.A2A_REVIEWER_PORT
    
    from app.agents import research_agent, reviewer_agent
    original_research_llm = research_agent.get_llm
    original_reviewer_llm = reviewer_agent.get_llm
    research_agent.get_llm = lambda: None
    reviewer_agent.get_llm = lambda: None
    
    try:
        # Route A2A client calls to our test server port
        client.A2A_RESEARCH_PORT = 8009
        client.A2A_REVIEWER_PORT = 8009
        
        # Test capabilities query
        caps = client.query_capabilities(8009)
        assert caps is not None
        assert len(caps) >= 2
        
        # Test task delegation (Research agent)
        state = create_initial_state("Analyze login bug", "/path/to/repo")
        state["current_task"] = "Locate login script"
        state["plan"] = {"current_task_id": "T1"}
        
        res = client.call_a2a_agent("research", state, task_id="test_t1")
        assert res is not None
        assert "research_results" in res
        assert "T1" in res["completed_tasks"]
        assert res["next_agent"] == "supervisor"
        
    finally:
        # Shutdown server and restore ports
        server.should_exit = True
        thread.join(timeout=2)
        client.A2A_RESEARCH_PORT = original_research_port
        client.A2A_REVIEWER_PORT = original_reviewer_port
        
        research_agent.get_llm = original_research_llm
        reviewer_agent.get_llm = original_reviewer_llm


def test_target_file_not_found_validation(tmp_path):
    """Verify that Research Agent fails immediately if target file does not exist."""
    repo = tmp_path / "mock_repo"
    repo.mkdir()
    
    state = create_initial_state("Fix bug in nonexistent.py", str(repo))
    res = research_node(state)
    
    assert len(res.get("errors", [])) > 0
    assert "was not found in the repository" in res["errors"][0]
    assert res["target_file"] == "nonexistent.py"
    assert res["next_agent"] == "supervisor"


def test_coding_agent_target_file_mismatch(tmp_path):
    """Verify that Coding Agent fails validation if proposed file does not match target file."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    (repo / "calculator.py").write_text("def multiply(a, b): return a + b", encoding="utf-8")
    
    state = create_initial_state("Fix bug in code.py", str(repo))
    state["target_file"] = "code.py"
    state["current_task"] = "Apply multiplication fix"
    state["plan"] = {"current_task_id": "T2"}
    
    from app.agents import coding_agent
    from app.agents.coding_agent import CodeChange
    from langchain_core.runnables import Runnable
    
    class MockLLM(Runnable):
        def invoke(self, input, config=None, **kwargs):
            return CodeChange(
                reasoning="Fixing wrong file",
                file_path="calculator.py",
                new_content="print('wrong')"
            )
            
    original_get_llm = coding_agent.get_llm
    coding_agent.get_llm = lambda: MockLLM()
    
    try:
        res = coding_node(state)
        assert len(res.get("errors", [])) > 0
        assert "Coding Agent validation failed" in res["errors"][0]
        assert res["next_agent"] == "supervisor"
    finally:
        coding_agent.get_llm = original_get_llm


def test_supervisor_target_mismatch_and_critical_abort():
    """Verify Supervisor terminates on target mismatch or missing target file errors."""
    from app.agents.supervisor import supervisor_node
    
    state1 = create_initial_state("Fix nonexistent.py", "/path")
    state1["errors"] = ["The requested target file nonexistent.py was not found in the repository."]
    res1 = supervisor_node(state1)
    assert res1["next_agent"] == "complete"
    
    state2 = create_initial_state("Fix code.py", "/path")
    state2["target_file"] = "code.py"
    state2["research_files"] = ["calculator.py"]
    res2 = supervisor_node(state2)
    assert res2["next_agent"] == "complete"
    assert len(res2.get("errors", [])) > 0
    assert "Target mismatch" in res2["errors"][0]


def test_testing_agent_syntax_validation_fallback(tmp_path):
    """Verify that Testing Agent performs syntax check validation when no tests are found."""
    from app.agents.testing_agent import testing_node
    repo = tmp_path / "sandbox"
    repo.mkdir()
    
    # Write python file with syntax error
    (repo / "calculator.py").write_text("def multiply(a, b):\n    return a + * b\n", encoding="utf-8")
    
    state = create_initial_state("Verify syntax validation", str(repo))
    state["target_file"] = "calculator.py"
    state["current_task"] = "Verify test suites"
    state["plan"] = {"current_task_id": "T3"}
    
    res = testing_node(state)
    assert "test_results" in res
    assert "Automated tests: Not available" in res["test_results"]
    assert "Python Syntax Error" in res["test_results"]
    
    # Write syntactically correct file
    (repo / "calculator.py").write_text("def multiply(a, b):\n    return a * b\n", encoding="utf-8")
    res2 = testing_node(state)
    assert "Automated tests: Not available" in res2["test_results"]
    assert "Python Syntax Check: PASSED" in res2["test_results"]

