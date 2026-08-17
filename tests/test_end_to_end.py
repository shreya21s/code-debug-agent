import os
import shutil
from pathlib import Path
import pytest

from app.state import create_initial_state
from app.graph.workflow import create_graph

def test_end_to_end_flow(tmp_path):
    """
    Verifies the complete execution flow (Research -> Coder -> Tester -> Reviewer)
    against a temporary repository with a calculator bug.
    """
    # 1. Setup mock repo in sandbox
    repo = tmp_path / "sandbox_project"
    repo.mkdir()
    
    # Buggy calculator.py
    (repo / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def multiply(a, b):\n"
        "    # Buggy addition\n"
        "    return a + b\n",
        encoding="utf-8"
    )
    
    # test_calculator.py
    (repo / "test_calculator.py").write_text(
        "from calculator import add, multiply\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n",
        encoding="utf-8"
    )
    
    # Save original WORKSPACE_ROOT
    original_root = os.getenv("WORKSPACE_ROOT")
    
    try:
        # Route MCP and config tools to target the temporary sandbox project
        os.environ["WORKSPACE_ROOT"] = str(repo)
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = repo
        
        # 2. Run the graph
        graph = create_graph()
        initial_state = create_initial_state(
            "Fix the multiply bug in calculator.py and verify tests pass",
            str(repo)
        )
        
        result = graph.invoke(initial_state)
        
        # 3. Assertions
        # Check iteration loop
        assert result["iteration_count"] > 0
        assert "T1" in result["completed_tasks"]
        assert "T2" in result["completed_tasks"]
        assert "T3" in result["completed_tasks"]
        assert "T4" in result["completed_tasks"]
        assert result["next_agent"] == "complete"
        
        # Verify file was modified to contain the fix
        fixed_code = (repo / "calculator.py").read_text()
        assert "a * b" in fixed_code
        assert "a + b" not in fixed_code.split("def multiply")[1]
        
        # Verify messages accumulated correctly
        assert len(result["messages"]) == 5
        
        # Check final output states
        assert "Research Diagnosis" in result["research_results"]
        assert "calculator.py" in result["code_changes"]
        assert "Test Success: True" in result["test_results"]
        assert "Approved: True" in result["review_results"]
        
    finally:
        # Restore environment settings
        if original_root:
            os.environ["WORKSPACE_ROOT"] = original_root
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = Path(original_root) if original_root else repo.parent
