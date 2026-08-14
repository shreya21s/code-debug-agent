import os
from pathlib import Path
import pytest

from app.config import WORKSPACE_ROOT
from app.mcp.tools import validate_safe_path, list_files, read_file, write_file, search_files
from app.mcp.client import call_mcp_tool

def test_validate_safe_path(tmp_path):
    """Verify boundaries are enforced and traversal is blocked."""
    workspace = tmp_path / "sandbox"
    workspace.mkdir()
    
    # Safe paths
    assert validate_safe_path("calc.py", workspace) == workspace / "calc.py"
    assert validate_safe_path("src/auth.py", workspace) == workspace / "src" / "auth.py"
    
    # Traversal attempts
    with pytest.raises(ValueError, match="Path traversal detected"):
        validate_safe_path("../outside.txt", workspace)
        
    # Absolute paths outside sandbox
    with pytest.raises(ValueError, match="Path traversal detected"):
        validate_safe_path(str(tmp_path / "outside.txt"), workspace)

def test_file_tools_direct(tmp_path):
    """Verify tool functions directly execute correctly."""
    # Write a test file in temporary workspace
    original_root = os.getenv("WORKSPACE_ROOT")
    try:
        # Override WORKSPACE_ROOT dynamically for tests
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = tmp_path
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)
        
        # Test write_file
        write_file("test.txt", "Hello World!")
        assert (tmp_path / "test.txt").read_text() == "Hello World!"
        
        # Test read_file
        content = read_file("test.txt")
        assert content == "Hello World!"
        
        # Test list_files
        write_file("src/main.py", "print('main')")
        files = list_files()
        assert "test.txt" in files
        assert "src/main.py" in files
        
        # Test search_files
        search_res = search_files("print")
        assert len(search_res) == 1
        assert search_res[0]["file_path"] == "src/main.py"
        assert "print" in search_res[0]["content"]
    finally:
        # Restore workspace root
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = Path(original_root) if original_root else WORKSPACE_ROOT
        if original_root:
            os.environ["WORKSPACE_ROOT"] = original_root
        else:
            os.environ.pop("WORKSPACE_ROOT", None)

def test_mcp_client_server_integration(tmp_path):
    """Tests the full client-server JSON-RPC communication over stdio."""
    original_root = os.getenv("WORKSPACE_ROOT")
    try:
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = tmp_path
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)
        
        # Write test file
        test_file = tmp_path / "integration.txt"
        test_file.write_text("MCP stdio integration test", encoding="utf-8")
        
        # Call read_file tool through client-server connection!
        logger_name = "app.mcp.client"
        res = call_mcp_tool("read_file", {"file_path": "integration.txt"})
        
        # The result returned by call_tool is a Content object with a text property
        assert res is not None
        assert len(res.content) > 0
        from mcp.types import TextContent
        first_content = res.content[0]
        assert isinstance(first_content, TextContent)
        assert "MCP stdio integration test" in first_content.text
        
        # Call write_file tool through client-server
        call_mcp_tool("write_file", {"file_path": "output.txt", "content": "written via client"})
        assert (tmp_path / "output.txt").read_text() == "written via client"
    finally:
        import app.mcp.tools as tools
        tools.WORKSPACE_ROOT = Path(original_root) if original_root else WORKSPACE_ROOT
        if original_root:
            os.environ["WORKSPACE_ROOT"] = original_root
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
