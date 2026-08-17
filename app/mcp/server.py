import os
import sys
import logging
from typing import List, Dict, Any

# Ensure correct package context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer
import app.mcp.tools as tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp_server")

# Initialize high-level MCPServer
server = MCPServer("ai-engineering-team-mcp-server")

@server.tool(name="list_files")
def list_files_tool(subdir: str = "") -> List[str]:
    """
    Lists relative paths of all files recursively within the allowed project workspace.
    
    Args:
        subdir: Optional subdirectory relative to the workspace root to scan.
    """
    logger.info(f"MCP Tool called: list_files (subdir='{subdir}')")
    return tools.list_files(subdir)

@server.tool(name="read_file")
def read_file_tool(file_path: str) -> str:
    """
    Reads the full text content of a file within the project workspace safely.
    
    Args:
        file_path: Relative or absolute path of the file to read (must reside inside workspace).
    """
    logger.info(f"MCP Tool called: read_file (file_path='{file_path}')")
    return tools.read_file(file_path)

@server.tool(name="write_file")
def write_file_tool(file_path: str, content: str) -> str:
    """
    Creates or overwrites a file in the project workspace with the specified text content.
    
    Args:
        file_path: Relative path to write the content to (must reside inside workspace).
        content: The text content to write.
    """
    logger.info(f"MCP Tool called: write_file (file_path='{file_path}', content_len={len(content)})")
    return tools.write_file(file_path, content)

@server.tool(name="search_files")
def search_files_tool(query: str) -> List[Dict[str, Any]]:
    """
    Searches for a specific text query across all text files in the project workspace (like grep).
    
    Args:
        query: The substring to search for.
    """
    logger.info(f"MCP Tool called: search_files (query='{query}')")
    return tools.search_files(query)

@server.tool(name="run_tests")
def run_tests_tool(command_arg: str = "") -> Dict[str, Any]:
    """
    Runs the pytest test suite inside the workspace directory.
    Strictly restricted to approved test targets.
    
    Args:
        command_arg: Optional specific file path or flag to pass to pytest.
    """
    logger.info(f"MCP Tool called: run_tests (command_arg='{command_arg}')")
    return tools.run_tests(command_arg)


if __name__ == "__main__":
    logger.info("Starting stdio MCP server...")
    server.run("stdio")
