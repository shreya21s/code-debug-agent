import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server import MCPServer
import app.mcp.tools as tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server")

server = MCPServer("ai-engineering-team-mcp-server")

@server.tool(name="list_files")
def list_files_tool(subdir: str = "") -> list:
    """Lists relative paths of all files recursively within the allowed project workspace."""
    logger.info("MCP Tool called: list_files (subdir=%r)", subdir)
    return tools.list_files(subdir)

@server.tool(name="read_file")
def read_file_tool(file_path: str) -> str:
    """Reads the full text content of a file within the project workspace safely."""
    logger.info("MCP Tool called: read_file (file_path=%r)", file_path)
    return tools.read_file(file_path)


@server.tool(name="write_file")
def write_file_tool(file_path: str, content: str) -> str:
    """Creates or overwrites a file in the project workspace with the specified text content."""
    logger.info("MCP Tool called: write_file (file_path=%r, content_len=%s)", file_path, len(content))
    return tools.write_file(file_path, content)


@server.tool(name="search_files")
def search_files_tool(query: str) -> list:
    """Searches for a specific text query across all text files in the project workspace."""
    logger.info("MCP Tool called: search_files (query=%r)", query)
    return tools.search_files(query)


@server.tool(name="run_tests")
def run_tests_tool(command_arg: str = "") -> dict:
    """Runs the pytest test suite inside the workspace directory."""
    logger.info("MCP Tool called: run_tests (command_arg=%r)", command_arg)
    return tools.run_tests(command_arg)


if __name__ == "__main__":
    logger.info("Starting stdio MCP server...")
    server.run(transport="stdio")
