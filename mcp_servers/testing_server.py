"""MCP server exposing test execution tools for the testing subagent."""

import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "./demo_workspace")).resolve()
mcp = FastMCP("TestingTools")


@mcp.tool()
def run_pytest(extra_args: str = "") -> str:
    """Run pytest in the workspace and return combined stdout/stderr."""
    cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
    if extra_args.strip():
        cmd.extend(extra_args.split())
    result = subprocess.run(
        cmd,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    status = "PASSED" if result.returncode == 0 else "FAILED"
    return f"Exit code: {result.returncode} ({status})\n\n{output}"


@mcp.tool()
def run_command(command: str) -> str:
    """Run a shell command in the workspace directory."""
    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    return f"Exit code: {result.returncode}\n\n{output}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
