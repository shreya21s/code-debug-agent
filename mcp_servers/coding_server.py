"""MCP server exposing filesystem tools for the coding subagent."""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "./demo_workspace")).resolve()
mcp = FastMCP("CodingTools")


def _resolve(path: str) -> Path:
    resolved = (WORKSPACE / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE)):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


@mcp.tool()
def list_directory(relative_path: str = ".") -> str:
    """List files and directories inside the workspace."""
    target = _resolve(relative_path)
    if not target.is_dir():
        return f"Not a directory: {relative_path}"
    entries = sorted(target.iterdir())
    lines = [f"{'[dir]' if e.is_dir() else '[file]'} {e.name}" for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
def read_file(relative_path: str) -> str:
    """Read a file from the workspace."""
    target = _resolve(relative_path)
    if not target.is_file():
        return f"File not found: {relative_path}"
    return target.read_text(encoding="utf-8")


@mcp.tool()
def write_file(relative_path: str, content: str) -> str:
    """Write or overwrite a file in the workspace."""
    target = _resolve(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {relative_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
