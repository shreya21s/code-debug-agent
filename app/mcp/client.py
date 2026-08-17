import os
import sys
import logging
import asyncio
import concurrent.futures
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

logger = logging.getLogger(__name__)

@asynccontextmanager
async def mcp_session():
    """Context manager to launch the stdio MCP server and establish a client session."""
    server_path = Path(__file__).resolve().parent / "server.py"
    
    # Configure the environment variables to make sure pythonpath is correct
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(project_root)
        
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=env
    )
    
    logger.info(f"Connecting to stdio MCP server at: {server_path}")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def call_mcp_tool_async(name: str, arguments: Dict[str, Any]) -> Any:
    """Asynchronously calls an MCP tool by name with arguments."""
    async with mcp_session() as session:
        logger.info(f"Calling MCP tool async: {name} with args {arguments}")
        result = await session.call_tool(name, arguments)
        try:
            from app.utils.logging import tracer
            summary = str(result)
            if hasattr(result, "content") and result.content:
                from mcp.types import TextContent
                first_content = result.content[0]
                if isinstance(first_content, TextContent):
                    summary = first_content.text
            tracer.record_tool_call(name, arguments, summary)
        except Exception as e:
            logger.debug(f"Tracer logging failed: {e}")
        return result


def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """
    Synchronously calls an MCP tool.
    Handles existing event loop contexts (like pytest-asyncio runs) safely via a thread executor.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        # Spawn in a separate thread to run the async loop safely
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(call_mcp_tool_async(name, arguments)))
            return future.result()
    else:
        return asyncio.run(call_mcp_tool_async(name, arguments))
