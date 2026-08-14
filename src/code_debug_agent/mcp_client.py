"""MCP client helpers for loading subagent tools."""

from langchain_mcp_adapters.client import MultiServerMCPClient

from code_debug_agent.config import CODING_MCP_SERVER, TESTING_MCP_SERVER, WORKSPACE_ROOT


def _server_env() -> dict[str, str]:
    return {"WORKSPACE_ROOT": str(WORKSPACE_ROOT)}


async def get_coding_tools():
    client = MultiServerMCPClient(
        {
            "coding": {
                "transport": "stdio",
                "command": "python",
                "args": [str(CODING_MCP_SERVER)],
                "env": _server_env(),
            }
        }
    )
    return await client.get_tools()


async def get_testing_tools():
    client = MultiServerMCPClient(
        {
            "testing": {
                "transport": "stdio",
                "command": "python",
                "args": [str(TESTING_MCP_SERVER)],
                "env": _server_env(),
            }
        }
    )
    return await client.get_tools()
