"""Testing subagent: runs tests via MCP shell tools."""

import subprocess

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from code_debug_agent.config import LLM_MODEL, WORKSPACE_ROOT
from code_debug_agent.mcp_client import get_testing_tools


def pytest_exit_code() -> int:
    """Authoritative pass/fail check for graph routing."""
    result = subprocess.run(
        ["python", "-m", "pytest", "-q"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode

TESTING_PROMPT = """You are a testing subagent. Your job is to verify code changes by running tests.

Use run_pytest to execute the test suite in the workspace.
Report clearly:
- Whether tests passed or failed
- Which tests failed and why
- Any relevant error output

If tests pass, say PASSED. If any fail, say FAILED and include details."""


async def create_testing_agent():
    tools = await get_testing_tools()
    return create_agent(
        model=LLM_MODEL,
        tools=tools,
        system_prompt=TESTING_PROMPT,
    )


async def run_testing(testing_agent, task: str, coding_summary: str) -> tuple[str, bool]:
    prompt = (
        f"Verify the fix for this task:\n{task}\n\n"
        f"Coding subagent reported:\n{coding_summary}\n\n"
        "Run pytest and report results."
    )

    result = await testing_agent.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 15},
    )
    output = result["messages"][-1].content
    passed = pytest_exit_code() == 0
    return output, passed
