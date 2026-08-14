"""Coding subagent: reads and edits code via MCP filesystem tools."""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from code_debug_agent.config import LLM_MODEL
from code_debug_agent.mcp_client import get_coding_tools

CODING_PROMPT = """You are a coding subagent specializing in debugging and fixing code.

You have MCP tools to list directories, read files, and write files in the workspace.
Always read relevant files before editing. Make minimal, targeted fixes.
After applying a fix, summarize what you changed and why."""


async def create_coding_agent():
    tools = await get_coding_tools()
    return create_agent(
        model=LLM_MODEL,
        tools=tools,
        system_prompt=CODING_PROMPT,
    )


async def run_coding(coding_agent, plan: str, task: str, test_feedback: str = "") -> str:
    prompt = f"User task:\n{task}\n\nSupervisor plan:\n{plan}"
    if test_feedback:
        prompt += f"\n\nTest failures to address:\n{test_feedback}"

    result = await coding_agent.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 25},
    )
    return result["messages"][-1].content
