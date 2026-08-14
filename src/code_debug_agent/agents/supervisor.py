"""Supervisor agent: analyzes the task and produces a plan."""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from code_debug_agent.config import LLM_MODEL

SUPERVISOR_PROMPT = """You are a supervisor/planner for a code-debugging team.

Given a user task, produce a concise plan for the coding and testing subagents.
Include:
1. Which files to inspect
2. What the likely bug is (if inferable)
3. What fix to attempt
4. How to verify with tests

Be specific and actionable. Keep the plan under 300 words."""


def create_supervisor():
    return create_agent(
        model=LLM_MODEL,
        tools=[],
        system_prompt=SUPERVISOR_PROMPT,
    )


async def plan_task(supervisor, task: str, prior_attempts: str = "") -> str:
    content = task
    if prior_attempts:
        content += f"\n\nPrevious attempts failed:\n{prior_attempts}"

    result = await supervisor.ainvoke(
        {"messages": [HumanMessage(content=content)]},
        config={"recursion_limit": 8},
    )
    return result["messages"][-1].content
