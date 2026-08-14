"""LangGraph workflow: User → Supervisor → Coding → Testing."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from code_debug_agent.agents.coding import create_coding_agent, run_coding
from code_debug_agent.agents.supervisor import create_supervisor, plan_task
from code_debug_agent.agents.testing import create_testing_agent, run_testing
from code_debug_agent.config import MAX_RETRIES


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: str
    coding_result: str
    testing_result: str
    tests_passed: bool
    retry_count: int
    status: str


async def build_graph():
    supervisor = create_supervisor()
    coding_agent = await create_coding_agent()
    testing_agent = await create_testing_agent()

    async def supervisor_node(state: AgentState) -> dict:
        prior = ""
        if state.get("retry_count", 0) > 0:
            prior = (
                f"Coding: {state.get('coding_result', '')}\n"
                f"Testing: {state.get('testing_result', '')}"
            )
        plan = await plan_task(supervisor, state["task"], prior)
        return {"plan": plan, "status": "coding"}

    async def coding_node(state: AgentState) -> dict:
        feedback = state.get("testing_result", "") if state.get("retry_count", 0) > 0 else ""
        result = await run_coding(
            coding_agent,
            state["plan"],
            state["task"],
            feedback,
        )
        return {"coding_result": result, "status": "testing"}

    async def testing_node(state: AgentState) -> dict:
        result, passed = await run_testing(
            testing_agent,
            state["task"],
            state["coding_result"],
        )
        return {
            "testing_result": result,
            "tests_passed": passed,
            "status": "done" if passed else "retry",
        }

    def route_after_testing(state: AgentState) -> Literal["supervisor", "__end__"]:
        if state.get("tests_passed"):
            return END
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return END
        return "supervisor"

    def increment_retry(state: AgentState) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("coding", coding_node)
    builder.add_node("testing", testing_node)
    builder.add_node("increment_retry", increment_retry)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "coding")
    builder.add_edge("coding", "testing")
    builder.add_conditional_edges("testing", route_after_testing, {"supervisor": "increment_retry", END: END})
    builder.add_edge("increment_retry", "supervisor")

    return builder.compile()


async def run_workflow(task: str) -> AgentState:
    graph = await build_graph()
    initial: AgentState = {
        "messages": [],
        "task": task,
        "plan": "",
        "coding_result": "",
        "testing_result": "",
        "tests_passed": False,
        "retry_count": 0,
        "status": "pending",
    }
    return await graph.ainvoke(initial)
