import os
import logging
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT

logger = logging.getLogger(__name__)


class TaskItem(BaseModel):
    task_id: str = Field(description="Unique short ID for the task, e.g. T1, T2")
    description: str = Field(description="Detailed description of what needs to be done")
    assigned_agent: str = Field(description="The agent to delegate this task to: research, coding, testing, reviewer")
    status: str = Field(default="pending", description="Status of the task: pending, in_progress, completed, failed")


class TaskPlan(BaseModel):
    reasoning: str = Field(description="Brief explanation of current state of progress and next step choice")
    tasks: List[TaskItem] = Field(description="The full ordered checklist of tasks to complete the user's request")
    next_agent: str = Field(description="The agent to run next: research, coding, testing, reviewer, or complete")
    current_task_id: str = Field(description="The task_id of the task that next_agent will work on, or empty if complete")


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "supervisor.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are the Supervisor of a Multi-Agent AI Software Engineering Team. "
        "Flow: research (analyze) -> coding (fix) -> testing -> reviewer. "
        "When finished, set next_agent to complete."
    )


def get_llm():
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    from app.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_HOST
    if LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_HOST,
                temperature=0.0,
            )
            return llm.with_structured_output(TaskPlan)
        except Exception as e:
            logger.error("Failed to initialize ChatOllama: %s", e)
            return None

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not set. Falling back to mock mode.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
        return llm.with_structured_output(TaskPlan)
    except Exception as e:
        logger.error("Failed to initialize ChatGoogleGenerativeAI: %s", e)
        return None


def run_supervisor_llm(state: AgentState, llm) -> TaskPlan:
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_system_prompt()),
        ("user", (
            "User Goal: {user_request}\n"
            "Target Repo: {repository_path}\n"
            "Target File: {target_file}\n"
            "Current Plan: {plan}\n"
            "Current Task: {current_task}\n"
            "Completed Tasks: {completed_tasks}\n"
            "Research Results: {research_results}\n"
            "Code Changes: {code_changes}\n"
            "Test Results: {test_results}\n"
            "Review Results: {review_results}\n"
            "Errors: {errors}\n\n"
            "Formulate the updated TaskPlan."
        )),
    ])

    chain = prompt | llm
    return chain.invoke({
        "user_request": state.get("user_request", ""),
        "repository_path": state.get("repository_path", ""),
        "target_file": state.get("target_file") or "",
        "plan": str(state.get("plan", {})),
        "current_task": state.get("current_task", ""),
        "completed_tasks": ", ".join(state.get("completed_tasks", [])),
        "research_results": state.get("research_results", ""),
        "code_changes": state.get("code_changes", ""),
        "test_results": state.get("test_results", ""),
        "review_results": state.get("review_results", ""),
        "errors": ", ".join(state.get("errors", [])),
    })


def run_supervisor_mock(state: AgentState) -> TaskPlan:
    completed = state.get("completed_tasks", [])

    tasks = [
        TaskItem(task_id="T1", description="Analyze the target file and diagnose the bug", assigned_agent="research", status="pending"),
        TaskItem(task_id="T2", description="Apply bug fix edits to the target file", assigned_agent="coding", status="pending"),
        TaskItem(task_id="T3", description="Run tests to verify the fix", assigned_agent="testing", status="pending"),
        TaskItem(task_id="T4", description="Review changes and final outcome", assigned_agent="reviewer", status="pending"),
    ]

    if "T1" in completed:
        tasks[0].status = "completed"
    if "T2" in completed:
        tasks[1].status = "completed"
    if "T3" in completed:
        tasks[2].status = "completed"
    if "T4" in completed:
        tasks[3].status = "completed"

    next_agent = "research"
    current_task_id = "T1"
    reasoning = "Start by analyzing the buggy file and locating the root cause."

    # Detect verification failure states
    test_results = state.get("test_results", "")
    review_results = state.get("review_results", "")
    test_failed = "test success: false" in test_results.lower()
    review_rejected = "approved: false" in review_results.lower()

    if "T1" in completed:
        next_agent = "coding"
        current_task_id = "T2"
        reasoning = "Analysis finished. Instructing Coding agent to apply the fix."
    if "T2" in completed:
        next_agent = "testing"
        current_task_id = "T3"
        reasoning = "Code modified. Instructing Testing agent to verify the fix."
    if "T3" in completed:
        if test_failed:
            next_agent = "coding"
            current_task_id = "T2"
            reasoning = "Tests failed. Instructing Coding agent to correct the code."
        else:
            next_agent = "reviewer"
            current_task_id = "T4"
            reasoning = "Testing finished. Instructing Reviewer agent to assess changes."
    if "T4" in completed:
        if review_rejected:
            next_agent = "coding"
            current_task_id = "T2"
            reasoning = "Review rejected. Instructing Coding agent to correct the code."
        else:
            next_agent = "complete"
            current_task_id = ""
            reasoning = "Analyze, fix, test, and review are complete."

    return TaskPlan(
        reasoning=reasoning,
        tasks=tasks,
        next_agent=next_agent,
        current_task_id=current_task_id,
    )


def _file_aliases(path: str) -> set[str]:
    normalized = path.replace("\\", "/").strip().lower()
    return {normalized, Path(normalized).name}


def supervisor_node(state: AgentState) -> dict:
    logger.info("Executing supervisor_agent node.")

    iteration = state.get("iteration_count", 0) + 1

    state_errors = state.get("errors", [])
    if any("was not found in the repository" in err for err in state_errors):
        logger.error("Critical error detected (missing target file). Terminating workflow.")
        return {
            "next_agent": "complete",
            "iteration_count": iteration,
        }

    target_file = state.get("target_file")
    research_files = state.get("research_files") or []
    if target_file and research_files:
        allowed = set()
        for item in research_files:
            allowed |= _file_aliases(item)
        if not (_file_aliases(target_file) & allowed):
            mismatch_err = (
                f"Target mismatch: Research suggested files {research_files} "
                f"but user explicitly requested '{target_file}'."
            )
            logger.error(mismatch_err)
            return {
                "errors": [mismatch_err],
                "next_agent": "complete",
                "iteration_count": iteration,
            }

    llm = get_llm()
    if llm:
        try:
            task_plan = run_supervisor_llm(state, llm)
        except Exception as e:
            logger.error("Supervisor LLM error: %s. Falling back to mock router.", e)
            task_plan = run_supervisor_mock(state)
    else:
        task_plan = run_supervisor_mock(state)

    next_agent = task_plan.next_agent
    errors_to_add = []

    # Reset completed tasks when looping back to guarantee clean state updates
    completed_tasks = list(state.get("completed_tasks", []))
    if next_agent == "coding":
        completed_tasks = [t for t in completed_tasks if t not in ("T2", "T3", "T4")]
        attempts = state.get("coding_attempts", 0)
        if attempts >= 3:
            limit_err = f"Coding retry limit reached ({attempts} attempts). TASK FAILED."
            logger.error(limit_err)
            next_agent = "complete"
            errors_to_add.append(limit_err)
    elif next_agent == "testing":
        completed_tasks = [t for t in completed_tasks if t not in ("T3", "T4")]
    elif next_agent == "reviewer":
        completed_tasks = [t for t in completed_tasks if t not in ("T4",)]

    current_desc = ""
    for task in task_plan.tasks:
        if task.task_id == task_plan.current_task_id:
            current_desc = task.description

    if target_file and next_agent == "coding":
        current_desc = f"Modify target file '{target_file}' to apply the requested bug fix."

    try:
        from app.utils.logging import tracer
        tracer.record_step(
            "supervisor_agent",
            "supervisor",
            "Update task plan",
            f"Next: {next_agent} | Reasoning: {task_plan.reasoning}",
        )
    except Exception:
        pass

    return {
        "iteration_count": iteration,
        "plan": task_plan.model_dump(),
        "next_agent": next_agent,
        "current_task": current_desc,
        "completed_tasks": completed_tasks,
        "errors": errors_to_add,
    }

