import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME

logger = logging.getLogger(__name__)

# Structured Output Definitions
class TaskItem(BaseModel):
    task_id: str = Field(description="Unique short ID for the task, e.g. T1, T2")
    description: str = Field(description="Detailed description of what needs to be done")
    assigned_agent: str = Field(description="The agent to delegate this task to: research, coding, testing, reviewer")
    status: str = Field(default="pending", description="Status of the task: pending, in_progress, completed, failed")

class TaskPlan(BaseModel):
    reasoning: str = Field(description="Brief explanation of the current state of progress and next step choice")
    tasks: List[TaskItem] = Field(description="The full ordered checklist of tasks to complete the user's request")
    next_agent: str = Field(description="The agent to run next: research, coding, testing, reviewer, or complete")
    current_task_id: str = Field(description="The task_id of the task that next_agent will work on, or empty if complete")


def get_llm():
    """Returns the Gemini LLM with structured output or None if not configured."""
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not set. LLM is unavailable; falling back to mock mode.")
        return None
    try:
        # Initialize Google GenAI chat model
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        return llm.with_structured_output(TaskPlan)
    except Exception as e:
        logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        return None


def run_supervisor_llm(state: AgentState, llm) -> TaskPlan:
    """Uses LLM to decide the next planning step."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Supervisor of a Multi-Agent AI Software Engineering Team.\n"
            "Your job is to orchestrate a team of specialized agents to satisfy the user's goal.\n"
            "The specialized agents are:\n"
            "1. research: retrieves files, searches codebase (RAG), and diagnoses issues.\n"
            "2. coding: inspects, edits, and writes source files (via MCP tools).\n"
            "3. testing: runs tests and checks test logs (via MCP tools).\n"
            "4. reviewer: reviews code diffs and test results, ensuring changes are correct.\n\n"
            "Rules:\n"
            "- Start by assigning 'research' to inspect/retrieve codebase info.\n"
            "- Once research identifies the target file/bug, assign 'coding' to write the fix.\n"
            "- Once coding is finished, assign 'testing' to run tests.\n"
            "- Once testing finishes, assign 'reviewer' to review everything.\n"
            "- If any subagent reports failure or if testing/review fails, update the plan and retry.\n"
            "- When all tasks are completed and verified, set next_agent to 'complete'.\n"
            "- Keep a maximum loop budget in mind (current iterations: {iteration_count}).\n"
        )),
        ("user", (
            "User Goal: {user_request}\n"
            "Target Repo: {repository_path}\n"
            "Current Plan: {plan}\n"
            "Current Task: {current_task}\n"
            "Completed Tasks: {completed_tasks}\n"
            "Research Results: {research_results}\n"
            "Code Changes: {code_changes}\n"
            "Test Results: {test_results}\n"
            "Review Results: {review_results}\n"
            "Errors: {errors}\n\n"
            "Formulate the updated TaskPlan."
        ))
    ])
    
    # Format inputs
    chain = prompt | llm
    result = chain.invoke({
        "user_request": state.get("user_request", ""),
        "repository_path": state.get("repository_path", ""),
        "plan": str(state.get("plan", {})),
        "current_task": state.get("current_task", ""),
        "completed_tasks": ", ".join(state.get("completed_tasks", [])),
        "research_results": state.get("research_results", ""),
        "code_changes": state.get("code_changes", ""),
        "test_results": state.get("test_results", ""),
        "review_results": state.get("review_results", ""),
        "errors": ", ".join(state.get("errors", [])),
        "iteration_count": state.get("iteration_count", 0)
    })
    return result


def run_supervisor_mock(state: AgentState) -> TaskPlan:
    """Mock Supervisor routing for development/testing without API keys."""
    iteration = state.get("iteration_count", 0)
    completed = state.get("completed_tasks", [])
    
    # Simple deterministic flow: Research -> Code -> Test -> Review -> Complete
    tasks = [
        TaskItem(task_id="T1", description="Research project and locate files", assigned_agent="research", status="pending"),
        TaskItem(task_id="T2", description="Apply bug fix or feature edits", assigned_agent="coding", status="pending"),
        TaskItem(task_id="T3", description="Run tests to verify changes", assigned_agent="testing", status="pending"),
        TaskItem(task_id="T4", description="Review changes and test outcome", assigned_agent="reviewer", status="pending")
    ]
    
    # Update statuses based on what's done
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
    reasoning = "Initial task. Delegating codebase analysis to Research agent."
    
    if "T1" in completed:
        next_agent = "coding"
        current_task_id = "T2"
        reasoning = "Research finished. Instructing Coding agent to edit files."
    if "T2" in completed:
        next_agent = "testing"
        current_task_id = "T3"
        reasoning = "Code modified. Instructing Testing agent to run test suite."
    if "T3" in completed:
        next_agent = "reviewer"
        current_task_id = "T4"
        reasoning = "Testing finished. Instructing Reviewer agent to assess the changes."
    if "T4" in completed:
        next_agent = "complete"
        current_task_id = ""
        reasoning = "All steps complete and verified."
        
    return TaskPlan(
        reasoning=reasoning,
        tasks=tasks,
        next_agent=next_agent,
        current_task_id=current_task_id
    )


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor Node: Coordinates the multi-agent task execution.
    """
    logger.info("Executing supervisor_agent node.")
    
    # Get current iteration count and increment it
    iteration = state.get("iteration_count", 0) + 1
    
    llm = get_llm()
    if llm:
        try:
            task_plan = run_supervisor_llm(state, llm)
        except Exception as e:
            logger.error(f"Supervisor LLM error: {e}. Falling back to mock router.")
            task_plan = run_supervisor_mock(state)
    else:
        task_plan = run_supervisor_mock(state)
        
    # Get current task description from plan
    current_desc = ""
    for task in task_plan.tasks:
        if task.task_id == task_plan.current_task_id:
            current_desc = task.description
            
    try:
        from app.utils.logging import tracer
        tracer.record_step(
            "supervisor_agent",
            "supervisor",
            "Update task plan",
            f"Next: {task_plan.next_agent} | Reasoning: {task_plan.reasoning}"
        )
    except:
        pass
        
    # Return changes to the state
    return {
        "iteration_count": iteration,
        "plan": task_plan.model_dump(),
        "next_agent": task_plan.next_agent,
        "current_task": current_desc
    }
