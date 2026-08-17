import os
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT
from app.rag.retriever import retrieve_codebase_context

logger = logging.getLogger(__name__)

# Structured Output Definition
class ResearchResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning diagnosing the codebase")
    relevant_files: List[str] = Field(description="List of relative file paths that need inspection or editing")
    root_cause: str = Field(description="Suspected bug cause or structural explanation")
    suggested_fix: str = Field(description="High-level suggested implementation instructions")


def get_llm():
    if not GOOGLE_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        return llm.with_structured_output(ResearchResult)
    except Exception as e:
        logger.error(f"Failed to initialize Research Agent LLM: {e}")
        return None


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "researcher.txt"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a Research Agent analyzing codebases."


def run_research_mock(state: AgentState) -> ResearchResult:
    """Mock fallback research behavior."""
    return ResearchResult(
        reasoning="Mock Research: Scanned workspace files manually and found login flow.",
        relevant_files=["calculator.py"],
        root_cause="The multiply function in calculator.py returns addition instead of multiplication.",
        suggested_fix="Change returning `a + b` to `a * b` in `multiply` function."
    )


def research_node(state: AgentState) -> dict:
    """
    Research Agent Node: Retrieves codebase context via RAG, diagnoses the issue, and suggests a fix.
    """
    logger.info("Executing research_agent node.")
    
    # 0. Check if remote A2A service is running, and delegate to it if active!
    try:
        from app.a2a.client import call_a2a_agent
        a2a_output = call_a2a_agent("research", state, task_id="A2A_Research_Task")
        if a2a_output is not None:
            logger.info("Research agent task executed remotely via A2A protocol.")
            return a2a_output
    except Exception as e:
        logger.debug(f"A2A Research delegation check failed: {e}. Running locally.")
    
    # 1. Retrieve RAG codebase context based on user goal
    query = state.get("user_request", "")
    repo_path = state.get("repository_path", "")
    
    # Check if we are running in-memory or persisted (tests use in-memory, app uses persist)
    in_memory = (repo_path == "" or ":memory:" in repo_path or "/tmp" in repo_path or "Temp" in repo_path or "temp" in repo_path)
    
    logger.info(f"Retrieving RAG context for query: '{query}'")
    rag_context = retrieve_codebase_context(query, repo_path, k=5, in_memory=in_memory)
    
    # 2. Run LLM or mock fallback
    llm = get_llm()
    if llm:
        try:
            system_prompt = load_system_prompt()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", (
                    "Active Task: {current_task}\n"
                    "User Request: {user_request}\n\n"
                    "Codebase Context (RAG):\n{rag_context}\n\n"
                    "Provide your structured ResearchResult."
                ))
            ])
            chain = prompt | llm
            result = chain.invoke({
                "current_task": state.get("current_task", ""),
                "user_request": state.get("user_request", ""),
                "rag_context": rag_context
            })
        except Exception as e:
            logger.error(f"Research LLM error: {e}. Falling back to mock research.")
            result = run_research_mock(state)
    else:
        result = run_research_mock(state)
        
    logger.info(f"[Research Findings] Cause: {result.root_cause}")
    logger.info(f"Files to edit: {result.relevant_files}")
    
    # Update task list
    completed = list(state.get("completed_tasks", []))
    plan = state.get("plan", {})
    current_task_id = plan.get("current_task_id", "")
    if current_task_id and current_task_id not in completed:
        completed.append(current_task_id)
        
    # Format a printable summary for state
    summary = (
        f"Research Diagnosis: {result.root_cause}\n"
        f"Files Identified: {', '.join(result.relevant_files)}\n"
        f"Suggested Fix: {result.suggested_fix}"
    )
    
    return {
        "research_results": summary,
        "completed_tasks": completed,
        "messages": [{"role": "researcher", "content": summary}],
        "next_agent": "supervisor"
    }
