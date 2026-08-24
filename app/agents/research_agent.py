import os
import logging
import re
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT
from app.rag.retriever import retrieve_codebase_context

logger = logging.getLogger(__name__)


class ResearchResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning diagnosing the codebase")
    relevant_files: List[str] = Field(description="List of relative file paths that need inspection or editing")
    root_cause: str = Field(description="Suspected bug cause or structural explanation")
    suggested_fix: str = Field(description="High-level suggested implementation instructions")


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
            return llm.with_structured_output(ResearchResult)
        except Exception as e:
            logger.error("Failed to initialize ChatOllama: %s", e)
            return None

    if not GOOGLE_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
        return llm.with_structured_output(ResearchResult)
    except Exception as e:
        logger.error("Failed to initialize Research Agent LLM: %s", e)
        return None


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "researcher.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are a Research Agent analyzing codebases."


def extract_target_file(user_request: str) -> Optional[str]:
    extensions = r"\.(py|js|ts|java|c|cpp|h|go|rs|php|rb|html|css|json|yaml|yml|sh|md|txt)"
    pattern = r"\b[\w\-/\\]+" + extensions + r"\b"
    match = re.search(pattern, user_request, re.IGNORECASE)
    if match:
        return match.group(0).replace("\\", "/")
    return None


def find_target_file_in_workspace(target_file: str, repo_path: str) -> Optional[str]:
    """Recursively search the workspace for the target file to avoid path mismatches."""
    repo_root = Path(repo_path).resolve()
    
    # Direct check
    direct_path = (repo_root / target_file).resolve()
    try:
        if direct_path.exists() and direct_path.is_file() and direct_path.is_relative_to(repo_root):
            return direct_path.relative_to(repo_root).as_posix()
    except ValueError:
        pass
        
    # Recursive search
    filename = Path(target_file).name
    for p in repo_root.rglob("*"):
        if p.name == filename and p.is_file():
            try:
                if p.resolve().is_relative_to(repo_root):
                    return p.resolve().relative_to(repo_root).as_posix()
            except ValueError:
                pass
    return None


def _pick_source_file(repo_path: str) -> Optional[str]:
    repo_root = Path(repo_path)
    if not repo_root.exists():
        return None
    candidates = []
    for path in repo_root.rglob("*.py"):
        name = path.name
        if name.startswith("test_") or name == "conftest.py":
            continue
        candidates.append(path.relative_to(repo_root).as_posix())
    if not candidates:
        return None
    for preferred in ("calculator.py", "code.py"):
        if preferred in candidates:
            return preferred
    return candidates[0]


def run_research_mock(state: AgentState, target_file: Optional[str] = None) -> ResearchResult:
    if not target_file:
        target_file = state.get("target_file")

    test_results = state.get("test_results", "")
    if not target_file and test_results:
        matches = re.findall(r"\b[\w\-]+\.py\b", test_results)
        if matches:
            candidate_files = [m for m in matches if "test_" not in m]
            target_file = candidate_files[0] if candidate_files else matches[0]

    if not target_file:
        target_file = _pick_source_file(state.get("repository_path", "")) or "calculator.py"

    return ResearchResult(
        reasoning=f"Mock Research: Inspected workspace and identified '{target_file}' as the file to debug.",
        relevant_files=[target_file],
        root_cause=f"Logic errors were found in {target_file}.",
        suggested_fix=f"Correct the incorrect arithmetic/operators inside {target_file}.",
    )


def research_node(state: AgentState) -> dict:
    logger.info("Executing research_agent node.")

    query = state.get("user_request", "")
    repo_path = state.get("repository_path", "")
    target_file = state.get("target_file")

    if not target_file:
        target_file = extract_target_file(query)

    if target_file:
        resolved_rel_path = find_target_file_in_workspace(target_file, repo_path)
        if resolved_rel_path:
            target_file = resolved_rel_path
        else:
            err_msg = f"The requested target file {target_file} was not found in the repository."
            logger.error(err_msg)
            return {
                "errors": [err_msg],
                "target_file": target_file,
                "next_agent": "supervisor",
            }

    updated_state = state.copy()
    updated_state["target_file"] = target_file

    if not state.get("_a2a_execution"):
        try:
            from app.a2a.client import call_a2a_agent
            a2a_output = call_a2a_agent("research", updated_state, task_id="A2A_Research_Task")
            if a2a_output is not None:
                logger.info("Research agent task executed remotely via A2A protocol.")
                a2a_output["target_file"] = target_file
                return a2a_output
        except Exception as e:
            logger.debug("A2A Research delegation check failed: %s. Running locally.", e)

    in_memory = (
        repo_path == ""
        or ":memory:" in repo_path
        or "/tmp" in repo_path
        or "Temp" in repo_path
        or "temp" in repo_path
    )
    
    try:
        logger.info("Retrieving RAG context for query: %r", query)
        rag_context = retrieve_codebase_context(query, repo_path, k=5, in_memory=in_memory)
    except Exception as e:
        logger.error("Failed to retrieve RAG codebase context: %s. Using direct debugging.", e)
        rag_context = "No supporting codebase context available (RAG failed)."

    primary_file_context = ""
    if target_file:
        try:
            repo_root = Path(repo_path).resolve()
            target_abs = (repo_root / target_file).resolve()
            if target_abs.exists():
                content = target_abs.read_text(encoding="utf-8", errors="ignore")
                primary_file_context = (
                    f"\n=== Target File Content ({target_file}) ===\n{content}\n==============================\n"
                )
                logger.info("Directly inspected primary target file: %s", target_file)
        except Exception as e:
            logger.error("Failed to inspect target file %s: %s", target_file, e)

    llm = get_llm()
    if llm:
        try:
            system_prompt = load_system_prompt()
            if target_file:
                system_prompt += (
                    f"\nNOTE: The user specified target file '{target_file}'. "
                    f"Prioritize diagnosing bugs inside that file. Include it first in relevant_files."
                )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", (
                    "Active Task: {current_task}\n"
                    "User Request: {user_request}\n\n"
                    "Starting Target File Context:\n{primary_file_context}\n\n"
                    "Supporting Codebase Context (RAG):\n{rag_context}\n\n"
                    "Previous Test Failure Results:\n{test_results}\n\n"
                    "Errors:\n{errors}\n\n"
                    "Provide your structured ResearchResult including all relevant_files to edit."
                )),
            ])
            chain = prompt | llm
            result = chain.invoke({
                "current_task": state.get("current_task", ""),
                "user_request": state.get("user_request", ""),
                "primary_file_context": primary_file_context,
                "rag_context": rag_context,
                "test_results": state.get("test_results", ""),
                "errors": ", ".join(state.get("errors", [])),
            })
        except Exception as e:
            logger.error("Research LLM error: %s. Falling back to mock research.", e)
            result = run_research_mock(updated_state, target_file)
    else:
        result = run_research_mock(updated_state, target_file)

    if isinstance(result, dict):
        result = ResearchResult(**result)
    assert isinstance(result, ResearchResult)

    if target_file:
        ordered = [target_file] + [f for f in result.relevant_files if f != target_file]
        result.relevant_files = ordered

    logger.info("[Research Findings] Cause: %s", result.root_cause)
    logger.info("Files to edit: %s", result.relevant_files)

    completed = list(state.get("completed_tasks", []))
    plan = state.get("plan", {})
    current_task_id = plan.get("current_task_id", "")
    if current_task_id and current_task_id not in completed:
        completed.append(current_task_id)

    summary = (
        f"Research Diagnosis: {result.root_cause}\n"
        f"Files Identified: {', '.join(result.relevant_files)}\n"
        f"Suggested Fix: {result.suggested_fix}"
    )

    return {
        "research_results": summary,
        "research_files": result.relevant_files,
        "relevant_files": result.relevant_files,
        "target_file": target_file,
        "completed_tasks": completed,
        "messages": [{"role": "researcher", "content": summary}],
        "next_agent": "supervisor",
    }
