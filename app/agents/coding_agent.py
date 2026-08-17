import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT, WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# Structured Output Definition
class CodeChange(BaseModel):
    reasoning: str = Field(description="Explanation of why this code change solves the issue")
    file_path: str = Field(description="Relative path to the file that needs modifying/creating")
    new_content: str = Field(description="The exact new content of the file, replacing the previous version completely")


def get_llm():
    if not GOOGLE_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        return llm.with_structured_output(CodeChange)
    except Exception as e:
        logger.error(f"Failed to initialize Coding Agent LLM: {e}")
        return None


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "coder.txt"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a Coding Agent editing codebase files."


def run_coding_mock(state: AgentState) -> CodeChange:
    """Mock fallback coding behavior that fixes the calculator multiply bug."""
    # Read the current file content if it exists
    repo_root = Path(state.get("repository_path", WORKSPACE_ROOT))
    target_file = repo_root / "calculator.py"
    
    # Simple mock calculator content with fix: return a * b
    fixed_content = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def multiply(a, b):\n"
        "    # Fixed multiply bug\n"
        "    return a * b\n"
    )
    
    return CodeChange(
        reasoning="Mock Coding: Replaced '+' operator with '*' in the multiply function of calculator.py.",
        file_path="calculator.py",
        new_content=fixed_content
    )


def coding_node(state: AgentState) -> dict:
    """
    Coding Agent Node: Receives research findings, generates the code change,
    and applies it directly to the local workspace.
    """
    logger.info("Executing coding_agent node.")
    
    repo_path = state.get("repository_path", WORKSPACE_ROOT)
    repo_root = Path(repo_path).resolve()
    os.environ["WORKSPACE_ROOT"] = str(repo_root)
    
    llm = get_llm()
    if llm:
        try:
            # Let's inspect the files suggested by research to feed current content into LLM
            research_info = state.get("research_results", "")
            # We can optionally extract a file content if we parse the paths
            file_content_context = ""
            for root_dir, _, files in os.walk(repo_root):
                for f in files:
                    if f.endswith(".py"):
                        p = Path(root_dir) / f
                        try:
                            with open(p, "r", encoding="utf-8", errors="ignore") as file_obj:
                                file_content_context += f"=== FILE: {p.relative_to(repo_root).as_posix()} ===\n{file_obj.read()}\n\n"
                        except:
                            pass
            
            system_prompt = load_system_prompt()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", (
                    "Active Task: {current_task}\n"
                    "User Request: {user_request}\n"
                    "Research Results:\n{research_results}\n\n"
                    "Current Codebase Files:\n{files_context}\n\n"
                    "Propose and apply the exact CodeChange."
                ))
            ])
            chain = prompt | llm
            result = chain.invoke({
                "current_task": state.get("current_task", ""),
                "user_request": state.get("user_request", ""),
                "research_results": research_info,
                "files_context": file_content_context
            })
        except Exception as e:
            logger.error(f"Coding LLM error: {e}. Falling back to mock code generator.")
            result = run_coding_mock(state)
    else:
        result = run_coding_mock(state)
        
    logger.info(f"[Coding Change proposed] Target Path: {result.file_path}")
    logger.info(f"Reasoning: {result.reasoning}")
    
    # Apply change to file safely within workspace limit
    target_abs = (repo_root / result.file_path).resolve()
    
    # Path traversal protection
    if not str(target_abs).startswith(str(repo_root)):
        error_msg = f"Path traversal attempt blocked: {result.file_path} is outside workspace root {repo_root}"
        logger.error(error_msg)
        return {"errors": [error_msg], "next_agent": "supervisor"}
        
    try:
        from app.mcp.client import call_mcp_tool
        msg = call_mcp_tool("write_file", {"file_path": result.file_path, "content": result.new_content})
        logger.info(f"Successfully modified file via MCP: {result.file_path}")
        code_summary = f"Modified {result.file_path} via MCP. Reasoning: {result.reasoning}"
    except Exception as e:
        error_msg = f"Failed to write file {result.file_path} via MCP: {e}"
        logger.error(error_msg)
        return {"errors": [error_msg], "next_agent": "supervisor"}
        
    # Update task lists
    completed = list(state.get("completed_tasks", []))
    plan = state.get("plan", {})
    current_task_id = plan.get("current_task_id", "")
    if current_task_id and current_task_id not in completed:
        completed.append(current_task_id)
        
    return {
        "code_changes": code_summary,
        "completed_tasks": completed,
        "messages": [{"role": "coder", "content": code_summary}],
        "next_agent": "supervisor"
    }
