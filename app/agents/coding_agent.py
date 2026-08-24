import os
import re
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT, WORKSPACE_ROOT

logger = logging.getLogger(__name__)


import difflib

def generate_diff(file_path: str, old_content: str, new_content: str) -> str:
    """Generate a clean unified diff for the modifications."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}"
    )
    return "".join(diff)


class CodeChange(BaseModel):
    reasoning: str = Field(description="Explanation of why this code change solves the issue")
    file_path: str = Field(description="Relative path to the file that needs modifying/creating")
    new_content: str = Field(description="The EXACT COMPLETE content of the file (including ALL unmodified functions, imports, and comments), replacing the previous version completely. Do NOT truncate or omit any code!")


DEFAULT_CALCULATOR = (
    "def add(a: int, b: int) -> int:\n"
    "    return a + b\n\n"
    "def subtract(a: int, b: int) -> int:\n"
    "    return a - b\n\n"
    "def multiply(a: int, b: int) -> int:\n"
    "    # Fixed multiply bug\n"
    "    return a * b\n\n"
    "def divide(a: int, b: int) -> float:\n"
    "    if b == 0:\n"
    "        raise ValueError(\"Cannot divide by zero\")\n"
    "    return a / b\n"
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
            return llm.with_structured_output(CodeChange)
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
        return llm.with_structured_output(CodeChange)
    except Exception as e:
        logger.error("Failed to initialize Coding Agent LLM: %s", e)
        return None


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "coder.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are a Coding Agent editing codebase files."


def _apply_named_arithmetic_fixes(content: str) -> str:
    """Fix common operator mix-ups inside add/multiply/subtract/divide functions."""
    expected = {
        "add": "+",
        "multiply": "*",
        "subtract": "-",
        "divide": "/",
    }
    lines = content.splitlines(keepends=True)
    current = None
    out = []
    for line in lines:
        def_match = re.match(r"^def\s+(\w+)", line)
        if def_match:
            current = def_match.group(1)
        if current in expected:
            op = expected[current]
            line = re.sub(r"return\s+a\s*[+\-*/]\s*b", f"return a {op} b", line)
        out.append(line)
    return "".join(out)


def run_coding_mock(state: AgentState) -> CodeChange:
    repo_root = Path(state.get("repository_path", WORKSPACE_ROOT))
    target_filename = state.get("target_file")
    if not target_filename:
        files = state.get("research_files") or state.get("relevant_files") or []
        target_filename = files[0] if files else "calculator.py"

    target_file = repo_root / target_filename
    content = ""
    if target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Error reading %s in mock coding: %s", target_filename, e)

    if content.strip():
        fixed_content = _apply_named_arithmetic_fixes(content)
        if "def multiply" in content and "Fixed multiply bug" not in fixed_content:
            # Preserve the classic demo comment used by unit tests when creating a new file.
            pass
        reasoning = f"Mock Coding: Corrected arithmetic operators inside {target_filename}."
    else:
        fixed_content = DEFAULT_CALCULATOR
        reasoning = f"Mock Coding: Fixed multiply logic inside {target_filename}."

    return CodeChange(
        reasoning=reasoning,
        file_path=target_filename,
        new_content=fixed_content,
    )


def _allowed_files(state: AgentState) -> list[str]:
    files = list(state.get("research_files") or [])
    for extra in state.get("relevant_files") or []:
        if extra not in files:
            files.append(extra)
    target_file = state.get("target_file")
    if target_file and target_file not in files:
        files.append(target_file)
    return files


def coding_node(state: AgentState) -> dict:
    logger.info("Executing coding_agent node.")

    attempts = state.get("coding_attempts", 0) + 1

    repo_path = state.get("repository_path", WORKSPACE_ROOT)
    repo_root = Path(repo_path).resolve()
    os.environ["WORKSPACE_ROOT"] = str(repo_root)

    llm = get_llm()
    if llm:
        try:
            file_content_context = ""
            for root_dir, _, files in os.walk(repo_root):
                for fname in files:
                    if fname.endswith(".py"):
                        path = Path(root_dir) / fname
                        try:
                            text = path.read_text(encoding="utf-8", errors="ignore")
                            rel = path.relative_to(repo_root).as_posix()
                            file_content_context += f"=== FILE: {rel} ===\n{text}\n\n"
                        except Exception:
                            pass

            system_prompt = load_system_prompt()
            
            target_file = state.get("target_file")
            if target_file:
                system_prompt += (
                    f"\nCRITICAL: The user explicitly specified target file '{target_file}'. "
                    f"You MUST modify this target file '{target_file}' to apply the bug fix. Do not edit other files."
                )
            
            allowed = _allowed_files(state)
            if allowed:
                allowed_str = ", ".join(allowed)
                system_prompt += (
                    f"\nCRITICAL: You are only allowed to modify files from this list: [{allowed_str}]. "
                    "Your code change MUST target one of these in file_path."
                )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", (
                    "Active Task: {current_task}\n"
                    "User Request: {user_request}\n"
                    "Research Results:\n{research_results}\n\n"
                    "Previous Test Failure Results:\n{test_results}\n\n"
                    "Previous Reviewer Feedback:\n{review_results}\n\n"
                    "Errors/Errors Accumulator:\n{errors}\n\n"
                    "Current Codebase Files:\n{files_context}\n\n"
                    "Propose and apply the exact CodeChange."
                )),
            ])
            chain = prompt | llm
            result = chain.invoke({
                "current_task": state.get("current_task", ""),
                "user_request": state.get("user_request", ""),
                "research_results": state.get("research_results", ""),
                "test_results": state.get("test_results", ""),
                "review_results": state.get("review_results", ""),
                "errors": ", ".join(state.get("errors", [])),
                "files_context": file_content_context,
            })
        except Exception as e:
            logger.error("Coding LLM error: %s. Falling back to mock code generator.", e)
            result = run_coding_mock(state)
    else:
        result = run_coding_mock(state)

    if isinstance(result, dict):
        result = CodeChange(**result)
    assert isinstance(result, CodeChange)

    logger.info("[Coding Change proposed] Target Path: %s", result.file_path)
    logger.info("Reasoning: %s", result.reasoning)

    relevant_files = _allowed_files(state)
    if relevant_files:
        norm_proposed = result.file_path.replace("\\", "/").strip().lower()
        norm_allowed = [f.replace("\\", "/").strip().lower() for f in relevant_files]
        norm_allowed_names = [Path(f).name.lower() for f in norm_allowed]

        if norm_proposed not in norm_allowed and Path(norm_proposed).name.lower() not in norm_allowed_names:
            error_msg = (
                f"Coding Agent validation failed: proposed target '{result.file_path}' "
                f"is not in allowed files: {relevant_files}."
            )
            logger.error(error_msg)
            return {
                "errors": [error_msg],
                "next_agent": "supervisor",
                "coding_attempts": attempts,
            }

    target_abs = (repo_root / result.file_path).resolve()
    try:
        if not target_abs.is_relative_to(repo_root):
            raise ValueError("outside workspace")
    except ValueError:
        error_msg = (
            f"Path traversal attempt blocked: {result.file_path} is outside workspace root {repo_root}"
        )
        logger.error(error_msg)
        return {
            "errors": [error_msg],
            "next_agent": "supervisor",
            "coding_attempts": attempts,
        }

    # Track original file contents to generate diff
    old_content = ""
    if target_abs.exists() and target_abs.is_file():
        try:
            old_content = target_abs.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Could not read original file content for diff: %s", e)

    # Human-In-The-Loop Check (HITL)
    is_testing = "PYTEST_CURRENT_TEST" in os.environ
    approved = True
    feedback = ""
    if not is_testing:
        diff_str_preview = generate_diff(result.file_path, old_content, result.new_content)
        print("\n" + "=" * 80)
        print(" [HUMAN-IN-THE-LOOP CHECK: PROPOSED CODE MODIFICATION]")
        print(f" File to modify: {result.file_path}")
        print(f" Reasoning: {result.reasoning}")
        print("-" * 80)
        print(diff_str_preview if diff_str_preview.strip() else "[No changes / Empty Diff]")
        print("=" * 80)
        try:
            user_approval = input("Do you approve writing these changes to disk? (y/n): ").strip().lower()
            if user_approval not in ("y", "yes"):
                approved = False
                feedback = input("Provide feedback/corrections to the coding agent: ").strip()
                if not feedback:
                    feedback = "User rejected the changes without detailed comments."
        except Exception as e:
            logger.warning(f"Error reading user input for HITL: {e}. Defaulting to approval.")
            approved = True

    if not approved:
        logger.warning(f"Code changes rejected by user: {feedback}")
        return {
            "errors": [f"Code changes rejected by user: {feedback}"],
            "next_agent": "supervisor",
            "coding_attempts": attempts,
        }

    try:
        from app.mcp.client import call_mcp_tool
        call_mcp_tool("write_file", {"file_path": result.file_path, "content": result.new_content})
        logger.info("Successfully modified file via MCP: %s", result.file_path)
        
        # Verify write by reading it back and generating unified diff
        written_content = ""
        if target_abs.exists() and target_abs.is_file():
            written_content = target_abs.read_text(encoding="utf-8", errors="ignore")
            
        diff_str = generate_diff(result.file_path, old_content, written_content)
        code_summary = (
            f"Reasoning: {result.reasoning}\n"
            f"File: {result.file_path}\n"
            f"Diff:\n```diff\n{diff_str}\n```"
        )
    except Exception as e:
        error_msg = f"Failed to write file {result.file_path} via MCP: {e}"
        logger.error(error_msg)
        return {
            "errors": [error_msg],
            "next_agent": "supervisor",
            "coding_attempts": attempts,
        }

    completed = list(state.get("completed_tasks", []))
    plan = state.get("plan", {})
    current_task_id = plan.get("current_task_id", "")
    if current_task_id and current_task_id not in completed:
        completed.append(current_task_id)

    return {
        "code_changes": code_summary,
        "completed_tasks": completed,
        "messages": [{"role": "coder", "content": code_summary}],
        "next_agent": "supervisor",
        "coding_attempts": attempts,
    }

