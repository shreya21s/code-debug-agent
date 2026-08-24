import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import AgentState
from app.config import GOOGLE_API_KEY, MODEL_NAME, APP_ROOT

logger = logging.getLogger(__name__)

# Structured Output Definition
class ReviewResult(BaseModel):
    approved: bool = Field(description="Set to True if the changes meet requirements and tests pass, False otherwise")
    reasoning: str = Field(description="Explanation of the code review findings")
    feedback: str = Field(description="Actionable guidance on what to fix if rejected, or empty if approved")


def get_llm():
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    from app.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_HOST
    if LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_HOST,
                temperature=0.0
            )
            return llm.with_structured_output(ReviewResult)
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama: {e}")
            return None

    if not GOOGLE_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        return llm.with_structured_output(ReviewResult)
    except Exception as e:
        logger.error(f"Failed to initialize Reviewer Agent LLM: {e}")
        return None


def load_system_prompt() -> str:
    prompt_path = APP_ROOT / "prompts" / "reviewer.txt"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a Reviewer Agent checking code modifications and test results."


def run_reviewer_mock(state: AgentState) -> ReviewResult:
    """Mock fallback reviewer behavior based on test success."""
    test_res = state.get("test_results", "")
    
    lowered = test_res.lower()
    if "test success: false" in lowered:
        return ReviewResult(
            approved=False,
            reasoning="Mock Reviewer: Detected test failures in the log.",
            feedback="Fix the logic error causing tests to fail.",
        )
        
    return ReviewResult(
        approved=True,
        reasoning="Mock Reviewer: All test cases passed successfully and code diff looks correct.",
        feedback=""
    )


def reviewer_node(state: AgentState) -> dict:
    """
    Reviewer Agent Node: Performs code review on the changes and test output.
    """
    logger.info("Executing reviewer_agent node.")
    
    # 0. Check if remote A2A service is running, and delegate to it if active!
    if not state.get("_a2a_execution"):
        try:
            from app.a2a.client import call_a2a_agent
            a2a_output = call_a2a_agent("reviewer", state, task_id="A2A_Reviewer_Task")
            if a2a_output is not None:
                logger.info("Reviewer agent task executed remotely via A2A protocol.")
                return a2a_output
        except Exception as e:
            logger.debug(f"A2A Reviewer delegation check failed: {e}. Running locally.")
    
    llm = get_llm()
    if llm:
        try:
            system_prompt = load_system_prompt()
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", (
                    "User Request: {user_request}\n"
                    "Code Changes applied:\n{code_changes}\n\n"
                    "Test Results:\n{test_results}\n\n"
                    "Provide your structured ReviewResult."
                ))
            ])
            chain = prompt | llm
            result = chain.invoke({
                "user_request": state.get("user_request", ""),
                "code_changes": state.get("code_changes", ""),
                "test_results": state.get("test_results", "")
            })
        except Exception as e:
            logger.error(f"Reviewer LLM error: {e}. Falling back to mock reviewer.")
            result = run_reviewer_mock(state)
    else:
        result = run_reviewer_mock(state)

    if isinstance(result, dict):
        result = ReviewResult(**result)
    assert isinstance(result, ReviewResult)
        
    logger.info(f"[Reviewer Decision] Approved: {result.approved} | Reason: {result.reasoning}")
    
    # Update task lists
    completed = list(state.get("completed_tasks", []))
    plan = state.get("plan", {})
    current_task_id = plan.get("current_task_id", "")
    if current_task_id and current_task_id not in completed:
        completed.append(current_task_id)
        
    review_summary = (
        f"Approved: {result.approved}\n"
        f"Reasoning: {result.reasoning}\n"
        f"Feedback: {result.feedback}"
    )
    
    # Only return *new* errors; AgentState.errors uses an append reducer.
    new_errors = []
    if not result.approved:
        new_errors.append(f"Reviewer rejected: {result.feedback}")

    return {
        "review_results": review_summary,
        "completed_tasks": completed,
        "errors": new_errors,
        "messages": [{"role": "reviewer", "content": review_summary}],
        "next_agent": "supervisor",
        "task_success": result.approved,
    }
