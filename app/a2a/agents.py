import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from app.state import AgentState
from app.agents.research_agent import research_node
from app.agents.reviewer_agent import reviewer_node

logger = logging.getLogger(__name__)

class A2ATaskRequest(BaseModel):
    task_id: str
    agent_type: str = Field(description="The target agent type: research or reviewer")
    state: AgentState = Field(description="The current AgentState context required for execution")

class A2ATaskResponse(BaseModel):
    status: str = Field(description="success or failed")
    output: Dict[str, Any] = Field(description="State modifications to merge back into the main state graph")
    error: str = Field(default="", description="Error details if status is failed")

class AgentCapability(BaseModel):
    agent_type: str
    description: str
    inputs_required: List[str]
    outputs_provided: List[str]

def get_agent_capabilities() -> List[AgentCapability]:
    """Advertises capabilities of A2A services."""
    return [
        AgentCapability(
            agent_type="research",
            description="Analyzes codebases, runs semantic search (RAG), and diagnoses bug root causes.",
            inputs_required=["user_request", "repository_path", "current_task"],
            outputs_provided=["research_results", "completed_tasks", "messages", "next_agent"]
        ),
        AgentCapability(
            agent_type="reviewer",
            description="Reviews code modifications and pytest results to approve/reject changes.",
            inputs_required=["user_request", "code_changes", "test_results"],
            outputs_provided=["review_results", "completed_tasks", "errors", "messages", "next_agent"]
        )
    ]


def process_a2a_task(request: A2ATaskRequest) -> A2ATaskResponse:
    """Invokes the appropriate local agent logic on the payload and returns the output."""
    logger.info(f"Processing A2A task {request.task_id} for agent: {request.agent_type}")
    
    agent_type = request.agent_type.lower()
    state = request.state
    state["_a2a_execution"] = True
    
    try:
        if agent_type == "research":
            # Call the Research agent node
            output = research_node(state)
            return A2ATaskResponse(status="success", output=output)
            
        elif agent_type == "reviewer":
            # Call the Reviewer agent node
            output = reviewer_node(state)
            return A2ATaskResponse(status="success", output=output)
            
        else:
            return A2ATaskResponse(
                status="failed", 
                output={}, 
                error=f"Unsupported A2A agent type: {request.agent_type}"
            )
    except Exception as e:
        logger.exception(f"A2A task processing exception: {e}")
        return A2ATaskResponse(status="failed", output={}, error=str(e))
