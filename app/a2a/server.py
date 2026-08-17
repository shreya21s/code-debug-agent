import uvicorn
from fastapi import FastAPI, HTTPException
from typing import List

from app.a2a.agents import (
    A2ATaskRequest, A2ATaskResponse, 
    AgentCapability, get_agent_capabilities, process_a2a_task
)

app = FastAPI(
    title="Agent-to-Agent (A2A) Service Server",
    description="Exposes Research and Reviewer agents as independent, network-accessible A2A services."
)

@app.get("/capabilities", response_model=List[AgentCapability])
def capabilities():
    """Endpoint advertising A2A capabilities."""
    return get_agent_capabilities()

@app.post("/execute", response_model=A2ATaskResponse)
def execute_task(request: A2ATaskRequest):
    """Endpoint to delegate tasks to independent agents."""
    response = process_a2a_task(request)
    if response.status == "failed":
        raise HTTPException(status_code=400, detail=response.error)
    return response


def run_a2a_server(port: int = 8001):
    """Helper to start uvicorn programmatically."""
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    import sys
    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Starting A2A Server on port {port}...")
    run_a2a_server(port)
