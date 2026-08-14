import logging
import requests
from typing import Dict, Any, Optional

from app.config import A2A_RESEARCH_PORT, A2A_REVIEWER_PORT
from app.a2a.agents import A2ATaskRequest, AgentCapability

logger = logging.getLogger(__name__)

def query_capabilities(port: int) -> Optional[list]:
    """Queries remote A2A service capabilities."""
    url = f"http://127.0.0.1:{port}/capabilities"
    try:
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.debug(f"A2A Server on port {port} offline or unreachable: {e}")
    return None


def call_a2a_agent(agent_type: str, state: Dict[str, Any], task_id: str = "A2A_T") -> Optional[Dict[str, Any]]:
    """
    Delegates a task to a remote A2A service.
    Resolves target port based on agent type.
    """
    port = A2A_RESEARCH_PORT if agent_type == "research" else A2A_REVIEWER_PORT
    url = f"http://127.0.0.1:{port}/execute"
    
    # Verify capabilities first to check if server is active and supports agent type
    caps = query_capabilities(port)
    if not caps:
        logger.info(f"A2A service for {agent_type} (port {port}) is offline. Falling back to local execution.")
        return None
        
    # Check if agent type is supported in capabilities advertisement
    supported = any(c.get("agent_type") == agent_type for c in caps)
    if not supported:
        logger.warning(f"Remote server on port {port} does not advertise capability for '{agent_type}'.")
        return None
        
    payload = A2ATaskRequest(
        task_id=task_id,
        agent_type=agent_type,
        state=state
    )
    
    logger.info(f"Sending A2A request to {url} for agent: {agent_type}")
    try:
        res = requests.post(url, json=payload.model_dump(), timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                logger.info(f"Remote A2A task {task_id} completed successfully.")
                try:
                    from app.utils.logging import tracer
                    tracer.record_tool_call(
                        f"a2a_{agent_type}", 
                        {"task_id": task_id}, 
                        f"Remote delegation successful. Output keys: {list(data.get('output', {}).keys())}"
                    )
                except Exception as e:
                    logger.debug(f"Tracer logging failed: {e}")
                return data.get("output")
            else:
                logger.error(f"Remote A2A task returned error status: {data.get('error')}")
        else:
            logger.error(f"A2A HTTP error code: {res.status_code} | details: {res.text}")
    except Exception as e:
        logger.error(f"A2A connection failed to {url}: {e}")
        
    return None
