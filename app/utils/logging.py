import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("agent_observability")

class ExecutionTracer:
    """
    Tracks and records the multi-agent execution path, steps, and tool calls
    for observability reporting.
    """
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.tool_calls_count = 0

    def reset(self):
        self.steps = []
        self.start_time = time.time()
        self.tool_calls_count = 0

    def record_step(self, node: str, agent: str, task: str, action_details: str, status: str = "success"):
        """Records a single agent state transition or tool invocation."""
        elapsed = time.time() - self.start_time
        step_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "graph_node": node,
            "agent": agent,
            "task": task,
            "action": action_details,
            "status": status
        }
        self.steps.append(step_record)
        logger.info(
            f"[{node}] Agent: {agent} | Task: {task} | Status: {status} | Details: {action_details[:100]}..."
        )

    def record_tool_call(self, tool_name: str, args: Dict[str, Any], result_summary: str):
        """Records an MCP or A2A remote call."""
        self.tool_calls_count += 1
        elapsed = time.time() - self.start_time
        tool_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "graph_node": "tool_invocation",
            "agent": "system",
            "task": f"Call tool: {tool_name}",
            "action": f"Args: {args} | Result: {result_summary}",
            "status": "success"
        }
        self.steps.append(tool_record)
        logger.info(f"[Tool Call #{self.tool_calls_count}] {tool_name} | Args: {args} | Result: {result_summary[:80]}...")

    def generate_report(self) -> str:
        """Generates a structured markdown execution report."""
        total_time = round(time.time() - self.start_time, 2)
        
        report_lines = [
            "==================================================",
            "             AGENT EXECUTION REPORT              ",
            "==================================================",
            f"Total Run Time: {total_time} seconds",
            f"Total Tool Calls: {self.tool_calls_count}",
            f"Steps Executed: {len(self.steps)}",
            "--------------------------------------------------"
        ]
        
        for idx, s in enumerate(self.steps):
            report_lines.append(
                f"{idx+1}. [{s['timestamp']}] ({s['elapsed_seconds']}s) - Node: {s['graph_node']}\n"
                f"   Agent: {s['agent']} | Task: {s['task']}\n"
                f"   Action: {s['action']}\n"
            )
            
        report_lines.append("==================================================")
        return "\n".join(report_lines)

# Global singleton tracer
tracer = ExecutionTracer()


def setup_logger(log_level: str = "INFO"):
    """Configures global python logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
        force=True  # Override root handlers if already set
    )
