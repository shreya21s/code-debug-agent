"""Configuration loaded from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", PROJECT_ROOT / "demo_workspace")).resolve()
LLM_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4o-mini")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

CODING_MCP_SERVER = PROJECT_ROOT / "mcp_servers" / "coding_server.py"
TESTING_MCP_SERVER = PROJECT_ROOT / "mcp_servers" / "testing_server.py"
