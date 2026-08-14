import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project directory structure
APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

# API Key for Gemini API
# langchain-google-genai can automatically read GOOGLE_API_KEY from environment,
# but we expose it here for clarity and manual verification.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

# Workspace root configuration for MCP filesystem restriction
# Resolve relative paths relative to the project root directory
workspace_env = os.getenv("WORKSPACE_ROOT", "./examples/sample_project")
WORKSPACE_ROOT = Path(workspace_env).resolve() if Path(workspace_env).is_absolute() else (PROJECT_ROOT / workspace_env).resolve()

# A2A and MCP service configurations
A2A_RESEARCH_PORT = int(os.getenv("A2A_RESEARCH_PORT", "8001"))
A2A_REVIEWER_PORT = int(os.getenv("A2A_REVIEWER_PORT", "8002"))
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8003"))

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
