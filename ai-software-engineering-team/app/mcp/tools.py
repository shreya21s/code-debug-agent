import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from app.config import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

def validate_safe_path(path: str, workspace_root: Path | None = None) -> Path:
    """
    Validates that the given path is safely within the workspace_root directory.
    Raises ValueError if path is out of bounds or traversal is attempted.
    """
    if workspace_root is None:
        workspace_root = WORKSPACE_ROOT
        
    resolved_root = workspace_root.resolve()
    target_path = Path(path)
    
    if not target_path.is_absolute():
        # Treat relative to workspace root
        target_path = resolved_root / target_path
        
    resolved_target = target_path.resolve()
    
    # Path boundary enforcement
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError(f"Path traversal detected! Target path '{resolved_target}' is outside workspace '{resolved_root}'.")
        
    return resolved_target


def list_files(subdir: str = "") -> List[str]:
    """Lists files recursively in the workspace (or a subdirectory)."""
    try:
        target_dir = validate_safe_path(subdir)
        if not target_dir.is_dir():
            return []
            
        file_paths = []
        # Ignore common system files and caches
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".chroma"}
        
        for root, dirs, files in os.walk(target_dir):
            # Modify dirs in place to prevent visiting ignored folders
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            
            for file in files:
                if file.startswith("."):
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(WORKSPACE_ROOT).as_posix()
                file_paths.append(rel_path)
                
        return file_paths
    except Exception as e:
        logger.error(f"Error in list_files: {e}")
        raise


def read_file(file_path: str) -> str:
    """Reads and returns the text content of a workspace file."""
    try:
        safe_path = validate_safe_path(file_path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not safe_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error in read_file: {e}")
        raise


def write_file(file_path: str, content: str) -> str:
    """Overwrites or creates a workspace file with the specified content."""
    try:
        safe_path = validate_safe_path(file_path)
        # Create directories if they do not exist
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"File written successfully via MCP tool: {safe_path}")
        return f"Successfully wrote {file_path} ({len(content)} characters)."
    except Exception as e:
        logger.error(f"Error in write_file: {e}")
        raise


def search_files(query: str) -> List[Dict[str, Any]]:
    """Simple text search inside workspace source files (equivalent to grep)."""
    try:
        results = []
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".chroma"}
        
        for root, dirs, files in os.walk(WORKSPACE_ROOT):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            
            for file in files:
                # Only search text files
                if not file.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini")):
                    continue
                    
                full_path = Path(root) / file
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    for line_idx, line in enumerate(lines):
                        if query.lower() in line.lower():
                            rel_path = full_path.relative_to(WORKSPACE_ROOT).as_posix()
                            results.append({
                                "file_path": rel_path,
                                "line_number": line_idx + 1,
                                "content": line.strip()
                            })
                except Exception as e:
                    # Skip files we cannot read
                    continue
                    
        return results[:50]  # Cap results at 50 to prevent flooding
    except Exception as e:
        logger.error(f"Error in search_files: {e}")
        raise


def run_tests(command_arg: str = "") -> Dict[str, Any]:
    """
    Runs pytest in the workspace safely.
    Strictly forbids running arbitrary commands; only 'pytest' or 'python -m pytest' is executed.
    """
    try:
        import shlex
        # Strictly enforce pytest command arguments only (e.g. specific file target)
        # Avoid arbitrary command injections
        python_exe = sys.executable
        cmd = [python_exe, "-m", "pytest", "-v"]
        
        # If command_arg specifies a file target and/or options, parse them safely
        if command_arg:
            clean_arg = command_arg.replace("pytest", "").replace("python -m", "").strip()
            if clean_arg:
                parts = shlex.split(clean_arg)
                i = 0
                while i < len(parts):
                    part = parts[i]
                    if part.startswith("-"):
                        cmd.append(part)
                        # If it's a flag that takes a value (like -k, -m, -c, -o, -p) and there is a next part
                        if part in {"-k", "-m", "-c", "-o", "-p", "--ignore"} and i + 1 < len(parts):
                            cmd.append(parts[i + 1])
                            i += 2
                        else:
                            i += 1
                    else:
                        # Validate it is a path within workspace
                        safe_target = validate_safe_path(part)
                        cmd.append(str(safe_target))
                        i += 1
                
        logger.info(f"Executing approved test command via MCP: {' '.join(cmd)}")
        res = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": (res.returncode == 0),
            "command": " ".join(cmd),
            "stdout": res.stdout or "",
            "stderr": res.stderr or "",
            "returncode": res.returncode
        }
    except Exception as e:
        logger.error(f"Error in run_tests: {e}")
        raise
