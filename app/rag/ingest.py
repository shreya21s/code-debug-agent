import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Extensions we care about for software engineering context
SUPPORTED_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", 
    ".toml", ".ini", ".cfg", ".sql", ".sh", ".js", ".ts", ".html", ".css"
}

# Directories to ignore
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", 
    "build", "dist", "eggs", "*.egg-info", ".pytest_cache", ".chroma", "data"
}

def should_ignore(path: Path, root_path: Path) -> bool:
    """Helper to check if a path should be skipped."""
    # Check parent directory names
    for part in path.relative_to(root_path).parts[:-1]:
        if part in IGNORED_DIRS:
            return True
        # Wildcard matches if needed
        if part.startswith(".") or part.endswith("-info"):
            return True
    # Ignore hidden files
    if path.name.startswith("."):
        return True
    return False

def scan_repository(repo_path: str) -> List[Dict[str, Any]]:
    """
    Scans the repository and returns a list of dictionaries with text content
    and file metadata for each valid file.
    """
    root = Path(repo_path).resolve()
    documents = []
    
    if not root.exists():
        logger.error(f"Repository path '{repo_path}' does not exist.")
        return []
        
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS:
            if should_ignore(p, root):
                continue
            try:
                # Read content as text
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Check for secrets or very large files
                if len(content) > 1000000: # 1MB limit for single file
                    logger.warning(f"Skipping too large file: {p}")
                    continue
                    
                documents.append({
                    "content": content,
                    "file_path": str(p.relative_to(root).as_posix()),
                    "file_type": p.suffix
                })
            except Exception as e:
                logger.error(f"Failed to read file {p}: {e}")
                
    return documents

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Splits documents into smaller chunks and retains metadata (path, chunk_index, type).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    for doc in documents:
        file_chunks = splitter.split_text(doc["content"])
        for idx, text in enumerate(file_chunks):
            chunks.append({
                "text": text,
                "metadata": {
                    "file_path": doc["file_path"],
                    "chunk": idx,
                    "file_type": doc["file_type"]
                }
            })
    return chunks
