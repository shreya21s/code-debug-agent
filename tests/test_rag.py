import os
from pathlib import Path
import pytest

from app.rag.ingest import scan_repository, chunk_documents, should_ignore
from app.rag.embeddings import get_embeddings, MockEmbeddings
from app.rag.vector_store import ChromaVectorStore
from app.rag.retriever import retrieve_codebase_context

def test_should_ignore():
    """Verify should_ignore filters out typical system and lock files/folders."""
    root = Path("/dummy/project")
    assert should_ignore(root / ".git" / "config", root) is True
    assert should_ignore(root / "__pycache__" / "main.pyc", root) is True
    assert should_ignore(root / "src" / "main.py", root) is False
    assert should_ignore(root / ".venv" / "lib" / "site-packages" / "pip", root) is True
    assert should_ignore(root / "node_modules" / "express" / "index.js", root) is True

def test_mock_embeddings():
    """Verify deterministic mock embedding size and consistency."""
    embeds = MockEmbeddings()
    vector1 = embeds.embed_query("hello world")
    vector2 = embeds.embed_query("hello world")
    vector3 = embeds.embed_query("different text")
    
    assert len(vector1) == 768
    assert vector1 == vector2
    assert vector1 != vector3

def test_scan_and_chunk_mock_repo(tmp_path):
    """Scan and chunk mock repository files in a temp directory."""
    # Create mock files
    repo = tmp_path / "mock_repo"
    repo.mkdir()
    
    file1 = repo / "main.py"
    file1.write_text("def main():\n    print('Hello World')\n", encoding="utf-8")
    
    file2 = repo / "README.md"
    file2.write_text("# Mock Repository\nThis is a RAG test.\n", encoding="utf-8")
    
    # Hidden folder - should ignore
    git_dir = repo / ".git"
    git_dir.mkdir()
    git_file = git_dir / "config"
    git_file.write_text("[core]\n    repositoryformatversion = 0\n", encoding="utf-8")
    
    docs = scan_repository(str(repo))
    assert len(docs) == 2
    paths = {doc["file_path"] for doc in docs}
    assert "main.py" in paths
    assert "README.md" in paths
    
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "file_path" in chunk["metadata"]
        assert "chunk" in chunk["metadata"]
        assert "file_type" in chunk["metadata"]

def test_chroma_indexing_and_retrieval(tmp_path):
    """Test Chroma index and retrieval in-memory using MockEmbeddings."""
    # Create mock repo
    repo = tmp_path / "rag_repo"
    repo.mkdir()
    
    file_auth = repo / "auth.py"
    file_auth.write_text(
        "def login_user(username, password):\n"
        "    if username == 'admin' and password == 'secret':\n"
        "        return {'status': 'success', 'token': 'jwt_token'}\n"
        "    return {'status': 'fail'}\n",
        encoding="utf-8"
    )
    
    file_calc = repo / "calc.py"
    file_calc.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8"
    )
    
    # Initialize in-memory store
    store = ChromaVectorStore(in_memory=True)
    store.index_repository(str(repo))
    
    assert store.collection.count() > 0
    
    # Test query retrieval
    results = store.similarity_search("login_user admin secret token", k=1)
    assert len(results) == 1
    assert "file_path" in results[0]["metadata"]
    assert "text" in results[0]
    
    # Test high-level retriever function
    context = retrieve_codebase_context("multiply", str(repo), k=1, in_memory=True)
    assert "calc.py" in context or "auth.py" in context

