import logging
from pathlib import Path
from app.rag.vector_store import ChromaVectorStore, DEFAULT_DB_DIR

logger = logging.getLogger(__name__)

def retrieve_codebase_context(query: str, repo_path: str, k: int = 5, in_memory: bool = False) -> str:
    """
    Performs similarity search on indexed codebase chunks and outputs
    a formatted text block of findings. Automatically indexes the repository if the store is empty.
    """
    # Create persistent store based on a hash of the repo path so repos don't conflict
    if in_memory:
        store = ChromaVectorStore(in_memory=True)
    else:
        # Create repo-specific subfolder in default persistent store path
        repo_name = Path(repo_path).name
        db_path = DEFAULT_DB_DIR / repo_name
        store = ChromaVectorStore(persist_dir=db_path, in_memory=False)
        
    # Index repository if store is empty
    count = store.collection.count()
    if count == 0:
        logger.info(f"Vector store is empty (count={count}). Indexing repo: {repo_path}")
        store.index_repository(repo_path)
    else:
        logger.info(f"Using existing vector store index with {count} chunks.")
        
    results = store.similarity_search(query, k=k)
    if not results:
        return "No relevant repository context found."
        
    context_blocks = []
    for item in results:
        meta = item["metadata"]
        file_path = meta.get("file_path", "unknown")
        chunk_idx = meta.get("chunk", 0)
        text = item["text"]
        
        block = (
            f"=== FILE: {file_path} [Chunk {chunk_idx}] ===\n"
            f"{text}\n"
            f"============================================\n"
        )
        context_blocks.append(block)
        
    return "\n".join(context_blocks)
