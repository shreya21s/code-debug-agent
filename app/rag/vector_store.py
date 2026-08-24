import os
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from app.config import PROJECT_ROOT
from app.rag.embeddings import get_embeddings
from app.rag.ingest import scan_repository, chunk_documents

logger = logging.getLogger(__name__)

# Default persistent directory for vector store
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "vector_store"

class SimpleVectorStore:
    """
    A pure Python & Numpy vector database.
    Used as a robust fallback on environments (like Windows) where the
    Chroma Rust SQLite module throws access violations.
    """
    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.data = []  # List of dicts with: id, text, metadata, embedding
        self.load()

    def load(self):
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "rb") as f:
                    self.data = pickle.load(f)
                logger.info(f"Loaded {len(self.data)} vector records from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load vector records: {e}")
                self.data = []

    def save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump(self.data, f)
            logger.info(f"Saved {len(self.data)} vector records to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save vector records: {e}")

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]):
        # Index by ID to avoid duplicates
        existing_ids = {item["id"]: idx for idx, item in enumerate(self.data)}
        
        for idx in range(len(ids)):
            record_id = ids[idx]
            record = {
                "id": record_id,
                "embedding": embeddings[idx],
                "metadata": metadatas[idx],
                "text": documents[idx]
            }
            if record_id in existing_ids:
                # Update existing
                self.data[existing_ids[record_id]] = record
            else:
                self.data.append(record)
        self.save()

    def query(
        self,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 5,
        query_embeddings: Optional[List[List[float]]] = None
    ) -> Any:
        if query_embedding is None:
            if query_embeddings is not None and len(query_embeddings) > 0:
                query_embedding = query_embeddings[0]
            else:
                raise ValueError("Either query_embedding or query_embeddings must be provided")

        if not self.data:
            if query_embeddings is not None:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            return []
            
        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        
        scores = []
        for item in self.data:
            i_vec = np.array(item["embedding"])
            i_norm = np.linalg.norm(i_vec)
            
            if q_norm == 0 or i_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(q_vec, i_vec) / (q_norm * i_norm))
                
            scores.append((similarity, item))
            
        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_k = scores[:n_results]
        
        if query_embeddings is not None:
            # Return Chroma's dict format
            return {
                "ids": [[item["id"] for similarity, item in top_k]],
                "documents": [[item["text"] for similarity, item in top_k]],
                "metadatas": [[item["metadata"] for similarity, item in top_k]],
                "distances": [[1.0 - similarity for similarity, item in top_k]]
            }
            
        # Format results to match Chroma's return format
        return [{
            "id": item["id"],
            "text": item["text"],
            "metadata": item["metadata"],
            "score": similarity
        } for similarity, item in top_k]

    def count(self) -> int:
        return len(self.data)


class ChromaVectorStore:
    """
    Wrapper around vector stores. Automatically selects SimpleVectorStore
    on Windows to prevent fatal DLL access violations, or uses Chroma Client.
    """
    def __init__(self, persist_dir: Path = DEFAULT_DB_DIR, in_memory: bool = False):
        self.persist_dir = persist_dir
        self.in_memory = in_memory
        self.embeddings = get_embeddings()
        
        # FORCE fallback vector store on Windows (os.name == 'nt') to guarantee process stability
        # or if explicitly configured via environment
        use_fallback = (os.name == "nt") or (os.getenv("VECTOR_DB_BACKEND", "simple") == "simple")
        
        if use_fallback:
            logger.info("Using SimpleVectorStore backend (portable pure-Python/numpy).")
            db_file = persist_dir / "vector_records.pkl" if not in_memory else Path(":memory:")
            # For in-memory fallback, write to a temp file path or just mock in memory
            if in_memory:
                import tempfile
                import uuid
                db_file = Path(tempfile.gettempdir()) / f"temp_in_memory_vectors_{uuid.uuid4().hex}.pkl"
            self.backend = SimpleVectorStore(persist_path=db_file)
            self.is_fallback = True
        else:
            logger.info("Using ChromaDB native backend.")
            try:
                import chromadb
                if in_memory:
                    self.client = chromadb.EphemeralClient()
                else:
                    self.persist_dir.mkdir(parents=True, exist_ok=True)
                    self.client = chromadb.PersistentClient(path=str(persist_dir))
                self.collection = self.client.get_or_create_collection(
                    name="codebase_chunks",
                    metadata={"hnsw:space": "cosine"}
                )
                self.is_fallback = False
            except Exception as e:
                logger.error(f"Failed to load ChromaDB: {e}. Falling back to SimpleVectorStore.")
                db_file = persist_dir / "vector_records.pkl"
                self.backend = SimpleVectorStore(persist_path=db_file)
                self.is_fallback = True

    @property
    def collection(self):
        # Compatibility property for count check
        if self.is_fallback:
            return self.backend
        return self._collection

    @collection.setter
    def collection(self, val):
        self._collection = val

    def index_repository(self, repo_path: str):
        """Scans, chunks, and indexes a repository's files."""
        logger.info(f"Scanning repository for indexing: {repo_path}")
        documents = scan_repository(repo_path)
        if not documents:
            logger.warning("No files found or scanned in repository.")
            return
            
        chunks = chunk_documents(documents)
        logger.info(f"Split {len(documents)} files into {len(chunks)} chunks.")
        
        # Prepare data
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{m['file_path']}_chunk_{m['chunk']}" for m in metadatas]
        
        # Generate embeddings
        logger.info("Generating embeddings for chunks...")
        embedded_list = self.embeddings.embed_documents(texts)
        
        if self.is_fallback:
            self.backend.upsert(
                ids=ids,
                embeddings=embedded_list,
                metadatas=metadatas,
                documents=texts
            )
        else:
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                self.collection.upsert(
                    ids=ids[i:end_idx],
                    embeddings=embedded_list[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    documents=texts[i:end_idx]
                )
        logger.info(f"Indexed {len(chunks)} chunks successfully.")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector store for top-k similar chunks."""
        query_vector = self.embeddings.embed_query(query)
        
        if self.is_fallback:
            return self.backend.query(query_vector, n_results=k)
            
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for idx in range(len(docs)):
                formatted_results.append({
                    "id": ids[idx],
                    "text": docs[idx],
                    "metadata": metadatas[idx],
                    "score": 1.0 - distances[idx]
                })
        return formatted_results
