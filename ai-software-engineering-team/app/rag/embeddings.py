import logging
import hashlib
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class MockEmbeddings(Embeddings):
    """
    Deterministic mock embeddings class for offline testing.
    Generates consistent float vectors based on content hashes.
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            # Generate deterministic values using hashlib
            hasher = hashlib.md5(text.encode('utf-8'))
            seed = int(hasher.hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            # Standard vector size of 768 dimensions
            vector = rng.uniform(-1, 1, 768).tolist()
            results.append(vector)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    """
    Returns GoogleGenAIEmbeddings if API key is set,
    otherwise falls back to MockEmbeddings.
    """
    if GOOGLE_API_KEY:
        try:
            logger.info("Initializing GoogleGenAIEmbeddings using models/text-embedding-004.")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=GOOGLE_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Embeddings: {e}. Using MockEmbeddings.")
            return MockEmbeddings()
    else:
        logger.warning("GOOGLE_API_KEY not configured. Falling back to deterministic MockEmbeddings.")
        return MockEmbeddings()
