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


class SafeGoogleGenAIEmbeddings(Embeddings):
    """
    Wrapper for GoogleGenerativeAIEmbeddings that catches network/API errors
    (such as unsupported models, invalid API keys, or connection timeouts)
    and falls back to MockEmbeddings instead of crashing.
    """
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.mock = MockEmbeddings()
        try:
            self.impl = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
        except Exception as e:
            logger.error(f"Failed to instantiate GoogleGenerativeAIEmbeddings: {e}")
            self.impl = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.impl:
            return self.mock.embed_documents(texts)
        try:
            return self.impl.embed_documents(texts)
        except Exception as e:
            logger.error(f"Google embedding API call failed: {e}. Falling back to MockEmbeddings.")
            return self.mock.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not self.impl:
            return self.mock.embed_query(text)
        try:
            return self.impl.embed_query(text)
        except Exception as e:
            logger.error(f"Google embedding query call failed: {e}. Falling back to MockEmbeddings.")
            return self.mock.embed_query(text)


class SafeOllamaEmbeddings(Embeddings):
    """
    Wrapper for OllamaEmbeddings that catches connection errors and falls back to MockEmbeddings.
    """
    def __init__(self, model: str, base_url: str):
        self.mock = MockEmbeddings()
        try:
            from langchain_ollama import OllamaEmbeddings
            self.impl = OllamaEmbeddings(
                model=model,
                base_url=base_url
            )
            # Eagerly verify embedding support to log warning early and fail gracefully
            self.impl.embed_query("test")
            logger.info("Ollama embeddings successfully verified.")
        except Exception as e:
            logger.warning(
                f"Ollama embeddings are unavailable or unsupported by the server: {e}. "
                "Falling back to MockEmbeddings."
            )
            self.impl = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.impl:
            return self.mock.embed_documents(texts)
        try:
            return self.impl.embed_documents(texts)
        except Exception as e:
            logger.error(f"Ollama embedding API call failed: {e}. Falling back to MockEmbeddings.")
            return self.mock.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not self.impl:
            return self.mock.embed_query(text)
        try:
            return self.impl.embed_query(text)
        except Exception as e:
            logger.error(f"Ollama embedding query call failed: {e}. Falling back to MockEmbeddings.")
            return self.mock.embed_query(text)


def get_embeddings() -> Embeddings:
    """
    Returns SafeGoogleGenAIEmbeddings or SafeOllamaEmbeddings based on configuration,
    otherwise falls back directly to MockEmbeddings.
    """
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        return MockEmbeddings()
    from app.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_HOST
    if LLM_PROVIDER == "ollama":
        logger.info(f"Initializing SafeOllamaEmbeddings wrapper (model={OLLAMA_MODEL}, url={OLLAMA_HOST}).")
        return SafeOllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    elif GOOGLE_API_KEY:
        logger.info("Initializing SafeGoogleGenAIEmbeddings wrapper.")
        return SafeGoogleGenAIEmbeddings(google_api_key=GOOGLE_API_KEY)
    else:
        logger.warning("GOOGLE_API_KEY not configured. Falling back to deterministic MockEmbeddings.")
        return MockEmbeddings()
