"""
Vector store abstraction layer.
Supports FAISS (local) and Qdrant (production).
Switch between them via VECTOR_STORE env var.
"""
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import get_settings
from app.core.logging import logger


def get_embeddings():
    """Shared embedding model used by both FAISS and Qdrant."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def get_vector_store(embeddings=None):
    """
    Factory — returns either FAISS or Qdrant store based on config.
    
    Usage:
        store = get_vector_store()
        store.add_documents(chunks)
        results = store.similarity_search("query", k=5)
    """
    settings = get_settings()
    embeddings = embeddings or get_embeddings()

    if settings.vector_store == "qdrant":
        return _get_qdrant_store(settings, embeddings)
    else:
        return _get_faiss_store(settings, embeddings)


def _get_qdrant_store(settings, embeddings):
    """Load or create Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # Create collection if it doesn't exist
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        logger.info(
            "qdrant_creating_collection",
            collection=settings.qdrant_collection,
        )
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=3072,        # gemini-embedding-001 output dimension
                distance=Distance.COSINE,
            ),
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    logger.info(
        "qdrant_store_ready",
        collection=settings.qdrant_collection,
        host=settings.qdrant_host,
    )
    return store


def _get_faiss_store(settings, embeddings):
    """Load existing FAISS index from disk."""
    from langchain_community.vectorstores import FAISS

    index_path = Path(settings.faiss_index_path)
    if not index_path.exists():
        return None

    store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("faiss_store_ready", path=str(index_path))
    return store