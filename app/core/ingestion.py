import hashlib
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import logger
from app.core.vector_store import get_embeddings, get_vector_store, _get_qdrant_store


class IngestionPipeline:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest(self, source: str, extra_metadata: Optional[dict] = None) -> int:
        path = Path(source)
        self._validate(path)

        t0 = time.perf_counter()
        logger.info("ingest_start", source=str(path))

        raw_docs = self._load(path)
        chunks = self._split(raw_docs, path, extra_metadata or {})
        self._store(chunks)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "ingest_complete",
            source=path.name,
            chunks=len(chunks),
            latency_ms=latency_ms,
        )
        return len(chunks)

    def _validate(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load(self, path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        logger.info("document_loaded", source=path.name, pages=len(docs))
        return docs

    def _split(
        self, docs: list[Document], path: Path, extra_metadata: dict
    ) -> list[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **extra_metadata,
            })
        logger.info("chunks_created", count=len(chunks), source=path.name)
        return chunks

    def _store(self, chunks: list[Document]):
        """Add chunks to vector store — works with both FAISS and Qdrant."""
        if self.settings.vector_store == "qdrant":
            store = _get_qdrant_store(self.settings, self.embeddings)
            store.add_documents(chunks)
            logger.info(
                "qdrant_chunks_added",
                chunks=len(chunks),
                collection=self.settings.qdrant_collection,
            )
        else:
            # FAISS path — keep existing behaviour
            from langchain_community.vectorstores import FAISS
            from pathlib import Path as P

            index_path = P(self.settings.faiss_index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            if index_path.exists():
                store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                store.add_documents(chunks)
            else:
                store = FAISS.from_documents(chunks, self.embeddings)
            store.save_local(str(index_path))
            logger.info("faiss_index_saved", path=str(index_path))