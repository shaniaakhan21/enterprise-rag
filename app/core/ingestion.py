import hashlib
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import logger          # ← import logger


class IngestionPipeline:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._index_path = Path(self.settings.faiss_index_path)

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

    def _split(self, docs: list[Document], path: Path, extra_metadata: dict) -> list[Document]:
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
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index_path.exists():
            logger.info("index_merging", new_chunks=len(chunks))
            store = FAISS.load_local(
                str(self._index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            store.add_documents(chunks)
        else:
            logger.info("index_creating", chunks=len(chunks))
            store = FAISS.from_documents(chunks, self.embeddings)

        store.save_local(str(self._index_path))
        logger.info("index_saved", path=str(self._index_path))