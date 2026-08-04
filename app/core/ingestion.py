import csv
import hashlib
import time
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import logger
from app.core.retry import with_retry
from app.core.vector_store import get_embeddings, get_vector_store, _get_qdrant_store


class IngestionPipeline:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

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
        if path.suffix.lower() == ".csv":
            # Rows are already grouped into right-sized chunks at load time —
            # re-splitting would just multiply the embedding API calls back up.
            chunks = self._tag_chunks(raw_docs, path, extra_metadata or {})
        else:
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
        suffix = path.suffix.lower()
        if suffix == ".csv":
            docs = self._load_csv_grouped(path, self.settings.csv_rows_per_chunk)
            logger.info("document_loaded", source=path.name, pages=len(docs))
            return docs

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        logger.info("document_loaded", source=path.name, pages=len(docs))
        return docs

    def _load_csv_grouped(self, path: Path, rows_per_chunk: int) -> list[Document]:
        """
        Group multiple CSV rows into each Document instead of one row per
        Document. A 1-row-per-chunk CSV loader turns a several-thousand-row
        file into as many embedding API calls, which blows through the free
        tier's rate limit. Grouping rows cuts that count by ~rows_per_chunk.
        """
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        docs = []
        for i in range(0, len(rows), rows_per_chunk):
            group = rows[i : i + rows_per_chunk]
            content = "\n\n".join(
                "\n".join(f"{col}: {val}" for col, val in row.items())
                for row in group
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "row_start": i,
                        "row_end": i + len(group) - 1,
                    },
                )
            )
        return docs

    def _split(
        self, docs: list[Document], path: Path, extra_metadata: dict
    ) -> list[Document]:
        chunks = self.splitter.split_documents(docs)
        return self._tag_chunks(chunks, path, extra_metadata)

    def _tag_chunks(
        self, chunks: list[Document], path: Path, extra_metadata: dict
    ) -> list[Document]:
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
            self._add_documents(store, chunks)
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
                self._add_documents(store, chunks)
            else:
                store = self._from_documents(chunks, self.embeddings)
            store.save_local(str(index_path))
            logger.info("faiss_index_saved", path=str(index_path))

    @with_retry(max_attempts=5, min_wait=2.0, max_wait=30.0)
    def _add_documents(self, store, chunks: list[Document]):
        """
        Embed + add chunks. Retries with backoff on rate-limit errors.

        batch_size caps how many texts go into a single embedding request —
        Gemini's free tier enforces a per-request input token quota, and a
        default-sized (64-100 text) batch of anything but very short chunks
        blows past it even though the account has plenty of daily quota left.
        """
        store.add_documents(chunks, batch_size=self.settings.embed_batch_size)

    @with_retry(max_attempts=5, min_wait=2.0, max_wait=30.0)
    def _from_documents(self, chunks: list[Document], embeddings):
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(chunks, embeddings)