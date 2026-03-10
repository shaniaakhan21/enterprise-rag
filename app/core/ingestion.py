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


class IngestionPipeline:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self):
        self.settings = get_settings()

        # This is what converts text → vectors
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )

        # This is what splits documents into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._index_path = Path(self.settings.faiss_index_path)

    def ingest(self, source: str, extra_metadata: Optional[dict] = None) -> int:
        """
        Full pipeline: load → split → embed → store.
        Returns number of chunks indexed.
        """
        path = Path(source)
        self._validate(path)

        print(f"[ingest] Loading: {path.name}")
        raw_docs = self._load(path)

        print(f"[ingest] Splitting into chunks...")
        chunks = self._split(raw_docs, path, extra_metadata or {})
        print(f"[ingest] Created {len(chunks)} chunks")

        print(f"[ingest] Embedding and storing in FAISS...")
        self._store(chunks)

        print(f"[ingest] Done. {len(chunks)} chunks indexed.")
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
        return loader.load()

    def _split(self, docs: list[Document], path: Path, extra_metadata: dict) -> list[Document]:
        chunks = self.splitter.split_documents(docs)

        # Attach metadata to every chunk so we know where it came from
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **extra_metadata,
            })
        return chunks

    def _store(self, chunks: list[Document]):
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists():
            # Index already exists → merge new chunks in
            print("[ingest] Existing index found, merging...")
            store = FAISS.load_local(
                str(self._index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            store.add_documents(chunks)
        else:
            # First time → create fresh index
            print("[ingest] Creating new index...")
            store = FAISS.from_documents(chunks, self.embeddings)

        store.save_local(str(self._index_path))
        print(f"[ingest] Index saved to {self._index_path}")
