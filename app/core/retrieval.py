import time
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from app.core.config import get_settings


FINANCE_PROMPT = """You are a senior financial analyst assistant.
Use ONLY the context excerpts below to answer the question.
If the answer is not in the context, say: "I could not find this information in the provided documents."
Always cite specific figures, dates, or sections where possible.
Be concise but precise.

Context:
{context}

Question: {question}

Answer:"""


class RetrievalChain:

    def __init__(self):
        self.settings = get_settings()
        self._store = None
        self._all_chunks = []    # ← we now keep chunks in memory for BM25
        self._chain = None
        self._index_path = Path(self.settings.faiss_index_path)
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def reload(self):
        self._load()

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        if not self.is_ready:
            raise RuntimeError("No index loaded. Ingest documents first.")

        k = top_k or self.settings.retrieval_top_k
        t0 = time.perf_counter()

        result = self._chain.invoke({"query": question})
        answer = result["result"]

        # Get scored sources for provenance
        scored_docs = self._store.similarity_search_with_relevance_scores(
            question, k=k
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        sources = [
            {
                "content": doc.page_content[:400],
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            }
            for doc, score in scored_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    def _load(self):
        if not self._index_path.exists():
            print("[retrieval] No index found yet. Ingest documents first.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )

        # Load FAISS
        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[retrieval] FAISS index loaded — {self._store.index.ntotal} chunks")

        # Extract all chunks from FAISS for BM25
        # BM25 needs the raw text, FAISS stores it in docstore
        self._all_chunks = list(self._store.docstore._dict.values())
        print(f"[retrieval] BM25 index built over {len(self._all_chunks)} chunks")

        llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            google_api_key=self.settings.google_api_key,
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=FINANCE_PROMPT,
        )

        # ── Hybrid retriever ──────────────────────────────────────
        # FAISS retriever — semantic similarity
        faiss_retriever = self._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.settings.retrieval_top_k,
                "fetch_k": self.settings.retrieval_top_k * 3,
            },
        )

        # BM25 retriever — exact keyword matching
        bm25_retriever = BM25Retriever.from_documents(
            self._all_chunks,
            k=self.settings.retrieval_top_k,
        )

        # Ensemble — combines both with equal weight
        # weight 0.5/0.5 means equal contribution from both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        self._chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,    # ← now using hybrid retriever
            chain_type_kwargs={"prompt": prompt},
        )
        print("[retrieval] Hybrid retrieval chain ready (FAISS + BM25).")