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
from sentence_transformers import CrossEncoder

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
        self._all_chunks = []
        self._chain = None
        self._reranker = None
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

        # Step 1 — hybrid retrieval: get broader candidate set
        # We fetch more than we need so reranker has good candidates to work with
        candidate_docs = self._get_candidates(question, k=k * 3)

        # Step 2 — rerank: score each candidate against the question precisely
        reranked_docs = self._rerank(question, candidate_docs)

        # Step 3 — generate: send reranked chunks to Gemini
        answer = self._generate(question, reranked_docs)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Build sources from reranked docs
        scored_docs = self._store.similarity_search_with_relevance_scores(
            question, k=k
        )
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

    def _get_candidates(self, question: str, k: int) -> list[Document]:
        """Hybrid retrieval — returns broader candidate pool for reranking."""
        faiss_retriever = self._store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
        bm25_retriever = BM25Retriever.from_documents(
            self._all_chunks, k=k
        )
        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        return ensemble.invoke(question)

    def _rerank(self, question: str, docs: list[Document]) -> list[Document]:
        """
        Cross-encoder reranker — scores each chunk against the question.
        Returns top reranker_top_k most relevant chunks.
        """
        if not docs:
            return docs

        # Build (question, chunk_text) pairs for the cross-encoder
        pairs = [[question, doc.page_content] for doc in docs]

        # Score all pairs — cross-encoder reads both together
        scores = self._reranker.predict(pairs)

        # Sort by score descending, keep top_k
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.settings.reranker_top_k]]

        print(f"[reranker] {len(docs)} candidates → top {len(top_docs)} after reranking")
        return top_docs

    def _generate(self, question: str, docs: list[Document]) -> str:
        """Send reranked chunks to Gemini and get answer."""
        # Build context from reranked chunks
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Build the prompt manually since we're bypassing the chain
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            google_api_key=self.settings.google_api_key,
        )

        prompt = FINANCE_PROMPT.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _load(self):
        if not self._index_path.exists():
            print("[retrieval] No index found yet. Ingest documents first.")
            return

        # Load reranker model (runs locally, no API needed)
        print(f"[retrieval] Loading reranker: {self.settings.reranker_model}")
        self._reranker = CrossEncoder(self.settings.reranker_model)
        print("[retrieval] Reranker loaded.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )

        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[retrieval] FAISS index loaded — {self._store.index.ntotal} chunks")

        self._all_chunks = list(self._store.docstore._dict.values())
        print(f"[retrieval] BM25 ready over {len(self._all_chunks)} chunks")
        print("[retrieval] Pipeline: Hybrid retrieval → Reranker → Gemini")