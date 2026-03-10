import time
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import logger


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

        candidate_docs = self._get_candidates(question, k=k * 3)
        reranked_docs = self._rerank(question, candidate_docs)
        answer = self._generate(question, reranked_docs)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

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

        logger.info(
            "query_complete",
            latency_ms=latency_ms,
            sources=len(sources),
            answer_len=len(answer),
            candidates=len(candidate_docs),
            reranked=len(reranked_docs),
        )

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
        }

    def _get_candidates(self, question: str, k: int) -> list[Document]:
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
        if not docs:
            return docs

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self._reranker.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:self.settings.reranker_top_k]]

        logger.info(
            "reranker_complete",
            candidates=len(docs),
            selected=len(top_docs),
        )
        return top_docs

    def _generate(self, question: str, docs: list[Document]) -> str:
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

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
            logger.info("no_index_found", path=str(self._index_path))
            return

        logger.info("reranker_loading", model=self.settings.reranker_model)
        self._reranker = CrossEncoder(self.settings.reranker_model)
        logger.info("reranker_ready")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )

        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("faiss_loaded", total_chunks=self._store.index.ntotal)

        self._all_chunks = list(self._store.docstore._dict.values())
        logger.info("bm25_ready", chunks=len(self._all_chunks))
        logger.info("retrieval_pipeline_ready", mode="hybrid+reranker")