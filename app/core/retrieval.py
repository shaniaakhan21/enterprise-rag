import time
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.config import get_settings


# This is the prompt Gemini receives for every question.
# Notice: we explicitly tell it to ONLY use the provided context.
# This prevents hallucination — Gemini can't make up answers.
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
    """Loads FAISS index and answers questions using Gemini."""

    def __init__(self):
        self.settings = get_settings()
        self._store = None   # FAISS index, loaded from disk
        self._chain = None   # The full RAG chain
        self._index_path = Path(self.settings.faiss_index_path)

        # Try to load index immediately if it exists
        self._load()

    @property
    def is_ready(self) -> bool:
        """Returns True if the index is loaded and ready for queries."""
        return self._store is not None

    def reload(self):
        """Reload index from disk — called after new documents are ingested."""
        self._load()

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Ask a question. Returns answer + source chunks.
        """
        if not self.is_ready:
            raise RuntimeError("No index loaded. Ingest documents first.")

        k = top_k or self.settings.retrieval_top_k

        t0 = time.perf_counter()

        # Run the chain
        result = self._chain.invoke({"query": question})
        answer = result["result"]

        # Get scored sources separately so we can show relevance scores
        scored_docs = self._store.similarity_search_with_relevance_scores(
            question, k=k
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Format sources nicely
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
        """Load FAISS index + build the chain. Silent if index doesn't exist yet."""
        if not self._index_path.exists():
            print("[retrieval] No index found yet. Ingest documents first.")
            return

        # Load embeddings — must be same model used during ingestion
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )

        # Load FAISS index from disk
        self._store = FAISS.load_local(
            str(self._index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[retrieval] Index loaded from {self._index_path}")

        # Set up Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            google_api_key=self.settings.google_api_key,
        )

        # Build the prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=FINANCE_PROMPT,
        )

        # MMR retriever — diverse + relevant chunks
        retriever = self._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.settings.retrieval_top_k,
                "fetch_k": self.settings.retrieval_top_k * 3,
            },
        )

        # Wire everything together into one chain
        self._chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )
        print("[retrieval] Chain ready.")