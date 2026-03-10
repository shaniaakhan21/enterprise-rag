from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini
    google_api_key: str

    # LLM
    llm_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 5

    # Paths
    faiss_index_path: str = "data/processed/faiss_index"
    raw_docs_path: str = "data/raw"

    # App
    log_level: str = "INFO"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()