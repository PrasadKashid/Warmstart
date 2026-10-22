from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # Google Gemini Settings
    gemini_api_key: str
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 3072
    llm_model: str = "gemma-4-26b-a4b-it"

    # Cache Settings
    similarity_threshold: float = 0.82
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "warmstart_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
