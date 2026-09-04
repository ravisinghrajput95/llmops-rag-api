"""Central configuration. Everything is env-driven; nothing is hardcoded.

Locally these come from a .env file (see .env.example). On Cloud Run they come
from `--set-env-vars` / Secret Manager. Defaults are chosen so that `pytest`
and `uvicorn` work on a clean checkout without any GCP resources existing.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service identity -------------------------------------------------
    service_name: str = "llmops-rag-api"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    gcp_project_id: str = ""

    # Cloud Run injects PORT. Never hardcode 8080 anywhere else.
    port: int = 8080

    # --- Auth -------------------------------------------------------------
    # Optional shared secret for /ingest and /query. Leave empty to disable.
    # Strongly recommended once the service is public: an unauthenticated
    # public endpoint spends *your* OpenAI balance on anyone who finds it.
    app_api_key: str = ""

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_output_tokens: int = 512
    temperature: float = 0.2
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 2

    # --- Retrieval --------------------------------------------------------
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    # Chunks scoring below this cosine similarity are dropped before they
    # reach the prompt. Fewer junk chunks == fewer input tokens == less money.
    min_similarity: float = 0.0
    max_ingest_chars: int = 200_000

    # --- MLflow -----------------------------------------------------------
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"
    mlflow_experiment: str = "llmops-rag-demo"
    # gs://<bucket>/mlflow — only used when the experiment is first created.
    mlflow_artifact_location: str = ""
    # Artifacts (prompt/response text) are the only thing that touches GCS at
    # request time. Turn off to keep the request path 100% GCP-free.
    mlflow_log_artifacts: bool = True

    # --- Cost estimation --------------------------------------------------
    # Used only to render a friendly rupee figure next to the USD estimate.
    usd_to_inr: float = 88.0

    @property
    def auth_enabled(self) -> bool:
        return bool(self.app_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
