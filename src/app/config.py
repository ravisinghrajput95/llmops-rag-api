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
    # reach the prompt. Fewer junk chunks == fewer input tokens == less money,
    # and when every chunk is dropped the LLM call is skipped entirely.
    #
    # 0.28 is calibrated on measured text-embedding-3-small scores over this
    # corpus against 61 answerable and 67 out-of-corpus questions. It is the
    # highest floor that costs nothing: it rejects 20.9% of out-of-corpus
    # questions against 11.9% at the old 0.2, with an identical answerable
    # rate on all ten held-out splits. Above ~0.30 answerable questions start
    # being refused.
    #
    # What this floor CANNOT do is reject a question about a topic the corpus
    # covers whose specific fact it lacks -- "what is Cloud Run's maximum
    # request timeout?" against a corpus that discusses Cloud Run concurrency
    # but never states a maximum. Those score identically to answerable
    # questions (AUC 0.518, i.e. chance), because the retriever is right: the
    # document IS the relevant one. Similarity measures topical relevance;
    # refusing needs factual sufficiency, which is a reading judgement the
    # model makes and geometry cannot. See README "Why the floor stops here".
    #
    # Model-dependent -- recalibrate if you change embeddings.
    min_similarity: float = 0.28
    # Additionally drop chunks scoring below this fraction of the best chunk's
    # score. The absolute floor cannot tell a weak match in a strong result set
    # from a strong match in a weak one, because similarity is not calibrated
    # across questions; this is, since it rescales per query. Worth having for
    # cost rather than for refusal: it cuts context from 3.95 to 2.87 chunks
    # per query -- 27% off the input tokens of every single query -- with no
    # measured loss of answerable rate. Confirmed end to end: retrieval hit
    # rate stayed at 100% and cost per eval case fell 30%. 0 disables.
    min_similarity_ratio: float = 0.60
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

    # --- Durability -------------------------------------------------------
    # Cloud Run's filesystem is a per-instance tmpfs, so without this the
    # vector store is lost on scale-to-zero. Set to the MLflow artifact bucket
    # to snapshot Chroma there on ingest and restore it on cold start. Empty
    # disables persistence entirely, which is the right default locally.
    gcs_bucket: str = ""
    chroma_snapshot_object: str = "snapshots/chroma.tar.gz"
    # The MLflow tracking DB lives on the same tmpfs and dies with it, which
    # made `make drift` blind to production: artifacts reach GCS but the run
    # metadata the monitor actually reads -- params, metrics, tags -- did not.
    # Snapshotting it costs one GCS write per `mlflow_snapshot_every` queries
    # rather than one per query, because Cloud Storage's free tier allows
    # 5,000 class A operations a month and a per-query write would spend them.
    # The cost of that batching is honest and bounded: up to that many runs are
    # lost if an instance dies between snapshots. Drift is measured over a
    # window of tens of queries, so losing a few skews nothing. 0 disables.
    mlflow_snapshot_object: str = "snapshots/mlflow.tar.gz"
    mlflow_snapshot_every: int = 25
    # Snapshotting adds a GCS round trip to /ingest (not to /query). Turn off
    # to keep ingest fast and accept ephemeral state.
    snapshot_on_ingest: bool = True

    # --- Spend and abuse ceilings ----------------------------------------
    # Hard daily cap on estimated OpenAI spend, in USD. GCP teardown does not
    # stop the OpenAI bill, so this is the only thing bounding it. 0 disables.
    # $0.25/day is ~1,200 typical queries on gpt-4o-mini -- far more than a
    # demo needs, and ~₹22 if something goes wrong for a whole day.
    daily_budget_usd: float = 0.25
    # Per-client request ceiling. 0 disables. Both limiters hold state in
    # process, so the effective allowance scales with instance count.
    rate_limit_per_minute: int = 30

    @property
    def auth_enabled(self) -> bool:
        return bool(self.app_api_key)

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.gcs_bucket)


@lru_cache
def get_settings() -> Settings:
    return Settings()
