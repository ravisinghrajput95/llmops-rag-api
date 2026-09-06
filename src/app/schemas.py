"""Request/response models. These double as the OpenAPI contract at /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
class Document(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text.")
    doc_id: str | None = Field(
        default=None, description="Stable id. Generated from a content hash if omitted."
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Arbitrary string tags stored alongside chunks."
    )


class IngestRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1, max_length=50)


class IngestResponse(BaseModel):
    ingested_documents: int
    ingested_chunks: int
    collection_size: int
    embedding_tokens: int
    estimated_cost_usd: float
    estimated_cost_inr: float
    latency_ms: float
    document_ids: list[str]


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    metadata: dict[str, str] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    usage: TokenUsage
    model: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    estimated_cost_usd: float
    estimated_cost_inr: float
    mlflow_run_id: str | None = None


# --------------------------------------------------------------------------
# Ops
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    collection: str
    collection_size: int
    mlflow_enabled: bool
    openai_configured: bool
    # Present so an operator can see remaining daily budget without shelling
    # into logs. Unauthenticated, so it deliberately exposes only aggregates.
    spend: dict = Field(default_factory=dict)
    persistence: dict = Field(default_factory=dict)
    # Which prompt version this instance is serving, so a deployed revision can
    # be matched to the eval run that measured it.
    prompts: dict = Field(default_factory=dict)
