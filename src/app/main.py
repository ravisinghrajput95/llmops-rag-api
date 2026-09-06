"""FastAPI application: /health, /ready, /ingest, /ingest/file, /query."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from openai import OpenAIError

from app.config import Settings, get_settings
from app.dependencies import build_pipeline, get_pipeline, require_api_key
from app.logging_config import (
    configure_logging,
    parse_cloud_trace_header,
    trace_context,
)
from app.rag.pipeline import RAGPipeline
from app.rate_limit import RateLimiter
from app.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SourceChunk,
    TokenUsage,
)
from app.tracking.spend_guard import BudgetExceededError

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_SUFFIXES = (".txt", ".md", ".markdown", ".text")

# Endpoints that cost money. /health and /ready stay unthrottled so uptime
# checks and Cloud Run startup probes can never be rate limited.
METERED_PATHS = ("/ingest", "/query")

# Configure logging at import time, not in lifespan: uvicorn emits its own
# startup lines before lifespan runs, and those would otherwise escape as
# unstructured plain text that Cloud Logging cannot parse.
_settings = get_settings()
configure_logging(
    level=_settings.log_level,
    service_name=_settings.service_name,
    project_id=_settings.gcp_project_id,
)

_rate_limiter = RateLimiter(requests_per_minute=_settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "starting service",
        extra={
            "version": settings.app_version,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "auth_enabled": settings.auth_enabled,
        },
    )
    try:
        app.state.pipeline = build_pipeline(settings)
    except Exception:
        # Do not crash the container: Cloud Run would restart-loop and burn
        # request quota. /health stays up so the failure is diagnosable, while
        # /ingest and /query return a clear 503.
        app.state.pipeline = None
        logger.error("pipeline initialisation failed", exc_info=True)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="LLMOps RAG API",
    description=(
        "RAG question answering over your own documents. "
        "OpenAI for generation, Chroma for retrieval, MLflow for per-call "
        "latency/token/cost tracking. Runs on Cloud Run, scale-to-zero."
    ),
    version=get_settings().app_version,
    lifespan=lifespan,
)


def _client_key(request: Request) -> str:
    """Identify the caller for throttling.

    Prefer the API key over the IP: behind Cloud Run every request arrives
    from Google's front end, and X-Forwarded-For is client-controlled, so IP
    alone is both unreliable and spoofable. Keys are hashed so no secret ever
    reaches a log line.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    forwarded = request.headers.get("X-Forwarded-For", "")
    client_ip = forwarded.split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    return "ip:" + client_ip


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _rate_limiter.enabled and request.url.path.startswith(METERED_PATHS):
        allowed, retry_after = _rate_limiter.allow(_client_key(request))
        if not allowed:
            logger.warning(
                "rate limited", extra={"path": request.url.path, "retry_after": retry_after}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
    return await call_next(request)


# Registered LAST, which in Starlette means it runs FIRST (outermost). That
# ordering matters: the rate limiter logs and returns 429 without calling the
# route, so if it sat outside this, every throttled request would log with no
# trace id -- losing correlation exactly when a client is being investigated.
@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Bind Cloud Trace id so every log line in a request is correlated."""
    token = trace_context.set(
        parse_cloud_trace_header(request.headers.get("X-Cloud-Trace-Context"))
    )
    try:
        return await call_next(request)
    finally:
        trace_context.reset(token)


@app.exception_handler(BudgetExceededError)
async def budget_exceeded_handler(request: Request, exc: BudgetExceededError) -> JSONResponse:
    """The daily spend ceiling. 429 rather than 503: the service is healthy,
    the caller simply may not spend more today."""
    logger.error(
        "daily budget exhausted -- refusing to spend",
        extra={
            "path": request.url.path,
            "spent_usd": exc.spent_usd,
            "budget_usd": exc.budget_usd,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Daily spend ceiling reached; no further LLM calls today.",
            "spent_usd": exc.spent_usd,
            "budget_usd": exc.budget_usd,
            "resets_in_seconds": exc.resets_in_seconds,
        },
        headers={"Retry-After": str(exc.resets_in_seconds)},
    )


@app.exception_handler(OpenAIError)
async def openai_error_handler(request: Request, exc: OpenAIError) -> JSONResponse:
    """Upstream provider failures are the most likely runtime error here.

    A bare 500 tells the caller nothing and buries the cause. 502 is the honest
    status: this service is fine, the thing it depends on is not.
    """
    logger.error(
        "openai request failed",
        extra={"path": request.url.path, "error_type": type(exc).__name__},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "detail": "Upstream LLM provider request failed.",
            "error_type": type(exc).__name__,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("bad request", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Ops endpoints (unauthenticated, no spend)
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness. Deliberately does no I/O so it stays free and instant."""
    return HealthResponse(
        status="ok", service=settings.service_name, version=settings.app_version
    )


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(
    settings: Settings = Depends(get_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ReadyResponse:
    """Readiness: confirms the vector store is reachable and reports its size."""
    return ReadyResponse(
        status="ready",
        collection=settings.chroma_collection,
        collection_size=pipeline.collection_size(),
        mlflow_enabled=settings.mlflow_enabled,
        openai_configured=bool(settings.openai_api_key),
        spend=pipeline.spend_info(),
        persistence=pipeline.persistence_info(),
        prompts=pipeline.prompt_info(),
    )


# --------------------------------------------------------------------------
# RAG endpoints
# --------------------------------------------------------------------------
@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    tags=["rag"],
)
def ingest(
    payload: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)
) -> IngestResponse:
    """Chunk, embed and store documents in Chroma."""
    result = pipeline.ingest(
        [(doc.text, doc.doc_id, doc.metadata) for doc in payload.documents]
    )
    return IngestResponse(
        ingested_documents=len(result.document_ids),
        ingested_chunks=result.chunk_count,
        collection_size=result.collection_size,
        embedding_tokens=result.embedding_tokens,
        estimated_cost_usd=result.cost_usd,
        estimated_cost_inr=result.cost_inr,
        latency_ms=result.latency_ms,
        document_ids=result.document_ids,
    )


@app.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    tags=["rag"],
)
async def ingest_file(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Upload a plain-text or Markdown file and ingest its contents."""
    filename = file.filename or "upload.txt"
    if not filename.lower().endswith(ALLOWED_UPLOAD_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only {', '.join(ALLOWED_UPLOAD_SUFFIXES)} files are supported.",
        )

    raw = await file.read()
    if len(raw) > settings.max_ingest_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_ingest_chars} bytes.",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from None

    if not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    result = pipeline.ingest([(text, None, {"source": filename})])
    return IngestResponse(
        ingested_documents=len(result.document_ids),
        ingested_chunks=result.chunk_count,
        collection_size=result.collection_size,
        embedding_tokens=result.embedding_tokens,
        estimated_cost_usd=result.cost_usd,
        estimated_cost_inr=result.cost_inr,
        latency_ms=result.latency_ms,
        document_ids=result.document_ids,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
    tags=["rag"],
)
def query(
    payload: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)
) -> QueryResponse:
    """Retrieve relevant chunks and answer the question with the LLM."""
    result = pipeline.query(payload.question, top_k=payload.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                similarity=hit.similarity,
                metadata=hit.metadata,
            )
            for hit in result.sources
        ],
        usage=TokenUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            embedding_tokens=result.embedding_tokens,
        ),
        model=result.model,
        latency_ms=result.latency_ms,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
        estimated_cost_usd=result.cost_usd,
        estimated_cost_inr=result.cost_inr,
        refused=result.refused,
        mlflow_run_id=result.mlflow_run_id,
    )
