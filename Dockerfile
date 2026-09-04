# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Two-stage build for Cloud Run.
#
# Stage 1 compiles wheels (chromadb pulls in packages that need a C toolchain);
# stage 2 keeps only the installed site-packages, so gcc and friends never ship.
# Smaller image => faster cold start => fewer billable container-seconds, and
# less Artifact Registry storage drawn against the trial credits.
# ---------------------------------------------------------------------------
# Cloud Run is x86_64 only. CI runners are already amd64; when building on an
# Apple Silicon Mac you MUST pass --platform linux/amd64 (see `make docker-build`)
# or the image will fail at startup with "exec format error".
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# Install into an isolated prefix we can copy wholesale into the runtime stage.
RUN pip install --prefix=/install -r requirements.txt

# chromadb declares the `kubernetes` client as a hard dependency for server
# deployment modes we never use (we run it embedded). It is 82 MB -- a sixth of
# site-packages -- and dropping it keeps three retained image versions inside
# Artifact Registry's free 0.5 GB.
#
# Note we do NOT prune onnxruntime, even though we always supply embeddings
# explicitly: chromadb builds its default ONNX embedding function at class
# definition time, so removing it breaks `import chromadb` outright.
# Removed by path, not `pip uninstall`: the packages live under the isolated
# --prefix tree above, which pip's uninstall does not target.
RUN rm -rf /install/lib/python3.12/site-packages/kubernetes \
           /install/lib/python3.12/site-packages/kubernetes-*.dist-info


# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED is required for Cloud Logging: buffered stdout means logs
# arrive late or are lost entirely when an instance is scaled to zero.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8080 \
    CHROMA_DIR=/tmp/chroma \
    MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db \
    ANONYMIZED_TELEMETRY=False \
    # MLflow probes for a git SHA to tag runs with. There is no git binary in
    # a slim image (and no repo), so GitPython prints a 13-line warning on
    # every cold start. We do not need the SHA: CI tags images by commit.
    GIT_PYTHON_REFRESH=quiet

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/

# Fail the build loudly if the dependency prune above broke anything, rather
# than shipping an image that dies on its first request.
RUN python -c "import chromadb, mlflow, openai, fastapi; from app.main import app; \
    assert len([r for r in app.routes if hasattr(r, 'methods')]) >= 5" \
    && echo "import check passed"

# Run as non-root. Cloud Run does not require it, but it costs nothing and
# limits the blast radius if the container is ever compromised.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run injects PORT and may change it; bind to it via the shell form
# rather than hardcoding 8080. Single worker on purpose: Cloud Run scales by
# adding instances, and extra workers would multiply memory inside the 512Mi
# limit for no throughput gain on an I/O-bound service.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 30
