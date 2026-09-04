"""Structured JSON logging that Cloud Logging parses natively.

Cloud Run captures stdout/stderr. If a line is valid JSON, Cloud Logging lifts
known fields out of it: `severity` drives the log level in the console, and
`logging.googleapis.com/trace` groups a request's logs under its trace so you
can click one request and see everything it did.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Set per-request by TraceMiddleware so every log line inside a request is
# automatically correlated, without threading a logger through every call.
trace_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cloud_trace", default=None
)

# Attributes LogRecord always carries; anything else was passed by the caller
# via `extra=` and should be surfaced as a structured field.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
    | {"asctime", "message", "taskName"}
)

_LEVEL_TO_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class CloudLoggingFormatter(logging.Formatter):
    """Render a LogRecord as a single-line Cloud Logging structured entry."""

    def __init__(self, service_name: str, project_id: str = "") -> None:
        super().__init__()
        self.service_name = service_name
        self.project_id = project_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": _LEVEL_TO_SEVERITY.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
            "service": self.service_name,
            "logging.googleapis.com/sourceLocation": {
                "file": record.pathname,
                "line": str(record.lineno),
                "function": record.funcName,
            },
        }

        trace_id = trace_context.get()
        if trace_id and self.project_id:
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self.project_id}/traces/{trace_id}"
            )
        elif trace_id:
            payload["trace_id"] = trace_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


# Remembered so ensure_structured_logging() can restore the configuration if a
# third-party library tears it down. See that function for why this is needed.
_ACTIVE_CONFIG: dict[str, str] = {}


def configure_logging(
    level: str = "INFO", service_name: str = "app", project_id: str = ""
) -> None:
    """Install the JSON formatter as the only root handler.

    Also re-points uvicorn's loggers at the root handler so access logs land in
    the same JSON stream instead of uvicorn's plain-text format.
    """
    _ACTIVE_CONFIG.update(
        {"level": level, "service_name": service_name, "project_id": project_id}
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter(service_name, project_id))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # These are chatty at INFO and every line costs Cloud Logging ingest.
    # alembic in particular emits ~30 migration lines the first time MLflow
    # creates its SQLite schema.
    for name in (
        "httpx",
        "chromadb",
        "openai",
        "urllib3",
        "chromadb.telemetry",
        "alembic",
        "mlflow",
        "mlflow.store",
        "mlflow.tracking",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def ensure_structured_logging() -> None:
    """Restore JSON logging if another library has taken over the root logger.

    MLflow calls logging.config.dictConfig() when it initialises its store,
    which replaces the root handler's formatter and silently reverts the whole
    process to plain-text logs. Cloud Logging then stops parsing our severity
    and trace fields, so structured logging quietly dies in production while
    still looking correct at startup.

    Cheap enough to call on any path that touches MLflow.
    """
    if not _ACTIVE_CONFIG:
        return
    root = logging.getLogger()
    if any(isinstance(h.formatter, CloudLoggingFormatter) for h in root.handlers):
        return
    configure_logging(**_ACTIVE_CONFIG)


def parse_cloud_trace_header(header: str | None) -> str | None:
    """Extract the trace id from `X-Cloud-Trace-Context: TRACE/SPAN;o=1`."""
    if not header:
        return None
    trace_id = header.split("/", 1)[0].strip()
    return trace_id or None
