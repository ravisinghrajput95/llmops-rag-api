"""Structured logging must stay parseable by Cloud Logging."""

from __future__ import annotations

import json
import logging

from app.logging_config import (
    CloudLoggingFormatter,
    configure_logging,
    parse_cloud_trace_header,
    trace_context,
)


def _record(level: int = logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord("app.test", level, "/src/app/x.py", 42, "hello", (), None)
    record.__dict__.update(extra)
    return record


def test_output_is_valid_json_with_cloud_logging_fields():
    payload = json.loads(CloudLoggingFormatter("svc", "proj").format(_record()))

    assert payload["severity"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["service"] == "svc"
    assert payload["logging.googleapis.com/sourceLocation"]["line"] == "42"


def test_levels_map_to_cloud_logging_severities():
    formatter = CloudLoggingFormatter("svc")

    assert json.loads(formatter.format(_record(logging.ERROR)))["severity"] == "ERROR"
    assert json.loads(formatter.format(_record(logging.WARNING)))["severity"] == "WARNING"


def test_extra_fields_are_promoted_to_structured_keys():
    payload = json.loads(
        CloudLoggingFormatter("svc").format(_record(cost_usd=0.0002, latency_ms=812))
    )

    assert payload["cost_usd"] == 0.0002
    assert payload["latency_ms"] == 812


def test_trace_id_is_attached_in_cloud_logging_format():
    token = trace_context.set("abc123")
    try:
        payload = json.loads(CloudLoggingFormatter("svc", "my-project").format(_record()))
    finally:
        trace_context.reset(token)

    assert payload["logging.googleapis.com/trace"] == "projects/my-project/traces/abc123"


def test_trace_header_parsing():
    assert parse_cloud_trace_header("TRACE_ID/SPAN;o=1") == "TRACE_ID"
    assert parse_cloud_trace_header(None) is None
    assert parse_cloud_trace_header("") is None


def test_exceptions_are_captured_as_a_field():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(CloudLoggingFormatter("svc").format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_installs_a_single_json_handler(capsys):
    configure_logging("INFO", "llmops-rag-api", "proj")
    logging.getLogger("app.demo").info("structured", extra={"tokens": 12})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "structured"
    assert payload["tokens"] == 12
    assert len(logging.getLogger().handlers) == 1


def test_structured_logging_survives_mlflow_hijacking_the_root_logger(tmp_path, capsys):
    """Regression test.

    MLflow calls logging.config.dictConfig() when it creates its SQLite schema,
    which replaces the root formatter and reverts the process to plain text.
    Cloud Logging then stops parsing severity and trace fields -- structured
    logging dies silently in production while startup logs still look correct.
    """
    from app.config import Settings
    from app.logging_config import CloudLoggingFormatter
    from app.tracking.mlflow_tracker import MLflowTracker, RunPayload

    configure_logging("INFO", "llmops-rag-api", "proj")
    assert isinstance(logging.getLogger().handlers[0].formatter, CloudLoggingFormatter)

    tracker = MLflowTracker(
        Settings(
            mlflow_enabled=True,
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'hijack.db'}",
            mlflow_experiment="hijack-regression",
        )
    )
    assert tracker.enabled, "tracker should have initialised"
    tracker.log_run("query", RunPayload(metrics={"latency_ms": 1.0}))

    # The formatter must still be ours after MLflow has run.
    assert isinstance(logging.getLogger().handlers[0].formatter, CloudLoggingFormatter)

    capsys.readouterr()  # discard MLflow's own startup chatter
    logging.getLogger("app.demo").info("still structured", extra={"cost_usd": 0.0003})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "still structured"
    assert payload["cost_usd"] == 0.0003
