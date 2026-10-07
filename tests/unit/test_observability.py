import json
import logging

from observability.context import (
    reset_request_id,
    reset_session_id,
    set_request_id,
    set_session_id,
)
from observability.logging import JsonFormatter


def test_json_formatter_includes_context() -> None:
    request_token = set_request_id("request-123")
    session_token = set_session_id("session-456")

    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        record.event = "test_event"
        record.duration_ms = 12.34

        output = JsonFormatter().format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["request_id"] == "request-123"
        assert parsed["session_id"] == "session-456"
        assert parsed["event"] == "test_event"
        assert parsed["duration_ms"] == 12.34

    finally:
        reset_session_id(session_token)
        reset_request_id(request_token)


def test_context_is_empty_by_default() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="No context",
        args=(),
        exc_info=None,
    )

    output = JsonFormatter().format(record)
    parsed = json.loads(output)

    assert "request_id" not in parsed
    assert "session_id" not in parsed
