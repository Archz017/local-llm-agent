import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from observability.context import request_id_var, session_id_var


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = (
            getattr(
                record,
                "request_id",
                None,
            )
            or request_id_var.get()
        )

        session_id = (
            getattr(
                record,
                "session_id",
                None,
            )
            or session_id_var.get()
        )

        if request_id is not None:
            log_entry["request_id"] = request_id

        if session_id is not None:
            log_entry["session_id"] = session_id

        for field in (
            "event",
            "duration_ms",
            "tool_name",
            "tool_call_id",
            "status_code",
            "error_type",
            "llm_round",
        ):
            value = getattr(record, field, None)

            if value is not None:
                log_entry[field] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
