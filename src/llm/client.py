import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        llm_round: int | None = None,
    ) -> dict[str, Any]:

        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        if tool_choice:
            payload["tool_choice"] = tool_choice

        start = time.perf_counter()

        logger.info(
            "LLM request started",
            extra={
                "event": "llm_request_started",
                "llm_round": llm_round,
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )

            response.raise_for_status()

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "LLM request failed",
                extra={
                    "event": "llm_request_failed",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                },
            )

            raise

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "LLM request completed",
            extra={
                "event": "llm_request_completed",
                "duration_ms": round(duration_ms, 2),
                "status_code": response.status_code,
                "llm_round": llm_round,
            },
        )

        return response.json()
