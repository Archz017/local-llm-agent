import logging
import time
from typing import Any

import httpx

from agent.errors import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        health_url: str = "http://localhost:8080/health",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.health_url = health_url
        self.timeout = timeout

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
                response = await client.get(self.health_url)

            return response.is_success

        except httpx.RequestError:
            return False

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

        except httpx.TimeoutException as exc:
            duration_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "LLM request timed out",
                extra={
                    "event": "llm_request_failed",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                    "llm_round": llm_round,
                },
            )

            raise LLMTimeoutError("LLM request timed out.") from exc

        except httpx.HTTPStatusError as exc:
            duration_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "LLM returned an error response",
                extra={
                    "event": "llm_request_failed",
                    "duration_ms": round(duration_ms, 2),
                    "status_code": exc.response.status_code,
                    "error_type": type(exc).__name__,
                    "llm_round": llm_round,
                },
            )

            raise LLMError(f"LLM returned HTTP {exc.response.status_code}.") from exc

        except httpx.RequestError as exc:
            duration_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "LLM connection failed",
                extra={
                    "event": "llm_request_failed",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                    "llm_round": llm_round,
                },
            )

            raise LLMError("Unable to communicate with the LLM.") from exc

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
