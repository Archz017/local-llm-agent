import logging
import time
from typing import Any

import httpx

from agent.errors import LLMError, LLMTimeoutError
from observability.metrics import (
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
)

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
            duration_seconds = time.perf_counter() - start

            LLM_REQUESTS_TOTAL.labels(
                outcome="timeout",
            ).inc()

            LLM_REQUEST_DURATION_SECONDS.observe(duration_seconds)

            raise LLMTimeoutError("LLM request timed out.") from exc

        except httpx.HTTPStatusError as exc:
            duration_seconds = time.perf_counter() - start

            LLM_REQUESTS_TOTAL.labels(
                outcome="http_error",
            ).inc()

            LLM_REQUEST_DURATION_SECONDS.observe(duration_seconds)

            raise LLMError(f"LLM returned HTTP {exc.response.status_code}.") from exc

        except httpx.RequestError as exc:
            duration_seconds = time.perf_counter() - start

            LLM_REQUESTS_TOTAL.labels(
                outcome="connection_error",
            ).inc()

            LLM_REQUEST_DURATION_SECONDS.observe(duration_seconds)

            raise LLMError("Unable to communicate with the LLM.") from exc

        data = response.json()

        duration_seconds = time.perf_counter() - start
        duration_ms = duration_seconds * 1000

        LLM_REQUESTS_TOTAL.labels(
            outcome="success",
        ).inc()

        LLM_REQUEST_DURATION_SECONDS.observe(duration_seconds)

        usage = data.get("usage")

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            if isinstance(prompt_tokens, int):
                LLM_TOKENS_TOTAL.labels(
                    type="prompt",
                ).inc(prompt_tokens)

            if isinstance(completion_tokens, int):
                LLM_TOKENS_TOTAL.labels(
                    type="completion",
                ).inc(completion_tokens)

        logger.info(
            "LLM request completed",
            extra={
                "event": "llm_request_completed",
                "llm_round": llm_round,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return data
