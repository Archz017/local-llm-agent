from typing import Any

import httpx


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
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )

        response.raise_for_status()
        return response.json()