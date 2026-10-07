import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from agent.errors import (
    LLMError,
    LLMTimeoutError,
    MaxToolRoundsError,
)
from api.app import app


def make_chat_services(
    *,
    runtime_error: Exception,
    messages: list[dict[str, object]] | None = None,
):
    session = SimpleNamespace(
        session_id="test-session",
        messages=list(messages or []),
        lock=asyncio.Lock(),
    )

    session_store = MagicMock()
    session_store.get = AsyncMock(return_value=session)

    runtime = MagicMock()
    runtime.run = AsyncMock(side_effect=runtime_error)

    services = MagicMock()
    services.session_store = session_store
    services.get_runtime.return_value = runtime

    return services, session


@pytest.mark.asyncio
async def test_chat_returns_503_when_llm_unavailable() -> None:
    services, session = make_chat_services(
        runtime_error=LLMError("Unable to communicate with the LLM.")
    )

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "session_id": "test-session",
                "message": "Hello",
            },
        )

    assert response.status_code == 503

    body = response.json()

    assert body["error"] == "llm_unavailable"
    assert body["detail"] == ("Unable to communicate with the LLM.")

    assert body["request_id"] == response.headers["X-Request-ID"]

    assert session.messages == []


@pytest.mark.asyncio
async def test_chat_returns_504_when_llm_times_out() -> None:
    services, session = make_chat_services(
        runtime_error=LLMTimeoutError("LLM request timed out.")
    )

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "session_id": "test-session",
                "message": "Hello",
            },
        )

    assert response.status_code == 504
    assert response.json()["error"] == "llm_timeout"
    assert session.messages == []


@pytest.mark.asyncio
async def test_chat_returns_500_when_max_tool_rounds_exceeded() -> None:
    services, session = make_chat_services(
        runtime_error=MaxToolRoundsError("Maximum tool rounds (5) exceeded.")
    )

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "session_id": "test-session",
                "message": "Keep using tools",
            },
        )

    assert response.status_code == 500
    assert response.json()["error"] == "max_tool_rounds_exceeded"

    assert session.messages == []


@pytest.mark.asyncio
async def test_failed_chat_preserves_existing_session_history() -> None:
    existing_messages = [
        {
            "role": "user",
            "content": "My fictional planet is Zephyria.",
        },
        {
            "role": "assistant",
            "content": "Understood.",
        },
    ]

    services, session = make_chat_services(
        runtime_error=LLMError("Unable to communicate with the LLM."),
        messages=existing_messages,
    )

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "session_id": "test-session",
                "message": "What is my fictional planet?",
            },
        )

    assert response.status_code == 503

    assert session.messages == existing_messages


@pytest.mark.asyncio
async def test_health_returns_200() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_ready_returns_503_when_llm_unavailable() -> None:
    services = AsyncMock()

    services.readiness.return_value = {
        "llm": False,
        "mcp": True,
    }

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/ready")

    assert response.status_code == 503

    assert response.json() == {
        "status": "not_ready",
        "dependencies": {
            "llm": "unavailable",
            "mcp": "ok",
        },
    }

    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_ready_returns_200_when_dependencies_ready() -> None:
    services = AsyncMock()

    services.readiness.return_value = {
        "llm": True,
        "mcp": True,
    }

    app.state.services = services

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/ready")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
        "dependencies": {
            "llm": "ok",
            "mcp": "ok",
        },
    }
