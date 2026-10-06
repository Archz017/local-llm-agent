import asyncio

import pytest

from agent.sessions import SessionStore


@pytest.mark.asyncio
async def test_create_and_get_session() -> None:
    store = SessionStore()

    session = await store.create()
    retrieved = await store.get(session.session_id)

    assert retrieved is session


@pytest.mark.asyncio
async def test_sessions_are_isolated() -> None:
    store = SessionStore()

    session_a = await store.create()
    session_b = await store.create()

    session_a.messages.append(
        {
            "role": "user",
            "content": "My favorite planet is Zephyria.",
        }
    )

    assert session_a.session_id != session_b.session_id
    assert len(session_a.messages) == 1
    assert session_b.messages == []


@pytest.mark.asyncio
async def test_delete_session() -> None:
    store = SessionStore()

    session = await store.create()

    deleted = await store.delete(session.session_id)

    assert deleted is True
    assert await store.get(session.session_id) is None


@pytest.mark.asyncio
async def test_same_session_lock_serializes_requests() -> None:
    store = SessionStore()
    session = await store.create()

    active = 0
    max_active = 0

    async def simulated_request() -> None:
        nonlocal active, max_active

        async with session.lock:
            active += 1
            max_active = max(max_active, active)

            await asyncio.sleep(0.05)

            active -= 1

    await asyncio.gather(
        simulated_request(),
        simulated_request(),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_different_sessions_can_run_concurrently() -> None:
    store = SessionStore()

    session_a = await store.create()
    session_b = await store.create()

    active = 0
    max_active = 0

    async def simulated_request(session) -> None:
        nonlocal active, max_active

        async with session.lock:
            active += 1
            max_active = max(max_active, active)

            await asyncio.sleep(0.05)

            active -= 1

    await asyncio.gather(
        simulated_request(session_a),
        simulated_request(session_b),
    )

    assert max_active == 2
