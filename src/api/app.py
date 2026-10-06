from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from api.dependencies import ApplicationServices
from api.schemas import ChatRequest, ChatResponse, CreateSessionResponse


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    services = ApplicationServices()

    await services.start()

    app.state.services = services

    try:
        yield
    finally:
        await services.stop()


app = FastAPI(
    title="Local LLM Agent",
    version="0.6.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Any,
) -> Any:
    request_id = str(uuid4())

    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    services: ApplicationServices = request.app.state.services

    session = await services.session_store.get(payload.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    runtime = services.get_runtime()

    async with session.lock:
        original_length = len(session.messages)

        session.messages.append(
            {
                "role": "user",
                "content": payload.message,
            }
        )

        try:
            answer = await runtime.run(session.messages)
        except Exception as exc:
            del session.messages[original_length:]

            raise HTTPException(
                status_code=500,
                detail="Agent execution failed.",
            ) from exc

        session.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    return ChatResponse(
        request_id=request.state.request_id,
        session_id=session.session_id,
        response=answer,
    )


@app.post(
    "/sessions",
    response_model=CreateSessionResponse,
)
async def create_session(
    request: Request,
) -> CreateSessionResponse:
    services: ApplicationServices = request.app.state.services

    session = await services.session_store.create()

    return CreateSessionResponse(
        session_id=session.session_id,
    )
