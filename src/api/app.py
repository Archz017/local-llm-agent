import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from api.dependencies import ApplicationServices
from api.schemas import ChatRequest, ChatResponse, CreateSessionResponse
from observability.context import (
    reset_request_id,
    reset_session_id,
    set_request_id,
    set_session_id,
)
from observability.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

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
    call_next,
):
    request_id = str(uuid4())
    request.state.request_id = request_id

    token = set_request_id(request_id)
    start = time.perf_counter()

    try:
        logger.info(
            "HTTP request started",
            extra={
                "event": "http_request_started",
            },
        )

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "duration_ms": round(duration_ms, 2),
                "status_code": response.status_code,
            },
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "HTTP request failed",
            extra={
                "event": "http_request_failed",
                "duration_ms": round(duration_ms, 2),
                "error_type": type(exc).__name__,
            },
        )

        raise

    finally:
        reset_request_id(token)


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

    session_token = set_session_id(session.session_id)

    try:
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
            except Exception:
                del session.messages[original_length:]
                raise

            session.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
    finally:
        reset_session_id(session_token)

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
