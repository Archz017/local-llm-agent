from contextvars import ContextVar, Token

request_id_var: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

session_id_var: ContextVar[str | None] = ContextVar(
    "session_id",
    default=None,
)


def set_request_id(request_id: str) -> Token:
    return request_id_var.set(request_id)


def set_session_id(session_id: str) -> Token:
    return session_id_var.set(session_id)


def reset_request_id(token: Token) -> None:
    request_id_var.reset(token)


def reset_session_id(token: Token) -> None:
    session_id_var.reset(token)
