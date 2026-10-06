from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str

    message: str = Field(
        min_length=1,
        max_length=10_000,
    )


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    response: str
