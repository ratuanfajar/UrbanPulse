"""Pydantic schemas for API request and response bodies."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        examples=["user-123"],
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=[
            "How can the government upgrade a riverside slum settlement in Jakarta?"
        ],
    )


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
    llm_ready: bool
    search_ready: bool
