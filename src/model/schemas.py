from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Input schema for the /v1/chat endpoint.

    Why:
    - prompt: The input text sent by the user/client to be matched or answered.
    - model: Allows the client to optionally specify an LLM (defaults to settings.llm_model).
    - threshold: Allows client-level customization of semantic distance (defaults to settings.similarity_threshold).
    """

    prompt: str = Field(..., description="The user prompt to evaluate or cache")
    model: Optional[str] = Field(
        default=None, description="Optional LLM model override"
    )
    threshold: Optional[float] = Field(
        default=None, description="Optional cosine similarity threshold override"
    )


class ChatResponse(BaseModel):
    """
    Output schema for the /v1/chat endpoint.

    Why:
    - response: The generated or cached textual answer.
    - cached: Boolean flag signaling whether the response was a cache hit.
    - similarity_score: Cosine similarity score (present on cache hit, None on miss).
    - latency_ms: Execution duration in milliseconds for benchmarking.
    - model: Which LLM or cache namespace served the request.
    """

    response: str
    cached: bool
    similarity_score: Optional[float] = None
    latency_ms: float
    model: str
