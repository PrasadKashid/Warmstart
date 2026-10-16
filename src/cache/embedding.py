from google import genai
from src.config import settings

client = genai.Client(api_key=settings.gemini_api_key)


def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector using Google's embedding model."""
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return response.embeddings[0].values
