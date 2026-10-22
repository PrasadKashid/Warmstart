from google import genai
from src.config import settings

# Initialize Google GenAI client using the configured API key
client = genai.Client(api_key=settings.gemini_api_key)


def generate_response(prompt: str, model: str = None) -> str:
    """
    Invokes Google Gemini when a cache miss occurs.

    Why this matters:
    - Uses settings.llm_model (models/gemini-2.5-flash) by default.
    - Extracts the generated text from response.text cleanly.
    """
    target_model = model or settings.llm_model

    response = client.models.generate_content(
        model=target_model,
        contents=prompt,
    )
    return response.text
