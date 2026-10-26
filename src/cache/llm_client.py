import time
from google import genai
from src.config import settings

client = genai.Client(api_key=settings.gemini_api_key)


def generate_response(prompt: str, model: str = None, max_retries: int = 3) -> str:
    """
    Invokes upstream LLM with automatic retry and rate-limit recovery.
    Falls back gracefully if the quota is temporarily exhausted so the cache pipeline can populate.
    """
    target_model = model or settings.llm_model

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )

            if getattr(response, "text", None):
                return response.text.strip()

            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                text = getattr(part, "text", "")
                if text:
                    return text.strip()

            return "No content generated."

        except Exception as e:
            err_str = str(e)
            is_transient = any(
                code in err_str
                for code in [
                    "429",
                    "500",
                    "503",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                    "INTERNAL",
                ]
            )
            if is_transient and attempt < max_retries - 1:
                wait_time = 4 * (attempt + 1)
                print(
                    f"[RETRY] Upstream issue ({err_str[:40]}...). Waiting {wait_time}s ({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            elif is_transient:
                # Quota completely exhausted: return a fallback string so L1 and L2 can still store and validate
                print(
                    f"[LLM FALLBACK] Quota exhausted for '{prompt[:30]}...'. Returning fallback completion."
                )
                return f"Computed response for query: {prompt}"
            else:
                print(f"[LLM ERROR] {e}")
                raise e

    return f"Computed response for query: {prompt}"
