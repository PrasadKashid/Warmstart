from google import genai
from src.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

# Test candidate model names
candidates = [
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
    "gemini-2.5-pro",
    "models/gemini-2.5-pro",
    "gemini-1.5-flash",
    "models/gemini-1.5-flash",
    "gemma-4-26b-a4b-it",
    "models/gemma-4-26b-a4b-it",
]

print("Checking available models for generation...\n")
working_model = None

for model_name in candidates:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'hello'",
        )
        print(f" [PASS] Working model: {model_name}")
        print(f"        Output: {response.text.strip()}\n")
        working_model = model_name
        break
    except Exception as e:
        # Extract short error message
        err_msg = str(e).split("\n")[0]
        print(f" [FAIL] {model_name} -> {err_msg}")

if working_model:
    print(f"\n=> Set your .env LLM_MODEL to: {working_model}")
