from google import genai
from src.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

print("Fetching active models supporting generateContent...\n")
active_models = []

try:
    for m in client.models.list():
        actions = getattr(m, "supported_actions", []) or []
        if "generateContent" in actions:
            # Model name often arrives as 'models/gemini-...'
            clean_name = m.name.replace("models/", "")
            active_models.append(clean_name)
except Exception as e:
    print(f"Failed to list models: {e}")

print(f"Discovered {len(active_models)} candidate models.")

working_model = None
for name in active_models:
    print(f"Testing: {name} ...", end=" ", flush=True)
    try:
        res = client.models.generate_content(
            model=name,
            contents="Say 'ready'",
        )
        output = res.text.strip() if res.text else "OK"
        print(f"[PASS] Response: {output}")
        working_model = name
        break
    except Exception as e:
        short_err = str(e).split("\n")[0][:60]
        print(f"[FAIL] ({short_err})")

if working_model:
    print(f"\n============================================")
    print(f"Set this in your .env:")
    print(f"LLM_MODEL={working_model}")
    print(f"============================================")
else:
    print("\nNo generation model succeeded. Check API status or credentials.")
