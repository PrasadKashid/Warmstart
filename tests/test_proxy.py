import httpx

BASE_URL = "http://127.0.0.1:8000"


def query(prompt: str):
    """
    Sends a POST request to the Warmstart proxy gateway.

    Why httpx?
    - Cleaner API than requests, with built-in HTTP/2 support and better timeout defaults.
    """
    response = httpx.post(
        f"{BASE_URL}/v1/chat",
        json={"prompt": prompt},
        timeout=30.0,
    )
    return response.json()


if __name__ == "__main__":
    prompt_1 = "Explain quantum computing in two sentences."
    print(f"\n1. Initial prompt: '{prompt_1}'")
    res1 = query(prompt_1)
    print(f"   Cached: {res1['cached']} | Similarity: {res1.get('similarity_score')} | Latency: {res1['latency_ms']} ms")

    prompt_2 = "Can you describe what quantum computing is in two short sentences?"
    print(f"\n2. Semantic variation: '{prompt_2}'")
    res2 = query(prompt_2)
    print(f"   Cached: {res2['cached']} | Similarity: {res2.get('similarity_score')} | Latency: {res2['latency_ms']} ms")

    prompt_3 = "Explain quantum computing in two sentences."
    print(f"\n3. Exact repeat of prompt 1 (L1 test): '{prompt_3}'")
    res3 = query(prompt_3)
    print(f"   Cached: {res3['cached']} | Similarity: {res3.get('similarity_score')} | Latency: {res3['latency_ms']} ms")