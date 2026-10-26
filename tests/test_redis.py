import time
from src.cache.exact_cache import ExactCache

if __name__ == "__main__":
    cache = ExactCache(default_ttl_seconds=2)

    prompt = "Explain quantum computing in two sentences."
    mock_response = "Quantum computers use qubits to compute in superposition."

    print("1. Testing Redis Write...")
    cache.set(prompt, mock_response)
    print("   Value written successfully.")

    print("\n2. Testing Exact Lookup...")
    cached_val = cache.get(prompt)
    assert cached_val == mock_response, "Cache value mismatch!"
    print(f"   [PASS] Retrieved: '{cached_val}'")

    print("\n3. Testing Case and Whitespace Normalization...")
    varied_prompt = "  Explain QUANTUM computing in two sentences.   "
    normalized_val = cache.get(varied_prompt)
    assert normalized_val == mock_response, "Normalization failed to hit cache!"
    print(f"   [PASS] Retrieved matching value for whitespace/casing variant!")

    print("\n4. Testing TTL Expiration (waiting 3 seconds)...")
    time.sleep(3)
    expired_val = cache.get(prompt)
    assert expired_val is None, "Cache value failed to expire!"
    print("   [PASS] Key expired as expected.")

    print("\nRedis L1 Cache verified and functioning properly!")
