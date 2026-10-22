from src.cache.embedding import get_embedding
from src.cache.vector_store import VectorStore

if __name__ == "__main__":
    store = VectorStore(location=":memory:")  # Fast in-memory test

    original_prompt = "What is the capital of France?"
    mock_response = "The capital of France is Paris."

    print("Embedding and storing initial prompt...")
    vector = get_embedding(original_prompt)
    store.store(prompt=original_prompt, vector=vector, response=mock_response)

    # Test 1: Semantically equivalent prompt
    similar_prompt = "Can you tell me what France's capital city is?"
    print(f"\nSearching with similar prompt: '{similar_prompt}'")
    similar_vector = get_embedding(similar_prompt)
    match = store.search_similar(similar_vector, threshold=0.80)

    if match:
        print("Semantic Cache HIT!")
        print(f"Similarity Score: {match['score']:.4f}")
        print(f"Cached Response: {match['payload']['response']}")
    else:
        print("Semantic Cache MISS!")

    # Test 2: Unrelated prompt
    unrelated_prompt = "How do you bake sourdough bread?"
    print(f"\nSearching with unrelated prompt: '{unrelated_prompt}'")
    unrelated_vector = get_embedding(unrelated_prompt)
    match2 = store.search_similar(unrelated_vector, threshold=0.80)

    if match2:
        print("Unexpected HIT!")
    else:
        print("Semantic Cache MISS (expected)!")