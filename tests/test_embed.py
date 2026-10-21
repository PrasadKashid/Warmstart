from src.cache.embedding import get_embedding

if __name__ == '__main__':
    prompt = "How does semantic caching reduce LLM latency?"
    vector = get_embedding(prompt)
    print("Embedding generated successfully!")
    print(f"Vector dimension: {len(vector)}")
    print(f"Sample values: {vector[:4]}")