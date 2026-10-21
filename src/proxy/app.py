import hashlib
import time
from fastapi import FastAPI
from src.model.schemas import ChatRequest, ChatResponse
from src.cache.embedding import get_embedding
from src.cache.llm_client import generate_response
from src.cache.vector_store import VectorStore
from src.config import settings

app = FastAPI(
    title="Warmstart Semantic Cache Gateway",
    version="0.1.0",
)

vector_store = VectorStore()

# Fast in-memory L1 cache: hash(prompt) -> response
l1_cache: dict[str, str] = {}


def get_l1_key(prompt: str) -> str:
    """Normalize text (trim + lowercase) and hash with SHA-256."""
    normalized = prompt.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "warmstart"}


@app.post("/v1/chat", response_model=ChatResponse)
def chat_completion(req: ChatRequest):
    start_time = time.perf_counter()
    selected_model = req.model or settings.llm_model
    target_threshold = req.threshold or settings.similarity_threshold

    # --- TIER 1: L1 Exact Match (NO NETWORK CALL) ---
    l1_key = get_l1_key(req.prompt)
    if l1_key in l1_cache:
        elapsed = (time.perf_counter() - start_time) * 1000
        print(
            f"[CACHE TIER 1 HIT - L1] Key: {l1_key[:8]}... | Latency: {elapsed:.2f} ms"
        )
        return ChatResponse(
            response=l1_cache[l1_key],
            cached=True,
            similarity_score=1.0,
            latency_ms=round(elapsed, 2),
            model=selected_model,
        )

    # --- TIER 2: L2 Semantic Match (Calls Embedding API + Qdrant) ---
    print(f"[CACHE TIER 1 MISS] Generating embedding for L2 check...")
    t_embed_start = time.perf_counter()
    query_vector = get_embedding(req.prompt)
    embed_duration = (time.perf_counter() - t_embed_start) * 1000
    print(f"[EMBEDDING DURATION] Google API took: {embed_duration:.2f} ms")

    cache_hit = vector_store.search_similar(
        query_vector=query_vector,
        threshold=target_threshold,
    )

    if cache_hit:
        elapsed = (time.perf_counter() - start_time) * 1000
        cached_text = cache_hit["payload"]["response"]

        # Save to L1 so identical calls skip the embedding step next time
        l1_cache[l1_key] = cached_text
        print(
            f"[CACHE TIER 2 HIT - L2] Similarity: {cache_hit['score']:.4f} | Total Latency: {elapsed:.2f} ms"
        )

        return ChatResponse(
            response=cached_text,
            cached=True,
            similarity_score=round(cache_hit["score"], 4),
            latency_ms=round(elapsed, 2),
            model=selected_model,
        )

    # --- CACHE MISS: Upstream LLM ---
    print(f"[CACHE MISS] Calling upstream LLM ({selected_model})...")
    llm_output = generate_response(prompt=req.prompt, model=selected_model)

    # Save to both tiers
    vector_store.store(
        prompt=req.prompt,
        vector=query_vector,
        response=llm_output,
        metadata={"model": selected_model},
    )
    l1_cache[l1_key] = llm_output

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[CACHE STORED] Total Latency: {elapsed:.2f} ms")
    return ChatResponse(
        response=llm_output,
        cached=False,
        similarity_score=None,
        latency_ms=round(elapsed, 2),
        model=selected_model,
    )
