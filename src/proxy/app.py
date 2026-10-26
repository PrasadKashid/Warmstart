import time
from fastapi import FastAPI, HTTPException
from src.model.schemas import ChatRequest, ChatResponse
from src.cache.exact_cache import ExactCache
from src.cache.embedding import get_embedding
from src.cache.llm_client import generate_response
from src.cache.vector_store import VectorStore
from src.config import settings

app = FastAPI(
    title="Warmstart Semantic Cache Gateway",
    version="0.1.0",
    description="Multi-tier LLM Caching Gateway (L1 Redis + L2 Qdrant)",
)

l1_cache = ExactCache()
vector_store = VectorStore()


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "warmstart"}


@app.post("/v1/chat", response_model=ChatResponse)
def chat_completion(req: ChatRequest):
    start_time = time.perf_counter()
    selected_model = req.model or settings.llm_model
    target_threshold = req.threshold or settings.similarity_threshold

    # --- TIER 1: L1 Redis Exact Match ---
    try:
        cached_exact = l1_cache.get(req.prompt)
        if cached_exact is not None:
            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"[L1 REDIS HIT] Latency: {elapsed:.2f} ms")
            return ChatResponse(
                response=cached_exact,
                cached=True,
                similarity_score=1.0,
                latency_ms=round(elapsed, 2),
                model=selected_model,
            )
    except Exception as e:
        print(f"[L1 WARNING] Redis lookup failed: {e}")

    # --- TIER 2: L2 Qdrant Semantic Match ---
    query_vector = None
    try:
        query_vector = get_embedding(req.prompt)
        cache_hit = vector_store.search_similar(
            query_vector=query_vector,
            threshold=target_threshold,
        )

        if cache_hit:
            elapsed = (time.perf_counter() - start_time) * 1000
            cached_response = cache_hit["payload"]["response"]

            # Populate L1 Redis so repeat exact queries bypass embedding
            try:
                l1_cache.set(req.prompt, cached_response)
            except Exception:
                pass

            print(
                f"[L2 QDRANT HIT] Similarity: {cache_hit['score']:.4f} | Latency: {elapsed:.2f} ms"
            )
            return ChatResponse(
                response=cached_response,
                cached=True,
                similarity_score=round(cache_hit["score"], 4),
                latency_ms=round(elapsed, 2),
                model=selected_model,
            )
    except Exception as e:
        print(f"[L2 WARNING] Semantic search failed: {e}")

    # --- CACHE MISS: Upstream LLM Generation ---
    print(f"[CACHE MISS] Calling upstream LLM: {selected_model}...")
    try:
        llm_output = generate_response(prompt=req.prompt, model=selected_model)
    except Exception as e:
        print(f"[LLM ERROR] Upstream generation failed: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {str(e)}")

    # Persist results to both tiers
    if query_vector:
        try:
            vector_store.store(
                prompt=req.prompt,
                vector=query_vector,
                response=llm_output,
                metadata={"model": selected_model},
            )
        except Exception as e:
            print(f"[QDRANT WARNING] Failed to persist vector: {e}")

    try:
        l1_cache.set(req.prompt, llm_output)
    except Exception as e:
        print(f"[REDIS WARNING] Failed to persist L1 cache: {e}")

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[CACHE STORED] Total Latency: {elapsed:.2f} ms")

    return ChatResponse(
        response=llm_output,
        cached=False,
        similarity_score=None,
        latency_ms=round(elapsed, 2),
        model=selected_model,
    )
