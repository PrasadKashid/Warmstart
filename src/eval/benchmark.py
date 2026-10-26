import json
import time
from datetime import datetime
from pathlib import Path
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FilterSelector
from src.cache.exact_cache import ExactCache
from src.config import settings

BASE_URL = "http://127.0.0.1:8000"
DATASET_PATH = Path("data/benchmark/dataset.json")
RESULTS_DIR = Path("data/results")


def reset_caches():
    print("[BENCHMARK SETUP] Flushing L1 Redis cache...")
    ExactCache().flush()

    print("[BENCHMARK SETUP] Clearing points from L2 Qdrant collection...")
    client = QdrantClient(url=settings.qdrant_url)
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection in collections:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=FilterSelector(filter=Filter()),
        )
    print("[BENCHMARK SETUP] Fresh environment ready.\n")


def run_benchmark():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    reset_caches()

    results = []
    print(f"Running benchmark on {len(test_cases)} test cases...\n")
    print(
        f"{'ID':<15} | {'Category':<12} | {'Expected':<8} | {'Status':<6} | {'Sim':<6} | {'Latency (ms)':<12}"
    )
    print("-" * 75)

    # 120s timeout handles long generation times on cache misses
    client = httpx.Client(base_url=BASE_URL, timeout=120.0)

    for item in test_cases:
        item_id = item["id"]
        prompt = item["prompt"]
        expected = item["expected_tier"]
        category = item["category"]

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                resp = client.post("/v1/chat", json={"prompt": prompt})

                if resp.status_code == 200:
                    res = resp.json()
                    cached = res["cached"]
                    sim_score = res.get("similarity_score")
                    latency = res["latency_ms"]

                    if not cached:
                        actual_tier = "miss"
                        # Generous pause after misses to avoid saturating Google RPM limits
                        time.sleep(5.0)
                    elif sim_score == 1.0:
                        actual_tier = "l1"
                        time.sleep(0.2)
                    else:
                        actual_tier = "l2"
                        time.sleep(1.0)

                    results.append(
                        {
                            "id": item_id,
                            "category": category,
                            "expected_tier": expected,
                            "actual_tier": actual_tier,
                            "cached": cached,
                            "similarity_score": sim_score,
                            "latency_ms": latency,
                            "prompt": prompt,
                            "response": (
                                (res["response"][:80] + "...")
                                if res.get("response")
                                else ""
                            ),
                        }
                    )

                    sim_str = f"{sim_score:.4f}" if sim_score is not None else "N/A"
                    print(
                        f"{item_id:<15} | {category:<12} | {expected:<8} | {actual_tier.upper():<6} | {sim_str:<6} | {latency:>10.2f} ms"
                    )
                    break
                else:
                    if attempt < max_attempts - 1:
                        print(
                            f"[{item_id}] HTTP {resp.status_code} - Waiting 8s before retry..."
                        )
                        time.sleep(8.0)
                    else:
                        print(
                            f"{item_id:<15} | {category:<12} | {expected:<8} | ERROR  | N/A    | HTTP {resp.status_code}"
                        )

            except Exception as e:
                if attempt < max_attempts - 1:
                    print(
                        f"[{item_id}] Request failed ({str(e)[:25]}) - Waiting 8s before retry..."
                    )
                    time.sleep(8.0)
                else:
                    print(
                        f"{item_id:<15} | {category:<12} | {expected:<8} | FAIL   | N/A    | {str(e)[:20]}"
                    )

    if not results:
        print("\nNo results gathered.")
        return

    total_queries = len(results)
    l1_hits = sum(1 for r in results if r["actual_tier"] == "l1")
    l2_hits = sum(1 for r in results if r["actual_tier"] == "l2")
    misses = sum(1 for r in results if r["actual_tier"] == "miss")

    l1_latencies = [r["latency_ms"] for r in results if r["actual_tier"] == "l1"]
    l2_latencies = [r["latency_ms"] for r in results if r["actual_tier"] == "l2"]
    miss_latencies = [r["latency_ms"] for r in results if r["actual_tier"] == "miss"]

    avg_l1 = sum(l1_latencies) / len(l1_latencies) if l1_latencies else 0.0
    avg_l2 = sum(l2_latencies) / len(l2_latencies) if l2_latencies else 0.0
    avg_miss = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0
    overall_hit_rate = ((l1_hits + l2_hits) / total_queries) * 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"benchmark_report_{timestamp}.md"
    json_file = RESULTS_DIR / f"benchmark_results_{timestamp}.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "summary": {
                    "total_queries": total_queries,
                    "overall_hit_rate_pct": round(overall_hit_rate, 2),
                    "l1_hits": l1_hits,
                    "l2_hits": l2_hits,
                    "misses": misses,
                    "avg_l1_latency_ms": round(avg_l1, 2),
                    "avg_l2_latency_ms": round(avg_l2, 2),
                    "avg_miss_latency_ms": round(avg_miss, 2),
                },
                "details": results,
            },
            f,
            indent=2,
        )

    speedup_l1 = f"{(avg_miss / avg_l1):.1f}x" if avg_l1 > 0 else "N/A"
    speedup_l2 = f"{(avg_miss / avg_l2):.1f}x" if avg_l2 > 0 else "N/A"

    markdown_content = f"""# Warmstart Benchmark Evaluation Report
**Timestamp:** {timestamp}  
**Target Model:** `{settings.llm_model}`  
**Embedding Model:** `{settings.embedding_model}` (`{settings.embedding_dim}` dims)  
**Similarity Threshold:** `{settings.similarity_threshold}`  

## 1. Executive Summary

| Metric | Result |
| :--- | :--- |
| **Total Completed Queries** | {total_queries} |
| **Overall Cache Hit Rate** | **{overall_hit_rate:.1f}%** |
| **L1 Exact Hits (Redis)** | {l1_hits} |
| **L2 Semantic Hits (Qdrant)** | {l2_hits} |
| **Cache Misses (LLM Calls)** | {misses} |
| **Average L1 Latency** | **{avg_l1:.2f} ms** |
| **Average L2 Latency** | **{avg_l2:.2f} ms** |
| **Average Miss Latency** | **{avg_miss:.2f} ms** |

## 2. Latency Speedup Factor

- **L1 Speedup vs Miss:** ~{speedup_l1} faster
- **L2 Speedup vs Miss:** ~{speedup_l2} faster

## 3. Query Breakdown

| ID | Category | Expected | Actual | Latency | Sim Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results:
        sim = f"{r['similarity_score']:.4f}" if r["similarity_score"] else "-"
        markdown_content += f"| `{r['id']}` | {r['category']} | `{r['expected_tier']}` | **`{r['actual_tier'].upper()}`** | {r['latency_ms']:.2f} ms | {sim} |\n"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print("-" * 75)
    print(f"\n[BENCHMARK COMPLETE]")
    print(
        f"Overall Hit Rate : {overall_hit_rate:.1f}% ({l1_hits + l2_hits}/{total_queries})"
    )
    print(f"Avg L1 Latency   : {avg_l1:.2f} ms")
    print(f"Avg L2 Latency   : {avg_l2:.2f} ms")
    print(f"Avg Miss Latency : {avg_miss:.2f} ms")
    print(f"Reports saved to : {report_file}")


if __name__ == "__main__":
    run_benchmark()
