# Warmstart Benchmark Evaluation Report
**Timestamp:** 20261026_205200  
**Target Model:** `gemma-4-26b-a4b-it`  
**Embedding Model:** `models/gemini-embedding-001` (`3072` dims)  
**Similarity Threshold:** `0.77`  

## 1. Executive Summary

| Metric | Result |
| :--- | :--- |
| **Total Completed Queries** | 10 |
| **Overall Cache Hit Rate** | **50.0%** |
| **L1 Exact Hits (Redis)** | 2 |
| **L2 Semantic Hits (Qdrant)** | 3 |
| **Cache Misses (LLM Calls)** | 5 |
| **Average L1 Latency** | **0.48 ms** |
| **Average L2 Latency** | **5368.61 ms** |
| **Average Miss Latency** | **39657.56 ms** |

## 2. Latency Speedup Factor

- **L1 Speedup vs Miss:** ~82619.9x faster
- **L2 Speedup vs Miss:** ~7.4x faster

## 3. Query Breakdown

| ID | Category | Expected | Actual | Latency | Sim Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `q1_base` | geography | `miss` | **`MISS`** | 5341.43 ms | - |
| `q1_exact` | geography | `l1` | **`L1`** | 0.63 ms | 1.0000 |
| `q1_semantic` | geography | `l2` | **`L2`** | 12699.79 ms | 0.8227 |
| `q2_base` | biology | `miss` | **`MISS`** | 42190.28 ms | - |
| `q2_semantic` | biology | `l2` | **`L2`** | 1830.66 ms | 0.7803 |
| `q2_exact` | biology | `l1` | **`L1`** | 0.33 ms | 1.0000 |
| `q3_base` | networking | `miss` | **`MISS`** | 41285.54 ms | - |
| `q3_semantic` | networking | `l2` | **`L2`** | 1575.37 ms | 0.8288 |
| `q4_unrelated` | art | `miss` | **`MISS`** | 69240.24 ms | - |
| `q5_unrelated` | physics | `miss` | **`MISS`** | 40230.33 ms | - |
