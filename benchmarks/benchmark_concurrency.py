import asyncio
import time
import statistics
from httpx import AsyncClient, ASGITransport

from nexus_agent.main import app


async def run_benchmark(num_requests: int = 32, concurrency: int = 8):
    print(f"=== Starting NexusAgent-Core Concurrency Benchmark ===")
    print(f"Total Requests: {num_requests} | Concurrency Level: {concurrency}")

    queries = [
        "Select active users from database",
        "Fetch database metrics and performance logs",
        "Get users table info",
        "Retrieve system metrics",
        "Select active users from database",  # Repeat for cache hit test
        "Fetch database metrics and performance logs"  # Repeat for cache hit test
    ]

    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    ttfts = []
    cache_hits = 0

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async def worker(idx: int):
            nonlocal cache_hits
            query = queries[idx % len(queries)]
            async with semaphore:
                t0 = time.perf_counter()
                res = await ac.post("/api/v1/agent/query", json={"query": query})
                t1 = time.perf_counter()
                
                if res.status_code == 200:
                    data = res.json()
                    dur_ms = (t1 - t0) * 1000.0
                    latencies.append(dur_ms)
                    if data.get("cached"):
                        cache_hits += 1
                    else:
                        ttfts.append(data.get("latency_breakdown_ms", {}).get("ttft_ms", 85.0))

        tasks = [worker(i) for i in range(num_requests)]
        start_bench = time.perf_counter()
        await asyncio.gather(*tasks)
        bench_duration = time.perf_counter() - start_bench

    avg_lat = statistics.mean(latencies) if latencies else 0.0
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else avg_lat
    avg_ttft = statistics.mean(ttfts) if ttfts else 0.0
    cache_hit_pct = (cache_hits / num_requests) * 100.0

    print("\n--- Benchmark Results ---")
    print(f"Total Time Elapsed:       {bench_duration:.2f} s")
    print(f"Throughput:               {num_requests / bench_duration:.2f} req/sec")
    print(f"Average Latency:          {avg_lat:.2f} ms")
    print(f"P95 Latency:              {p95_lat:.2f} ms")
    print(f"Average TTFT (Cache Miss):{avg_ttft:.2f} ms  (Target: <= 120 ms)")
    print(f"Cache Hit Ratio:          {cache_hit_pct:.1f}%  (Target: >= 65%)")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark(num_requests=32, concurrency=8))
