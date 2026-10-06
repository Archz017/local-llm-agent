import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx

BASE_URL = "http://localhost:8080/v1"

CONCURRENCY_LEVELS = [1, 2, 4, 8]

TOTAL_REQUESTS = 20

concurrency = 4

PROMPT = (
    "Do not use extended reasoning. "
    "Explain asynchronous programming in Python "
    "with a practical example."
)

MAX_TOKENS = 128


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    ordered = sorted(values)

    index = round((len(ordered) - 1) * percentile_value)

    return ordered[index]


@dataclass
class RequestResult:
    latency_seconds: float
    completion_tokens: int


async def run_request(
    client: httpx.AsyncClient,
) -> RequestResult:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }

    start = time.perf_counter()

    response = await client.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
    )

    response.raise_for_status()

    latency = time.perf_counter() - start

    data = response.json()

    completion_tokens = data.get(
        "usage",
        {},
    ).get(
        "completion_tokens",
        0,
    )

    return RequestResult(
        latency_seconds=latency,
        completion_tokens=completion_tokens,
    )


async def benchmark_level(
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=300.0,
    ) as client:
        # Warmup outside measured workload.
        await run_request(client)

        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                bounded_request(
                    client,
                    semaphore,
                )
                for _ in range(TOTAL_REQUESTS)
            ]
        )

        wall_time = time.perf_counter() - start

    latencies = [result.latency_seconds for result in results]

    total_tokens = sum(result.completion_tokens for result in results)

    requests_per_second = TOTAL_REQUESTS / wall_time

    aggregate_tokens_per_second = total_tokens / wall_time

    print(f"\nConcurrency: {concurrency}")
    print("-" * 30)

    print(f"Requests:        {TOTAL_REQUESTS}")
    print(f"Wall time:       {wall_time:.2f}s")
    print(f"Requests/sec:    {requests_per_second:.3f}")
    print(f"Aggregate tok/s: {aggregate_tokens_per_second:.2f}")
    print(f"Mean latency:    {statistics.mean(latencies):.2f}s")
    print(f"P50 latency:     {percentile(latencies, 0.50):.2f}s")
    print(f"P95 latency:     {percentile(latencies, 0.95):.2f}s")


async def bounded_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        return await run_request(client)


async def main() -> None:
    print("LLM Concurrent Inference Benchmark")
    print("==================================")

    for concurrency in CONCURRENCY_LEVELS:
        await benchmark_level(concurrency)


if __name__ == "__main__":
    asyncio.run(main())
