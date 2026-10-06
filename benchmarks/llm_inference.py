import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "http://localhost:8080/v1"

PROMPT = (
    "Do not use extended reasoning. "
    "Write a detailed technical explanation of asynchronous "
    "programming in Python."
)

RUNS = 10
WARMUP_RUNS = 2
MODEL_NAME = "qwen3-8b-q4_k_m-native-metal"
RESULTS_DIR = Path("benchmarks/results")


@dataclass
class BenchmarkResult:
    run: int
    ttft_ms: float
    total_latency_ms: float
    completion_tokens: int
    effective_tokens_per_second: float
    generation_tokens_per_second: float


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

    ordered = sorted(values)

    index = round((len(ordered) - 1) * percentile_value)

    return ordered[index]


async def benchmark() -> list[BenchmarkResult]:
    print(f"Warming up with {WARMUP_RUNS} requests...")

    for _ in range(WARMUP_RUNS):
        await run_streaming_inference()

    print(f"\nRunning {RUNS} measured requests...\n")

    results = []

    for run in range(1, RUNS + 1):
        result = await run_streaming_inference()

        benchmark_result = BenchmarkResult(
            run=run,
            ttft_ms=result["ttft_ms"],
            total_latency_ms=result["total_latency_ms"],
            completion_tokens=result["completion_tokens"],
            effective_tokens_per_second=result["effective_tokens_per_second"],
            generation_tokens_per_second=result["generation_tokens_per_second"],
        )

        results.append(benchmark_result)

        print(
            f"Run {run:02d}: "
            f"TTFT={benchmark_result.ttft_ms:.1f}ms | "
            f"latency="
            f"{benchmark_result.total_latency_ms:.1f}ms | "
            f"tok/s="
            f"{result['generation_tokens_per_second']:.2f} | "
            f"tok/s="
            f"{benchmark_result.generation_tokens_per_second:.2f}"
        )

    return results


def print_metric(
    name: str,
    values: list[float],
    unit: str,
) -> None:
    mean = statistics.mean(values)
    p5 = percentile(values, 0.05)
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Mean: {mean:.2f} {unit}")
    print(f"P5:   {p5:.2f} {unit}")
    print(f"P50:  {p50:.2f} {unit}")
    print(f"P95:  {p95:.2f} {unit}")


def print_summary(
    results: list[BenchmarkResult],
) -> None:
    print("\n\nLLM Inference Benchmark")
    print("=======================")

    print_metric(
        "TTFT",
        [result.ttft_ms for result in results],
        "ms",
    )

    print_metric(
        "Total Latency",
        [result.total_latency_ms for result in results],
        "ms",
    )

    print_metric(
        "Generation Throughput",
        [result.generation_tokens_per_second for result in results],
        "tok/s",
    )

    print_metric(
        "Completion Tokens",
        [float(result.completion_tokens) for result in results],
        "tokens",
    )


def save_results(
    results: list[BenchmarkResult],
    model_name: str,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RESULTS_DIR / f"{model_name}.json"

    data = {
        "model": model_name,
        "runs": RUNS,
        "warmup_runs": WARMUP_RUNS,
        "results": [asdict(result) for result in results],
    }

    output.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )

    print(f"\nResults saved to {output}")


async def run_streaming_inference() -> dict[str, Any]:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0

    async with (
        httpx.AsyncClient(
            timeout=120.0,
        ) as client,
        client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        received_chunks = 0

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            raw_data = line[6:]

            if raw_data == "[DONE]":
                break

            data = json.loads(raw_data)
            received_chunks += 1
            choices = data.get("choices", [])

            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                reasoning_content = delta.get("reasoning_content")

                generated_text = content or reasoning_content

                if generated_text and first_token_time is None:
                    first_token_time = time.perf_counter()

            usage = data.get("usage")

            if usage:
                completion_tokens = usage.get(
                    "completion_tokens",
                    completion_tokens,
                )
    end = time.perf_counter()

    total_seconds = end - start

    if first_token_time is None:
        raise RuntimeError("No generated content received from llama.cpp.")

    ttft_seconds = first_token_time - start

    generation_seconds = end - first_token_time

    effective_tokens_per_second = (
        completion_tokens / total_seconds if total_seconds > 0 else 0.0
    )

    generation_tokens_per_second = (
        completion_tokens / generation_seconds if generation_seconds > 0 else 0.0
    )

    return {
        "ttft_ms": ttft_seconds * 1000,
        "total_latency_ms": total_seconds * 1000,
        "completion_tokens": completion_tokens,
        "effective_tokens_per_second": (effective_tokens_per_second),
        "generation_tokens_per_second": (generation_tokens_per_second),
    }


async def run_inference() -> dict[str, Any]:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    start = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=120.0,
    ) as client:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
        )

    total_seconds = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    usage = data.get("usage", {})

    completion_tokens = usage.get(
        "completion_tokens",
        0,
    )

    prompt_tokens = usage.get(
        "prompt_tokens",
        0,
    )

    tokens_per_second = completion_tokens / total_seconds if total_seconds > 0 else 0

    return {
        "total_seconds": total_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
    }


async def main() -> None:
    results = await benchmark()

    print_summary(results)

    save_results(
        results,
        MODEL_NAME,
    )


if __name__ == "__main__":
    asyncio.run(main())
