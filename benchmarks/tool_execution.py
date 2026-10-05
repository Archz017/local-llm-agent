import asyncio
import statistics
import time

from agent.tool_executor import ToolExecutor
from mcp_client.client import MCPClient


def percentile(
    values: list[float],
    percent: float,
) -> float:
    ordered = sorted(values)

    index = int((len(ordered) - 1) * percent)

    return ordered[index]


TOOL_CALLS = [
    {
        "id": "call_1",
        "function": {
            "name": "wait",
            "arguments": {
                "seconds": 1.0,
                "label": "A",
            },
        },
    },
    {
        "id": "call_2",
        "function": {
            "name": "wait",
            "arguments": {
                "seconds": 1.0,
                "label": "B",
            },
        },
    },
    {
        "id": "call_3",
        "function": {
            "name": "wait",
            "arguments": {
                "seconds": 1.0,
                "label": "C",
            },
        },
    },
]


async def measure(
    executor: ToolExecutor,
    strategy: str,
) -> float:
    start = time.perf_counter()

    if strategy == "sequential":
        await executor.execute_sequential(TOOL_CALLS)

    elif strategy == "parallel":
        await executor.execute_parallel(TOOL_CALLS)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return time.perf_counter() - start


async def benchmark(
    runs: int = 5,
) -> None:
    async with MCPClient() as client:
        executor = ToolExecutor(
            client,
            max_concurrency=4,
        )

        sequential_times = []
        parallel_times = []

        print(f"Running {runs} benchmark iterations...\n")

        for run in range(1, runs + 1):
            sequential = await measure(
                executor,
                "sequential",
            )

            parallel = await measure(
                executor,
                "parallel",
            )

            sequential_times.append(sequential)
            parallel_times.append(parallel)

            print(f"Run {run}: sequential={sequential:.2f}s, parallel={parallel:.2f}s")

    sequential_mean = statistics.mean(sequential_times)

    parallel_mean = statistics.mean(parallel_times)

    sequential_p50 = percentile(
        sequential_times,
        0.50,
    )

    sequential_p95 = percentile(
        sequential_times,
        0.95,
    )

    parallel_p50 = percentile(
        parallel_times,
        0.50,
    )

    parallel_p95 = percentile(
        parallel_times,
        0.95,
    )

    speedup = sequential_mean / parallel_mean

    reduction = (sequential_mean - parallel_mean) / sequential_mean * 100

    print("\nTool Execution Benchmark")
    print("========================")
    print(f"Runs:            {runs}")
    print(f"Tools/run:       {len(TOOL_CALLS)}")

    print()
    print("Sequential")
    print(f"  Mean: {sequential_mean:.2f}s")
    print(f"  P50:  {sequential_p50:.2f}s")
    print(f"  P95:  {sequential_p95:.2f}s")

    print()
    print("Parallel")
    print(f"  Mean: {parallel_mean:.2f}s")
    print(f"  P50:  {parallel_p50:.2f}s")
    print(f"  P95:  {parallel_p95:.2f}s")

    print(f"Speedup:         {speedup:.2f}x")
    print(f"Latency reduced: {reduction:.1f}%")


if __name__ == "__main__":
    asyncio.run(benchmark())
