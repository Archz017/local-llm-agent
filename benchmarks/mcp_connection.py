import asyncio
import statistics
import time

from mcp_client.client import MCPClient

RUNS = 10


async def persistent_connection() -> list[float]:
    timings = []

    async with MCPClient() as client:
        for _ in range(RUNS):
            start = time.perf_counter()

            await client.call_tool(
                "calculator_add",
                {
                    "a": 1,
                    "b": 2,
                },
            )

            timings.append((time.perf_counter() - start) * 1000)

    return timings


async def reconnect_each_time() -> list[float]:
    timings = []

    for _ in range(RUNS):
        start = time.perf_counter()

        async with MCPClient() as client:
            await client.call_tool(
                "calculator_add",
                {
                    "a": 1,
                    "b": 2,
                },
            )

        timings.append((time.perf_counter() - start) * 1000)

    return timings


async def main() -> None:
    persistent = await persistent_connection()
    reconnect = await reconnect_each_time()

    persistent_mean = statistics.mean(persistent)
    reconnect_mean = statistics.mean(reconnect)

    speedup = reconnect_mean / persistent_mean

    print("\nMCP Connection Benchmark")
    print("========================")
    print(f"Runs:                    {RUNS}")
    print(f"Persistent mean:         {persistent_mean:.2f} ms")
    print(f"Reconnect-each-time:     {reconnect_mean:.2f} ms")
    print(f"Speedup:                 {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
