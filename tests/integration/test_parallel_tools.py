import time

import pytest

from agent.tool_executor import ToolExecutor
from mcp_client.client import MCPClient

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


@pytest.mark.asyncio
async def test_parallel_execution_is_faster() -> None:
    async with MCPClient() as client:
        executor = ToolExecutor(client)

        start = time.perf_counter()

        await executor.execute_sequential(TOOL_CALLS)

        sequential_duration = time.perf_counter() - start

        start = time.perf_counter()

        await executor.execute_parallel(TOOL_CALLS)

        parallel_duration = time.perf_counter() - start

    print(f"\nSequential: {sequential_duration:.2f}s")
    print(f"Parallel:   {parallel_duration:.2f}s")

    assert parallel_duration < sequential_duration
