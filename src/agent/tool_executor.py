import asyncio
import time
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult

from agent.tool_parser import parse_tool_arguments
from mcp_client.client import MCPClient
from observability.metrics import (
    TOOL_EXECUTION_DURATION_SECONDS,
    TOOL_EXECUTIONS_TOTAL,
)


@dataclass
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None
    duration_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ToolExecutor:
    def __init__(
        self,
        mcp_client: MCPClient,
        max_concurrency: int = 4,
    ) -> None:
        self.mcp = mcp_client
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        start = time.perf_counter()

        async with self.semaphore:
            result: CallToolResult = await self.mcp.call_tool(
                tool_name,
                arguments,
            )

        duration_seconds = time.perf_counter() - start

        TOOL_EXECUTION_DURATION_SECONDS.labels(
            tool_name=tool_name,
        ).observe(duration_seconds)

        outcome = "error" if result.is_error else "success"

        TOOL_EXECUTIONS_TOTAL.labels(
            tool_name=tool_name,
            outcome=outcome,
        ).inc()

        if result.is_error:
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                duration_ms=duration_seconds * 1000,
                error="MCP tool reported an execution error.",
            )

        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_seconds * 1000,
        )

    async def execute_sequential(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolExecutionResult]:
        results = []

        for tool_call in tool_calls:
            function = tool_call["function"]

            arguments = parse_tool_arguments(function.get("arguments", {}))

            result = await self.execute(
                tool_call_id=tool_call["id"],
                tool_name=function["name"],
                arguments=arguments,
            )

            results.append(result)

        return results

    async def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolExecutionResult]:
        tasks = []

        for tool_call in tool_calls:
            function = tool_call["function"]

            arguments = parse_tool_arguments(function.get("arguments", {}))

            task = self.execute(
                tool_call_id=tool_call["id"],
                tool_name=function["name"],
                arguments=arguments,
            )

            tasks.append(task)

        return list(await asyncio.gather(*tasks))
