import pytest
from mcp.types import CallToolResult, TextContent

from agent.tool_executor import ToolExecutor


class FakeMCP:
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> CallToolResult:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="success",
                )
            ],
            isError=False,
        )


class FailingMCP:
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> CallToolResult:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="tool failed",
                )
            ],
            isError=True,
        )


@pytest.mark.asyncio
async def test_execute_success() -> None:
    executor = ToolExecutor(FakeMCP())

    result = await executor.execute(
        tool_call_id="call-1",
        tool_name="working_tool",
        arguments={},
    )

    assert result.succeeded is True
    assert result.error is None
    assert result.result is not None
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_captures_mcp_tool_error() -> None:
    executor = ToolExecutor(FailingMCP())

    result = await executor.execute(
        tool_call_id="call-1",
        tool_name="broken_tool",
        arguments={},
    )

    assert result.succeeded is False
    assert result.error == ("MCP tool reported an execution error.")
    assert result.result is not None
    assert result.result.is_error is True


@pytest.mark.asyncio
async def test_execute_propagates_unexpected_exception() -> None:
    class BrokenMCP:
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict,
        ):
            raise RuntimeError("MCP infrastructure failed")

    executor = ToolExecutor(BrokenMCP())

    with pytest.raises(
        RuntimeError,
        match="MCP infrastructure failed",
    ):
        await executor.execute(
            tool_call_id="call-1",
            tool_name="broken_tool",
            arguments={},
        )


@pytest.mark.asyncio
async def test_parallel_execution_preserves_partial_results() -> None:
    executor = ToolExecutor(PartialMCP())

    tool_calls = [
        {
            "id": "call-1",
            "function": {
                "name": "working",
                "arguments": {},
            },
        },
        {
            "id": "call-2",
            "function": {
                "name": "broken",
                "arguments": {},
            },
        },
    ]

    results = await executor.execute_parallel(tool_calls)

    assert len(results) == 2

    assert results[0].succeeded is True
    assert results[0].error is None

    assert results[1].succeeded is False
    assert results[1].error == ("MCP tool reported an execution error.")


class PartialMCP:
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> CallToolResult:
        if tool_name == "broken":
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="failed",
                    )
                ],
                isError=True,
            )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="success",
                )
            ],
            isError=False,
        )
