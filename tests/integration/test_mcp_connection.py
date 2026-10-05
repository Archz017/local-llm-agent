import pytest

from mcp_client.client import MCPClient


@pytest.mark.asyncio
async def test_persistent_mcp_connection() -> None:
    async with MCPClient() as client:
        tools = await client.list_tools()

        names = {tool.name for tool in tools}

        assert "calculator_add" in names
        assert "calculator_multiply" in names
        assert "system_info" in names
        assert "wait" in names

        first = await client.call_tool(
            "calculator_add",
            {
                "a": 10,
                "b": 20,
            },
        )

        second = await client.call_tool(
            "calculator_multiply",
            {
                "a": 5,
                "b": 6,
            },
        )

        assert first is not None
        assert second is not None
