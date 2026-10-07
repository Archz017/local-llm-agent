import pytest

from mcp_client.client import MCPClient
from mcp_client.errors import MCPNotConnectedError


@pytest.mark.asyncio
async def test_call_tool_requires_connection() -> None:
    client = MCPClient()

    with pytest.raises(
        MCPNotConnectedError,
        match="MCPClient is not connected",
    ):
        await client.call_tool(
            "system_info",
            {},
        )


@pytest.mark.asyncio
async def test_list_tools_requires_connection() -> None:
    client = MCPClient()

    with pytest.raises(
        MCPNotConnectedError,
        match="MCPClient is not connected",
    ):
        await client.list_tools()
