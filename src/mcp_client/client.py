from typing import Any

from mcp import Client, StdioServerParameters


class MCPClient:
    def __init__(self) -> None:
        self.server = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "python",
                "-m",
                "mcp_server.server",
            ],
        )

    async def list_tools(self) -> list[Any]:
        async with Client(self.server) as client:
            result = await client.list_tools()
            return result.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        async with Client(self.server) as client:
            return await client.call_tool(name, arguments)