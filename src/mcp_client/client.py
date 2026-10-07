from types import TracebackType
from typing import Any, Self

from mcp import Client, MCPError, StdioServerParameters

from mcp_client.errors import MCPNotConnectedError


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

        self._client: Client | None = None

    async def __aenter__(self) -> Self:
        self._client = Client(self.server)

        await self._client.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )

            self._client = None

    def _require_client(self) -> Client:
        if self._client is None:
            raise MCPNotConnectedError(
                "MCPClient is not connected. Use 'async with MCPClient() as client:'."
            )

        return self._client

    async def list_tools(self) -> list[Any]:
        client = self._require_client()

        result = await client.list_tools()

        return result.tools

    async def health(self) -> bool:
        try:
            await self.list_tools()
        except MCPError:
            return False

        return True

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        client = self._require_client()

        return await client.call_tool(
            name,
            arguments,
        )
