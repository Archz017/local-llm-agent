import asyncio
import json

from agent.tool_adapter import mcp_tools_to_openai
from mcp_client.client import MCPClient


async def main() -> None:
    async with MCPClient() as client:
        tools = await client.list_tools()

    llm_tools = mcp_tools_to_openai(tools)

    print(
        json.dumps(
            llm_tools,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
