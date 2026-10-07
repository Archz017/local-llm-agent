import pytest

from agent.runtime import AgentRuntime
from llm.client import LLMClient
from mcp_client.client import MCPClient


@pytest.mark.asyncio
async def test_agent_calculator() -> None:
    async with MCPClient() as mcp_client:
        runtime = AgentRuntime(
            llm_client=LLMClient(),
            mcp_client=mcp_client,
        )

        messages = [
            {
                "role": "user",
                "content": ("Use the calculator tool to multiply 12 by 13."),
            }
        ]

        response = await runtime.run(messages)

        assert "156" in response
