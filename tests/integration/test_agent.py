import pytest

from agent.runtime import AgentRuntime
from llm.client import LLMClient
from mcp_client.client import MCPClient


@pytest.mark.asyncio
async def test_agent_calculator() -> None:
    runtime = AgentRuntime(
        llm_client=LLMClient(),
        mcp_client=MCPClient(),
    )

    response = await runtime.run(
        "Use the calculator to multiply 12 by 13."
    )

    assert "156" in response