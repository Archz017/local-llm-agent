from agent.runtime import AgentRuntime
from agent.sessions import SessionStore
from llm.client import LLMClient
from mcp_client.client import MCPClient


class ApplicationServices:
    def __init__(self) -> None:
        self.mcp_client = MCPClient()
        self.session_store = SessionStore()
        self.runtime: AgentRuntime | None = None

    async def start(self) -> None:
        await self.mcp_client.__aenter__()

        self.runtime = AgentRuntime(
            llm_client=LLMClient(),
            mcp_client=self.mcp_client,
        )

    async def stop(self) -> None:
        await self.mcp_client.__aexit__(
            None,
            None,
            None,
        )

        self.runtime = None

    def get_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            raise RuntimeError("Application services are not initialized.")

        return self.runtime
