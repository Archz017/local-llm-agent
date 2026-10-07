from agent.runtime import AgentRuntime
from agent.sessions import SessionStore
from llm.client import LLMClient
from mcp_client.client import MCPClient
from mcp_client.errors import MCPNotConnectedError


class ApplicationServices:
    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.mcp_client = MCPClient()
        self.session_store = SessionStore()
        self.runtime: AgentRuntime | None = None

    async def start(self) -> None:
        await self.mcp_client.__aenter__()

        self.runtime = AgentRuntime(
            llm_client=self.llm_client,
            mcp_client=self.mcp_client,
        )

    async def stop(self) -> None:
        await self.mcp_client.__aexit__(
            None,
            None,
            None,
        )

        self.runtime = None

    async def readiness(self) -> dict[str, bool]:
        llm_ready = await self.llm_client.health()

        try:
            await self.mcp_client.list_tools()
            mcp_ready = True
        except MCPNotConnectedError:
            mcp_ready = False

        return {
            "llm": llm_ready,
            "mcp": mcp_ready,
        }

    def get_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            raise RuntimeError("Application services are not initialized.")

        return self.runtime
