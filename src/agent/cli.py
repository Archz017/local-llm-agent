import asyncio

from agent.runtime import AgentRuntime
from llm.client import LLMClient
from mcp_client.client import MCPClient


async def main() -> None:
    async with MCPClient() as mcp_client:
        runtime = AgentRuntime(
            llm_client=LLMClient(),
            mcp_client=mcp_client,
        )

        print("Local LLM Agent")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in {
                "exit",
                "quit",
            }:
                break

            if not user_input:
                continue

            answer = await runtime.run(user_input)

            print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
