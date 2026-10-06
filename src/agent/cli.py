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

        messages: list[dict[str, object]] = []

        while True:
            user_input = input("You: ")

            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            try:
                answer = await runtime.run(messages)
            except Exception:
                messages.pop()
                raise

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            print(f"Assistant: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
