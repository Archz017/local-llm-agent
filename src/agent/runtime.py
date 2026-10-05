import json
from typing import Any

from agent.tool_adapter import mcp_tools_to_openai
from agent.tool_parser import parse_tool_arguments
from llm.client import LLMClient
from mcp_client.client import MCPClient


class AgentRuntime:
    def __init__(
        self,
        llm_client: LLMClient,
        mcp_client: MCPClient,
        max_tool_rounds: int = 5,
    ) -> None:
        self.llm = llm_client
        self.mcp = mcp_client
        self.max_tool_rounds = max_tool_rounds

    async def run(self, user_message: str) -> str:
        mcp_tools = await self.mcp.list_tools()
        tools = mcp_tools_to_openai(mcp_tools)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a local AI assistant. "
                    "Use tools when they are useful. "
                    "Do not invent tool results."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        for _ in range(self.max_tool_rounds):
            response = await self.llm.chat(
                messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )

            message = response["choices"][0]["message"]

            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return message.get("content", "")

            messages.append(message)

            for tool_call in tool_calls:
                function = tool_call["function"]

                tool_name = function["name"]

                arguments = parse_tool_arguments(
                    function.get("arguments", {})
                )

                print(
                    f"\n[tool] {tool_name}"
                    f"({json.dumps(arguments)})"
                )

                result = await self.mcp.call_tool(
                    tool_name,
                    arguments,
                )

                tool_result = self._serialize_tool_result(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    }
                )

        raise RuntimeError(
            f"Maximum tool rounds ({self.max_tool_rounds}) exceeded."
        )

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        structured = getattr(
            result,
            "structured_content",
            None,
        )

        if structured is not None:
            return json.dumps(structured)

        content = getattr(result, "content", None)

        if content is not None:
            return str(content)

        return str(result)