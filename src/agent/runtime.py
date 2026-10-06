import json
from typing import Any

from agent.tool_adapter import mcp_tools_to_openai
from agent.tool_executor import ToolExecutor
from llm.client import LLMClient
from mcp_client.client import MCPClient

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to tools.

Answer general questions and conversational requests normally.
Use the conversation history to understand follow-up questions.

Do not claim that the user previously provided information unless
that information appears in the current conversation history. If a
question depends on missing conversation context, say that the
information was not provided.

Use tools when they are useful or necessary to answer the user's
request. Otherwise, answer directly.
""".strip()


class AgentRuntime:
    def __init__(
        self,
        llm_client: LLMClient,
        mcp_client: MCPClient,
        max_tool_rounds: int = 5,
        parallel_tools: bool = True,
    ) -> None:
        self.llm = llm_client
        self.mcp = mcp_client
        self.executor = ToolExecutor(mcp_client)
        self.max_tool_rounds = max_tool_rounds
        self.parallel_tools = parallel_tools

    async def run(
        self,
        messages: list[dict[str, object]],
    ) -> str:
        llm_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ]

        mcp_tools = await self.mcp.list_tools()
        tools = mcp_tools_to_openai(mcp_tools)

        for _ in range(self.max_tool_rounds):
            response = await self.llm.chat(
                llm_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )

            message = response["choices"][0]["message"]

            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return message.get("content", "")

            llm_messages.append(message)

            if self.parallel_tools:
                execution_results = await self.executor.execute_parallel(tool_calls)
            else:
                execution_results = await self.executor.execute_sequential(tool_calls)

            for execution in execution_results:
                print(
                    f"[tool] {execution.tool_name} "
                    f"completed in "
                    f"{execution.duration_ms:.2f} ms"
                )

                tool_result = self._serialize_tool_result(execution.result)

                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": execution.tool_call_id,
                        "content": tool_result,
                    }
                )

        raise RuntimeError(f"Maximum tool rounds ({self.max_tool_rounds}) exceeded.")

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
