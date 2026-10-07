import json
import logging
import time
from typing import Any

from agent.errors import MaxToolRoundsError
from agent.tool_adapter import mcp_tools_to_openai
from agent.tool_executor import ToolExecutor
from llm.client import LLMClient
from mcp_client.client import MCPClient
from observability.metrics import (
    AGENT_EXECUTION_DURATION_SECONDS,
    AGENT_EXECUTIONS_TOTAL,
    AGENT_LLM_ROUNDS,
    AGENT_TOOL_CALLS,
)

logger = logging.getLogger(__name__)

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

    async def run(self, messages: list[dict[str, object]]) -> str:
        start = time.perf_counter()

        logger.info(
            "Agent execution started",
            extra={"event": "agent_started"},
        )

        try:
            answer = await self._run_agent_loop(messages)

        except Exception as exc:
            duration_seconds = time.perf_counter() - start
            duration_ms = duration_seconds * 1000

            AGENT_EXECUTIONS_TOTAL.labels(
                outcome="error",
            ).inc()

            AGENT_EXECUTION_DURATION_SECONDS.observe(duration_seconds)

            logger.exception(
                "Agent execution failed",
                extra={
                    "event": "agent_failed",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                },
            )
            raise

        duration_seconds = time.perf_counter() - start
        duration_ms = duration_seconds * 1000

        AGENT_EXECUTIONS_TOTAL.labels(
            outcome="success",
        ).inc()

        AGENT_EXECUTION_DURATION_SECONDS.observe(duration_seconds)

        logger.info(
            "Agent execution completed",
            extra={
                "event": "agent_completed",
                "duration_ms": round(duration_ms, 2),
            },
        )

        return answer

    async def _run_agent_loop(
        self,
        messages: list[dict[str, object]],
    ) -> str:
        tool_call_count = 0
        llm_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ]

        mcp_tools = await self.mcp.list_tools()
        tools = mcp_tools_to_openai(mcp_tools)

        for llm_round in range(
            1,
            self.max_tool_rounds + 1,
        ):
            response = await self.llm.chat(
                llm_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                llm_round=llm_round,
            )

            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return message.get("content") or ""

            llm_messages.append(message)
            tool_call_count += len(tool_calls)

            if self.parallel_tools:
                execution_results = await self.executor.execute_parallel(tool_calls)
            else:
                execution_results = await self.executor.execute_sequential(tool_calls)

            for execution in execution_results:
                if execution.succeeded:
                    logger.info(
                        "Tool execution completed",
                        extra={
                            "event": "tool_completed",
                            "tool_name": execution.tool_name,
                            "tool_call_id": execution.tool_call_id,
                            "duration_ms": round(
                                execution.duration_ms,
                                2,
                            ),
                        },
                    )

                    tool_result = self._serialize_tool_result(execution.result)

                else:
                    logger.warning(
                        "Tool execution failed",
                        extra={
                            "event": "tool_failed",
                            "tool_name": execution.tool_name,
                            "tool_call_id": execution.tool_call_id,
                            "duration_ms": round(
                                execution.duration_ms,
                                2,
                            ),
                        },
                    )

                    tool_result = f"Tool execution failed. Error: {execution.error}"

                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": execution.tool_call_id,
                        "content": tool_result,
                    }
                )

                AGENT_LLM_ROUNDS.observe(llm_round)
                AGENT_TOOL_CALLS.observe(tool_call_count)

        raise MaxToolRoundsError(
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
