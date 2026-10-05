from typing import Any


def mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP tool definition into an OpenAI-compatible tool schema."""

    input_schema = getattr(tool, "inputSchema", None)

    if input_schema is None:
        input_schema = getattr(tool, "input_schema", None)

    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {},
        }

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": input_schema,
        },
    }


def mcp_tools_to_openai(
    tools: list[Any],
) -> list[dict[str, Any]]:
    return [mcp_tool_to_openai(tool) for tool in tools]
