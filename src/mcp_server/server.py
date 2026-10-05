from mcp.server import MCPServer

from mcp_server.tools.calculator import add, multiply
from mcp_server.tools.system_info import get_system_info

mcp = MCPServer(
    name="local-llm-tools",
    instructions=(
        "Local tools available to the Local LLM Agent. "
        "Use these tools for calculations and system information."
    ),
)


@mcp.tool()
def calculator_add(a: float, b: float) -> float:
    """Add two numbers."""
    return add(a, b)


@mcp.tool()
def calculator_multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return multiply(a, b)


@mcp.tool()
def system_info() -> dict[str, str]:
    """Return information about the machine running this MCP server."""
    return get_system_info()

if __name__ == "__main__":
    mcp.run()