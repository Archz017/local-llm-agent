class MCPError(Exception):
    """Base exception for MCP infrastructure failures."""


class MCPNotConnectedError(MCPError):
    """Raised when the MCP client is not connected."""
