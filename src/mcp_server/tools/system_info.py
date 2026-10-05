import platform


def get_system_info() -> dict[str, str]:
    """Return basic information about the machine running the MCP server."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
