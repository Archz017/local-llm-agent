import json
from typing import Any


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize tool arguments returned by different llama.cpp versions."""

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        parsed = json.loads(arguments)

        if not isinstance(parsed, dict):
            raise TypeError("Tool arguments must decode to a JSON object.")

        return parsed

    raise TypeError(f"Unsupported tool argument type: {type(arguments).__name__}")
