class AgentError(Exception):
    """Base exception for expected agent failures."""


class LLMError(AgentError):
    """Raised when communication with the LLM fails."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""


class ToolExecutionError(AgentError):
    """Raised when a tool cannot be executed successfully."""


class MaxToolRoundsError(AgentError):
    """Raised when the agent exceeds its tool-call limit."""
