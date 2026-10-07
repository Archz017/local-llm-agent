from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "agent_http_requests_total",
    "Total HTTP requests handled by the agent API.",
    ["method", "route", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "route"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
    ),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "agent_http_requests_in_progress",
    "Number of HTTP requests currently being processed.",
)

AGENT_EXECUTIONS_TOTAL = Counter(
    "agent_executions_total",
    "Total agent executions.",
    ["outcome"],
)

AGENT_EXECUTION_DURATION_SECONDS = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration in seconds.",
    buckets=(
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
        120.0,
    ),
)

LLM_REQUESTS_TOTAL = Counter(
    "agent_llm_requests_total",
    "Total LLM requests.",
    ["outcome"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "agent_llm_request_duration_seconds",
    "LLM request duration in seconds.",
    buckets=(
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
        120.0,
    ),
)

TOOL_EXECUTIONS_TOTAL = Counter(
    "agent_tool_executions_total",
    "Total tool executions.",
    ["tool_name", "outcome"],
)

TOOL_EXECUTION_DURATION_SECONDS = Histogram(
    "agent_tool_execution_duration_seconds",
    "Tool execution duration in seconds.",
    ["tool_name"],
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

AGENT_LLM_ROUNDS = Histogram(
    "agent_llm_rounds",
    "Number of LLM rounds required per successful agent execution.",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10),
)

AGENT_TOOL_CALLS = Histogram(
    "agent_tool_calls",
    "Number of tool calls made per successful agent execution.",
    buckets=(0, 1, 2, 3, 4, 5, 8, 10),
)

LLM_TOKENS_TOTAL = Counter(
    "agent_llm_tokens_total",
    "Total tokens processed by the LLM.",
    ["type"],
)
