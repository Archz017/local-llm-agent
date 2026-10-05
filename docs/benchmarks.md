# Performance Benchmarks

## v0.4.0 — Concurrent Tool Execution

Configuration:

- Model: Qwen3 8B Q4_K_M
- Inference runtime: llama.cpp
- MCP transport: stdio
- Tool calls: 3
- Artificial delay per tool: 1 second
- Maximum concurrency: 4

### Tool Execution

| Strategy | Mean Latency |
|---|---:|
| Sequential | 3.03s |
| Parallel | 1.01s |

Speedup: 3.01x

Latency reduction: 66.7%

## MCP Connection Overhead

| Strategy | Mean Latency |
|---|---:|
| Reconnect per call | 3.99 ms |
| Persistent connection | 384.46 ms |

Speedup: 96.25x