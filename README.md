# Local LLM Agent

An open-source local LLM agent runtime exploring:

- Local LLM inference
- MCP client/server architecture
- Tool calling
- Asynchronous tool execution
- Parallel execution
- LLM inference optimization
- Quantization
- Continuous batching
- Speculative decoding
- Performance benchmarking

## Architecture

                         ┌─────────────────────────┐
                         │       User / CLI         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Agent Runtime       │
                         │        Python           │
                         │                         │
                         │ • conversation manager  │
                         │ • tool router           │
                         │ • MCP client            │
                         │ • async executor        │
                         │ • response synthesizer  │
                         └───────┬─────────┬───────┘
                                 │         │
                    MCP          │         │ OpenAI-compatible API
                                 │         │
              ┌──────────────────┘         └──────────────┐
              ▼                                           ▼
   ┌─────────────────────┐                    ┌──────────────────────┐
   │     MCP Server      │                    │    llama.cpp         │
   │       Python        │                    │    LLM Server        │
   │                     │                    │                      │
   │ • calculator        │                    │ Qwen 3.x / similar   │
   │ • filesystem        │                    │ GGUF quantization    │
   │ • web/search*       │                    │                      │
   │ • system info       │                    │ GPU / CPU inference  │
   │ • database          │                    │ KV cache              │
   └─────────────────────┘                    │ batching              │
                                              │ speculative decoding  │
                                              └──────────────────────┘

                         ┌─────────────────────────┐
                         │     Observability       │
                         │                         │
                         │ OpenTelemetry           │
                         │ Prometheus              │
                         │ Grafana                 │
                         └─────────────────────────┘

## Development

### Requirements

- Python 3.12
- uv
- Docker

### Setup

```bash
uv sync
