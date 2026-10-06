# LLM Benchmark Environment

## Model

- Model: Qwen3 8B
- Quantization: Q4_K_M
- Format: GGUF

## Runtime

- Inference engine: llama.cpp
- Deployment: Docker
- API: OpenAI-compatible HTTP API

## llama.cpp Configuration

- Context size: 8192
- Tool-call Jinja templates: enabled

## Hardware

- Platform: macOS
- Architecture: Apple Silicon

### Q4_K_M — Docker Baseline

- Mean TTFT: 445.85 ms
- P50 TTFT: 422.71 ms
- P95 TTFT: 628.81 ms
- Mean latency: 22440.59 ms
- P50 latency: 21295.21 ms
- P95 latency: 36992.54 ms
- Mean generation throughput: 12.20 tok/s
- P50 generation throughput: 12.26 tok/s
- P5 generation throughput: 6.99 tok/s
- Completion tokens: 256

## Runtime Optimization Results

### Q4_K_M — Docker

- Mean TTFT: 445.85 ms
- P50 TTFT: 422.71 ms
- P95 TTFT: 628.81 ms
- Mean latency: 22440.59 ms
- P50 latency: 21295.21 ms
- P95 latency: 36992.54 ms
- Mean generation throughput: 12.20 tok/s
- P50 generation throughput: 12.26 tok/s
- P5 generation throughput: 6.99 tok/s

### Q4_K_M — Native llama.cpp

- Mean TTFT: 173.27 ms
- P50 TTFT: 170.76 ms
- P95 TTFT: 188.01 ms
- Mean latency: 7865.70 ms
- P50 latency: 7814.50 ms
- P95 latency: 7975.74 ms
- Mean generation throughput: 33.28 tok/s
- P50 generation throughput: 33.45 tok/s
- P5 generation throughput: 32.81 tok/s

### Improvement

Moving Qwen3-8B Q4_K_M from the Docker-based llama.cpp
deployment to native llama.cpp on Apple Silicon:

- Increased mean generation throughput from 12.20 to
  33.28 tok/s: 2.73x throughput.
- Reduced mean end-to-end latency from 22.44s to 7.87s:
  64.9% reduction.
- Reduced mean TTFT from 445.85ms to 173.27ms:
  61.1% reduction.
- Reduced P95 latency from 36.99s to 7.98s:
  78.4% reduction.

  ## Quantization Comparison — Docker

| Metric          | Q4_K_M    | Q5_K_M     |
|-----------------|----------:|-----------:|
| Mean TTFT       | 445.85 ms | 1015.42 ms |
| P50 TTFT        | 422.71 ms | 520.03 ms  |
| P95 TTFT        | 628.81 ms | 5213.87 ms |
| Mean latency    | 22.44 s   | 25.59 s    |
| Mean generation |
| throughput      |12.20 tok/s| 10.98 tok/s|
| Completion.     |
| tokens          | 256       | 256        |

Q4_K_M provided approximately 11.1% higher mean generation
throughput than Q5_K_M in the controlled Docker benchmark.

## Concurrent Inference — Native llama.cpp Baseline

| Concurrency | Requests/s | Aggregate tok/s | Mean Latency |
|---:|---:|---:|---:|
| 1 | 0.253 | 32.36 | 3.96s |
| 2 | 0.293 | 37.47 | 6.82s |
| 4 | 0.523 | 66.95 | 7.63s |
| 8 | 0.541 | 69.23 | 11.12s |

### Findings

Aggregate inference throughput increased from 32.36 tok/s
at concurrency 1 to 66.95 tok/s at concurrency 4, representing
a 2.07x throughput improvement.

Increasing concurrency from 4 to 8 produced only approximately
3.4% additional aggregate throughput while increasing mean
request latency by approximately 45.7%.

For this workload and hardware configuration, concurrency 4
provided the strongest balance between aggregate throughput
and request latency.

## Concurrent Inference Load Test

The final load test executed 20 requests at each concurrency
level using native llama.cpp with four server slots.

| Concurrency | Requests/s | Aggregate tok/s | Mean Latency | P50 Latency | P95 Latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.246 | 31.44 | 4.07s | 4.02s | 4.19s |
| 2 | 0.298 | 38.09 | 6.72s | 6.72s | 6.79s |
| 4 | 0.533 | 68.24 | 7.50s | 7.43s | 7.75s |
| 8 | 0.542 | 69.43 | 13.27s | 14.67s | 14.90s |

### Findings

Increasing concurrency from 1 to 4 increased aggregate
generation throughput from 31.44 tok/s to 68.24 tok/s,
representing a 2.17x improvement.

Increasing concurrency from 4 to 8 produced only approximately
1.7% additional aggregate throughput while increasing mean
latency by approximately 76.9% and P95 latency by approximately
92.3%.

For this workload and hardware configuration, four concurrent
requests provided the strongest balance between serving
throughput and latency.