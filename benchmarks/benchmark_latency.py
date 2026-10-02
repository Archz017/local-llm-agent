import asyncio
import time

from llm.client import LLMClient


async def benchmark() -> None:
    client = LLMClient()

    prompt = [
        {
            "role": "user",
            "content": (
                "Explain the difference between concurrency "
                "and parallelism in 150 words."
            ),
        }
    ]

    start = time.perf_counter()

    response = await client.chat(
        prompt,
        max_tokens=200,
    )

    elapsed = time.perf_counter() - start

    usage = response.get("usage", {})

    print(f"Latency: {elapsed:.3f}s")
    print(f"Usage: {usage}")


if __name__ == "__main__":
    asyncio.run(benchmark())