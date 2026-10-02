import asyncio

from llm.client import LLMClient


async def main() -> None:
    client = LLMClient()

    response = await client.chat(
        [
            {
                "role": "user",
                "content": "What is the capital of Australia?",
            }
        ]
    )

    answer = response["choices"][0]["message"]["content"]

    print("\nLLM:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())