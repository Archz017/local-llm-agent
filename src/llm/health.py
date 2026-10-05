import httpx


async def check_llm() -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8080/health",
            timeout=5,
        )

    return response.status_code == 200
