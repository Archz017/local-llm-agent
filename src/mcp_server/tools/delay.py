import asyncio


async def delay(
    seconds: float,
    label: str,
) -> dict[str, str | float]:
    """Wait for a specified duration and return a label."""

    if seconds < 0:
        raise ValueError("Delay cannot be negative.")

    if seconds > 10:
        raise ValueError("Delay cannot exceed 10 seconds.")

    await asyncio.sleep(seconds)

    return {
        "label": label,
        "delay_seconds": seconds,
    }
