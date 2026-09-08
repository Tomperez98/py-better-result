"""Cancel CPU-heavy async work at explicit chunk boundaries."""

from __future__ import annotations

import asyncio

from better_result import CancellationToken, TryContext, try_async


async def compute_checksum(context: TryContext) -> int:
    """Do bounded CPU work, yielding so task cancellation can be delivered."""
    checksum = 0
    for start in range(0, 100_000_000, 10_000):
        checksum += sum(
            (number * number) % 97 for number in range(start, start + 10_000)
        )

        # Automatic cancellation handles the await below. The explicit check
        # also stops promptly between CPU chunks if the token was cancelled.
        if context.cancel_token is not None:
            context.cancel_token.raise_if_cancelled()
        await asyncio.sleep(0)

    return checksum


async def main() -> None:
    token = CancellationToken()
    task = asyncio.create_task(try_async(compute_checksum, cancel_token=token))

    await asyncio.sleep(0.01)
    token.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("CPU work cancelled at a chunk boundary")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
