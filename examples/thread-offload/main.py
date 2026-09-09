"""Offload synchronous CPU work to a thread in an async Result workflow."""

from __future__ import annotations

import asyncio
import threading

from better_result import Ok


def checksum(
    limit: int,
    loop: asyncio.AbstractEventLoop,
    started: asyncio.Event,
    release: threading.Event,
) -> int:
    """Perform synchronous CPU work that would otherwise block the event loop."""
    loop.call_soon_threadsafe(started.set)
    if not release.wait(timeout=1):
        message = "event loop did not remain responsive"
        raise RuntimeError(message)
    return sum((number * number) % 97 for number in range(limit))


async def offloaded_checksum(
    limit: int,
    loop: asyncio.AbstractEventLoop,
    started: asyncio.Event,
    release: threading.Event,
) -> int:
    """Adapt the synchronous operation to the awaitable callback API."""
    return await asyncio.to_thread(checksum, limit, loop, started, release)


async def main() -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    async def callback(limit: int) -> int:
        return await offloaded_checksum(limit, loop, started, release)

    result_task = asyncio.create_task(Ok(100_000).map_async(callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    # The worker is waiting for the event loop to release it. If the CPU
    # function were running on the event-loop thread, this assertion would
    # never be reached before the worker timed out.
    assert not result_task.done()
    release.set()

    result = await result_task
    assert result == Ok(4_800_196)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
