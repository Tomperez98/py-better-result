"""Cancel an in-flight async operation with CancellationToken."""

from __future__ import annotations

import asyncio

from better_result import CancellationToken, TryContext, try_async


async def main() -> None:
    token = CancellationToken()
    operation_started = asyncio.Event()

    async def long_running_operation(_context: TryContext) -> str:
        operation_started.set()
        await asyncio.sleep(60)
        return "unreachable"

    task = asyncio.create_task(try_async(long_running_operation, cancel_token=token))
    await operation_started.wait()

    token.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("operation cancelled")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
