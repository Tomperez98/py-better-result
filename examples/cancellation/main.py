"""Cancel in-flight async work and an interrupted retry wait."""

from __future__ import annotations

import asyncio

from better_result import CancellationToken, RetryPolicy, TryContext, try_async


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
        print("operation cancelled")

    retry_token = CancellationToken()

    async def transient_failure(_: TryContext) -> str:
        message = "temporary failure"
        raise TimeoutError(message)

    async def cancel_retry_wait() -> None:
        await asyncio.sleep(0.01)
        retry_token.cancel()

    cancellation_task = asyncio.create_task(cancel_retry_wait())
    try:
        await try_async(
            transient_failure,
            retry=RetryPolicy.constant(times=1, delay=60),
            cancel_token=retry_token,
        )
    except asyncio.CancelledError:
        print("retry wait cancelled")
    finally:
        await cancellation_task


if __name__ == "__main__":
    asyncio.run(main())
