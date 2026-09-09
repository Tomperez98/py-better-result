"""Retry async Result-producing operations and cancel asyncio tasks."""

from __future__ import annotations

import asyncio
from enum import Enum, auto

from better_result import Err, Ok, Result, RetryPolicy, retry_async


class RequestError(Enum):
    TEMPORARY = auto()


async def main() -> None:
    one_shot_policy = RetryPolicy.immediate(0)
    one_shot = await retry_async(
        _one_shot,
        one_shot_policy,
    )
    assert one_shot == Ok("one-shot value")
    print(f"one-shot: {one_shot}")

    policy = RetryPolicy.fixed(2, 0).with_predicate(
        lambda context: context.error is RequestError.TEMPORARY,
    )
    attempts = 0

    async def operation() -> Result[str, RequestError]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return Err(RequestError.TEMPORARY)
        return Ok("async response")

    result = await retry_async(operation, policy)
    assert result == Ok("async response")
    assert attempts == 3
    print(f"retried async operation: {result}, attempts={attempts}")

    operation_task = asyncio.create_task(_pending_operation())
    await asyncio.sleep(0)
    operation_task.cancel()
    try:
        await operation_task
    except asyncio.CancelledError:
        print("cancelled an in-flight operation")

    slow_policy = RetryPolicy.fixed(1, 60)
    retry_task = asyncio.create_task(
        retry_async(_temporary_failure, slow_policy),
    )
    await asyncio.sleep(0)
    retry_task.cancel()
    try:
        await retry_task
    except asyncio.CancelledError:
        print("cancelled a retry delay")


async def _pending_operation() -> None:
    await asyncio.Future()


async def _one_shot() -> Ok[str]:
    return Ok("one-shot value")


async def _temporary_failure() -> Err[str]:
    return Err("temporary failure")


if __name__ == "__main__":
    asyncio.run(main())
