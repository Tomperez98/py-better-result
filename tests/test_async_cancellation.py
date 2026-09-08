"""Examples of cancelling in-flight asynchronous Result operations."""

from __future__ import annotations

import asyncio

import pytest

from better_result import Ok, Result
from better_result._collections import all_results_async
from better_result._retry import TryAsyncContext, try_async


@pytest.mark.asyncio
async def test_cancelling_an_async_result_operation_reaches_the_operation() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    never = asyncio.Event()

    async def load(_: int) -> Result[int, str]:
        started.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            cleaned_up.set()
            raise
        return Ok(42)

    task = asyncio.create_task(Ok(1).and_then_async(load))
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_cancelling_a_retry_operation_does_not_convert_cancellation_to_err() -> (
    None
):
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    never = asyncio.Event()
    attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        started.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            cleaned_up.set()
            raise
        return 42

    task = asyncio.create_task(try_async(operation))
    await started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned_up.is_set()
    assert attempts == [1]


@pytest.mark.asyncio
async def test_cancelling_all_results_async_cancels_in_flight_siblings() -> None:
    started = [asyncio.Event(), asyncio.Event()]
    cleaned_up: list[int] = []
    never = asyncio.Event()

    async def load(index: int) -> Result[int, str]:
        started[index].set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            cleaned_up.append(index)
            raise
        return Ok(index)

    task = asyncio.create_task(
        all_results_async([load(0), load(1)]),
    )
    await asyncio.gather(*(event.wait() for event in started))

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(cleaned_up) == [0, 1]
