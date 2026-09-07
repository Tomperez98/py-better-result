"""Tests for asynchronous Result-level utilities."""

from __future__ import annotations

import asyncio
from typing import Never

import pytest

from better_result.collections import all_results_async, partition_async
from better_result.core import Err, Ok, PanicError, Result
from better_result.retry import AsyncRetryConfig, TryAsyncContext, try_async


@pytest.mark.asyncio
async def test_try_async_retries_with_context_and_custom_catch() -> None:
    attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        if context.attempt < 3:
            msg = "not ready"
            raise ValueError(msg)
        return 42

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](times=2),
    )

    assert isinstance(result, Ok)
    assert result.value == 42
    assert attempts == [1, 2, 3]


@pytest.mark.asyncio
async def test_try_async_stops_when_retry_predicate_rejects() -> None:
    attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        msg = "fatal"
        raise ValueError(msg)

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](
            times=3,
            should_retry=lambda error, _context: error != "fatal",
        ),
    )

    assert isinstance(result, Err)
    assert result.error == "fatal"
    assert attempts == [1]


@pytest.mark.asyncio
async def test_try_async_panics_on_invalid_jitter_and_catch_failure() -> None:
    async def operation(_context: TryAsyncContext) -> int:
        msg = "operation failed"
        raise ValueError(msg)

    with pytest.raises(PanicError, match="jitter"):
        await try_async(
            operation,
            retry=AsyncRetryConfig[object](times=1, jitter=2.0),
        )

    async def broken_catch(_cause: BaseException) -> str:
        msg = "catch failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="catch handler threw"):
        await try_async(operation, broken_catch)


@pytest.mark.asyncio
async def test_async_result_operations_use_instance_methods() -> None:
    success = Ok(2)
    failure = Err("bad")

    chained = await success.and_then_async(
        lambda value: _completed(Ok(str(value))),
    )
    assert isinstance(chained, Ok)
    assert chained.value == "2"

    recovered = await failure.try_recover_async(
        lambda value: _completed(Ok(len(value))),
    )
    assert isinstance(recovered, Ok)
    assert recovered.value == 3

    seen: list[object] = []
    tapped = await success.tap_async(lambda value: _completed(seen.append(value)))
    assert tapped is success
    tapped_error = await failure.tap_error_async(
        lambda value: _completed(seen.append(value)),
    )
    assert tapped_error is failure
    assert seen == [2, "bad"]


@pytest.mark.asyncio
async def test_all_async_and_partition_async_preserve_input_order() -> None:
    async def delayed(value: int) -> Result[int, Never]:
        await asyncio.sleep(0)
        return Ok(value)

    collected = await all_results_async([delayed(1), Ok(2), delayed(3)])
    assert isinstance(collected, Ok)
    assert collected.value == [1, 2, 3]

    partitioned = await partition_async([delayed(1), Err("bad"), delayed(3)])
    assert partitioned == ([1, 3], ["bad"])

    async def rejected() -> Result[int, str]:
        msg = "network failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="input awaitable rejected"):
        await all_results_async([rejected()])


@pytest.mark.asyncio
async def test_try_async_validates_retry_policy_and_preserves_panic() -> None:
    async def operation(_context: TryAsyncContext) -> int:
        msg = "temporary"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="unsupported retry backoff"):
        await try_async(
            operation,
            retry=AsyncRetryConfig[object](
                backoff="invalid",  # ty: ignore[invalid-argument-type]
            ),
        )

    with pytest.raises(PanicError):
        await try_async(
            lambda _context: _raise_panic(),
            retry=AsyncRetryConfig[object](times=1),
        )


async def _raise_panic() -> int:
    msg = "bug"
    raise PanicError(msg)


@pytest.mark.asyncio
async def test_all_async_cancels_siblings_when_an_input_rejects() -> None:
    cancelled = asyncio.Event()

    async def rejected() -> Result[int, str]:
        await asyncio.sleep(0)
        msg = "network failed"
        raise RuntimeError(msg)

    async def sibling() -> Result[int, str]:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return Ok(1)

    with pytest.raises(PanicError, match="input awaitable rejected"):
        await all_results_async([rejected(), sibling()])

    assert cancelled.is_set()


async def _completed[T](value: T) -> T:
    await asyncio.sleep(0)
    return value
