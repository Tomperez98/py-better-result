"""Tests for asynchronous Result-level utilities."""

from __future__ import annotations

import asyncio
from typing import Any, Never, cast

import pytest

from better_result.core import Err, Ok, Panic, Result, err, ok
from better_result.result import (
    AsyncRetryConfig,
    TryAsyncContext,
    all_async,
    and_then_async,
    partition_async,
    tap_async,
    tap_both_async,
    tap_error_async,
    try_async,
    try_recover_async,
)


@pytest.mark.asyncio
async def test_try_async_retries_with_context_and_custom_catch() -> None:
    attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        if context.attempt < 3:
            raise ValueError("not ready")
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
        raise ValueError("fatal")

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
        raise ValueError("operation failed")

    with pytest.raises(Panic, match="jitter"):
        await try_async(
            operation,
            retry=AsyncRetryConfig[object](times=1, jitter=2.0),
        )

    async def broken_catch(_cause: BaseException) -> str:
        raise RuntimeError("catch failed")

    with pytest.raises(Panic, match="catch handler threw"):
        await try_async(operation, broken_catch)


@pytest.mark.asyncio
async def test_async_combinators_support_data_first_and_data_last_forms() -> None:
    success = ok(2)
    failure = err("bad")

    chained = await and_then_async(success, lambda value: _completed(ok(str(value))))
    assert isinstance(chained, Ok)
    assert chained.value == "2"

    chain_later = and_then_async(lambda value: _completed(ok(str(value))))
    chained_later = await chain_later(success)
    assert isinstance(chained_later, Ok)
    assert chained_later.value == "2"

    recovered = await try_recover_async(
        failure,
        lambda value: _completed(ok(len(value))),
    )
    assert isinstance(recovered, Ok)
    assert recovered.value == 3

    seen: list[object] = []
    tapped = await tap_async(success, lambda value: _completed(seen.append(value)))
    assert tapped is success
    tapped_error = await tap_error_async(
        failure,
        lambda value: _completed(seen.append(value)),
    )
    assert tapped_error is failure
    both = await tap_both_async(
        failure,
        {"ok": _completed, "err": lambda value: _completed(seen.append(value))},
    )
    assert both is failure
    assert seen == [2, "bad", "bad"]


@pytest.mark.asyncio
async def test_all_async_and_partition_async_preserve_input_order() -> None:
    async def delayed(value: int) -> Result[int, Never]:
        await asyncio.sleep(0)
        return ok(value)

    collected = await all_async([delayed(1), ok(2), delayed(3)])
    assert isinstance(collected, Ok)
    assert collected.value == [1, 2, 3]

    partitioned = await partition_async([delayed(1), err("bad"), delayed(3)])
    assert partitioned == ([1, 3], ["bad"])

    async def rejected() -> Result[int, str]:
        raise RuntimeError("network failed")

    with pytest.raises(Panic, match="input awaitable rejected"):
        await all_async([rejected()])


@pytest.mark.asyncio
async def test_try_async_validates_retry_policy_and_preserves_panic() -> None:
    async def operation(_context: TryAsyncContext) -> int:
        raise ValueError("temporary")

    with pytest.raises(ValueError, match="unsupported retry backoff"):
        await try_async(
            operation,
            retry=AsyncRetryConfig[object](backoff=cast("Any", "invalid")),
        )

    with pytest.raises(Panic):
        await try_async(
            lambda _context: _raise_panic(),
            retry=AsyncRetryConfig[object](times=1),
        )


async def _raise_panic() -> int:
    raise Panic("bug")


@pytest.mark.asyncio
async def test_all_async_cancels_siblings_when_an_input_rejects() -> None:
    cancelled = asyncio.Event()

    async def rejected() -> Result[int, str]:
        await asyncio.sleep(0)
        raise RuntimeError("network failed")

    async def sibling() -> Result[int, str]:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return Ok[int, str](1)

    with pytest.raises(Panic, match="input awaitable rejected"):
        await all_async([rejected(), sibling()])

    assert cancelled.is_set()


async def _completed[T](value: T) -> T:
    await asyncio.sleep(0)
    return value
