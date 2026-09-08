"""Additional regression coverage for internal async extensions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, assert_type, cast

import pytest

from better_result import Err, Ok, PanicError, Result
from better_result._collections import (
    all_results,
    all_results_async,
    partition,
    partition_async,
)
from better_result._retry import AsyncRetryConfig, TryAsyncContext, try_async

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backoff", "expected_delays"),
    [
        ("constant", [0.01, 0.01, 0.01]),
        ("linear", [0.01, 0.02, 0.03]),
        ("exponential", [0.01, 0.02, 0.04]),
    ],
)
async def test_static_retry_backoff_policies_schedule_expected_delays(
    monkeypatch: pytest.MonkeyPatch,
    backoff: str,
    expected_delays: list[float],
) -> None:
    delays: list[float] = []
    retry_backoff = cast("Literal['constant', 'linear', 'exponential']", backoff)

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def operation(_context: TryAsyncContext) -> int:
        message = "temporary"
        raise RuntimeError(message)

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](times=3, delay_ms=10, backoff=retry_backoff),
    )

    assert isinstance(result, Err)
    assert result.error == "temporary"
    assert delays == expected_delays


@pytest.mark.asyncio
async def test_retry_callbacks_receive_context_and_control_retries() -> None:
    attempts: list[int] = []
    retry_attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        message = "temporary"
        raise RuntimeError(message)

    def should_retry(_error: str, context: TryAsyncContext) -> bool:
        retry_attempts.append(context.attempt)
        return context.attempt < 3

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](times=4, should_retry=should_retry, delay_ms=0),
    )

    assert isinstance(result, Err)
    assert attempts == [1, 2, 3]
    assert retry_attempts == [1, 2, 3]


@pytest.mark.asyncio
async def test_retry_policy_callback_failures_are_panics() -> None:
    async def operation(_context: TryAsyncContext) -> int:
        message = "temporary"
        raise RuntimeError(message)

    def broken_should_retry(_error: str, _context: TryAsyncContext) -> bool:
        message = "bad retry policy"
        raise ValueError(message)

    with pytest.raises(PanicError, match="should_retry predicate threw"):
        await try_async(
            operation,
            str,
            AsyncRetryConfig[str](times=1, should_retry=broken_should_retry),
        )


def test_all_stops_at_the_first_error_and_partition_keeps_every_branch() -> None:
    def results() -> Iterator[Result[int, str]]:
        yield Ok(1)
        yield Err("first")
        message = "all() iterated after its first error"
        raise AssertionError(message)

    assert all_results(results()) == Err("first")
    assert partition([Ok(1), Err("first"), Ok(2), Err("second")]) == (
        [1, 2],
        ["first", "second"],
    )


@pytest.mark.asyncio
async def test_all_async_and_partition_async_preserve_order_and_panic_on_rejection() -> (
    None
):
    async def load(value: int) -> Result[int, str]:
        await asyncio.sleep(0)
        return Ok(value)

    collected = await all_results_async([load(1), Err("bad"), load(3)])
    assert collected == Err("bad")

    requests: list[Result[int, str] | Awaitable[Result[int, str]]] = [
        load(1),
        Err("bad"),
        load(3),
    ]
    partitioned = await partition_async(requests)
    assert partitioned == ([1, 3], ["bad"])
    assert_type(partitioned, tuple[list[int], list[str]])

    async def rejected() -> Result[int, str]:
        message = "broken input"
        raise RuntimeError(message)

    with pytest.raises(PanicError, match="input awaitable rejected"):
        await partition_async([rejected()])
