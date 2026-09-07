"""Additional regression coverage for the async Result guides."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, assert_type, cast

import pytest

from better_result.collections import (
    all_results,
    all_results_async,
    partition,
    partition_async,
)
from better_result.retry import AsyncRetryConfig, TryAsyncContext, try_async

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

from better_result.core import Err, Ok, PanicError, Result


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
        msg = "temporary"
        raise RuntimeError(msg)

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](times=3, delay_ms=10, backoff=retry_backoff),
    )

    assert isinstance(result, Err)
    assert result.error == "temporary"
    assert delays == expected_delays


@pytest.mark.asyncio
async def test_retry_callbacks_receive_failed_attempt_context_and_control_retries() -> (
    None
):
    attempts: list[int] = []
    retry_attempts: list[int] = []
    delay_attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        msg = "temporary"
        raise RuntimeError(msg)

    def should_retry(_error: str, context: TryAsyncContext) -> bool:
        retry_attempts.append(context.attempt)
        return context.attempt < 3

    def delay(_error: str, context: TryAsyncContext) -> float:
        delay_attempts.append(context.attempt)
        return 0

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](
            times=4,
            should_retry=should_retry,
            delay_ms=delay,
        ),
    )

    assert isinstance(result, Err)
    assert result.error == "temporary"
    assert attempts == [1, 2, 3]
    assert retry_attempts == [1, 2, 3]
    assert delay_attempts == [1, 2]


@pytest.mark.asyncio
async def test_retry_policy_callback_failures_are_panics() -> None:
    async def operation(_context: TryAsyncContext) -> int:
        msg = "temporary"
        raise RuntimeError(msg)

    def broken_should_retry(_error: str, _context: TryAsyncContext) -> bool:
        msg = "bad retry policy"
        raise ValueError(msg)

    with pytest.raises(PanicError, match="should_retry predicate threw"):
        await try_async(
            operation,
            str,
            AsyncRetryConfig[str](times=1, should_retry=broken_should_retry),
        )

    def broken_delay(_error: str, _context: TryAsyncContext) -> float:
        msg = "bad delay policy"
        raise ValueError(msg)

    with pytest.raises(PanicError, match="delay_ms callback threw"):
        await try_async(
            operation,
            str,
            AsyncRetryConfig[str](times=1, delay_ms=broken_delay),
        )


def test_all_stops_at_the_first_error_and_partition_keeps_every_branch() -> None:
    def results() -> Iterator[Result[int, str]]:
        yield Ok(1)
        yield Err("first")
        msg = "all() iterated after its first error"
        raise AssertionError(msg)

    collected = all_results(results())
    assert isinstance(collected, Err)
    assert collected.error == "first"

    values, errors = partition([Ok(1), Err("first"), Ok(2), Err("second")])
    assert values == [1, 2]
    assert errors == ["first", "second"]


def test_observers_use_static_forms_and_preserve_the_original_result() -> None:
    success = Ok(7)
    failure = Err("missing")
    seen: list[object] = []

    def record_error(error: str) -> None:
        seen.append(error)

    assert success.tap(seen.append) is success
    assert failure.tap_error(record_error) is failure
    assert failure.tap_error(record_error) is failure
    assert seen == [7, "missing", "missing"]


def test_each_observer_defect_is_a_panic() -> None:
    with pytest.raises(PanicError, match="tap_error callback threw"):
        Err("missing").tap_error(
            lambda _error: (_ for _ in ()).throw(RuntimeError("log failed")),
        )


@pytest.mark.asyncio
async def test_async_observers_use_instance_methods_and_preserve_identity() -> None:
    success = Ok(7)
    failure = Err("missing")
    seen: list[object] = []

    async def observe_error(error: str) -> None:
        seen.append(error)

    async def observe_value(value: int) -> None:
        seen.append(value)

    assert await failure.tap_error_async(observe_error) is failure

    assert await success.tap_async(observe_value) is success
    assert seen == ["missing", 7]


@pytest.mark.asyncio
async def test_all_async_and_partition_async_preserve_order_and_panic_on_rejection() -> (
    None
):
    async def load(value: int) -> Result[int, str]:
        await asyncio.sleep(0)
        return Ok(value)

    collected = await all_results_async([load(1), Err("bad"), load(3)])
    assert isinstance(collected, Err)
    assert collected.error == "bad"

    requests: list[Result[int, str] | Awaitable[Result[int, str]]] = [
        load(1),
        Err("bad"),
        load(3),
    ]
    partitioned = await partition_async(requests)
    assert partitioned == ([1, 3], ["bad"])
    assert_type(partitioned, tuple[list[int], list[str]])

    async def rejected() -> Result[int, str]:
        msg = "broken input"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="input awaitable rejected"):
        await partition_async([rejected()])


def test_unwrap_or_leaves_results_safely() -> None:
    failure = Err("missing")
    success = Ok(8080)

    assert failure.unwrap_or(3000) == 3000
    assert success.unwrap_or(3000) == 8080

    with pytest.raises(PanicError) as raised:
        failure.unwrap("configuration must be valid")
    assert raised.value.message == "configuration must be valid"
    assert raised.value.cause == "missing"
