"""Additional regression coverage for the async Result guides."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, assert_type, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

from better_result.core import Err, Ok, Panic, Result, err, ok
from better_result.result import (
    AsyncRetryConfig,
    TryAsyncContext,
    all as all_results,
    all_async,
    partition,
    partition_async,
    tap,
    tap_both,
    tap_both_async,
    tap_error,
    tap_error_async,
    try_async,
    unwrap_or,
)


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
    retry_backoff = cast(
        "Literal['constant', 'linear', 'exponential']", backoff
    )

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def operation(_context: TryAsyncContext) -> int:
        raise RuntimeError("temporary")

    result = await try_async(
        operation,
        str,
        AsyncRetryConfig[str](times=3, delay_ms=10, backoff=retry_backoff),
    )

    assert isinstance(result, Err)
    assert result.error == "temporary"
    assert delays == expected_delays


@pytest.mark.asyncio
async def test_retry_callbacks_receive_failed_attempt_context_and_control_retries() -> None:
    attempts: list[int] = []
    retry_attempts: list[int] = []
    delay_attempts: list[int] = []

    async def operation(context: TryAsyncContext) -> int:
        attempts.append(context.attempt)
        raise RuntimeError("temporary")

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
        raise RuntimeError("temporary")

    def broken_should_retry(_error: str, _context: TryAsyncContext) -> bool:
        raise ValueError("bad retry policy")

    with pytest.raises(Panic, match="should_retry predicate threw"):
        await try_async(
            operation,
            str,
            AsyncRetryConfig[str](times=1, should_retry=broken_should_retry),
        )

    def broken_delay(_error: str, _context: TryAsyncContext) -> float:
        raise ValueError("bad delay policy")

    with pytest.raises(Panic, match="delay_ms callback threw"):
        await try_async(
            operation,
            str,
            AsyncRetryConfig[str](times=1, delay_ms=broken_delay),
        )


def test_all_stops_at_the_first_error_and_partition_keeps_every_branch() -> None:
    def results() -> Iterator[Result[int, str]]:
        yield Ok[int, str](1)
        yield err("first")
        raise AssertionError("all() iterated after its first error")

    collected = all_results(results())
    assert isinstance(collected, Err)
    assert collected.error == "first"

    values, errors = partition([ok(1), err("first"), ok(2), err("second")])
    assert values == [1, 2]
    assert errors == ["first", "second"]


def test_observers_use_static_forms_and_preserve_the_original_result() -> None:
    success = ok(7)
    failure = err("missing")
    seen: list[object] = []

    def record_error(error: str) -> None:
        seen.append(error)

    assert tap(success, seen.append) is success
    assert tap_error(failure, record_error) is failure
    assert tap_error(record_error)(failure) is failure
    assert tap_both({"ok": seen.append, "err": record_error})(success) is success
    assert tap_both({"ok": seen.append, "err": record_error})(failure) is failure
    assert seen == [7, "missing", "missing", 7, "missing"]


def test_each_observer_defect_is_a_panic() -> None:
    with pytest.raises(Panic, match="tap_error callback threw"):
        err("missing").tap_error(lambda _error: (_ for _ in ()).throw(RuntimeError("log failed")))

    with pytest.raises(Panic, match="tap_both err callback threw"):
        err("missing").tap_both(
            {"ok": lambda _value: None, "err": lambda _error: (_ for _ in ()).throw(RuntimeError("log failed"))}
        )


@pytest.mark.asyncio
async def test_async_observers_use_data_last_forms_and_preserve_identity() -> None:
    success = ok(7)
    failure = err("missing")
    seen: list[object] = []

    async def observe_error(error: str) -> None:
        seen.append(error)

    async def observe_value(value: int) -> None:
        seen.append(value)

    observed_error = tap_error_async(observe_error)
    assert await observed_error(failure) is failure

    observed_both = tap_both_async(
        {"ok": observe_value, "err": observe_error}
    )
    assert await observed_both(success) is success
    assert seen == ["missing", 7]

    async def broken_error(_error: str) -> None:
        raise RuntimeError("async log failed")

    with pytest.raises(Panic, match="tap_both_async err callback threw"):
        await tap_both_async(
            {"ok": observe_value, "err": broken_error}
        )(failure)


@pytest.mark.asyncio
async def test_all_async_and_partition_async_preserve_order_and_panic_on_rejection() -> None:
    async def load(value: int) -> Result[int, str]:
        await asyncio.sleep(0)
        return Ok[int, str](value)

    collected = await all_async([load(1), Err[int, str]("bad"), load(3)])
    assert isinstance(collected, Err)
    assert collected.error == "bad"

    requests: list[Result[int, str] | Awaitable[Result[int, str]]] = [
        load(1),
        Err[int, str]("bad"),
        load(3),
    ]
    partitioned = await partition_async(requests)
    assert partitioned == ([1, 3], ["bad"])
    assert_type(partitioned, tuple[list[int], list[str]])

    async def rejected() -> Result[int, str]:
        raise RuntimeError("broken input")

    with pytest.raises(Panic, match="input awaitable rejected"):
        await partition_async([rejected()])


def test_unwrap_or_leaves_results_safely_and_supports_both_forms() -> None:
    failure = err("missing")
    success = ok(8080)

    assert unwrap_or(failure, 3000) == 3000
    assert unwrap_or(3000)(failure) == 3000
    assert unwrap_or(success, 3000) == 8080

    with pytest.raises(Panic) as raised:
        failure.unwrap("configuration must be valid")
    assert raised.value.message == "configuration must be valid"
    assert raised.value.cause == "missing"
