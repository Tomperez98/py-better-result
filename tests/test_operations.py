"""Behavioral tests for Result operations outside the core variants."""

from __future__ import annotations

import asyncio
from typing import Never

import pytest

from better_result import (
    CancellationToken,
    Err,
    Ok,
    RetryPolicy,
    TryContext,
    all_results,
    all_results_async,
    flatten_result,
    partition_results,
    partition_results_async,
    try_async,
    try_result,
)


def test_all_results_collects_successes_and_short_circuits_first_error() -> None:
    assert all_results([Ok(1), Ok(2)]) == Ok((1, 2))
    assert all_results([Ok(1), Err("first"), Err("second")]) == Err("first")
    assert all_results([]) == Ok(())


def test_partition_results_preserves_relative_order() -> None:
    assert partition_results([Ok(1), Err("a"), Ok(2), Err("b")]) == (
        [1, 2],
        ["a", "b"],
    )
    assert partition_results([]) == ([], [])


def test_flatten_result_propagates_inner_or_outer_error() -> None:
    assert flatten_result(Ok(Ok(42))) == Ok(42)
    assert flatten_result(Ok(Err("inner"))) == Err("inner")
    assert flatten_result(Err("outer")) == Err("outer")


def test_try_result_captures_expected_exceptions_without_forcing_error_types() -> None:
    assert try_result(lambda _: 42) == Ok(42)

    captured = try_result(_raise_zero)
    assert isinstance(captured, Err)
    assert isinstance(captured.err_value, ZeroDivisionError)
    assert try_result(_raise_zero, catch=str) == Err("division by zero")


def test_try_result_exposes_attempt_context_and_retries_immediately() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 2:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    assert try_result(operation, retry=1) == Ok("ok")
    assert attempts == [1, 2]


def test_try_result_propagates_mapper_bugs() -> None:
    def broken_mapper(_: Exception) -> str:
        message = "mapper broken"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="mapper broken"):
        try_result(_raise_zero, catch=broken_mapper)


def _raise_zero(_: TryContext) -> Never:
    message = "division by zero"
    raise ZeroDivisionError(message)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_try_async_forwards_cancellation_token_and_stops_retry_delay() -> None:
    token = CancellationToken()
    assert not token.is_cancelled
    token.raise_if_cancelled()
    contexts: list[CancellationToken | None] = []
    cancel_task: asyncio.Task[None] | None = None

    async def cancel_later() -> None:
        await asyncio.sleep(0)
        token.cancel()

    async def operation(context: TryContext) -> str:
        nonlocal cancel_task
        contexts.append(context.cancel_token)
        cancel_task = asyncio.create_task(cancel_later())
        message = "temporary"
        raise RuntimeError(message)

    result = await try_async(
        operation,
        catch=str,
        cancel_token=token,
        retry=RetryPolicy(times=3, delay=10),
    )

    assert result == Err("temporary")
    assert contexts == [token]
    assert cancel_task is not None
    await cancel_task
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()
    assert token.is_cancelled


@pytest.mark.asyncio
async def test_try_async_captures_errors_and_retries_successful_attempts() -> None:
    attempts: list[int] = []

    async def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 3:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    result = await try_async(
        operation,
        catch=str,
        retry=RetryPolicy(times=3, delay=0),
    )
    assert result == Ok("ok")
    assert attempts == [1, 2, 3]


@pytest.mark.asyncio
async def test_try_async_can_stop_retries_with_a_predicate_and_support_async_catch() -> (
    None
):
    attempts: list[int] = []
    retry_errors: list[str] = []

    async def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        message = "permanent"
        raise RuntimeError(message)

    async def catch(exc: Exception) -> str:
        return str(exc)

    result = await try_async(
        operation,
        catch=catch,
        retry=RetryPolicy(
            times=3,
            delay=0,
            should_retry=lambda error, _: retry_errors.append(error) or False,
        ),
    )
    assert result == Err("permanent")
    assert attempts == [1]
    assert retry_errors == ["permanent"]


@pytest.mark.asyncio
async def test_try_async_applies_backoff_and_dynamic_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fails(_: TryContext) -> str:
        message = "failed"
        raise RuntimeError(message)

    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy(times=2, delay=2, backoff="linear"),
    )
    assert result == Err("failed")
    assert delays == [2, 4]

    delays.clear()
    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy(times=2, delay=lambda _, context: context.attempt / 10),
    )
    assert result == Err("failed")
    assert delays == [0.1, 0.2]

    delays.clear()
    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy(
            times=2,
            delay=2,
            backoff="exponential",
            jitter=0.5,
        ),
    )
    assert result == Err("failed")
    assert len(delays) == 2
    assert 1 <= delays[0] <= 2
    assert 2 <= delays[1] <= 4


@pytest.mark.asyncio
async def test_try_async_validates_retry_policy_and_preserves_cancellation() -> None:
    async def unreachable(_: TryContext) -> str:
        return "unreachable"

    with pytest.raises(ValueError, match="jitter"):
        await try_async(
            unreachable,
            retry=RetryPolicy(times=1, delay=0, jitter=1.1),
        )

    async def cancelled(_: TryContext) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(cancelled)

    async def fails(_: TryContext) -> str:
        message = "failed"
        raise RuntimeError(message)

    no_mapper = await try_async(fails)
    assert isinstance(no_mapper, Err)
    assert isinstance(no_mapper.err_value, RuntimeError)

    timeout_token = CancellationToken()
    timeout_result = await try_async(
        fails,
        retry=RetryPolicy(times=1, delay=0),
        cancel_token=timeout_token,
    )
    assert isinstance(timeout_result, Err)

    pre_cancelled = CancellationToken()
    pre_cancelled.cancel()
    pre_cancelled_result = await try_async(
        fails,
        retry=RetryPolicy(times=1, delay=10),
        cancel_token=pre_cancelled,
    )
    assert isinstance(pre_cancelled_result, Err)

    with pytest.raises(ValueError, match="retry"):
        try_result(lambda _: "unused", retry=-1)
    with pytest.raises(ValueError, match="retry"):
        await try_async(unreachable, retry=RetryPolicy(times=-1))
    with pytest.raises(ValueError, match="backoff"):
        await try_async(
            unreachable,
            retry=RetryPolicy(
                times=1,
                backoff="invalid",
            ),
        )
    with pytest.raises(ValueError, match="delay"):
        await try_async(unreachable, retry=RetryPolicy(times=1, delay=-1))
    with pytest.raises(ValueError, match="backoff"):
        await try_async(
            unreachable,
            retry=RetryPolicy(
                times=1,
                delay=lambda _, __: 0,
                backoff="linear",
            ),
        )
    with pytest.raises(ValueError, match="jitter"):
        await try_async(
            unreachable,
            retry=RetryPolicy(
                times=1,
                delay=lambda _, __: 0,
                jitter=0.5,
            ),
        )
    with pytest.raises(ValueError, match="delay"):
        await try_async(
            fails,
            retry=RetryPolicy(times=1, delay=lambda _, __: -1),
        )


@pytest.mark.asyncio
async def test_async_collection_operations_await_in_input_order() -> None:
    async def result(value: int) -> Ok[int]:
        await asyncio.sleep(0)
        return Ok(value)

    assert await all_results_async([result(1), Ok(2)]) == Ok((1, 2))
    assert await partition_results_async([result(1), Err("bad"), result(2)]) == (
        [1, 2],
        ["bad"],
    )
    assert await all_results_async([]) == Ok(())
    assert await partition_results_async([]) == ([], [])


@pytest.mark.asyncio
async def test_async_collection_operations_propagate_unexpected_rejections() -> None:
    async def broken() -> Ok[int]:
        message = "broken"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="broken"):
        await all_results_async([broken()])

    with pytest.raises(RuntimeError, match="broken"):
        await partition_results_async([broken()])
