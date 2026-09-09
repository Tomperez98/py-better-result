"""Behavioral tests for the Rust-parity retry API."""

from __future__ import annotations

import asyncio
from typing import Never, cast

import pytest

from better_result import (
    AlwaysRetry,
    Err,
    ExponentialBackoff,
    FixedBackoff,
    InvalidBackoffMultiplier,
    InvalidJitterFactor,
    Jittered,
    LinearBackoff,
    Ok,
    Result,
    RetryAfter,
    RetryContext,
    RetryPolicy,
    StopRetry,
    retry,
    retry_async,
)


def test_backoff_schedules_match_rust_attempt_semantics() -> None:
    context = RetryContext(error="temporary", attempt=3)

    assert FixedBackoff(2).decide(context) == RetryAfter(2)
    assert LinearBackoff(2).decide(context) == RetryAfter(6)
    assert ExponentialBackoff(2, 2).decide(context) == RetryAfter(8)


def test_invalid_exponential_multiplier_raises_configuration_error() -> None:
    with pytest.raises(InvalidBackoffMultiplier, match="at least 1"):
        ExponentialBackoff(1, 0)
    with pytest.raises(InvalidBackoffMultiplier, match="at least 1"):
        RetryPolicy.exponential(1, 1, 0)


def test_jitter_constructor_and_schedule() -> None:
    context = RetryContext(error="temporary", attempt=1)
    fixed = FixedBackoff(1)

    assert Jittered(fixed, 0).decide(context) == RetryAfter(1)
    jittered = Jittered(fixed, 1).decide(context)
    assert isinstance(jittered, RetryAfter)
    assert 0 <= jittered.delay <= 1

    with pytest.raises(InvalidJitterFactor, match="between 0 and 1"):
        Jittered(fixed, 2)
    with pytest.raises(InvalidJitterFactor, match="between 0 and 1"):
        RetryPolicy.fixed(1, 1).with_jitter(2)


def test_policy_predicate_and_custom_schedule() -> None:
    policy = RetryPolicy.immediate(10).with_predicate(
        lambda context: context.error == "temporary",
    )
    assert isinstance(policy._decide("temporary", 1), RetryAfter)
    assert isinstance(policy._decide("permanent", 1), StopRetry)

    seen: list[int] = []

    def schedule(context: RetryContext[str]) -> RetryAfter | StopRetry:
        seen.append(context.attempt)
        return RetryAfter(0) if context.attempt < 2 else StopRetry()

    custom = RetryPolicy.new(10, schedule)
    assert custom._decide("temporary", 1) == RetryAfter(0)
    assert custom._decide("temporary", 2) == StopRetry()
    assert seen == [1, 2]
    assert AlwaysRetry().should_retry(RetryContext("error", 1))


def test_retry_returns_success_and_respects_retry_bound() -> None:
    calls = 0
    policy = RetryPolicy.immediate(2)

    def operation() -> Result[str, str]:
        nonlocal calls
        calls += 1
        return Err("temporary") if calls < 3 else Ok("done")

    assert retry(operation, policy) == Ok("done")
    assert calls == 3

    assert retry(lambda: Err("temporary"), RetryPolicy.immediate(0)) == Err("temporary")


def test_retry_does_not_catch_callback_exceptions() -> None:
    with pytest.raises(RuntimeError, match="programmer error"):
        retry(_raise_runtime_error, RetryPolicy.immediate(0))


def test_retry_rejects_non_result_callback_values() -> None:
    with pytest.raises(TypeError, match="must return a Result"):
        retry(_invalid_operation, RetryPolicy.immediate(0))


@pytest.mark.asyncio
async def test_retry_async_matches_sync_and_task_cancellation() -> None:
    calls = 0
    policy = RetryPolicy.immediate(2)

    async def operation() -> Result[str, str]:
        nonlocal calls
        calls += 1
        return Err("temporary") if calls < 2 else Ok("done")

    assert await retry_async(operation, policy) == Ok("done")
    assert calls == 2
    assert await retry_async(_async_error, RetryPolicy.immediate(0)) == Err("temporary")

    slow = RetryPolicy.fixed(1, 60)
    task = asyncio.create_task(retry_async(_async_error, slow))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _invalid_operation() -> Result[int, str]:
    return cast("Result[int, str]", cast("object", 42))


async def _async_error() -> Result[str, str]:
    return Err("temporary")


def _raise_runtime_error() -> Never:
    message = "programmer error"
    raise RuntimeError(message)
