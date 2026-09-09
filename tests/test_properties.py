"""Property tests for Result and Rust-parity retry contracts."""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st
import pytest

from better_result import (
    Err,
    ExponentialBackoff,
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


@given(
    is_ok=st.booleans(),
    value=st.integers(),
    error=st.text(),
    left=st.integers(),
    right=st.integers(),
)
def test_result_map_identity_and_composition_laws(
    *, is_ok: bool, value: int, error: str, left: int, right: int
) -> None:
    result: Result[int, str] = Ok(value) if is_ok else Err(error)
    assert result.map(lambda item: item) == result
    assert result.map(lambda item: item + left).map(
        lambda item: item * right,
    ) == result.map(lambda item: (item + left) * right)


@settings(max_examples=100)
@given(
    initial=st.integers(min_value=0, max_value=100),
    multiplier=st.integers(min_value=1, max_value=4),
    attempt=st.integers(min_value=1, max_value=8),
    jitter=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_backoff_schedules_preserve_bounds(
    initial: int, multiplier: int, attempt: int, jitter: float
) -> None:
    context = RetryContext(error="temporary", attempt=attempt)
    linear = LinearBackoff(initial).decide(context)
    exponential = ExponentialBackoff(initial, multiplier).decide(context)
    jittered = Jittered(
        ExponentialBackoff(initial, multiplier),
        jitter,
    ).decide(context)

    assert isinstance(linear, RetryAfter)
    assert isinstance(exponential, RetryAfter)
    assert isinstance(jittered, RetryAfter)
    assert linear.delay == initial * attempt
    assert exponential.delay == initial * multiplier ** (attempt - 1)
    assert math.isfinite(jittered.delay)
    assert 0 <= jittered.delay <= exponential.delay


@settings(max_examples=100)
@given(failures=st.integers(min_value=0, max_value=12), retries=st.integers(0, 12))
def test_retry_never_exceeds_its_retry_bound(failures: int, retries: int) -> None:
    attempts = 0

    def operation() -> Result[str, str]:
        nonlocal attempts
        attempts += 1
        return Err("temporary") if attempts <= failures else Ok("ok")

    result = retry(operation, RetryPolicy.immediate(retries))
    assert attempts == min(failures, retries) + 1
    assert result == (Ok("ok") if failures <= retries else Err("temporary"))


@settings(max_examples=100)
@given(times=st.integers(0, 12), attempt=st.integers(1, 20))
def test_policy_decision_checks_bound_before_schedule(times: int, attempt: int) -> None:
    seen: list[int] = []

    def schedule(context: RetryContext[str]) -> RetryDecisionLike:
        seen.append(context.attempt)
        return RetryAfter(0)

    policy = RetryPolicy(times, schedule)
    decision = policy._decide("temporary", attempt)
    if attempt <= times:
        assert decision == RetryAfter(0)
        assert seen == [attempt]
    else:
        assert decision == StopRetry()
        assert seen == []


@pytest.mark.asyncio
async def test_async_retry_matches_sync_retry() -> None:
    async def operation() -> Result[str, str]:
        return Ok("ok")

    assert await retry_async(operation, RetryPolicy.immediate(0)) == Ok("ok")


RetryDecisionLike = RetryAfter | StopRetry
