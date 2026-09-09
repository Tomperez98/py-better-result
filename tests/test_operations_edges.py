"""Edge-case coverage for retry validation and dispatch."""

from __future__ import annotations

import math

import pytest

from better_result import (
    Err,
    ExponentialBackoff,
    FixedBackoff,
    InvalidBackoffMultiplier,
    InvalidJitterFactor,
    Jittered,
    LinearBackoff,
    RetryContext,
    RetryPolicy,
    StopRetry,
)


def test_retry_context_and_delay_validation() -> None:
    with pytest.raises(TypeError, match="attempt"):
        RetryContext(error="error", attempt=True)
    with pytest.raises(ValueError, match="attempt"):
        RetryContext("error", 0)

    with pytest.raises(TypeError, match="delay"):
        FixedBackoff(object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="delay"):
        LinearBackoff(object())  # ty: ignore[invalid-argument-type]
    for constructor in (FixedBackoff, LinearBackoff):
        with pytest.raises(ValueError, match="delay"):
            constructor(-1)
        with pytest.raises(ValueError, match="delay"):
            constructor(math.inf)

    with pytest.raises(ValueError, match="max retries"):
        RetryPolicy.immediate(-1)
    with pytest.raises(ValueError, match="max retries"):
        RetryPolicy.immediate(max_retries=True)


def test_overflowing_backoffs_stop_retrying() -> None:
    context = RetryContext("error", 2)
    assert isinstance(LinearBackoff(1e308).decide(context), StopRetry)
    assert isinstance(
        ExponentialBackoff(1, 2).decide(RetryContext("error", 2049)), StopRetry
    )
    assert isinstance(ExponentialBackoff(1e308, 2).decide(context), StopRetry)


def test_invalid_exponential_and_jitter_configuration() -> None:
    with pytest.raises(InvalidBackoffMultiplier):
        ExponentialBackoff(1, 0)
    with pytest.raises(InvalidBackoffMultiplier):
        ExponentialBackoff(1, multiplier=True)
    with pytest.raises(InvalidBackoffMultiplier):
        ExponentialBackoff(1, multiplier=object())  # ty: ignore[invalid-argument-type]

    schedule = FixedBackoff(1)
    for factor in (-1, 2, math.nan, math.inf, True, object()):
        with pytest.raises(InvalidJitterFactor):
            Jittered(schedule, factor)  # ty: ignore[invalid-argument-type]


def test_jitter_and_policy_dispatch_fail_fast_on_invalid_components() -> None:
    context = RetryContext("error", 1)
    stopped = Jittered(lambda _: StopRetry(), 0)
    assert stopped.decide(context) == StopRetry()

    with pytest.raises(TypeError, match="schedule"):
        Jittered(object(), 0).decide(context)

    with pytest.raises(TypeError, match="schedule"):
        RetryPolicy.new(1, object())._decide("error", 1)

    with pytest.raises(TypeError, match="RetryDecision"):
        RetryPolicy.new(1, lambda _: object())._decide("error", 1)

    with pytest.raises(TypeError, match="predicate"):
        RetryPolicy.new(1, FixedBackoff(0)).with_predicate(object())._decide("error", 1)

    assert isinstance(RetryPolicy.fixed(1, 1).with_jitter(0.5).schedule, Jittered)
    assert str(InvalidBackoffMultiplier()) == "backoff multiplier must be at least 1"
    assert (
        str(InvalidJitterFactor()) == "jitter factor must be finite and between 0 and 1"
    )
    assert Err("bad").is_err()
