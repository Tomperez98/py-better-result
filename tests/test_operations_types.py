"""Static type contracts for the Rust-parity API."""

from __future__ import annotations

from typing import Never, assert_type

from better_result import (
    AlwaysRetry,
    Err,
    ExponentialBackoff,
    Ok,
    Result,
    RetryPolicy,
    retry,
)


def test_result_method_types() -> None:
    success: Ok[int] = Ok(1)
    failure: Err[str] = Err("bad")

    assert_type(success.map(str), Ok[str])
    assert_type(failure.map_err(len), Err[int])
    assert_type(success.and_then(lambda value: Ok(str(value))), Result[str, Never])
    assert_type(failure.or_else(lambda error: Ok(len(error))), Result[int, Never])
    assert_type(success.unwrap_or(0), int)
    assert_type(failure.unwrap_or_else(len), int)


def test_retry_constructor_and_operation_types() -> None:
    schedule = ExponentialBackoff(1, 2)
    policy = RetryPolicy.exponential(1, 1, 2)
    direct_policy = RetryPolicy(1, schedule)
    assert_type(schedule, ExponentialBackoff)
    assert_type(policy, RetryPolicy[ExponentialBackoff, AlwaysRetry])
    assert_type(direct_policy, RetryPolicy[ExponentialBackoff, AlwaysRetry])
    assert_type(retry(lambda: Ok(1), RetryPolicy.immediate(1)), Result[int, Never])
    assert_type(Err("bad"), Err[str])
