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
    capture,
    capture_async,
    is_err,
    is_ok,
    retry,
    try_async,
    try_result,
)


def test_result_method_types() -> None:
    success = Ok(1)
    failure = Err("bad")

    assert_type(success.map(str), Ok[str])
    assert_type(failure.map_err(len), Err[int])
    assert_type(success.and_then(lambda value: Ok(str(value))), Result[str, Never])
    assert_type(failure.or_else(lambda error: Ok(len(error))), Result[int, Never])
    assert_type(success.unwrap_or(0), int)
    assert_type(failure.unwrap_or_else(len), int)


async def test_capture_types() -> None:
    async def operation() -> int:
        return 42

    assert_type(capture(_return_int), Result[int, Exception])
    assert_type(capture(_return_int, catch=str), Result[int, str])
    assert_type(
        capture(_return_int, exceptions=ValueError),
        Result[int, ValueError],
    )
    assert_type(
        capture(_return_int, exceptions=(ValueError, TypeError)),
        Result[int, ValueError | TypeError],
    )
    assert_type(await capture_async(operation), Result[int, Exception])
    assert_type(await capture_async(operation, catch=str), Result[int, str])
    assert_type(
        await capture_async(operation, exceptions=ValueError),
        Result[int, ValueError],
    )


def _return_int() -> int:
    return 42


def test_result_type_guards_narrow_variants() -> None:
    result = _typed_result(success=True)
    if is_ok(result):
        assert_type(result, Ok[int])
    elif is_err(result):
        assert_type(result, Err[str])


def _typed_result(*, success: bool) -> Result[int, str]:
    return Ok(1) if success else Err("bad")


async def test_exception_boundary_types() -> None:
    async def operation() -> int:
        return 42

    assert_type(try_result(_return_int), Result[int, Exception])
    assert_type(
        try_result(_return_int, RetryPolicy.immediate(1)), Result[int, Exception]
    )
    assert_type(try_result(_return_int, catch=str), Result[int, str])
    assert_type(
        try_result(_return_int, exceptions=ValueError),
        Result[int, ValueError],
    )
    assert_type(await try_async(operation), Result[int, Exception])
    assert_type(
        await try_async(operation, RetryPolicy.immediate(1), catch=str),
        Result[int, str],
    )
    assert_type(
        await try_async(operation, exceptions=ValueError),
        Result[int, ValueError],
    )


def test_retry_constructor_and_operation_types() -> None:
    schedule = ExponentialBackoff(1, 2)
    policy = RetryPolicy.exponential(1, 1, 2)
    direct_policy = RetryPolicy(1, schedule)
    assert_type(schedule, ExponentialBackoff)
    assert_type(policy, RetryPolicy[ExponentialBackoff, AlwaysRetry])
    assert_type(direct_policy, RetryPolicy[ExponentialBackoff, AlwaysRetry])
    assert_type(retry(lambda: Ok(1), RetryPolicy.immediate(1)), Result[int, Never])
    assert_type(Err("bad"), Err[str])
