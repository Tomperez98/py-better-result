"""Tests for converting exception-based boundaries into Results."""

from __future__ import annotations

import asyncio
from typing import assert_type

import pytest

from better_result import (
    Err,
    Ok,
    Result,
    RetryPolicy,
    capture,
    capture_async,
    try_async,
    try_result,
)


def test_capture_wraps_success_and_preserves_the_value() -> None:
    assert capture(lambda: 42) == Ok(42)


def test_capture_returns_the_original_exception_without_a_mapper() -> None:
    result = capture(_raise_value_error)

    assert isinstance(result, Err)
    assert isinstance(result.err(), ValueError)
    assert str(result.err()) == "bad input"


def test_capture_maps_exceptions_at_the_boundary() -> None:
    result = capture(_raise_value_error, catch=str)

    assert result == Err("bad input")


def test_capture_rejects_base_exception_filters() -> None:
    with pytest.raises(TypeError, match="Exception subclasses"):
        capture(  # ty: ignore[no-matching-overload]
            _return_int,
            exceptions=BaseException,
        )


def test_capture_only_catches_selected_exception_types() -> None:
    with pytest.raises(ValueError, match="bad input"):
        capture(_raise_value_error, exceptions=TypeError)

    assert capture(
        _raise_value_error,
        catch=str,
        exceptions=(ValueError,),
    ) == Err("bad input")


def test_capture_does_not_hide_mapper_failures() -> None:
    def broken_mapper(_: Exception) -> str:
        message = "mapper bug"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="mapper bug"):
        capture(_raise_value_error, catch=broken_mapper)


@pytest.mark.asyncio
async def test_capture_async_wraps_success_and_maps_sync_errors() -> None:
    async def operation() -> int:
        message = "bad input"
        raise ValueError(message)

    assert await capture_async(operation, catch=str) == Err("bad input")


@pytest.mark.asyncio
async def test_capture_async_awaits_an_async_error_mapper() -> None:
    async def operation() -> int:
        message = "bad input"
        raise ValueError(message)

    async def map_error(error: Exception) -> str:
        await asyncio.sleep(0)
        return str(error)

    assert await capture_async(operation, catch=map_error) == Err("bad input")


@pytest.mark.asyncio
async def test_capture_async_preserves_success() -> None:
    async def operation() -> int:
        return 42

    assert await capture_async(operation) == Ok(42)


@pytest.mark.asyncio
async def test_capture_async_returns_original_exception_without_mapper() -> None:
    async def operation() -> int:
        message = "bad input"
        raise ValueError(message)

    result = await capture_async(operation)

    assert isinstance(result, Err)
    assert isinstance(result.err(), ValueError)


def test_try_result_captures_and_retries_exception_operations() -> None:
    attempts = 0

    def operation() -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            _raise_value_error()
        return 42

    assert try_result(
        operation,
        RetryPolicy.immediate(1),
        catch=str,
        exceptions=ValueError,
    ) == Ok(42)
    assert attempts == 2


def test_try_result_can_capture_without_a_retry_policy() -> None:
    assert try_result(_return_int) == Ok(42)
    assert try_result(_raise_value_error, catch=str) == Err("bad input")


def test_try_result_stops_after_the_retry_bound() -> None:
    assert try_result(
        _raise_value_error,
        RetryPolicy.immediate(0),
        catch=str,
    ) == Err("bad input")


@pytest.mark.asyncio
async def test_capture_async_does_not_capture_cancellation() -> None:
    async def operation() -> int:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await capture_async(operation)


@pytest.mark.asyncio
async def test_try_async_can_capture_without_a_retry_policy() -> None:
    async def operation() -> int:
        return 42

    assert await try_async(operation) == Ok(42)


@pytest.mark.asyncio
async def test_try_async_captures_and_retries_exception_operations() -> None:
    attempts = 0

    async def operation() -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            message = "temporary"
            raise TimeoutError(message)
        return 42

    result = await try_async(
        operation,
        RetryPolicy.immediate(1),
        catch=str,
        exceptions=TimeoutError,
    )

    assert result == Ok(42)
    assert attempts == 2

    async def permanent_failure() -> int:
        message = "permanent"
        raise TimeoutError(message)

    assert await try_async(
        permanent_failure,
        RetryPolicy.immediate(0),
        catch=str,
    ) == Err("permanent")


def test_capture_signatures_expose_the_error_type() -> None:
    assert_type(capture(_return_int), Result[int, Exception])
    assert_type(capture(_return_int, catch=str), Result[int, str])


def _return_int() -> int:
    return 42


def _raise_value_error() -> int:
    message = "bad input"
    raise ValueError(message)
