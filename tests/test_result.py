"""Tests for Result-level utilities and internal extensions."""

from __future__ import annotations

from typing import Never, assert_type

import pytest

from better_result import Err, Ok, PanicError, Result
from better_result._collections import all_results, partition
from better_result._error import UnhandledError
from better_result._retry import RetryConfig, TryContext, try_result


def test_try_result_returns_success_and_tracks_attempts() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> int:
        attempts.append(context.attempt)
        if context.attempt < 3:
            message = "not ready"
            raise ValueError(message)
        return 42

    result = try_result(operation, retry=RetryConfig(times=2))
    assert result == Ok(42)
    assert attempts == [1, 2, 3]


def test_try_result_returns_unhandled_exception_without_a_catch_handler() -> None:
    result = try_result(lambda _context: (_ for _ in ()).throw(ValueError("bad")))
    assert isinstance(result, Err)
    assert isinstance(result.error, UnhandledError)


def test_try_result_preserves_process_control_and_panics() -> None:
    with pytest.raises(KeyboardInterrupt):
        try_result(lambda _context: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(PanicError):
        try_result(lambda _context: (_ for _ in ()).throw(PanicError("bug")))

    def broken_catch(_cause: BaseException) -> str:
        message = "catch failed"
        raise RuntimeError(message)

    with pytest.raises(PanicError, match="catch handler threw"):
        try_result(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")),
            broken_catch,
        )


def test_canonical_result_transformations_and_collections() -> None:
    success = Ok(2)
    failure = Err("bad")

    assert success.map(lambda value: value * 2) == Ok(4)
    assert failure.map_error(str.upper) == Err("BAD")
    assert success.and_then(lambda value: Ok(str(value))) == Ok("2")
    assert success.match(lambda value: value * 2, lambda _: 0) == 4
    assert failure.unwrap_or(99) == 99

    results: list[Result[int, str]] = [Ok(1), Err("bad"), Ok(2)]
    assert all_results(results) == Err("bad")
    assert partition(results) == ([1, 2], ["bad"])
    typed_results: list[Result[int, Never]] = [Ok(1), Ok(2)]
    assert_type(all_results(typed_results), Result[list[int], Never])
