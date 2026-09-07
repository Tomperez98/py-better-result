"""Tests for Result-level synchronous utilities."""

from __future__ import annotations

from typing import Never, assert_type

import pytest

from better_result.collections import all_results, partition
from better_result.core import Err, Ok, PanicError, Result
from better_result.error import UnhandledError
from better_result.retry import RetryConfig, TryContext, try_result


def test_try_result_returns_success_and_tracks_attempts() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> int:
        attempts.append(context.attempt)
        if context.attempt < 3:
            msg = "not ready"
            raise ValueError(msg)
        return 42

    result = try_result(operation, retry=RetryConfig(times=2))

    assert isinstance(result, Ok)
    assert result.value == 42
    assert attempts == [1, 2, 3]


def test_try_result_returns_unhandled_exception_without_a_catch_handler() -> None:
    result = try_result(lambda _context: (_ for _ in ()).throw(ValueError("bad")))

    assert isinstance(result, Err)
    assert isinstance(result.error, UnhandledError)
    assert isinstance(result.error.cause, ValueError)


def test_try_result_preserves_process_control_exceptions() -> None:
    with pytest.raises(KeyboardInterrupt):
        try_result(lambda _context: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(PanicError):
        try_result(lambda _context: (_ for _ in ()).throw(PanicError("bug")))


def test_try_result_uses_catch_handler_and_panics_if_catch_fails() -> None:
    result = try_result(
        lambda _context: (_ for _ in ()).throw(ValueError("bad")),
        lambda cause: f"handled: {cause}",
    )
    assert isinstance(result, Err)
    assert result.error == "handled: bad"

    def broken_catch(_cause: BaseException) -> str:
        msg = "catch failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="catch handler threw"):
        try_result(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")),
            broken_catch,
        )


def test_result_transformations_use_instance_methods() -> None:
    success = Ok(2)
    failure = Err("bad")

    mapped = success.map(lambda value: value * 2)
    assert isinstance(mapped, Ok)
    assert mapped.value == 4
    mapped_error = failure.map_error(str.upper)
    assert isinstance(mapped_error, Err)
    assert mapped_error.error == "BAD"
    chained = success.and_then(lambda value: Ok(str(value)))
    assert isinstance(chained, Ok)
    assert chained.value == "2"
    recovered = failure.try_recover(lambda value: Ok(len(value)))
    assert isinstance(recovered, Ok)
    assert recovered.value == 3


def test_result_observers_and_collectors_preserve_order() -> None:
    results: list[Result[int, str]] = [
        Ok(1),
        Err("bad"),
        Ok(2),
    ]
    seen: list[int] = []

    assert results[0].match(lambda value: value * 2, lambda _: 0) == 2
    assert results[0].tap(seen.append) is results[0]
    assert seen == [1]
    assert results[1].unwrap_or(99) == 99
    assert results[0].unwrap_or(None) == 1
    collected = all_results(results)
    assert isinstance(collected, Err)
    assert collected.error == "bad"
    collected_successes = all_results([Ok(1), Ok(2)])
    assert isinstance(collected_successes, Ok)
    assert collected_successes.value == [1, 2]
    assert partition(results) == ([1, 2], ["bad"])


def test_flatten_collapses_both_nested_variants() -> None:
    flattened_success = Ok(Ok(42)).flatten()
    assert isinstance(flattened_success, Ok)
    assert flattened_success.value == 42
    flattened_inner_error = Ok(Err("inner")).flatten()
    assert isinstance(flattened_inner_error, Err)
    assert flattened_inner_error.error == "inner"
    flattened_outer_error = Err("outer").flatten()
    assert isinstance(flattened_outer_error, Err)
    assert flattened_outer_error.error == "outer"


def test_result_utility_types_are_inferred() -> None:
    success: Result[int, Never] = Ok(2)
    assert_type(success.map(str), Ok[str])
    successes: list[Result[int, Never]] = [Ok(1), Ok(2)]
    assert_type(all_results(successes), Result[list[int], Never])
