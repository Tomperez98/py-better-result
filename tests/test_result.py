"""Tests for Result-level synchronous utilities."""

from __future__ import annotations

from typing import Never, assert_type

import pytest

from better_result.collections import all_results, flatten, partition
from better_result.combinators import (
    and_then,
    map_error,
    map_result,
    match,
    tap,
    try_recover,
    unwrap_or,
)
from better_result.core import Err, Ok, Panic, Result, err, ok
from better_result.error import UnhandledException
from better_result.retry import RetryConfig, TryContext, try_result


def test_try_result_returns_success_and_tracks_attempts() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> int:
        attempts.append(context.attempt)
        if context.attempt < 3:
            raise ValueError("not ready")
        return 42

    result = try_result(operation, retry=RetryConfig(times=2))

    assert isinstance(result, Ok)
    assert result.value == 42
    assert attempts == [1, 2, 3]


def test_try_result_returns_unhandled_exception_without_a_catch_handler() -> None:
    result = try_result(lambda _context: (_ for _ in ()).throw(ValueError("bad")))

    assert isinstance(result, Err)
    assert isinstance(result.error, UnhandledException)
    assert isinstance(result.error.cause, ValueError)


def test_try_result_preserves_process_control_exceptions() -> None:
    with pytest.raises(KeyboardInterrupt):
        try_result(lambda _context: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(Panic):
        try_result(lambda _context: (_ for _ in ()).throw(Panic("bug")))


def test_try_result_uses_catch_handler_and_panics_if_catch_fails() -> None:
    result = try_result(
        lambda _context: (_ for _ in ()).throw(ValueError("bad")),
        lambda cause: f"handled: {cause}",
    )
    assert isinstance(result, Err)
    assert result.error == "handled: bad"

    def broken_catch(_cause: BaseException) -> str:
        raise RuntimeError("catch failed")

    with pytest.raises(Panic, match="catch handler threw"):
        try_result(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")),
            broken_catch,
        )


def test_result_combinators_support_data_first_and_data_last_forms() -> None:
    success = ok(2)
    failure = err("bad")

    mapped = map_result(success, lambda value: value * 2)
    assert isinstance(mapped, Ok)
    assert mapped.value == 4
    map_value = map_result(lambda value: value * 2)
    mapped_later = map_value(success)
    assert isinstance(mapped_later, Ok)
    assert mapped_later.value == 4
    mapped_error = map_error(failure, str.upper)
    assert isinstance(mapped_error, Err)
    assert mapped_error.error == "BAD"
    mapped_error_later = map_error(str.upper)(failure)
    assert isinstance(mapped_error_later, Err)
    assert mapped_error_later.error == "BAD"
    chained = and_then(success, lambda value: ok(str(value)))
    assert isinstance(chained, Ok)
    assert chained.value == "2"
    chained_later = and_then(lambda value: ok(str(value)))(success)
    assert isinstance(chained_later, Ok)
    assert chained_later.value == "2"
    recovered = try_recover(failure, lambda value: ok(len(value)))
    assert isinstance(recovered, Ok)
    assert recovered.value == 3
    recovered_later = try_recover(lambda value: ok(len(value)))(failure)
    assert isinstance(recovered_later, Ok)
    assert recovered_later.value == 3


def test_result_observers_and_collectors_preserve_order() -> None:
    results: list[Result[int, str]] = [
        Ok[int, str](1),
        Err[int, str]("bad"),
        Ok[int, str](2),
    ]
    seen: list[int] = []

    assert match(results[0], {"ok": lambda value: value * 2, "err": lambda _: 0}) == 2
    assert tap(results[0], seen.append) is results[0]
    assert seen == [1]
    assert unwrap_or(results[1], 99) == 99
    assert unwrap_or(results[0], None) == 1
    assert unwrap_or(99)(results[0]) == 1
    assert unwrap_or(None)(results[0]) == 1
    collected = all_results(results)
    assert isinstance(collected, Err)
    assert collected.error == "bad"
    collected_successes = all_results([ok(1), ok(2)])
    assert isinstance(collected_successes, Ok)
    assert collected_successes.value == [1, 2]
    assert partition(results) == ([1, 2], ["bad"])


def test_flatten_collapses_both_nested_variants() -> None:
    flattened_success = flatten(ok(ok(42)))
    assert isinstance(flattened_success, Ok)
    assert flattened_success.value == 42
    flattened_inner_error = flatten(ok(err("inner")))
    assert isinstance(flattened_inner_error, Err)
    assert flattened_inner_error.error == "inner"
    flattened_outer_error = flatten(Err[Result[int, str], str]("outer"))
    assert isinstance(flattened_outer_error, Err)
    assert flattened_outer_error.error == "outer"


def test_result_utility_types_are_inferred() -> None:
    success = ok(2)
    assert_type(map_result(success, str), Result[str, Never])
    assert_type(all_results([ok(1), ok(2)]), Result[list[int], Never])
