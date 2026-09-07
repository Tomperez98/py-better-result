"""Executable coverage for the creating, narrowing, and transforming guides."""

from __future__ import annotations

from typing import assert_type

import pytest

from better_result.core import Err, Ok, Panic, Result, err, is_err, is_error, is_ok, ok
from better_result.error import TaggedError, UnhandledException
from better_result.result import (
    and_then,
    map_error,
    match,
    try_async,
    try_recover,
)


class ParseFailed(TaggedError, tag="ParseFailed"):
    """The input could not be parsed."""

    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Could not parse input", input=input_value)


class ValidationFailed(TaggedError, tag="ValidationFailed"):
    """The parsed value failed validation."""

    value: int

    def __init__(self, value: int) -> None:
        super().__init__(message="Value is invalid", value=value)


class SaveFailed(TaggedError, tag="SaveFailed"):
    """The validated value could not be saved."""

    def __init__(self) -> None:
        super().__init__(message="Could not save value")


@pytest.mark.asyncio
async def test_try_async_without_catch_returns_unhandled_exception() -> None:
    original = ValueError("request failed")

    async def operation(_context: object) -> int:
        raise original

    result = await try_async(operation)
    assert_type(result, Result[int, UnhandledException])
    assert isinstance(result, Err)
    assert isinstance(result.error, UnhandledException)
    assert result.error.cause is original


def test_narrowing_and_matching_support_both_forms() -> None:
    success: Result[int, ParseFailed] = Ok[int, ParseFailed](3)
    failure: Result[int, ParseFailed] = err(ParseFailed("bad"))

    assert success.status == "ok"
    assert success.value == 3
    assert failure.status == "error"
    assert isinstance(failure.error, ParseFailed)

    assert is_ok(success)
    assert not is_err(success)
    assert not is_error(success)
    assert not is_ok(failure)
    assert is_err(failure)
    assert is_error(failure)
    assert success.is_ok()
    assert not success.is_err()
    assert failure.is_err()
    assert not failure.is_ok()

    handlers = {
        "ok": lambda value: f"value={value}",
        "err": lambda error: f"error={error.input}",
    }
    assert match(success, handlers) == "value=3"
    assert match(handlers)(failure) == "error=bad"
    assert success.match(handlers) == "value=3"
    assert failure.match(handlers) == "error=bad"


def test_map_nests_results_but_and_then_flattens_them() -> None:
    nested = ok(2).map(lambda value: ok(value * 3))
    assert isinstance(nested, Ok)
    assert isinstance(nested.value, Ok)
    assert nested.value.value == 6

    def validate(value: int) -> Result[str, ValidationFailed]:
        if value <= 0:
            return err(ValidationFailed(value))
        return Ok[str, ValidationFailed](f"valid:{value}")

    chained = and_then(Ok[int, ParseFailed](2), validate)
    assert_type(chained, Result[str, ParseFailed | ValidationFailed])
    assert isinstance(chained, Ok)
    assert chained.value == "valid:2"

    rejected = and_then(Ok[int, ParseFailed](0), validate)
    assert isinstance(rejected, Err)
    assert isinstance(rejected.error, ValidationFailed)

    called = False

    def should_not_run(_value: int) -> Result[str, ValidationFailed]:
        nonlocal called
        called = True
        return Ok[str, ValidationFailed]("unexpected")

    short_circuited = and_then(
        Err[int, ParseFailed](ParseFailed("bad")),
        should_not_run,
    )
    assert isinstance(short_circuited, Err)
    assert isinstance(short_circuited.error, ParseFailed)
    assert not called


def test_map_error_translates_only_errors_and_recovery_returns_results() -> None:
    failure: Result[int, ParseFailed] = Err[int, ParseFailed](ParseFailed("bad"))

    def translate(error: ParseFailed) -> ValidationFailed:
        return ValidationFailed(len(error.input))

    translated = map_error(failure, translate)

    assert isinstance(translated, Err)
    assert isinstance(translated.error, ValidationFailed)
    assert translated.error.value == 3

    success: Result[int, ParseFailed] = Ok[int, ParseFailed](7)
    mapped_success = map_error(success, lambda _error: ValidationFailed(0))
    assert mapped_success is success

    recovered = try_recover(
        failure,
        lambda _error: Ok[int, SaveFailed](99),
    )
    assert_type(recovered, Result[int, SaveFailed])
    assert isinstance(recovered, Ok)
    assert recovered.value == 99

    untouched = try_recover(
        Ok[int, ParseFailed](7),
        lambda _error: Err[int, SaveFailed](SaveFailed()),
    )
    assert isinstance(untouched, Ok)
    assert untouched.value == 7

    with pytest.raises(Panic, match="try_recover callback threw"):
        failure.try_recover(lambda _error: (_ for _ in ()).throw(RuntimeError("bug")))
