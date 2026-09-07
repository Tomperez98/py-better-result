"""Executable coverage for the creating, narrowing, and transforming guides."""

from __future__ import annotations

from typing import assert_type

import pytest

from better_result.core import Err, Ok, PanicError, Result, is_err, is_ok
from better_result.error import TaggedError, UnhandledError
from better_result.retry import try_async


class ParseFailedError(TaggedError, tag="ParseFailed"):
    """The input could not be parsed."""

    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Could not parse input", input=input_value)


class ValidationFailedError(TaggedError, tag="ValidationFailed"):
    """The parsed value failed validation."""

    value: int

    def __init__(self, value: int) -> None:
        super().__init__(message="Value is invalid", value=value)


class SaveFailedError(TaggedError, tag="SaveFailed"):
    """The validated value could not be saved."""

    def __init__(self) -> None:
        super().__init__(message="Could not save value")


@pytest.mark.asyncio
async def test_try_async_without_catch_returns_unhandled_exception() -> None:
    original = ValueError("request failed")

    async def operation(_context: object) -> int:
        raise original

    result = await try_async(operation)
    assert_type(result, Result[int, UnhandledError])
    assert isinstance(result, Err)
    assert isinstance(result.error, UnhandledError)
    assert result.error.cause is original


def test_narrowing_and_matching_use_both_callbacks() -> None:
    success: Result[int, ParseFailedError] = Ok(3)
    failure: Result[int, ParseFailedError] = Err(ParseFailedError("bad"))

    assert success.value == 3
    assert isinstance(failure.error, ParseFailedError)

    assert is_ok(success)
    assert not is_err(success)
    assert not is_ok(failure)
    assert is_err(failure)

    def on_ok(value: int) -> str:
        return f"value={value}"

    def on_err(error: ParseFailedError) -> str:
        return f"error={error.input}"

    assert success.match(on_ok, on_err) == "value=3"
    assert failure.match(on_ok, on_err) == "error=bad"


def test_map_nests_results_but_and_then_flattens_them() -> None:
    nested = Ok(2).map(lambda value: Ok(value * 3))
    assert isinstance(nested, Ok)
    assert isinstance(nested.value, Ok)
    assert nested.value.value == 6

    def validate(value: int) -> Result[str, ValidationFailedError]:
        if value <= 0:
            return Err(ValidationFailedError(value))
        return Ok(f"valid:{value}")

    def chain(
        result: Result[int, ParseFailedError],
    ) -> Result[str, ParseFailedError | ValidationFailedError]:
        return result.and_then(validate)

    initial: Result[int, ParseFailedError] = Ok(2)
    chained = chain(initial)
    assert_type(chained, Result[str, ParseFailedError | ValidationFailedError])
    assert isinstance(chained, Ok)
    assert chained.value == "valid:2"

    rejected = Ok(0).and_then(validate)
    assert isinstance(rejected, Err)
    assert isinstance(rejected.error, ValidationFailedError)

    called = False

    def should_not_run(_value: int) -> Result[str, ValidationFailedError]:
        nonlocal called
        called = True
        return Ok("unexpected")

    short_circuited = Err(ParseFailedError("bad")).and_then(should_not_run)
    assert isinstance(short_circuited, Err)
    assert isinstance(short_circuited.error, ParseFailedError)
    assert not called


def test_map_error_translates_only_errors_and_recovery_returns_results() -> None:
    failure: Result[int, ParseFailedError] = Err(ParseFailedError("bad"))

    def translate(error: ParseFailedError) -> ValidationFailedError:
        return ValidationFailedError(len(error.input))

    translated = failure.map_error(translate)

    assert isinstance(translated, Err)
    assert isinstance(translated.error, ValidationFailedError)
    assert translated.error.value == 3

    success: Result[int, ParseFailedError] = Ok(7)
    mapped_success = success.map_error(lambda _error: ValidationFailedError(0))
    assert mapped_success is success

    def recover(_error: ParseFailedError) -> Result[int, SaveFailedError]:
        return Ok(99)

    def recover_result(
        result: Result[int, ParseFailedError],
    ) -> Result[int, SaveFailedError]:
        return result.try_recover(recover)

    recovered = recover_result(failure)
    assert_type(recovered, Result[int, SaveFailedError])
    assert isinstance(recovered, Ok)
    assert recovered.value == 99

    untouched = Ok(7).try_recover(lambda _error: Err(SaveFailedError()))
    assert isinstance(untouched, Ok)
    assert untouched.value == 7

    with pytest.raises(PanicError, match="try_recover callback threw"):
        failure.try_recover(lambda _error: (_ for _ in ()).throw(RuntimeError("bug")))
