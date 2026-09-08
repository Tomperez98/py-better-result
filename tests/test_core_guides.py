"""Executable coverage for the canonical Result workflow."""

from __future__ import annotations

from typing import assert_type

from better_result import Err, Ok, Result, TaggedError


class ParseFailedError(TaggedError, tag="ParseFailed"):
    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Could not parse input", input=input_value)


class ValidationFailedError(TaggedError, tag="ValidationFailed"):
    value: int

    def __init__(self, value: int) -> None:
        super().__init__(message="Value is invalid", value=value)


def test_narrowing_and_matching_use_both_callbacks() -> None:
    success: Result[int, ParseFailedError] = Ok(3)
    failure: Result[int, ParseFailedError] = Err(ParseFailedError("bad"))

    assert (
        success.match(lambda value: f"value={value}", lambda error: error.message)
        == "value=3"
    )
    assert failure.match(str, lambda error: f"error={error.input}") == "error=bad"


def test_map_and_and_then_are_the_canonical_transformations() -> None:
    def validate(value: int) -> Result[str, ValidationFailedError]:
        if value <= 0:
            return Err(ValidationFailedError(value))
        return Ok(f"valid:{value}")

    initial: Result[int, ParseFailedError] = Ok(2)
    chained = initial.map(lambda value: value + 1).and_then(validate)
    assert_type(chained, Result[str, ValidationFailedError])
    assert chained == Ok("valid:3")

    rejected = Ok(0).and_then(validate)
    assert isinstance(rejected, Err)
    assert isinstance(rejected.error, ValidationFailedError)


def test_map_error_translates_only_errors() -> None:
    failure: Result[int, ParseFailedError] = Err(ParseFailedError("bad"))
    translated = failure.map_error(
        lambda error: ValidationFailedError(len(error.input))
    )

    assert isinstance(translated, Err)
    assert translated.error.value == 3
    success: Result[int, ParseFailedError] = Ok(7)
    assert success.map_error(lambda _: ValidationFailedError(0)) is success
