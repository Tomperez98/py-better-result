"""Static type contracts for the reduced Result API."""

from __future__ import annotations

from typing import Never, assert_type

from better_result._core import Err, Ok, Result


def test_variant_types_are_precise() -> None:
    success = typed_success()
    failure = typed_failure()

    assert_type(success, Ok[int])
    assert_type(failure, Err[str])
    assert_type(success.ok(), int)
    assert_type(success.err(), None)
    assert_type(failure.ok(), None)
    assert_type(failure.err(), str)
    assert_type(success.unwrap(), int)
    assert_type(failure.unwrap_err(), str)
    assert_type(success.unwrap_or("unused"), int)
    default = typed_zero()
    assert_type(failure.unwrap_or(default), int)
    assert_type(success.unwrap_or_else(lambda _: 0), int)
    assert_type(failure.unwrap_or_else(len), int)
    assert_type(success.unwrap_or_default(), int)
    assert_type(failure.unwrap_or_default(), None)
    assert_type(success.expect("unused"), int)
    assert_type(failure.expect_err("unused"), str)


def test_result_combinators_preserve_the_active_variant() -> None:
    success = typed_success()
    failure = typed_failure()

    assert_type(success.map(str), Ok[str])
    assert_type(failure.map(str), Err[str])
    assert_type(success.map_err(str), Ok[int])
    assert_type(failure.map_err(len), Err[int])
    default = typed_zero()
    assert_type(success.map_or(default, str), int | str)
    assert_type(failure.map_or(default, str), int | str)
    assert_type(success.map_or_else(lambda: default, str), int | str)
    assert_type(failure.map_or_else(lambda: default, str), int | str)
    assert_type(success.and_then(to_float), Result[float, Never])
    assert_type(failure.and_then(to_float), Err[str])
    assert_type(success.or_else(lambda _: Err(ValueError())), Ok[int])
    assert_type(failure.or_else(to_float_result), Result[float, Never])
    assert_type(success.inspect(str), Ok[int])
    assert_type(failure.inspect(str), Err[str])
    assert_type(success.inspect_err(str), Ok[int])
    assert_type(failure.inspect_err(str), Err[str])


def test_result_type_narrowing_is_lsp_friendly() -> None:
    result = typed_result(success=False)
    assert_type(result, Result[int, str])

    if isinstance(result, Ok):
        assert_type(result, Ok[int])
        assert_type(result.ok(), int)
    elif isinstance(result, Err):
        assert_type(result, Err[str])
        assert_type(result.err(), str)


def typed_zero() -> int:
    return 0


def typed_success() -> Ok[int]:
    value: int = 1
    return Ok(value)


def typed_failure() -> Err[str]:
    value: str = "bad"
    return Err(value)


def typed_result(*, success: bool) -> Result[int, str]:
    return typed_success() if success else typed_failure()


def to_float(value: int) -> Result[float, Never]:
    return Ok(float(value))


def to_float_result(_: str) -> Result[float, Never]:
    return Ok(1.0)
