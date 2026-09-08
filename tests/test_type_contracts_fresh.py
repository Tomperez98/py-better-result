"""Static type contracts for the fresh Result API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, assert_type

import pytest

from better_result._core import Err, Ok, Result, UnwrapError, is_err, is_ok


def test_variant_types_are_precise() -> None:
    success = typed_success()
    failure = typed_failure()

    assert_type(success, Ok[int])
    assert_type(failure, Err[str])
    assert_type(success.ok(), int)
    assert_type(success.err(), None)
    assert_type(success.ok_value, int)
    assert_type(failure.ok(), None)
    assert_type(failure.err(), str)
    assert_type(failure.err_value, str)
    assert_type(success.unwrap(), int)
    assert_type(failure.unwrap_err(), str)
    assert_type(success.unwrap_or("unused"), int)
    default = typed_zero()
    assert_type(failure.unwrap_or(default), int)
    assert_type(success.unwrap_or_else(lambda _: 0), int)
    assert_type(failure.unwrap_or_else(len), int)
    assert_type(success.unwrap_or_raise(ValueError), int)
    assert_type(success.expect("unused"), int)
    assert_type(failure.expect_err("unused"), str)

    if TYPE_CHECKING:
        assert_type(failure.unwrap_or_raise(ValueError), Never)


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
    # The predicates are runtime checks; ty represents their TypeIs metadata
    # separately from bool. Concrete isinstance checks verify LSP narrowing.
    assert is_ok(result) is False
    assert is_err(result) is True

    if isinstance(result, Ok):
        assert_type(result, Ok[int])
        assert_type(result.ok_value, int)
    else:
        assert_type(result, Err[str])
        assert_type(result.err_value, str)

    result = typed_result(success=True)
    if isinstance(result, Err):
        assert_type(result, Err[str])
        assert_type(result.err_value, str)
    else:
        assert_type(result, Ok[int])
        assert_type(result.ok_value, int)


def test_result_success_type_is_covariant() -> None:
    animal = accept_animal(typed_dog_result())

    assert_type(animal, Result[Animal, str])


@pytest.mark.asyncio
async def test_async_combinators_have_precise_types() -> None:
    success = typed_success()
    failure = typed_failure()

    assert_type(await success.map_async(to_text), Ok[str])
    assert_type(await failure.map_async(to_text), Err[str])
    assert_type(await success.and_then_async(to_result), Result[str, ValueError])
    assert_type(await failure.and_then_async(to_result), Err[str])


def test_unwrap_error_retains_a_result_shape() -> None:
    try:
        Err("bad").unwrap()
    except UnwrapError as error:
        assert_type(error.result, Ok[Any] | Err[Any])


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


class Animal:
    pass


class Dog(Animal):
    pass


def typed_dog_result() -> Result[Dog, str]:
    return Ok(Dog())


def accept_animal(result: Result[Animal, str]) -> Result[Animal, str]:
    return result


def to_float(value: int) -> Result[float, Never]:
    return Ok(float(value))


def to_float_result(_: str) -> Result[float, Never]:
    return Ok(1.0)


async def to_text(value: int) -> str:
    return str(value)


async def to_result(value: int) -> Result[str, ValueError]:
    return Ok(str(value))
