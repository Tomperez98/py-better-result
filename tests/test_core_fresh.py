"""Behavioral tests for the fresh Result core API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from typing import TYPE_CHECKING, Never, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from better_result._core import Err, Ok, Result, UnwrapError, is_err, is_ok


def test_variants_have_symmetric_shape_and_value_access() -> None:
    success = Ok(3)
    failure = Err("bad")

    assert repr(success) == "Ok(value=3)"
    assert repr(failure) == "Err(value='bad')"
    assert success == Ok(3)
    assert failure == Err("bad")
    assert success != failure
    assert success.is_ok() is True
    assert success.is_err() is False
    assert failure.is_ok() is False
    assert failure.is_err() is True
    assert success.ok() == 3
    assert success.err() is None
    assert failure.ok() is None
    assert failure.err() == "bad"
    assert success.ok_value == 3
    assert failure.err_value == "bad"
    assert is_ok(success)
    assert is_err(failure)


def test_variants_support_public_positional_pattern_matching() -> None:
    success = Ok(3)
    failure = Err("bad")

    match success:
        case Ok(value):
            assert value == 3
        case _:
            pytest.fail("Ok did not match its public value")

    match failure:
        case Err(error):
            assert error == "bad"
        case _:
            pytest.fail("Err did not match its public value")


def test_variants_are_frozen_dataclasses() -> None:
    assert is_dataclass(Ok)
    assert is_dataclass(Err)


def test_variants_are_immutable_and_unhashable() -> None:
    success = Ok(3)
    failure = Err("bad")

    with pytest.raises(FrozenInstanceError):
        success.__setattr__("value", 4)
    with pytest.raises(FrozenInstanceError):
        failure.__setattr__("value", "changed")

    assert Ok.__hash__ is None
    assert Err.__hash__ is None

    with pytest.raises(TypeError):
        hash(success)
    with pytest.raises(TypeError):
        hash(failure)
    with pytest.raises(TypeError):
        hash(Ok([]))
    with pytest.raises(TypeError):
        hash(Err([]))


def test_equality_does_not_cross_subclass_boundaries() -> None:
    class SpecialOk(Ok[int]):
        pass

    class SpecialErr(Err[str]):
        pass

    assert Ok(1) != SpecialOk(1)
    assert Err("bad") != SpecialErr("bad")


def test_active_and_inactive_sync_operations() -> None:
    success: Result[int, str] = Ok(2)
    failure: Result[int, str] = Err("bad")
    calls: list[str] = []

    assert success.map(lambda value: value * 2) == Ok(4)
    assert failure.map(lambda _: pytest.fail("map callback ran")) is failure
    assert success.map_err(lambda _: pytest.fail("map_err callback ran")) is success
    assert failure.map_err(str.upper) == Err("BAD")
    assert success.map_or(0, str) == "2"
    assert failure.map_or(0, str) == 0
    assert success.map_or_else(lambda: 0, str) == "2"
    assert failure.map_or_else(lambda: 0, str) == 0
    assert success.and_then(lambda value: Ok(str(value))) == Ok("2")
    assert failure.and_then(lambda _: pytest.fail("and_then callback ran")) is failure
    assert success.or_else(lambda _: pytest.fail("or_else callback ran")) is success
    assert failure.or_else(lambda error: Ok(error.upper())) == Ok("BAD")
    assert success.inspect(lambda value: calls.append(f"ok:{value}")) is success
    assert failure.inspect(lambda _: pytest.fail("inspect callback ran")) is failure
    assert (
        success.inspect_err(lambda _: pytest.fail("inspect_err callback ran"))
        is success
    )
    assert failure.inspect_err(lambda error: calls.append(f"err:{error}")) is failure
    assert calls == ["ok:2", "err:bad"]


def test_match_and_inspect_both_select_only_the_active_branch() -> None:
    success = Ok(2)
    failure = Err("bad")
    calls: list[str] = []

    assert success.match(ok=lambda value: value * 2, err=lambda _: 0) == 4
    assert failure.match(ok=lambda _: 0, err=str.upper) == "BAD"
    assert (
        success.inspect_both(
            ok=lambda value: calls.append(f"ok:{value}"),
            err=lambda _: pytest.fail("err callback ran"),
        )
        is success
    )
    assert (
        failure.inspect_both(
            ok=lambda _: pytest.fail("ok callback ran"),
            err=lambda error: calls.append(f"err:{error}"),
        )
        is failure
    )
    assert calls == ["ok:2", "err:bad"]


def test_unwrap_and_fallback_operations() -> None:
    success = Ok(3)
    failure = Err("bad")

    assert success.expect("unused") == 3
    assert failure.expect_err("unused") == "bad"
    assert success.unwrap() == 3
    assert failure.unwrap_err() == "bad"
    assert success.unwrap_or(0) == 3
    assert failure.unwrap_or(0) == 0
    assert success.unwrap_or_else(lambda _: 0) == 3
    assert failure.unwrap_or_else(len) == 3
    assert success.unwrap_or_raise(ValueError) == 3

    with pytest.raises(UnwrapError, match="custom message"):
        success.expect_err("custom message")
    with pytest.raises(UnwrapError, match="custom: 'bad'"):
        failure.expect("custom")
    with pytest.raises(UnwrapError, match="Result\\.unwrap\\(\\)"):
        failure.unwrap()
    with pytest.raises(UnwrapError, match="unwrap_err"):
        success.unwrap_err()
    with pytest.raises(ValueError, match="bad"):
        failure.unwrap_or_raise(ValueError)

    cause = ValueError("root cause")
    failure_with_exception = Err(cause)
    with pytest.raises(UnwrapError) as expect_info:
        failure_with_exception.expect("custom")
    assert expect_info.value.__cause__ is cause
    with pytest.raises(UnwrapError) as unwrap_info:
        failure_with_exception.unwrap()
    assert unwrap_info.value.__cause__ is cause


def test_and_then_rejects_non_result_callbacks() -> None:
    invalid = cast("Callable[[int], Result[object, object]]", lambda _: 42)
    with pytest.raises(TypeError, match="must return a Result"):
        Ok(1).and_then(invalid)


@pytest.mark.asyncio
async def test_async_operations_short_circuit_and_validate() -> None:
    success = Ok(2)
    failure = Err("bad")

    assert await success.map_async(lambda value: _constant(value * 2)) == Ok(4)
    assert (
        await failure.map_async(lambda _: pytest.fail("map_async callback ran"))
        is failure
    )

    async def map_error(error: str) -> int:
        return len(error)

    assert await failure.map_err_async(map_error) == Err(3)
    assert (
        await success.map_err_async(lambda _: pytest.fail("map_err_async callback ran"))
        is success
    )

    async def next_result(value: int) -> Result[str, Never]:
        return Ok(str(value))

    assert await success.and_then_async(next_result) == Ok("2")
    assert (
        await failure.and_then_async(
            lambda _: pytest.fail("and_then_async callback ran")
        )
        is failure
    )
    assert await failure.or_else_async(
        lambda error: _constant(Ok(error.upper()))
    ) == Ok("BAD")
    assert (
        await success.or_else_async(lambda _: pytest.fail("or_else_async callback ran"))
        is success
    )

    async_calls: list[str] = []
    assert (
        await success.inspect_async(
            lambda value: _constant(async_calls.append(f"ok:{value}")),
        )
        is success
    )
    assert (
        await failure.inspect_err_async(
            lambda error: _constant(async_calls.append(f"err:{error}")),
        )
        is failure
    )
    assert (
        await success.inspect_both_async(
            ok=lambda value: _constant(async_calls.append(f"both-ok:{value}")),
            err=lambda _: pytest.fail("async err callback ran"),
        )
        is success
    )
    assert async_calls == ["ok:2", "err:bad", "both-ok:2"]
    assert (
        await success.inspect_err_async(
            lambda _: pytest.fail("async inactive err callback ran"),
        )
        is success
    )
    assert (
        await failure.inspect_async(
            lambda _: pytest.fail("async inactive ok callback ran"),
        )
        is failure
    )
    assert (
        await failure.inspect_both_async(
            ok=lambda _: pytest.fail("async inactive both ok callback ran"),
            err=lambda error: _constant(async_calls.append(f"both-err:{error}")),
        )
        is failure
    )
    assert async_calls == ["ok:2", "err:bad", "both-ok:2", "both-err:bad"]

    invalid = cast(
        "Callable[[int], Awaitable[Result[object, object]]]",
        lambda _: _constant(42),
    )
    with pytest.raises(TypeError, match="must return a Result"):
        await success.and_then_async(invalid)


@pytest.mark.asyncio
async def test_callback_exceptions_propagate_as_defects() -> None:
    with pytest.raises(ZeroDivisionError):
        Ok(1).map(lambda _: _raise_zero_division())

    with pytest.raises(ZeroDivisionError):
        await Ok(1).map_async(lambda _: _raise_zero_division())

    with pytest.raises(RuntimeError, match="broken"):
        Ok(1).and_then(lambda _: _raise_runtime_error())

    with pytest.raises(RuntimeError, match="broken"):
        await Ok(1).and_then_async(lambda _: _raise_runtime_error_async())


def test_results_are_not_iterables() -> None:
    assert not hasattr(Ok(1), "__iter__")
    assert not hasattr(Err("bad"), "__iter__")


async def _constant[T](value: T) -> T:
    return value


def _raise_zero_division() -> Never:
    message = "broken"
    raise ZeroDivisionError(message)


def _raise_runtime_error() -> Never:
    message = "broken"
    raise RuntimeError(message)


async def _raise_runtime_error_async() -> Never:
    message = "broken"
    raise RuntimeError(message)
