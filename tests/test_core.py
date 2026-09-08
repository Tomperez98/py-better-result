"""Tests for the narrow synchronous Result API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Never, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from better_result import Err, Ok, PanicError, Result


def test_result_construction_equality_and_immutability() -> None:
    success = Ok(42)
    failure = Err("missing")

    assert success == Ok(42)
    assert failure == Err("missing")
    assert repr(success) == "Ok(42)"
    assert repr(failure) == "Err('missing')"
    with pytest.raises(AttributeError, match="immutable"):
        success.__setattr__("value", 7)
    with pytest.raises(AttributeError, match="immutable"):
        failure.__setattr__("error", "changed")


def test_map_and_map_error_only_touch_the_active_branch() -> None:
    success: Result[int, str] = Ok(2)
    failure: Result[int, str] = Err("bad")

    mapped = success.map(lambda value: value * 2)
    mapped_error = failure.map_error(str.upper)

    assert mapped == Ok(4)
    assert mapped_error == Err("BAD")
    assert success.map_error(lambda _: "unused") is success
    assert failure.map(lambda _: 99) is failure


def test_and_then_short_circuits_and_validates_callbacks() -> None:
    assert Ok(2).and_then(lambda value: Ok(str(value))) == Ok("2")
    assert Err("bad").and_then(lambda _: Ok("unused")) == Err("bad")

    invalid = cast("Callable[[int], Result[object, object]]", lambda _: "not a Result")
    with pytest.raises(PanicError, match="must return a Result"):
        Ok(1).and_then(invalid)

    with pytest.raises(PanicError, match="callback threw"):
        Ok(1).and_then(lambda _: (_ for _ in ()).throw(RuntimeError("broken")))


def test_match_is_the_canonical_consumer() -> None:
    result: Result[int, str] = Ok(3)
    failure: Result[int, str] = Err("bad")

    assert result.match(lambda value: f"value={value}", lambda _: "error") == "value=3"
    assert (
        failure.match(lambda _: "value", lambda error: f"error={error}") == "error=bad"
    )

    invalid_ok = cast("Callable[[int], str]", object())
    invalid_err = cast("Callable[[Never], str]", object())
    with pytest.raises(PanicError, match="callbacks must be callable"):
        result.match(invalid_ok, invalid_err)
    with pytest.raises(PanicError, match="handler threw"):
        result.match(
            lambda _: (_ for _ in ()).throw(RuntimeError("broken")), lambda _: "error"
        )
    with pytest.raises(PanicError, match="callbacks must be callable"):
        failure.match(
            cast("Callable[[Never], str]", object()),
            cast("Callable[[str], str]", object()),
        )


def test_unwrap_or_is_the_only_fallback_operation() -> None:
    assert Ok(1).unwrap_or(99) == 1
    assert Err("bad").unwrap_or(99) == 99


def test_panic_serialization_preserves_cause_and_safe_removes_stacks() -> None:
    cause = ValueError("broken")
    error = PanicError("pipeline failed", cause)

    assert error.cause is cause
    assert error.__cause__ is cause
    assert error.to_json()["message"] == "pipeline failed"
    assert "stack" in error.to_json()
    assert "stack" not in error.to_safe_json()
    assert isinstance(PanicError("object", cause=object()).to_json()["cause"], str)


def test_result_types_are_concrete() -> None:
    success: Result[int, Never] = Ok(2)
    assert success.map(str) == Ok("2")
