"""Behavioral tests for the std::result::Result-shaped core API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from typing import Never, cast

import pytest

from better_result._core import Err, Ok, Result, is_err, is_ok


def test_variants_have_symmetric_shape_and_value_access() -> None:
    success = Ok(3)
    failure = Err("bad")

    assert repr(success) == "Ok(value=3)"
    assert repr(failure) == "Err(value='bad')"
    assert success == Ok(3)
    assert failure == Err("bad")
    assert success != failure
    assert success.ok() == 3
    assert success.err() is None
    assert failure.ok() is None
    assert failure.err() == "bad"
    assert success.is_ok()
    assert not success.is_err()
    assert not failure.is_ok()
    assert failure.is_err()


def test_variants_support_pattern_matching_and_are_immutable() -> None:
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

    assert is_dataclass(Ok)
    assert is_dataclass(Err)
    with pytest.raises(FrozenInstanceError):
        success.__setattr__("value", 4)
    assert Ok.__hash__ is None
    assert Err.__hash__ is None
    with pytest.raises(TypeError):
        hash(success)


def test_type_guards_identify_the_active_variant() -> None:
    success: Result[int, str] = Ok(1)
    failure: Result[int, str] = Err("bad")

    assert is_ok(success)
    assert not is_err(success)
    assert is_err(failure)
    assert not is_ok(failure)


def test_result_operations_only_run_on_the_active_branch() -> None:
    success: Result[int, str] = Ok(2)
    failure: Result[int, str] = Err("bad")
    calls: list[str] = []

    assert success.is_ok_and(lambda value: value == 2)
    assert not failure.is_ok_and(lambda _: pytest.fail("inactive callback"))
    assert failure.is_err_and(lambda error: error == "bad")
    assert not success.is_err_and(lambda _: pytest.fail("inactive callback"))

    assert success.map(lambda value: value * 2) == Ok(4)
    assert failure.map(lambda _: pytest.fail("map callback ran")) is failure
    assert success.map_err(lambda _: pytest.fail("map_err callback ran")) is success
    assert failure.map_err(str.upper) == Err("BAD")
    assert success.map_or(0, str) == "2"
    assert failure.map_or(0, str) == 0
    assert success.map_or_else(lambda: 0, str) == "2"
    assert failure.map_or_else(lambda: 3, str) == 3
    assert success.and_(Err("ignored")) == Err("ignored")
    assert failure.and_(Ok(3)) is failure
    assert success.and_then(lambda value: Ok(str(value))) == Ok("2")
    assert failure.and_then(lambda _: pytest.fail("and_then callback ran")) is failure
    assert success.or_(Err("ignored")) is success
    assert failure.or_(Ok(3)) == Ok(3)
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


def test_unwrap_and_fallback_operations_match_result_semantics() -> None:
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
    assert success.unwrap_or_default() == 3
    assert failure.unwrap_or_default() is None


def test_wrong_variant_unwrap_operations_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_abort() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("better_result._core.os.abort", fake_abort)

    assert Ok(1).unwrap_err() is None
    assert Ok(1).expect_err("unused") is None
    assert Err("bad").unwrap() is None
    assert Err("bad").expect("unused") is None
    assert calls == 4


def test_callbacks_and_invalid_result_values_fail_fast() -> None:
    with pytest.raises(ZeroDivisionError):
        Ok(1).map(_division_by_zero)
    with pytest.raises(TypeError, match="must return a Result"):
        Ok(1).and_then(_invalid_next)
    with pytest.raises(TypeError, match="must return a Result"):
        Err("bad").or_else(_invalid_recovery)


def test_results_are_not_iterables_and_subclasses_do_not_compare_equal() -> None:
    assert not hasattr(Ok(1), "__iter__")
    assert not hasattr(Err("bad"), "__iter__")

    class SpecialOk(Ok[int]):
        pass

    class SpecialErr(Err[str]):
        pass

    assert Ok(1) != SpecialOk(1)
    assert Err("bad") != SpecialErr("bad")


def _division_by_zero(_: int) -> int:
    message = "division by zero"
    raise ZeroDivisionError(message)


def _invalid_next(_: int) -> Result[int, str]:
    return cast("Result[int, str]", cast("object", 42))


def _invalid_recovery(_: str) -> Result[int, str]:
    return cast("Result[int, str]", cast("object", 42))


def _never(_: Never) -> None:
    message = "unreachable"
    raise AssertionError(message)
