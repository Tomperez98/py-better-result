"""Tests for the synchronous Result core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Never, assert_type, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

from better_result.core import (
    Err,
    Ok,
    PanicError,
    Result,
    is_err,
    is_ok,
    is_panic,
    panic,
)


def assert_ok(result: object, expected: object) -> None:
    assert isinstance(result, Ok)
    assert result.value == expected


def assert_err(result: object, expected: object) -> None:
    assert isinstance(result, Err)
    assert result.error == expected


def assert_panic_raised(
    fn: Callable[[], object],
    message_contains: str | None = None,
) -> PanicError:
    try:
        fn()
    except PanicError as error:
        if message_contains is not None and message_contains not in error.message:
            message = f"Expected {message_contains!r} in {error.message!r}"
            raise AssertionError(message) from None
        return error
    msg = "Expected Panic to be raised"
    raise AssertionError(msg)


def test_result_constructors_preserve_their_static_payload_types() -> None:
    assert_type(Ok(None), Ok[None])
    assert_type(Ok(int("42")), Ok[int])

    def missing() -> str:
        return "missing"

    assert_type(Err(missing()), Err[str])


def test_concrete_variants_fit_annotated_results_and_narrow_branches() -> None:
    def inspect(result: Result[int, str]) -> None:
        assert_type(result, Result[int, str])
        if isinstance(result, Ok):
            assert_type(result.value, int)
        if isinstance(result, Err):
            assert_type(result.error, str)

    def as_result(result: Result[int, str]) -> Result[int, str]:
        return result

    success: Result[int, str] = Ok(42)
    failure: Result[int, str] = Err("missing")
    assert_type(as_result(Ok(42)), Result[int, str])
    assert_type(as_result(Err("missing")), Result[int, str])
    inspect(success)
    inspect(failure)

    def handle_success(value: int) -> str:
        return str(value)

    def handle_failure(error: str) -> str:
        return error

    def match_result(result: Result[int, str]) -> str:
        return result.match(handle_success, handle_failure)

    assert_type(match_result(success), str)


def test_ok_and_err_constructors_expose_payload() -> None:
    success = Ok(42)
    failure = Err("missing")

    assert isinstance(success, Ok)
    assert success.value == 42
    assert isinstance(failure, Err)
    assert failure.error == "missing"
    assert Ok(None).value is None


def test_results_have_value_equality_and_repr() -> None:
    assert Ok(42) == Ok(42)
    assert Ok(42) != Ok(7)
    assert Err("missing") == Err("missing")
    assert Err("missing") != Err("other")
    assert Ok(42) != Err(42)
    assert Err(42) != Ok(42)
    assert repr(Ok(42)) == "Ok(42)"
    assert repr(Err("missing")) == "Err('missing')"


def test_guards_narrow_the_result_variants() -> None:
    success = Ok(42)
    failure = Err("missing")

    assert is_ok(success)
    assert not is_err(success)
    assert not is_ok(failure)
    assert is_err(failure)


def test_map_transforms_only_ok_values() -> None:
    assert_ok(Ok(2).map(lambda value: value * 3), 6)

    called = False

    def should_not_run(_: object) -> object:
        nonlocal called
        called = True
        return "unexpected"

    result = Err("bad").map(should_not_run)
    assert_err(result, "bad")
    assert not called


def test_map_error_transforms_only_err_values() -> None:
    assert_err(Err("bad").map_error(str.upper), "BAD")

    called = False

    def should_not_run(_: object) -> object:
        nonlocal called
        called = True
        return "unexpected"

    result = Ok(42).map_error(should_not_run)
    assert_ok(result, 42)
    assert not called


def test_try_recover_short_circuits_or_recovers() -> None:
    called = False

    def should_not_run(_: object) -> Result[str, str]:
        nonlocal called
        called = True
        return Err("unexpected")

    result = Ok(42).try_recover(should_not_run)
    assert_ok(result, 42)
    assert not called

    assert_ok(Err("missing").try_recover(lambda error: Ok(len(error))), 7)
    assert_err(
        Err("invalid").try_recover(lambda _: Err("still invalid")),
        "still invalid",
    )


def test_and_then_chains_ok_and_short_circuits_err() -> None:
    assert_ok(Ok(2).and_then(lambda value: Ok(value + 1)), 3)
    assert_err(Ok(2).and_then(lambda _: Err("rejected")), "rejected")

    called = False

    def should_not_run(_: object) -> Result[str, Never]:
        nonlocal called
        called = True
        return Ok("unexpected")

    result = Err("earlier").and_then(should_not_run)
    assert_err(result, "earlier")
    assert not called


def test_match_calls_only_the_active_handler() -> None:
    calls: list[str] = []

    def on_ok(value: int) -> int:
        calls.append(f"ok:{value}")
        return value * 2

    def on_err(error: str) -> int:
        calls.append(f"err:{error}")
        return 0

    assert Ok(3).match(on_ok, on_err) == 6
    assert calls == ["ok:3"]
    calls.clear()
    assert Err("bad").match(on_ok, on_err) == 0
    assert calls == ["err:bad"]


def test_unwrap_and_unwrap_or() -> None:
    assert Ok(42).unwrap() == 42
    assert Ok(42).unwrap("ignored") == 42
    assert Ok(42).unwrap_or("fallback") == 42
    assert Err("bad").unwrap_or(42) == 42

    panic_error = assert_panic_raised(lambda: Err("bad").unwrap())
    assert "Unwrap called on Err" in panic_error.message
    assert panic_error.cause == "bad"

    custom = assert_panic_raised(lambda: Err("bad").unwrap("custom message"))
    assert custom.message == "custom message"


def test_tap_methods_run_the_active_side_effect_and_return_self() -> None:
    calls: list[object] = []
    success = Ok(7)
    failure = Err("bad")

    assert success.tap(calls.append) is success
    assert calls == [7]
    assert failure.tap(calls.append) is failure
    assert calls == [7]

    assert success.tap_error(calls.append) is success
    assert calls == [7]
    assert failure.tap_error(calls.append) is failure
    assert calls == [7, "bad"]


def test_callback_failures_are_wrapped_in_panic() -> None:
    original = ValueError("broken")

    with pytest.raises(PanicError, match="map callback threw") as raised:
        Ok(1).map(lambda _: (_ for _ in ()).throw(original))

    panic_error = raised.value
    assert panic_error.cause is original
    assert panic_error.__cause__ is original

    def explode_map_error(_: str) -> int:
        raise ZeroDivisionError

    def explode_and_then(_: int) -> Result[object, object]:
        msg = "broken chain"
        raise RuntimeError(msg)

    def explode_match(_: int) -> int:
        msg = "broken match"
        raise RuntimeError(msg)

    def explode_tap(_: int) -> None:
        msg = "broken tap"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="map_error callback threw"):
        Err("bad").map_error(explode_map_error)
    with pytest.raises(PanicError, match="and_then callback threw"):
        Ok(1).and_then(explode_and_then)
    with pytest.raises(PanicError, match="match ok handler threw"):
        Ok(1).match(explode_match, lambda _: 0)
    with pytest.raises(PanicError, match="tap callback threw"):
        Ok(1).tap(explode_tap)


def test_panic_serializes_causes_and_has_a_guard() -> None:
    cause = ValueError("root cause")
    panic_error = PanicError("wrapped", cause=cause)
    serialized = panic_error.to_dict()

    assert is_panic(panic_error)
    assert is_panic(panic_error)
    assert not is_panic(ValueError("not a panic"))
    assert serialized["_tag"] == "Panic"
    assert serialized["name"] == "Panic"
    assert serialized["message"] == "wrapped"
    cause_data = serialized["cause"]
    assert isinstance(cause_data, dict)
    assert cause_data["name"] == "ValueError"
    assert cause_data["message"] == "root cause"
    assert PanicError("plain").to_json()["cause"] is None
    assert PanicError("plain", cause="text").to_dict()["cause"] == "text"
    assert "stack" in panic_error.to_json()
    assert "stack" not in panic_error.to_safe_json()
    assert panic_error.to_safe_json()["message"] == "wrapped"

    nested = PanicError(
        "nested",
        cause={"stack": "secret", "items": [{"stack": "nested secret"}]},
    ).to_safe_json()
    assert nested["cause"] == {"items": [{}]}
    assert panic_error.__cause__ is cause


def test_panic_function_always_raises() -> None:
    with pytest.raises(PanicError, match="fatal"):
        panic("fatal")


def test_result_iterators_support_yield_from_and_short_circuiting() -> None:
    success_iterator = iter(Ok(5))
    with pytest.raises(StopIteration) as success_stop:
        next(success_iterator)
    success_value = success_stop.value.value
    assert isinstance(success_value, int)
    assert success_value == 5

    failure_iterator = iter(Err("broken"))
    yielded = next(failure_iterator)
    assert_err(yielded, "broken")
    with pytest.raises(PanicError, match="Unreachable"):
        next(failure_iterator)

    def pipeline() -> Generator[object, None, Ok[int]]:
        value = yield from Ok(2)
        return Ok(value * 4)

    pipeline_iterator = pipeline()
    with pytest.raises(StopIteration) as pipeline_stop:
        next(pipeline_iterator)
    returned = pipeline_stop.value.value
    assert isinstance(returned, Ok)
    assert returned.value == 8


def test_assertion_helpers_fail_when_shape_is_wrong() -> None:
    assert_ok(Ok(1), 1)
    assert_err(Err("bad"), "bad")

    with pytest.raises(AssertionError):
        assert_ok(Err("bad"), 1)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        assert_err(Ok(1), "bad")  # type: ignore[arg-type]


def test_result_variants_are_immutable_and_guards_agree() -> None:
    success = Ok(1)
    failure = Err("bad")

    with pytest.raises(AttributeError):
        success.value = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        failure.error = "changed"  # ty: ignore[invalid-assignment]

    assert is_ok(success)
    assert not is_err(success)
    assert not is_ok(failure)
    assert is_err(failure)


def test_result_callbacks_must_return_results() -> None:
    def invalid_success(_: int) -> Result[object, object]:
        return cast("Result[object, object]", object())

    def invalid_error(_: str) -> Result[object, object]:
        return cast("Result[object, object]", object())

    with pytest.raises(PanicError, match="and_then callback must return a Result"):
        Ok(1).and_then(invalid_success)
    with pytest.raises(PanicError, match="try_recover callback must return a Result"):
        Err("bad").try_recover(invalid_error)

    with pytest.raises(PanicError, match="match callbacks must be callable"):
        Ok(1).match(lambda value: value, cast("Callable[[Never], int]", object()))
    with pytest.raises(TypeError):
        cast("Callable[[], object]", Err("bad").match)()
    with pytest.raises(PanicError, match="match callbacks must be callable"):
        Err("bad").match(cast("Callable[[Never], int]", object()), lambda _: 0)
