"""Tests for the synchronous Result core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Never, assert_type, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from better_result.core import (
    Err,
    Ok,
    Panic,
    Result,
    assert_err,
    assert_ok,
    assert_panic_raised,
    err,
    is_err,
    is_error,
    is_ok,
    is_panic,
    ok,
    panic,
)


def test_result_constructors_preserve_their_static_payload_types() -> None:
    assert_type(ok(), Ok[None, Never])
    assert_type(ok(42), Ok[int, Never])
    assert_type(err("missing"), Err[Never, str])


def test_ok_and_err_constructors_expose_status_and_payload() -> None:
    success = ok(42)
    failure = err("missing")

    assert isinstance(success, Ok)
    assert success.status == "ok"
    assert success.value == 42
    assert isinstance(failure, Err)
    assert failure.status == "error"
    assert failure.error == "missing"
    assert ok().value is None


def test_guards_narrow_the_result_variants() -> None:
    success = ok(42)
    failure = err("missing")

    assert is_ok(success)
    assert not is_err(success)
    assert not is_error(success)
    assert not is_ok(failure)
    assert is_err(failure)
    assert is_error(failure)
    assert success.is_ok()
    assert not success.is_err()
    assert not failure.is_ok()
    assert failure.is_err()


def test_map_transforms_only_ok_values() -> None:
    assert_ok(ok(2).map(lambda value: value * 3), 6)

    called = False

    def should_not_run(_: object) -> object:
        nonlocal called
        called = True
        return "unexpected"

    result = err("bad").map(should_not_run)
    assert_err(result, "bad")
    assert not called


def test_map_error_transforms_only_err_values() -> None:
    assert_err(err("bad").map_error(str.upper), "BAD")

    called = False

    def should_not_run(_: object) -> object:
        nonlocal called
        called = True
        return "unexpected"

    result = ok(42).map_error(should_not_run)
    assert_ok(result, 42)
    assert not called


def test_try_recover_short_circuits_or_recovers() -> None:
    called = False

    def should_not_run(_: object) -> Result[str, str]:
        nonlocal called
        called = True
        return err("unexpected")

    result = ok(42).try_recover(should_not_run)
    assert_ok(result, 42)
    assert not called

    assert_ok(err("missing").try_recover(lambda error: ok(len(error))), 7)
    assert_err(
        err("invalid").try_recover(lambda _: err("still invalid")),
        "still invalid",
    )


def test_and_then_chains_ok_and_short_circuits_err() -> None:
    assert_ok(ok(2).and_then(lambda value: ok(value + 1)), 3)
    assert_err(ok(2).and_then(lambda _: err("rejected")), "rejected")

    called = False

    def should_not_run(_: object) -> Result[str, Never]:
        nonlocal called
        called = True
        return ok("unexpected")

    result = err("earlier").and_then(should_not_run)
    assert_err(result, "earlier")
    assert not called


def test_match_calls_only_the_active_handler() -> None:
    calls: list[str] = []
    handlers = {
        "ok": lambda value: calls.append(f"ok:{value}") or value * 2,
        "err": lambda error: calls.append(f"err:{error}") or 0,
    }

    assert ok(3).match(handlers) == 6
    assert calls == ["ok:3"]
    calls.clear()
    assert err("bad").match(handlers) == 0
    assert calls == ["err:bad"]


def test_unwrap_and_unwrap_or() -> None:
    assert ok(42).unwrap() == 42
    assert ok(42).unwrap("ignored") == 42
    assert ok(42).unwrap_or("fallback") == 42
    assert err("bad").unwrap_or(42) == 42

    panic_error = assert_panic_raised(lambda: err("bad").unwrap())
    assert "Unwrap called on Err" in panic_error.message
    assert panic_error.cause == "bad"

    custom = assert_panic_raised(lambda: err("bad").unwrap("custom message"))
    assert custom.message == "custom message"


def test_tap_methods_run_the_active_side_effect_and_return_self() -> None:
    calls: list[object] = []
    success = ok(7)
    failure = err("bad")

    assert success.tap(calls.append) is success
    assert calls == [7]
    assert failure.tap(calls.append) is failure
    assert calls == [7]

    assert success.tap_error(calls.append) is success
    assert calls == [7]
    assert failure.tap_error(calls.append) is failure
    assert calls == [7, "bad"]

    assert success.tap_both({"ok": calls.append, "err": calls.append}) is success
    assert failure.tap_both({"ok": calls.append, "err": calls.append}) is failure
    assert calls[-2:] == [7, "bad"]


def test_callback_failures_are_wrapped_in_panic() -> None:
    original = ValueError("broken")

    with pytest.raises(Panic, match="map callback threw") as raised:
        ok(1).map(lambda _: (_ for _ in ()).throw(original))

    panic_error = raised.value
    assert panic_error.cause is original
    assert panic_error.__cause__ is original

    def explode_map_error(_: str) -> int:
        raise ZeroDivisionError

    def explode_and_then(_: int) -> Result[object, object]:
        raise RuntimeError("broken chain")

    def explode_match(_: int) -> int:
        raise RuntimeError("broken match")

    def explode_tap(_: int) -> None:
        raise RuntimeError("broken tap")

    with pytest.raises(Panic, match="map_error callback threw"):
        err("bad").map_error(explode_map_error)
    with pytest.raises(Panic, match="and_then callback threw"):
        ok(1).and_then(explode_and_then)
    with pytest.raises(Panic, match="match ok handler threw"):
        ok(1).match({"ok": explode_match, "err": lambda _: 0})
    with pytest.raises(Panic, match="tap callback threw"):
        ok(1).tap(explode_tap)


def test_panic_serializes_causes_and_has_a_guard() -> None:
    cause = ValueError("root cause")
    panic_error = Panic("wrapped", cause=cause)
    serialized = panic_error.to_dict()

    assert is_panic(panic_error)
    assert Panic.is_panic(panic_error)
    assert not is_panic(ValueError("not a panic"))
    assert serialized["_tag"] == "Panic"
    assert serialized["name"] == "Panic"
    assert serialized["message"] == "wrapped"
    cause_data = serialized["cause"]
    assert isinstance(cause_data, dict)
    assert cause_data["name"] == "ValueError"
    assert cause_data["message"] == "root cause"
    assert Panic("plain").to_json()["cause"] is None
    assert Panic("plain", cause="text").to_dict()["cause"] == "text"
    assert panic_error.__cause__ is cause


def test_panic_function_always_raises() -> None:
    with pytest.raises(Panic, match="fatal"):
        panic("fatal")


def test_result_iterators_support_yield_from_and_short_circuiting() -> None:
    success_iterator = iter(ok(5))
    with pytest.raises(StopIteration) as success_stop:
        next(success_iterator)
    success_value = success_stop.value.value
    assert isinstance(success_value, int)
    assert success_value == 5

    failure_iterator = iter(err("broken"))
    yielded = next(failure_iterator)
    assert_err(yielded, "broken")
    with pytest.raises(Panic, match="Unreachable"):
        next(failure_iterator)

    def pipeline() -> Generator[object, None, Ok[int, Never]]:
        value = yield from ok(2)
        return ok(value * 4)

    pipeline_iterator = pipeline()
    with pytest.raises(StopIteration) as pipeline_stop:
        next(pipeline_iterator)
    returned = pipeline_stop.value.value
    assert isinstance(returned, Ok)
    assert returned.value == 8


def test_assertion_helpers_fail_when_shape_is_wrong() -> None:
    assert_ok(ok(1), 1)
    assert_err(err("bad"), "bad")

    with pytest.raises(AssertionError):
        assert_ok(err("bad"), 1)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        assert_err(ok(1), "bad")  # type: ignore[arg-type]


def test_result_variants_are_immutable_and_guards_agree() -> None:
    success = ok(1)
    failure = err("bad")

    with pytest.raises(AttributeError):
        success.value = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        success.status = "error"  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        failure.error = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        failure.status = "ok"  # ty: ignore[invalid-assignment]

    assert is_ok(success)
    assert not is_err(success)
    assert not is_ok(failure)
    assert is_err(failure)


def test_result_callbacks_must_return_results() -> None:
    def invalid_success(_: int) -> Result[object, object]:
        return cast("Result[object, object]", object())

    def invalid_error(_: str) -> Result[object, object]:
        return cast("Result[object, object]", object())

    with pytest.raises(Panic, match="and_then callback must return a Result"):
        ok(1).and_then(invalid_success)
    with pytest.raises(Panic, match="try_recover callback must return a Result"):
        err("bad").try_recover(invalid_error)
