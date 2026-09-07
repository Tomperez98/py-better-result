"""Boundary and failure-path tests for complete coverage."""

from __future__ import annotations

import asyncio
import math
from typing import Any, cast

import pytest

from better_result.codec import SchemaFailure, codec, codec_config
from better_result.collections import all_results_async
from better_result.combinators import (
    and_then,
    and_then_async,
    map_error,
    map_result,
    match,
    tap,
    tap_async,
    tap_both,
    tap_both_async,
    tap_error,
    tap_error_async,
    try_recover,
    try_recover_async,
    unwrap_or,
)
from better_result.core import Err, Ok, Panic, assert_panic_raised, err, ok
from better_result.dual import dual
from better_result.error import (
    TaggedError,
    UnhandledException,
    match_error,
    match_error_partial,
    tagged_error,
)
from better_result.retry import (
    AsyncRetryConfig,
    RetryConfig,
    TryAsyncContext,
    TryContext,
    _jitter_factor,
    _retry_times,
    _sleep_for_retry,
    _static_retry_delay,
    _validate_delay,
    try_async,
    try_result,
)


class ValidatingSchema:
    def validate(self, value: object) -> str:
        return str(value)


def test_codec_schema_objects_and_schema_failure_branches() -> None:
    result_codec = codec(
        codec_config(
            serialize_ok=ValidatingSchema(),
            serialize_err=lambda value: SchemaFailure(({"value": value},)),
            deserialize_ok=ValidatingSchema(),
            deserialize_err=lambda value: SchemaFailure(({"value": value},)),
        ),
    )

    serialized = result_codec.serialize(ok(7))
    assert isinstance(serialized, Ok)
    assert serialized.value == {"status": "ok", "value": "7"}

    serialization_failure = result_codec.serialize(err("bad"))
    assert isinstance(serialization_failure, Err)
    assert serialization_failure.error.value == "bad"

    deserialized = result_codec.deserialize({"status": "ok", "value": 7})
    assert isinstance(deserialized, Ok)
    assert deserialized.value == "7"

    deserialization_failure = result_codec.deserialize(
        {"status": "error", "error": "bad"}
    )
    assert isinstance(deserialization_failure, Err)
    assert deserialization_failure.error.value == "bad"


def test_codec_wraps_sync_and_async_schema_defects() -> None:
    def sync_failure(_value: int) -> str:
        raise RuntimeError("sync schema failed")

    def sync_panic(_value: int) -> str:
        raise Panic("already a panic")

    with pytest.raises(Panic, match="serialize schema threw"):
        codec(
            codec_config(
                serialize_ok=sync_failure,
                serialize_err=str,
                deserialize_ok=str,
                deserialize_err=str,
            ),
        ).serialize(ok(1))
    with pytest.raises(Panic, match="already a panic"):
        codec(
            codec_config(
                serialize_ok=sync_panic,
                serialize_err=str,
                deserialize_ok=str,
                deserialize_err=str,
            ),
        ).serialize(ok(1))

    async def async_failure(_value: int) -> str:
        raise RuntimeError("async schema failed")

    async def async_panic(_value: int) -> str:
        raise Panic("already a panic")

    async def exercise() -> None:
        with pytest.raises(Panic, match="serialize schema threw"):
            await codec(
                codec_config(
                    serialize_ok=async_failure,
                    serialize_err=str,
                    deserialize_ok=str,
                    deserialize_err=str,
                ),
            ).serialize_async(ok(1))
        with pytest.raises(Panic, match="already a panic"):
            await codec(
                codec_config(
                    serialize_ok=async_panic,
                    serialize_err=str,
                    deserialize_ok=str,
                    deserialize_err=str,
                ),
            ).serialize_async(ok(1))

    asyncio.run(exercise())


def test_codec_async_helpers_accept_sync_operations_and_unsafe_success() -> None:
    result_codec = codec(
        codec_config(
            serialize_ok=str,
            serialize_err=str,
            deserialize_ok=int,
            deserialize_err=str,
        ),
    )

    serialized = asyncio.run(result_codec.serialize_async(ok(3)))
    assert isinstance(serialized, Ok)
    deserialized = asyncio.run(
        result_codec.deserialize_async({"status": "ok", "value": "3"})
    )
    assert isinstance(deserialized, Ok)
    assert deserialized.value == 3

    unsafe = result_codec.deserialize_unsafe({"status": "ok", "value": "3"})
    assert isinstance(unsafe, Ok)
    assert unsafe.value == 3


def test_panic_iteration_and_assertion_helper_failures() -> None:
    panic_error = Panic("broken")
    iterator = iter(panic_error)
    yielded = next(iterator)
    assert isinstance(yielded, Err)
    with pytest.raises(Panic, match="Unreachable"):
        next(iterator)

    nested: dict[str, object] = {}
    nested["self"] = nested
    panic_json = Panic("cycle", cause=[ValueError("nested")]).to_json()
    assert isinstance(panic_json["cause"], list)
    assert Panic("cycle", cause=nested).to_json()["cause"] == {"self": "<cycle>"}

    with pytest.raises(AssertionError, match="Expected 'wanted'"):
        assert_panic_raised(lambda: (_ for _ in ()).throw(Panic("actual")), "wanted")
    with pytest.raises(AssertionError, match="Expected Panic"):
        assert_panic_raised(lambda: None)


def test_dual_partial_callback_rejects_wrong_arity() -> None:
    partial = cast("Any", dual(2, lambda left, right: left + right)(1))
    with pytest.raises(TypeError, match="expected 1 arguments"):
        partial()
    with pytest.raises(TypeError, match="expected 1 arguments"):
        partial(2, 3)


def test_tagged_error_validation_and_matching_failure_paths() -> None:
    with pytest.raises(TypeError, match="must not define _tag"):

        class InvalidTag(TaggedError, tag="InvalidTag"):
            _tag = "shadowed"

    with pytest.raises(ValueError, match="tag must not be empty"):
        tagged_error("")

    runtime_match_error = cast("Any", match_error)
    with pytest.raises(TypeError, match="handlers must be a mapping"):
        runtime_match_error(object())
    with pytest.raises(TypeError, match="error must be a TaggedError"):
        runtime_match_error(object(), {})

    matcher = cast("Any", match_error({"TaggedError": lambda error: error.message}))
    with pytest.raises(Panic, match="match_error handler threw"):
        matcher(object())

    def panic_handler(_error: TaggedError) -> object:
        raise Panic("bug")

    with pytest.raises(Panic, match="bug"):
        match_error(TaggedError(message="x"), {"TaggedError": panic_handler})

    runtime_partial_match = cast("Any", match_error_partial)
    with pytest.raises(TypeError, match="on_unhandled must be callable"):
        runtime_partial_match({}, {})
    with pytest.raises(TypeError, match="handlers must be a mapping"):
        runtime_partial_match(TaggedError(message="x"), None)

    def partial_panic_handler(_error: TaggedError) -> object:
        raise Panic("partial bug")

    with pytest.raises(Panic, match="partial bug"):
        match_error_partial(
            TaggedError(message="x"), {"TaggedError": partial_panic_handler}
        )

    assert UnhandledException(None).message == "Unhandled exception: null"
    assert UnhandledException(cause=True).message == "Unhandled exception: true"
    assert UnhandledException(cause=False).message == "Unhandled exception: false"


def test_retry_helpers_cover_validation_and_backoff() -> None:
    assert _retry_times(None) == 0
    assert _retry_times(RetryConfig(times=2)) == 2
    with pytest.raises(ValueError, match="must not be negative"):
        _retry_times(-1)
    with pytest.raises(ValueError, match="must not be negative"):
        _retry_times(RetryConfig(times=-1))

    assert _static_retry_delay(10, "constant", 2) == 10
    assert _static_retry_delay(10, "linear", 2) == 30
    assert _static_retry_delay(10, "exponential", 2) == 40
    assert _jitter_factor(jitter=True) == 1
    assert _jitter_factor(jitter=False) == 0
    assert _jitter_factor(jitter=0.25) == 0.25
    with pytest.raises(Panic, match="jitter"):
        _jitter_factor(jitter=2)
    with pytest.raises(Panic, match="finite"):
        _validate_delay(math.inf)
    with pytest.raises(Panic, match="number"):
        _validate_delay(cast("Any", "not a number"))


def test_try_result_breaks_after_a_retry_succeeds() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> int:
        attempts.append(context.attempt)
        if context.attempt == 1:
            raise ValueError("retry")
        return 42

    result = try_result(operation, retry=RetryConfig(times=2))
    assert isinstance(result, Ok)
    assert attempts == [1, 2]

    def panic_catch(_cause: BaseException) -> str:
        raise Panic("catch panic")

    with pytest.raises(Panic, match="catch panic"):
        try_result(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")), panic_catch
        )


@pytest.mark.asyncio
async def test_retry_sleep_and_async_retry_failure_paths() -> None:
    event = asyncio.Event()
    assert await _sleep_for_retry(0, event) is True
    event.set()
    assert await _sleep_for_retry(1, event) is False

    event.clear()
    assert await _sleep_for_retry(0.001, event) is True

    async def set_event() -> None:
        await asyncio.sleep(0)
        event.set()

    event.clear()
    setter = asyncio.create_task(set_event())
    assert await _sleep_for_retry(100, event) is False
    await setter

    async def operation(_context: TryAsyncContext) -> int:
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="must not be negative"):
        await try_async(operation, retry=AsyncRetryConfig(times=-1))
    with pytest.raises(Panic, match="finite"):
        await try_async(operation, retry=AsyncRetryConfig(delay_ms=-1))

    async def cancelled_operation(_context: TryAsyncContext) -> int:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(cancelled_operation)

    async def cancelled_catch(_cause: BaseException) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(operation, cancelled_catch)

    async def panic_catch(_cause: BaseException) -> str:
        raise Panic("catch panic")

    with pytest.raises(Panic, match="catch panic"):
        await try_async(operation, panic_catch)

    async def bad_predicate(_error: object, _context: TryAsyncContext) -> bool:
        raise RuntimeError("predicate failed")

    with pytest.raises(Panic, match="should_retry predicate threw"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, should_retry=bad_predicate),
        )

    async def cancelled_predicate(_error: object, _context: TryAsyncContext) -> bool:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, should_retry=cancelled_predicate),
        )

    async def panic_predicate(_error: object, _context: TryAsyncContext) -> bool:
        raise Panic("predicate panic")

    with pytest.raises(Panic, match="predicate panic"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, should_retry=panic_predicate),
        )

    def bad_delay(_error: object, _context: TryAsyncContext) -> float:
        raise RuntimeError("delay failed")

    with pytest.raises(Panic, match="delay_ms callback threw"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, delay_ms=bad_delay),
        )

    async def cancelled_delay(_error: object, _context: TryAsyncContext) -> float:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, delay_ms=cancelled_delay),
        )

    async def panic_delay(_error: object, _context: TryAsyncContext) -> float:
        raise Panic("delay panic")

    with pytest.raises(Panic, match="delay panic"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, delay_ms=panic_delay),
        )


@pytest.mark.asyncio
async def test_async_result_wrapper_forms_and_type_errors() -> None:
    success = ok(2)
    failure = err("bad")

    async def async_identity(value: object) -> object:
        return await _completed(value)

    def return_result(value: object) -> Ok[object, Any]:
        return ok(value)

    runtime_map = cast("Any", map_result)
    runtime_map_error = cast("Any", map_error)
    runtime_try_recover = cast("Any", try_recover)
    runtime_try_recover_async = cast("Any", try_recover_async)
    runtime_and_then = cast("Any", and_then)
    runtime_and_then_async = cast("Any", and_then_async)
    runtime_match = cast("Any", match)
    runtime_tap = cast("Any", tap)
    runtime_tap_async = cast("Any", tap_async)
    runtime_tap_error = cast("Any", tap_error)
    runtime_tap_error_async = cast("Any", tap_error_async)
    runtime_tap_both = cast("Any", tap_both)
    runtime_tap_both_async = cast("Any", tap_both_async)
    runtime_unwrap_or = cast("Any", unwrap_or)

    with pytest.raises(TypeError):
        runtime_map(success)
    with pytest.raises(TypeError):
        runtime_map(return_result, success)
    with pytest.raises(TypeError):
        runtime_map_error(failure)
    with pytest.raises(TypeError):
        runtime_map_error(return_result, failure)
    with pytest.raises(TypeError):
        runtime_try_recover(failure)
    with pytest.raises(TypeError):
        runtime_try_recover(return_result, failure)
    with pytest.raises(TypeError):
        runtime_try_recover_async(success)
    with pytest.raises(TypeError):
        runtime_try_recover_async(async_identity, failure)
    with pytest.raises(TypeError):
        runtime_and_then(success)
    with pytest.raises(TypeError):
        runtime_and_then(return_result, success)
    with pytest.raises(TypeError):
        runtime_and_then_async(success)
    with pytest.raises(TypeError):
        runtime_and_then_async(async_identity, success)
    with pytest.raises(TypeError):
        runtime_match(success)
    with pytest.raises(TypeError):
        runtime_match({}, success)
    with pytest.raises(TypeError):
        runtime_tap(success)
    with pytest.raises(TypeError):
        runtime_tap(return_result, success)
    with pytest.raises(TypeError):
        runtime_tap_async(success)
    with pytest.raises(TypeError):
        runtime_tap_async(async_identity, success)
    with pytest.raises(TypeError):
        runtime_tap_error(failure)
    with pytest.raises(TypeError):
        runtime_tap_error(return_result, failure)
    with pytest.raises(TypeError):
        runtime_tap_error_async(failure)
    with pytest.raises(TypeError):
        runtime_tap_error_async(async_identity, failure)
    with pytest.raises(TypeError):
        runtime_tap_both(success)
    with pytest.raises(TypeError):
        runtime_tap_both({}, success)
    with pytest.raises(TypeError):
        runtime_tap_both_async(success)
    with pytest.raises(TypeError):
        runtime_tap_both_async({}, success)
    with pytest.raises(TypeError):
        runtime_unwrap_or(success)
    with pytest.raises(TypeError):
        runtime_unwrap_or(42, success)

    assert isinstance(map_result(lambda value: value + 1)(success), Ok)
    assert isinstance(map_error(str.upper)(failure), Err)
    assert isinstance(try_recover(lambda value: ok(len(value)))(failure), Ok)
    assert isinstance(
        await try_recover_async(lambda value: _completed(ok(len(value))))(failure), Ok
    )
    assert isinstance(and_then(lambda value: ok(value + 1))(success), Ok)
    assert isinstance(
        await and_then_async(lambda value: _completed(ok(value + 1)))(success), Ok
    )
    assert match({"ok": lambda value: value, "err": lambda value: value})(success) == 2
    assert tap(lambda _value: None)(success) is success
    assert await tap_async(lambda _value: _completed(None))(success) is success
    assert tap_error(lambda _value: None)(failure) is failure
    assert await tap_error_async(lambda _value: _completed(None))(failure) is failure
    handlers = {"ok": lambda _value: None, "err": lambda _value: None}
    assert tap_both(success, handlers) is success
    assert tap_both(handlers)(success) is success
    assert (
        await tap_both_async({"ok": _completed, "err": _completed})(failure) is failure
    )
    assert unwrap_or(99)(failure) == 99


@pytest.mark.asyncio
async def test_all_async_cancels_for_cancelled_and_panic_inputs() -> None:
    async def cancelled() -> Ok[int, str]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await all_results_async([cancelled()])

    async def panicking() -> Ok[int, str]:
        raise Panic("input panic")

    with pytest.raises(Panic, match="input panic"):
        await all_results_async([panicking()])


async def _completed[T](value: T) -> T:
    await asyncio.sleep(0)
    return value
