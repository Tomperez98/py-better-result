"""Boundary and failure-path tests for complete coverage."""

from __future__ import annotations

import asyncio
import math

import pytest

from better_result import Err, Ok, PanicError, TaggedError
from better_result._codec import SchemaFailure, async_codec, codec
from better_result._collections import all_results_async
from better_result._error import ResultCodecIssue, UnhandledError
from better_result._retry import (
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


def value_issue(value: object) -> ResultCodecIssue:
    return {"value": value}


def test_codec_schema_objects_and_schema_failure_branches() -> None:
    result_codec = codec(
        serialize_ok=ValidatingSchema(),
        serialize_err=lambda value: SchemaFailure((value_issue(value),)),
        deserialize_ok=ValidatingSchema(),
        deserialize_err=lambda value: SchemaFailure((value_issue(value),)),
    )

    serialized = result_codec.serialize(Ok(7))
    assert isinstance(serialized, Ok)
    assert serialized.value == {"status": "ok", "value": "7"}

    serialization_failure = result_codec.serialize(Err("bad"))
    assert isinstance(serialization_failure, Err)
    assert serialization_failure.error.value == "bad"

    deserialized = result_codec.deserialize({"status": "ok", "value": 7})
    assert isinstance(deserialized, Ok)
    assert deserialized.value == "7"

    deserialization_failure = result_codec.deserialize(
        {"status": "error", "error": "bad"},
    )
    assert isinstance(deserialization_failure, Err)
    assert deserialization_failure.error.value == "bad"


def test_codec_wraps_sync_and_async_schema_defects() -> None:
    def sync_failure(_value: int) -> str:
        msg = "sync schema failed"
        raise RuntimeError(msg)

    def sync_panic(_value: int) -> str:
        msg = "already a panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="serialize schema threw"):
        codec(
            serialize_ok=sync_failure,
            serialize_err=str,
            deserialize_ok=str,
            deserialize_err=str,
        ).serialize(Ok(1))
    with pytest.raises(PanicError, match="already a panic"):
        codec(
            serialize_ok=sync_panic,
            serialize_err=str,
            deserialize_ok=str,
            deserialize_err=str,
        ).serialize(Ok(1))

    async def async_failure(_value: int) -> str:
        msg = "async schema failed"
        raise RuntimeError(msg)

    async def async_panic(_value: int) -> str:
        msg = "already a panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="received an async schema"):
        codec(
            serialize_ok=async_failure,
            serialize_err=str,
            deserialize_ok=str,
            deserialize_err=str,
        ).serialize(Ok(1))

    class FutureLike:
        def __await__(self) -> object:
            if False:
                yield None
            return "value"

    with pytest.raises(PanicError, match="received an async schema"):
        codec(
            serialize_ok=lambda _value: FutureLike(),
            serialize_err=str,
            deserialize_ok=str,
            deserialize_err=str,
        ).serialize(Ok(1))

    async def exercise() -> None:
        with pytest.raises(PanicError, match="serialize schema threw"):
            await async_codec(
                serialize_ok=async_failure,
                serialize_err=str,
                deserialize_ok=str,
                deserialize_err=str,
            ).serialize(Ok(1))
        with pytest.raises(PanicError, match="already a panic"):
            await async_codec(
                serialize_ok=async_panic,
                serialize_err=str,
                deserialize_ok=str,
                deserialize_err=str,
            ).serialize(Ok(1))

    asyncio.run(exercise())


def test_async_codec_accepts_sync_schemas() -> None:
    result_codec = async_codec(
        serialize_ok=str,
        serialize_err=str,
        deserialize_ok=int,
        deserialize_err=str,
    )

    serialized = asyncio.run(result_codec.serialize(Ok(3)))
    assert isinstance(serialized, Ok)
    deserialized = asyncio.run(result_codec.deserialize({"status": "ok", "value": "3"}))
    assert isinstance(deserialized, Ok)
    assert deserialized.value == 3


def test_panic_and_error_serialization_failures() -> None:
    nested: dict[str, object] = {}
    nested["self"] = nested
    panic_json = PanicError("cycle", cause=[ValueError("nested")]).to_json()
    assert isinstance(panic_json["cause"], list)
    assert PanicError("cycle", cause=nested).to_json()["cause"] == {"self": "<cycle>"}


def test_tagged_error_validation_paths() -> None:
    with pytest.raises(TypeError, match="must not define _tag"):

        class InvalidTagError(TaggedError, tag="InvalidTag"):
            _tag = "shadowed"

    with pytest.raises(ValueError, match="must not be empty"):
        type("EmptyTag", (TaggedError,), {}, tag="   ")
    with pytest.raises(TypeError, match="reserved"):
        TaggedError(to_json="shadowed")

    assert UnhandledError(None).message == "Unhandled exception: null"
    assert UnhandledError(cause=True).message == "Unhandled exception: true"
    assert UnhandledError(cause=False).message == "Unhandled exception: false"


def test_retry_helpers_cover_validation_and_backoff() -> None:
    assert _retry_times(None) == 0
    assert _retry_times(RetryConfig(times=2)) == 2
    with pytest.raises(ValueError, match="must not be negative"):
        _retry_times(RetryConfig(times=-1))

    assert _static_retry_delay(10, "constant", 2) == 10
    assert _static_retry_delay(10, "linear", 2) == 30
    assert _static_retry_delay(10, "exponential", 2) == 40
    assert _jitter_factor(jitter=0.25) == 0.25
    with pytest.raises(PanicError, match="jitter"):
        _jitter_factor(jitter=2)
    with pytest.raises(PanicError, match="finite"):
        _validate_delay(math.inf)


def test_try_result_breaks_after_a_retry_succeeds() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> int:
        attempts.append(context.attempt)
        if context.attempt == 1:
            msg = "retry"
            raise ValueError(msg)
        return 42

    result = try_result(operation, retry=RetryConfig(times=2))
    assert isinstance(result, Ok)
    assert attempts == [1, 2]

    def panic_catch(_cause: BaseException) -> str:
        msg = "catch panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="catch panic"):
        try_result(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")),
            panic_catch,
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
        msg = "operation failed"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="must not be negative"):
        await try_async(operation, retry=AsyncRetryConfig(times=-1))
    with pytest.raises(PanicError, match="finite"):
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
        msg = "catch panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="catch panic"):
        await try_async(operation, panic_catch)

    async def bad_predicate(_error: object, _context: TryAsyncContext) -> bool:
        msg = "predicate failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="should_retry predicate threw"):
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
        msg = "predicate panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="predicate panic"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, should_retry=panic_predicate),
        )

    def bad_delay(_error: object, _context: TryAsyncContext) -> float:
        msg = "delay failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="delay_ms callback threw"):
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
        msg = "delay panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="delay panic"):
        await try_async(
            operation,
            retry=AsyncRetryConfig(times=1, delay_ms=panic_delay),
        )


@pytest.mark.asyncio
async def test_async_result_instance_methods_cover_each_branch() -> None:
    success = Ok(2)
    failure = Err("bad")

    assert await success.and_then_async(lambda value: _completed(Ok(value + 1))) == Ok(
        3
    )
    assert await failure.and_then_async(lambda _value: _completed(Ok(99))) is failure
    assert failure.unwrap_or(99) == 99


@pytest.mark.asyncio
async def test_all_async_cancels_for_cancelled_and_panic_inputs() -> None:
    async def cancelled() -> Ok[int]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await all_results_async([cancelled()])

    async def panicking() -> Ok[int]:
        msg = "input panic"
        raise PanicError(msg)

    with pytest.raises(PanicError, match="input panic"):
        await all_results_async([panicking()])


async def _completed[T](value: T) -> T:
    await asyncio.sleep(0)
    return value
