"""Small, typed codecs for crossing a Result serialization boundary."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from ._core import Err, Ok, Result, UnwrapError


class CodecIssue(TypedDict, total=False):
    """A structured validation issue returned by a schema."""

    message: str
    path: Sequence[str | int]


@dataclass(frozen=True, slots=True)
class SchemaFailure:
    """Expected validation failure returned by a schema."""

    issues: Sequence[CodecIssue]


@dataclass(frozen=True, slots=True)
class ResultSerializationError:
    """A Result payload was rejected while being encoded."""

    value: object
    issues: Sequence[CodecIssue]


@dataclass(frozen=True, slots=True)
class ResultDeserializationError:
    """An envelope or payload was rejected while being decoded."""

    value: object
    issues: Sequence[CodecIssue] | None = None


class SerializedOk[T](TypedDict):
    status: Literal["ok"]
    value: T


class SerializedErr[E](TypedDict):
    status: Literal["error"]
    error: E


type SerializedResult[T, E] = SerializedOk[T] | SerializedErr[E]


type SyncSchema[T, U] = Callable[[T], U | SchemaFailure]
type AsyncSchema[T, U] = Callable[[T], Awaitable[U | SchemaFailure]]


def _envelope(value: object) -> Mapping[object, object] | None:
    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    if status not in ("ok", "error"):
        return None
    return cast("Mapping[object, object]", value)


def _finish[T, E](
    value: T | SchemaFailure,
    original: object,
    make_error: Callable[[object, Sequence[CodecIssue]], E],
) -> Result[T, E]:
    if isinstance(value, SchemaFailure):
        return Err(make_error(original, value.issues))
    return Ok(value)


def _run_sync[T, U, E](
    schema: SyncSchema[T, U],
    value: T,
    original: object,
    make_error: Callable[[object, Sequence[CodecIssue]], E],
) -> Result[U, E]:
    validated = schema(value)
    if inspect.isawaitable(validated):
        # A sync codec should fail at the call site, and close a coroutine so
        # callers do not get a second "never awaited" warning.
        if inspect.iscoroutine(validated):
            validated.close()
        message = "sync codec schema returned an awaitable; use async_codec"
        raise TypeError(message)
    return cast(
        "Result[U, E]",
        _finish(cast("U | SchemaFailure", validated), original, make_error),
    )


async def _run_async[T, U, E](
    schema: AsyncSchema[T, U],
    value: T,
    original: object,
    make_error: Callable[[object, Sequence[CodecIssue]], E],
) -> Result[U, E]:
    validated = await schema(value)
    return cast(
        "Result[U, E]",
        _finish(validated, original, make_error),
    )


class ResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Encode and decode Result envelopes with synchronous schemas."""

    def __init__(
        self,
        *,
        serialize_ok: SyncSchema[OkInput, OkWire],
        serialize_err: SyncSchema[ErrInput, ErrWire],
        deserialize_ok: SyncSchema[OkWire, OkOutput],
        deserialize_err: SyncSchema[ErrWire, ErrOutput],
    ) -> None:
        self._serialize_ok = serialize_ok
        self._serialize_err = serialize_err
        self._deserialize_ok = deserialize_ok
        self._deserialize_err = deserialize_err

    def serialize(
        self,
        result: Result[OkInput, ErrInput],
    ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
        if isinstance(result, Ok):
            encoded = _run_sync(
                self._serialize_ok,
                result.ok_value,
                result.ok_value,
                ResultSerializationError,
            )
            if isinstance(encoded, Err):
                return Err(encoded.err_value)
            envelope: SerializedOk[OkWire] = {
                "status": "ok",
                "value": encoded.ok_value,
            }
            return Ok(envelope)

        encoded = _run_sync(
            self._serialize_err,
            result.err_value,
            result.err_value,
            ResultSerializationError,
        )
        if isinstance(encoded, Err):
            return Err(encoded.err_value)
        envelope: SerializedErr[ErrWire] = {
            "status": "error",
            "error": encoded.ok_value,
        }
        return Ok(envelope)

    def deserialize(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
        envelope = _envelope(value)
        if envelope is None:
            return Err(ResultDeserializationError(value))

        if envelope["status"] == "ok":
            decoded = cast(
                "Result[OkOutput, ResultDeserializationError]",
                _run_sync(
                    self._deserialize_ok,
                    cast("OkWire", envelope.get("value")),
                    value,
                    ResultDeserializationError,
                ),
            )
            if isinstance(decoded, Err):
                return Err(decoded.err_value)
            return Ok(decoded.ok_value)

        decoded = cast(
            "Result[ErrOutput, ResultDeserializationError]",
            _run_sync(
                self._deserialize_err,
                cast("ErrWire", envelope.get("error")),
                value,
                ResultDeserializationError,
            ),
        )
        if isinstance(decoded, Err):
            return Err(decoded.err_value)
        return Err(decoded.ok_value)

    def serialize_unsafe(
        self,
        result: Result[OkInput, ErrInput],
    ) -> SerializedResult[OkWire, ErrWire]:
        encoded = self.serialize(result)
        if isinstance(encoded, Err):
            raise UnwrapError(encoded, "ResultCodec.serialize_unsafe failed")
        return encoded.ok_value

    def deserialize_unsafe(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput]:
        decoded = self.deserialize(value)
        if isinstance(decoded, Err) and isinstance(
            decoded.err_value, ResultDeserializationError
        ):
            raise UnwrapError(decoded, "ResultCodec.deserialize_unsafe failed")
        return cast("Result[OkOutput, ErrOutput]", decoded)


class AsyncResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Encode and decode Result envelopes with asynchronous schemas."""

    def __init__(
        self,
        *,
        serialize_ok: AsyncSchema[OkInput, OkWire],
        serialize_err: AsyncSchema[ErrInput, ErrWire],
        deserialize_ok: AsyncSchema[OkWire, OkOutput],
        deserialize_err: AsyncSchema[ErrWire, ErrOutput],
    ) -> None:
        self._serialize_ok = serialize_ok
        self._serialize_err = serialize_err
        self._deserialize_ok = deserialize_ok
        self._deserialize_err = deserialize_err

    async def serialize(
        self,
        result: Result[OkInput, ErrInput],
    ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
        if isinstance(result, Ok):
            encoded = await _run_async(
                self._serialize_ok,
                result.ok_value,
                result.ok_value,
                ResultSerializationError,
            )
            if isinstance(encoded, Err):
                return Err(encoded.err_value)
            return Ok({"status": "ok", "value": encoded.ok_value})

        encoded = await _run_async(
            self._serialize_err,
            result.err_value,
            result.err_value,
            ResultSerializationError,
        )
        if isinstance(encoded, Err):
            return Err(encoded.err_value)
        return Ok({"status": "error", "error": encoded.ok_value})

    async def deserialize(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
        envelope = _envelope(value)
        if envelope is None:
            return Err(ResultDeserializationError(value))

        if envelope["status"] == "ok":
            decoded = cast(
                "Result[OkOutput, ResultDeserializationError]",
                await _run_async(
                    self._deserialize_ok,
                    cast("OkWire", envelope.get("value")),
                    value,
                    ResultDeserializationError,
                ),
            )
            if isinstance(decoded, Err):
                return Err(decoded.err_value)
            return Ok(decoded.ok_value)

        decoded = cast(
            "Result[ErrOutput, ResultDeserializationError]",
            await _run_async(
                self._deserialize_err,
                cast("ErrWire", envelope.get("error")),
                value,
                ResultDeserializationError,
            ),
        )
        if isinstance(decoded, Err):
            return Err(decoded.err_value)
        return Err(decoded.ok_value)

    async def serialize_unsafe(
        self,
        result: Result[OkInput, ErrInput],
    ) -> SerializedResult[OkWire, ErrWire]:
        encoded = await self.serialize(result)
        if isinstance(encoded, Err):
            raise UnwrapError(encoded, "AsyncResultCodec.serialize_unsafe failed")
        return encoded.ok_value

    async def deserialize_unsafe(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput]:
        decoded = await self.deserialize(value)
        if isinstance(decoded, Err) and isinstance(
            decoded.err_value, ResultDeserializationError
        ):
            raise UnwrapError(decoded, "AsyncResultCodec.deserialize_unsafe failed")
        return cast("Result[OkOutput, ErrOutput]", decoded)


def codec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    *,
    serialize_ok: SyncSchema[OkInput, OkWire],
    serialize_err: SyncSchema[ErrInput, ErrWire],
    deserialize_ok: SyncSchema[OkWire, OkOutput],
    deserialize_err: SyncSchema[ErrWire, ErrOutput],
) -> ResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Build a synchronous Result codec."""
    return ResultCodec(
        serialize_ok=serialize_ok,
        serialize_err=serialize_err,
        deserialize_ok=deserialize_ok,
        deserialize_err=deserialize_err,
    )


def async_codec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    *,
    serialize_ok: AsyncSchema[OkInput, OkWire],
    serialize_err: AsyncSchema[ErrInput, ErrWire],
    deserialize_ok: AsyncSchema[OkWire, OkOutput],
    deserialize_err: AsyncSchema[ErrWire, ErrOutput],
) -> AsyncResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Build a Result codec from asynchronous schemas."""
    return AsyncResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
        serialize_ok=serialize_ok,
        serialize_err=serialize_err,
        deserialize_ok=deserialize_ok,
        deserialize_err=deserialize_err,
    )
