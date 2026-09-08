"""Small, typed codecs for crossing a Result serialization boundary."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from ._core import Err, Ok, Result, UnwrapError, is_err, is_ok


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


def _require_ok[T, E](value: Result[T, E]) -> Ok[T]:
    if is_ok(value):
        return value
    message = "expected an Ok Result"
    raise TypeError(message)


def _require_err[T, E](value: Result[T, E]) -> Err[E]:
    if is_err(value):
        return value
    message = "expected an Err Result"
    raise TypeError(message)


def _run_sync[T, U](schema: SyncSchema[T, U], value: T) -> U | SchemaFailure:
    validated = schema(value)
    if isinstance(validated, Awaitable):
        # A sync codec should fail at the call site, and close a coroutine so
        # callers do not get a second "never awaited" warning.
        if inspect.iscoroutine(validated):
            validated.close()
        message = "sync codec schema returned an awaitable; use async_codec"
        raise TypeError(message)
    return cast("U | SchemaFailure", validated)


async def _run_async[T, U](
    schema: AsyncSchema[T, U],
    value: T,
) -> U | SchemaFailure:
    return await schema(value)


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
        if is_ok(result):
            encoded = _run_sync(self._serialize_ok, result.ok_value)
            if isinstance(encoded, SchemaFailure):
                return Err(ResultSerializationError(result.ok_value, encoded.issues))
            envelope: SerializedOk[OkWire] = {
                "status": "ok",
                "value": encoded,
            }
            return Ok(envelope)

        result = _require_err(result)
        encoded = _run_sync(self._serialize_err, result.err_value)
        if isinstance(encoded, SchemaFailure):
            return Err(ResultSerializationError(result.err_value, encoded.issues))
        envelope: SerializedErr[ErrWire] = {
            "status": "error",
            "error": encoded,
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
            decoded = _run_sync(
                self._deserialize_ok,
                cast("OkWire", envelope.get("value")),
            )
            if isinstance(decoded, SchemaFailure):
                return Err(ResultDeserializationError(value, decoded.issues))
            return Ok(cast("OkOutput", decoded))

        decoded = _run_sync(
            self._deserialize_err,
            cast("ErrWire", envelope.get("error")),
        )
        if isinstance(decoded, SchemaFailure):
            return Err(ResultDeserializationError(value, decoded.issues))
        return Err(cast("ErrOutput", decoded))

    def serialize_unsafe(
        self,
        result: Result[OkInput, ErrInput],
    ) -> SerializedResult[OkWire, ErrWire]:
        encoded = self.serialize(result)
        if is_err(encoded):
            raise UnwrapError(encoded, "ResultCodec.serialize_unsafe failed")
        return _require_ok(encoded).ok_value

    def deserialize_unsafe(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput]:
        decoded = self.deserialize(value)
        if is_err(decoded) and isinstance(
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
        if is_ok(result):
            encoded = await _run_async(self._serialize_ok, result.ok_value)
            if isinstance(encoded, SchemaFailure):
                return Err(ResultSerializationError(result.ok_value, encoded.issues))
            return Ok({"status": "ok", "value": encoded})

        result = _require_err(result)
        encoded = await _run_async(self._serialize_err, result.err_value)
        if isinstance(encoded, SchemaFailure):
            return Err(ResultSerializationError(result.err_value, encoded.issues))
        return Ok({"status": "error", "error": encoded})

    async def deserialize(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
        envelope = _envelope(value)
        if envelope is None:
            return Err(ResultDeserializationError(value))

        if envelope["status"] == "ok":
            decoded = await _run_async(
                self._deserialize_ok,
                cast("OkWire", envelope.get("value")),
            )
            if isinstance(decoded, SchemaFailure):
                return Err(ResultDeserializationError(value, decoded.issues))
            return Ok(cast("OkOutput", decoded))

        decoded = await _run_async(
            self._deserialize_err,
            cast("ErrWire", envelope.get("error")),
        )
        if isinstance(decoded, SchemaFailure):
            return Err(ResultDeserializationError(value, decoded.issues))
        return Err(cast("ErrOutput", decoded))

    async def serialize_unsafe(
        self,
        result: Result[OkInput, ErrInput],
    ) -> SerializedResult[OkWire, ErrWire]:
        encoded = await self.serialize(result)
        if is_err(encoded):
            raise UnwrapError(encoded, "AsyncResultCodec.serialize_unsafe failed")
        return _require_ok(encoded).ok_value

    async def deserialize_unsafe(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput]:
        decoded = await self.deserialize(value)
        if is_err(decoded) and isinstance(
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
