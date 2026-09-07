"""
Validated Result codecs for serialization boundaries.

A codec combines four schemas: success and error payloads in each direction.
Schemas may be synchronous or asynchronous. A schema should return a payload
or :class:`SchemaFailure`; throwing from validation is a defect and becomes
``Panic``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, cast

from better_result.core import Err, Ok, PanicError, Result, panic
from better_result.error import (
    ResultCodecIssue,
    ResultDeserializationError,
    ResultSerializationError,
)


@dataclass(frozen=True, slots=True)
class SchemaFailure:
    """Validation issues returned by a schema instead of raised."""

    issues: Sequence[ResultCodecIssue]


class Schema[T, U](Protocol):
    """A synchronous validator for a Result boundary."""

    def validate(self, value: T) -> U | SchemaFailure: ...


type SyncSchemaLike[T, U] = Schema[T, U] | Callable[[T], U | SchemaFailure]
type SchemaLike[T, U] = (
    Schema[T, U] | Callable[[T], U | SchemaFailure | Awaitable[U | SchemaFailure]]
)


class SerializeSchemas[OkInput, ErrInput, OkWire, ErrWire](TypedDict):
    """The two schemas used to encode Result payloads."""

    ok: SchemaLike[OkInput, OkWire]
    err: SchemaLike[ErrInput, ErrWire]


class DeserializeSchemas[OkWire, ErrWire, OkOutput, ErrOutput](TypedDict):
    """The two schemas used to decode Result payloads."""

    ok: SchemaLike[OkWire, OkOutput]
    err: SchemaLike[ErrWire, ErrOutput]


class ResultCodecConfig[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    TypedDict,
):
    """Four named schemas defining a Result boundary."""

    serialize: SerializeSchemas[OkInput, ErrInput, OkWire, ErrWire]
    deserialize: DeserializeSchemas[OkWire, ErrWire, OkOutput, ErrOutput]


class SerializedOk[T](TypedDict):
    status: Literal["ok"]
    value: T


class SerializedErr[E](TypedDict):
    status: Literal["error"]
    error: E


type SerializedResult[T, E] = SerializedOk[T] | SerializedErr[E]


def _validate[T, U](
    schema: SchemaLike[T, U],
    value: T,
) -> U | SchemaFailure | Awaitable[U | SchemaFailure]:
    if callable(schema):
        callback = cast(
            "Callable[[T], U | SchemaFailure | Awaitable[U | SchemaFailure]]",
            schema,
        )
        return callback(value)
    return schema.validate(value)


def _finish_validation[T, U, E](
    validation: U | SchemaFailure,
    original: T,
    make_error: Callable[[T, Sequence[ResultCodecIssue]], E],
) -> Result[U, E]:
    if isinstance(validation, SchemaFailure):
        return Err(make_error(original, validation.issues))
    return Ok(validation)


def _run_validation[T, U, E](
    schema: SchemaLike[T, U],
    value: T,
    make_error: Callable[[T, Sequence[ResultCodecIssue]], E],
    panic_message: str,
    *,
    allow_async: bool = True,
) -> Result[U, E] | Awaitable[Result[U, E]]:
    try:
        validation = _validate(schema, value)
    except PanicError:
        raise
    except Exception as cause:
        panic(panic_message, cause)

    if not allow_async and inspect.isawaitable(validation):
        cast("Coroutine[None, None, U | SchemaFailure]", validation).close()
        panic("ResultCodec received an async schema; use async_codec")

    if inspect.isawaitable(validation):

        async def finish_async() -> Result[U, E]:
            try:
                resolved = await validation
            except asyncio.CancelledError:
                raise
            except PanicError:
                raise
            except Exception as cause:
                panic(panic_message, cause)
            return cast(
                "Result[U, E]",
                _finish_validation(
                    cast("U | SchemaFailure", resolved),
                    value,
                    make_error,
                ),
            )

        return finish_async()
    return _finish_validation(validation, value, make_error)


async def _await_operation[T, E](
    operation: T | Awaitable[T],
) -> T:
    if inspect.isawaitable(operation):
        return cast("T", await operation)
    return operation


def _is_envelope(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("status") in {"ok", "error"}


class _CodecBase[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Shared implementation for synchronous and asynchronous codecs."""

    def __init__(
        self,
        config: ResultCodecConfig[
            OkInput,
            ErrInput,
            OkWire,
            ErrWire,
            OkOutput,
            ErrOutput,
        ],
    ) -> None:
        self._config = config

    def _serialize(
        self,
        result: Result[OkInput, ErrInput],
        *,
        allow_async: bool = True,
    ) -> (
        Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]
        | Awaitable[Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]]
    ):
        if isinstance(result, Ok):
            operation = _run_validation(
                self._config["serialize"]["ok"],
                result.value,
                ResultSerializationError,
                "Result.codec serialize schema threw",
                allow_async=allow_async,
            )

            def finish_ok(
                resolved: Result[OkWire, ResultSerializationError],
            ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
                if isinstance(resolved, Err):
                    return Err(resolved.error)
                envelope: SerializedOk[OkWire] = {
                    "status": "ok",
                    "value": resolved.value,
                }
                return Ok(
                    envelope,
                )

            return _map_operation(operation, finish_ok)

        operation = _run_validation(
            self._config["serialize"]["err"],
            result.error,
            ResultSerializationError,
            "Result.codec serialize schema threw",
            allow_async=allow_async,
        )

        def finish_err(
            resolved: Result[ErrWire, ResultSerializationError],
        ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
            if isinstance(resolved, Err):
                return Err(
                    resolved.error,
                )
            envelope: SerializedErr[ErrWire] = {
                "status": "error",
                "error": resolved.value,
            }
            return Ok(
                envelope,
            )

        return _map_operation(operation, finish_err)

    def _deserialize(
        self,
        value: object,
        *,
        allow_async: bool = True,
    ) -> (
        Result[OkOutput, ErrOutput | ResultDeserializationError]
        | Awaitable[Result[OkOutput, ErrOutput | ResultDeserializationError]]
    ):
        if not _is_envelope(value):
            return Err(
                ResultDeserializationError(value),
            )
        assert isinstance(value, Mapping)
        if value["status"] == "ok":
            operation = _run_validation(
                self._config["deserialize"]["ok"],
                cast("OkWire", value.get("value")),
                ResultDeserializationError,
                "Result.codec deserialize schema threw",
                allow_async=allow_async,
            )

            def finish_ok(
                resolved: Result[OkOutput, ResultDeserializationError],
            ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
                if isinstance(resolved, Err):
                    return Err(
                        resolved.error,
                    )
                return Ok(
                    resolved.value,
                )

            return _map_operation(operation, finish_ok)

        operation = _run_validation(
            self._config["deserialize"]["err"],
            cast("ErrWire", value.get("error")),
            ResultDeserializationError,
            "Result.codec deserialize schema threw",
            allow_async=allow_async,
        )

        def finish_err(
            resolved: Result[ErrOutput, ResultDeserializationError],
        ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
            if isinstance(resolved, Err):
                return Err(
                    resolved.error,
                )
            return Err(resolved.value)

        return _map_operation(operation, finish_err)


class ResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    _CodecBase[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput],
):
    """Encode and decode Result envelopes with synchronous schemas."""

    def serialize(
        self,
        result: Result[OkInput, ErrInput],
    ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
        return cast(
            "Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]",
            self._serialize(result, allow_async=False),
        )

    def deserialize(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
        return cast(
            "Result[OkOutput, ErrOutput | ResultDeserializationError]",
            self._deserialize(value, allow_async=False),
        )


class AsyncResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    _CodecBase[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput],
):
    """Encode and decode Result envelopes with sync or async schemas."""

    async def serialize(
        self,
        result: Result[OkInput, ErrInput],
    ) -> Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]:
        return cast(
            "Result[SerializedResult[OkWire, ErrWire], ResultSerializationError]",
            await _await_operation(self._serialize(result)),
        )

    async def deserialize(
        self,
        value: object,
    ) -> Result[OkOutput, ErrOutput | ResultDeserializationError]:
        return cast(
            "Result[OkOutput, ErrOutput | ResultDeserializationError]",
            await _await_operation(self._deserialize(value)),
        )


def _map_operation[T, U, E](
    operation: Result[T, E] | Awaitable[Result[T, E]],
    transform: Callable[[Result[T, E]], U],
) -> U | Awaitable[U]:
    if inspect.isawaitable(operation):

        async def finish_async() -> U:
            return transform(cast("Result[T, E]", await operation))

        return finish_async()
    return transform(operation)


def _codec_config[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    *,
    serialize_ok: SchemaLike[OkInput, OkWire],
    serialize_err: SchemaLike[ErrInput, ErrWire],
    deserialize_ok: SchemaLike[OkWire, OkOutput],
    deserialize_err: SchemaLike[ErrWire, ErrOutput],
) -> ResultCodecConfig[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    return {
        "serialize": {"ok": serialize_ok, "err": serialize_err},
        "deserialize": {"ok": deserialize_ok, "err": deserialize_err},
    }


def codec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    *,
    serialize_ok: SyncSchemaLike[OkInput, OkWire],
    serialize_err: SyncSchemaLike[ErrInput, ErrWire],
    deserialize_ok: SyncSchemaLike[OkWire, OkOutput],
    deserialize_err: SyncSchemaLike[ErrWire, ErrOutput],
) -> ResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Build a codec with synchronous schemas."""
    return ResultCodec(
        _codec_config(
            serialize_ok=serialize_ok,
            serialize_err=serialize_err,
            deserialize_ok=deserialize_ok,
            deserialize_err=deserialize_err,
        ),
    )


def async_codec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput](
    *,
    serialize_ok: SchemaLike[OkInput, OkWire],
    serialize_err: SchemaLike[ErrInput, ErrWire],
    deserialize_ok: SchemaLike[OkWire, OkOutput],
    deserialize_err: SchemaLike[ErrWire, ErrOutput],
) -> AsyncResultCodec[OkInput, ErrInput, OkWire, ErrWire, OkOutput, ErrOutput]:
    """Build a codec with synchronous or asynchronous schemas."""
    return AsyncResultCodec(
        _codec_config(
            serialize_ok=serialize_ok,
            serialize_err=serialize_err,
            deserialize_ok=deserialize_ok,
            deserialize_err=deserialize_err,
        ),
    )
