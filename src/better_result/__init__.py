"""A small, typed Result type with boundary codecs."""

from __future__ import annotations

from ._codec import (
    AsyncResultCodec,
    AsyncSchema,
    CodecIssue,
    ResultCodec,
    ResultDeserializationError,
    ResultSerializationError,
    SchemaFailure,
    SerializedErr,
    SerializedOk,
    SerializedResult,
    SyncSchema,
    async_codec,
    codec,
)
from ._core import Err, Ok, Result, UnwrapError, is_err, is_ok

__all__ = [
    "AsyncResultCodec",
    "AsyncSchema",
    "CodecIssue",
    "Err",
    "Ok",
    "Result",
    "ResultCodec",
    "ResultDeserializationError",
    "ResultSerializationError",
    "SchemaFailure",
    "SerializedErr",
    "SerializedOk",
    "SerializedResult",
    "SyncSchema",
    "UnwrapError",
    "async_codec",
    "codec",
    "is_err",
    "is_ok",
]
