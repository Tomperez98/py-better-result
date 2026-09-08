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
from ._operations import (
    CancellationToken,
    RetryPolicy,
    TryContext,
    all_results,
    all_results_async,
    flatten_result,
    partition_results,
    partition_results_async,
    try_async,
    try_result,
)

__all__ = [
    "AsyncResultCodec",
    "AsyncSchema",
    "CancellationToken",
    "CodecIssue",
    "Err",
    "Ok",
    "Result",
    "ResultCodec",
    "ResultDeserializationError",
    "ResultSerializationError",
    "RetryPolicy",
    "SchemaFailure",
    "SerializedErr",
    "SerializedOk",
    "SerializedResult",
    "SyncSchema",
    "TryContext",
    "UnwrapError",
    "all_results",
    "all_results_async",
    "async_codec",
    "codec",
    "flatten_result",
    "is_err",
    "is_ok",
    "partition_results",
    "partition_results_async",
    "try_async",
    "try_result",
]
