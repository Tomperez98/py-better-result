"""Edge cases for complete Result codec coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from better_result import (
    CodecIssue,
    Err,
    Ok,
    Result,
    ResultCodec,
    SchemaFailure,
    SyncSchema,
    UnwrapError,
    async_codec,
    codec,
)
from better_result._codec import _require_err, _require_ok

ISSUE: CodecIssue = {"message": "rejected"}


def reject(_: object) -> SchemaFailure:
    return SchemaFailure([ISSUE])


def identity(value: object) -> object:
    return value


async def async_identity(value: object) -> object:
    return value


async def async_reject(_: object) -> SchemaFailure:
    return SchemaFailure([ISSUE])


class NonCoroutineAwaitable:
    def __await__(self) -> Iterator[object]:
        """Return an await iterator without being a coroutine."""
        return iter(())


def test_require_helpers_reject_unknown_result_variants() -> None:
    invalid = cast("Result[int, str]", object())

    with pytest.raises(TypeError, match="expected an Ok Result"):
        _require_ok(invalid)
    with pytest.raises(TypeError, match="expected an Err Result"):
        _require_err(invalid)


def test_sync_codec_covers_error_branches_and_rejects_async_schemas() -> None:
    result_codec = codec(
        serialize_ok=identity,
        serialize_err=reject,
        deserialize_ok=identity,
        deserialize_err=reject,
    )

    assert isinstance(result_codec.serialize(Err("bad")), Err)
    assert isinstance(
        result_codec.deserialize({"status": "error", "error": "bad"}),
        Err,
    )

    async_schema_codec = ResultCodec(
        serialize_ok=cast("SyncSchema[int, int]", async_identity),
        serialize_err=identity,
        deserialize_ok=identity,
        deserialize_err=identity,
    )
    with pytest.raises(TypeError, match="use async_codec"):
        async_schema_codec.serialize(Ok(1))

    non_coroutine_codec = ResultCodec(
        serialize_ok=cast(
            "SyncSchema[int, int]",
            lambda _: NonCoroutineAwaitable(),
        ),
        serialize_err=identity,
        deserialize_ok=identity,
        deserialize_err=identity,
    )
    with pytest.raises(TypeError, match="use async_codec"):
        non_coroutine_codec.serialize(Ok(1))


@pytest.mark.asyncio
async def test_async_codec_covers_all_branches_and_unsafe_methods() -> None:
    result_codec = async_codec(
        serialize_ok=async_reject,
        serialize_err=async_reject,
        deserialize_ok=async_reject,
        deserialize_err=async_reject,
    )

    assert isinstance(await result_codec.serialize(Ok(1)), Err)
    assert isinstance(await result_codec.serialize(Err("bad")), Err)
    assert isinstance(await result_codec.deserialize(None), Err)
    assert isinstance(
        await result_codec.deserialize({"status": "ok", "value": 1}),
        Err,
    )
    assert isinstance(
        await result_codec.deserialize({"status": "error", "error": "bad"}),
        Err,
    )

    identity_codec = async_codec(
        serialize_ok=async_identity,
        serialize_err=async_identity,
        deserialize_ok=async_identity,
        deserialize_err=async_identity,
    )
    assert await identity_codec.serialize_unsafe(Ok(1)) == {
        "status": "ok",
        "value": 1,
    }
    assert await identity_codec.serialize(Err("bad")) == Ok(
        {"status": "error", "error": "bad"},
    )
    assert await identity_codec.deserialize_unsafe(
        {"status": "error", "error": "bad"},
    ) == Err("bad")

    with pytest.raises(UnwrapError, match="serialize_unsafe"):
        await result_codec.serialize_unsafe(Ok(1))
    with pytest.raises(UnwrapError, match="deserialize_unsafe"):
        await result_codec.deserialize_unsafe(None)
