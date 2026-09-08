"""Behavioral tests for Result codecs."""

from __future__ import annotations

from typing import cast

import pytest

from better_result import (
    CodecIssue,
    Err,
    Ok,
    ResultDeserializationError,
    ResultSerializationError,
    SchemaFailure,
    UnwrapError,
    async_codec,
    codec,
)
from tests.codec_helpers import (
    assert_async_codec_roundtrip,
    assert_codec_roundtrip,
)


def identity(value: object) -> object:
    return value


def test_codec_serializes_and_deserializes_both_branches() -> None:
    result_codec = codec(
        serialize_ok=lambda value: {"name": value},
        serialize_err=lambda error: {"code": error},
        deserialize_ok=lambda value: value["name"],
        deserialize_err=lambda value: value["code"],
    )

    assert_codec_roundtrip(
        result_codec,
        Ok("Ada"),
        expected_wire={"status": "ok", "value": {"name": "Ada"}},
        expected_result=Ok("Ada"),
    )
    assert_codec_roundtrip(
        result_codec,
        Err(404),
        expected_wire={"status": "error", "error": {"code": 404}},
        expected_result=Err(404),
    )


def test_codec_returns_expected_errors_without_catching_schema_bugs() -> None:
    issue = cast("CodecIssue", {"message": "expected an integer"})

    def reject(_: object) -> SchemaFailure:
        return SchemaFailure([issue])

    result_codec = codec(
        serialize_ok=reject,
        serialize_err=identity,
        deserialize_ok=reject,
        deserialize_err=identity,
    )

    serialized = result_codec.serialize(Ok("not-an-int"))
    assert isinstance(serialized, Err)
    assert isinstance(serialized.err_value, ResultSerializationError)
    assert serialized.err_value.value == "not-an-int"
    assert serialized.err_value.issues == [issue]

    deserialized = result_codec.deserialize({"status": "ok", "value": "bad"})
    assert isinstance(deserialized, Err)
    assert isinstance(deserialized.err_value, ResultDeserializationError)
    assert deserialized.err_value.value == {"status": "ok", "value": "bad"}
    assert deserialized.err_value.issues == [issue]

    malformed = result_codec.deserialize({"status": "unknown"})
    assert isinstance(malformed, Err)
    assert isinstance(malformed.err_value, ResultDeserializationError)
    assert malformed.err_value.issues is None

    unhashable_status = result_codec.deserialize({"status": []})
    assert isinstance(unhashable_status, Err)
    assert isinstance(unhashable_status.err_value, ResultDeserializationError)
    assert unhashable_status.err_value.value == {"status": []}

    def broken(_: object) -> object:
        message = "schema bug"
        raise RuntimeError(message)

    broken_codec = codec(
        serialize_ok=broken,
        serialize_err=identity,
        deserialize_ok=identity,
        deserialize_err=identity,
    )
    with pytest.raises(RuntimeError, match="schema bug"):
        broken_codec.serialize(Ok("value"))


def test_unsafe_methods_only_remove_codec_errors() -> None:
    result_codec = codec(
        serialize_ok=lambda value: value,
        serialize_err=lambda error: error,
        deserialize_ok=lambda value: value,
        deserialize_err=lambda value: value,
    )

    assert result_codec.serialize_unsafe(Ok(1)) == {"status": "ok", "value": 1}
    assert result_codec.deserialize_unsafe({"status": "error", "error": "nope"}) == Err(
        "nope"
    )

    def reject(_: object) -> SchemaFailure:
        return SchemaFailure([cast("CodecIssue", {"message": "rejected"})])

    rejecting_codec = codec(
        serialize_ok=reject,
        serialize_err=identity,
        deserialize_ok=identity,
        deserialize_err=identity,
    )
    with pytest.raises(UnwrapError, match="serialize_unsafe"):
        rejecting_codec.serialize_unsafe(Ok(object()))

    with pytest.raises(UnwrapError, match="deserialize_unsafe"):
        result_codec.deserialize_unsafe(None)


@pytest.mark.asyncio
async def test_async_codec_accepts_async_schemas() -> None:
    async def stringify(value: object) -> str:
        return str(value)

    result_codec = async_codec(
        serialize_ok=stringify,
        serialize_err=stringify,
        deserialize_ok=stringify,
        deserialize_err=stringify,
    )

    await assert_async_codec_roundtrip(
        result_codec,
        Ok(42),
        expected_wire={"status": "ok", "value": "42"},
        expected_result=Ok("42"),
    )
    await assert_async_codec_roundtrip(
        result_codec,
        Err(404),
        expected_wire={"status": "error", "error": "404"},
        expected_result=Err("404"),
    )
