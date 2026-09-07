"""Tests for Result codec serialization boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from better_result.codec import SchemaFailure, codec, codec_config
from better_result.core import Err, Ok, Panic
from better_result.error import ResultDeserializationError, ResultSerializationError


def encode_number(value: int) -> str | SchemaFailure:
    if value < 0:
        return SchemaFailure(({"message": "number must be non-negative"},))
    return str(value)


def encode_error(value: str) -> str:
    return value


def decode_number(value: str | None) -> int | SchemaFailure:
    if not isinstance(value, str) or not value.isdecimal():
        return SchemaFailure(({"message": "expected digits"},))
    return int(value)


def decode_error(value: str) -> str:
    return value


CONFIG = codec_config(
    serialize_ok=encode_number,
    serialize_err=encode_error,
    deserialize_ok=decode_number,
    deserialize_err=decode_error,
)

GOLDEN_DIR = Path(__file__).with_name("golden") / "codec"


def _golden_json(filename: str) -> object:
    return json.loads((GOLDEN_DIR / filename).read_text(encoding="utf-8"))


def _stable_codec_error(
    error: ResultDeserializationError | ResultSerializationError,
) -> dict[str, object | None]:
    payload = error.to_json()
    stable = {
        key: payload[key]
        for key in ("_tag", "name", "message", "value", "issues")
        if key in payload
    }
    return cast("dict[str, object | None]", json.loads(json.dumps(stable)))


def test_codec_serializes_and_deserializes_both_result_branches() -> None:
    result_codec = codec(CONFIG)

    encoded_success = result_codec.serialize(Ok(42))
    assert isinstance(encoded_success, Ok)
    assert encoded_success.value == {"status": "ok", "value": "42"}

    encoded_error = result_codec.serialize(Err("missing"))
    assert isinstance(encoded_error, Ok)
    assert encoded_error.value == {"status": "error", "error": "missing"}

    decoded_success = result_codec.deserialize({"status": "ok", "value": "42"})
    assert isinstance(decoded_success, Ok)
    assert decoded_success.value == 42

    decoded_error = result_codec.deserialize({"status": "error", "error": "missing"})
    assert isinstance(decoded_error, Err)
    assert decoded_error.error == "missing"


def test_codec_returns_typed_errors_for_invalid_envelopes_and_payloads() -> None:
    result_codec = codec(CONFIG)

    invalid_envelope = result_codec.deserialize({"status": "unknown"})
    assert isinstance(invalid_envelope, Err)
    assert isinstance(invalid_envelope.error, ResultDeserializationError)
    assert invalid_envelope.error.value == {"status": "unknown"}

    invalid_payload = result_codec.deserialize({"status": "ok", "value": "nope"})
    assert isinstance(invalid_payload, Err)
    assert isinstance(invalid_payload.error, ResultDeserializationError)
    assert invalid_payload.error.issues == ({"message": "expected digits"},)

    invalid_output = result_codec.serialize(Ok(-1))
    assert isinstance(invalid_output, Err)
    assert isinstance(invalid_output.error, ResultSerializationError)
    assert invalid_output.error.value == -1


def test_codec_wire_envelopes_match_golden_protocol_files() -> None:
    result_codec = codec(CONFIG)

    encoded_success = result_codec.serialize(Ok(42))
    assert isinstance(encoded_success, Ok)
    assert encoded_success.value == _golden_json("ok-envelope.json")

    encoded_error = result_codec.serialize(Err("missing"))
    assert isinstance(encoded_error, Ok)
    assert encoded_error.value == _golden_json("error-envelope.json")


def test_codec_fixture_inputs_and_boundary_errors_match_golden_files() -> None:
    result_codec = codec(CONFIG)

    decoded_success = result_codec.deserialize(_golden_json("ok-envelope.json"))
    assert isinstance(decoded_success, Ok)
    assert decoded_success.value == 42

    decoded_error = result_codec.deserialize(_golden_json("error-envelope.json"))
    assert isinstance(decoded_error, Err)
    assert decoded_error.error == "missing"

    invalid_envelope = result_codec.deserialize(_golden_json("invalid-envelope.json"))
    assert isinstance(invalid_envelope, Err)
    assert isinstance(invalid_envelope.error, ResultDeserializationError)
    assert _stable_codec_error(invalid_envelope.error) == _golden_json(
        "invalid-envelope-error.json",
    )

    invalid_payload = result_codec.deserialize(_golden_json("invalid-payload.json"))
    assert isinstance(invalid_payload, Err)
    assert isinstance(invalid_payload.error, ResultDeserializationError)
    assert _stable_codec_error(invalid_payload.error) == _golden_json(
        "invalid-payload-error.json",
    )

    invalid_output = result_codec.serialize(Ok(-1))
    assert isinstance(invalid_output, Err)
    assert isinstance(invalid_output.error, ResultSerializationError)
    assert _stable_codec_error(invalid_output.error) == _golden_json(
        "serialization-error.json",
    )


def test_codec_unsafe_methods_panic_only_on_codec_errors() -> None:
    result_codec = codec(CONFIG)

    assert result_codec.serialize_unsafe(Ok(42)) == {
        "status": "ok",
        "value": "42",
    }
    decoded_error = result_codec.deserialize_unsafe(
        {"status": "error", "error": "missing"},
    )
    assert isinstance(decoded_error, Err)
    assert decoded_error.error == "missing"

    with pytest.raises(Panic, match="serialize_unsafe failed"):
        result_codec.serialize_unsafe(Ok(-1))
    with pytest.raises(Panic, match="deserialize_unsafe failed"):
        result_codec.deserialize_unsafe({"status": "ok", "value": "nope"})


def test_missing_payload_is_passed_to_the_selected_schema() -> None:
    result_codec = codec(CONFIG)

    decoded = result_codec.deserialize({"status": "ok"})
    assert isinstance(decoded, Err)
    assert isinstance(decoded.error, ResultDeserializationError)
    assert decoded.error.value is None


@pytest.mark.asyncio
async def test_codec_preserves_schema_cancellation() -> None:
    async def cancel(_value: int) -> str:
        raise asyncio.CancelledError

    result_codec = codec(
        codec_config(
            serialize_ok=cancel,
            serialize_err=encode_error,
            deserialize_ok=decode_number,
            deserialize_err=decode_error,
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await result_codec.serialize_async(Ok(1))


@pytest.mark.asyncio
async def test_codec_supports_async_schemas() -> None:
    async def async_encode(value: int) -> str:
        await asyncio.sleep(0)
        return str(value)

    async def async_decode(value: str) -> int:
        await asyncio.sleep(0)
        return int(value)

    async_config = codec_config(
        serialize_ok=async_encode,
        serialize_err=encode_error,
        deserialize_ok=async_decode,
        deserialize_err=decode_error,
    )
    result_codec = codec(async_config)

    encoded = await result_codec.serialize_async(Ok(7))
    assert isinstance(encoded, Ok)
    assert encoded.value == {"status": "ok", "value": "7"}

    decoded = await result_codec.deserialize_async({"status": "ok", "value": "7"})
    assert isinstance(decoded, Ok)
    assert decoded.value == 7


@pytest.mark.asyncio
async def test_async_codec_uses_the_same_golden_wire_protocol() -> None:
    async def async_encode(value: int) -> str:
        await asyncio.sleep(0)
        return str(value)

    async def async_decode(value: str) -> int:
        await asyncio.sleep(0)
        return int(value)

    async_config = codec_config(
        serialize_ok=async_encode,
        serialize_err=encode_error,
        deserialize_ok=async_decode,
        deserialize_err=decode_error,
    )
    result_codec = codec(async_config)

    encoded = await result_codec.serialize_async(Ok(42))
    assert isinstance(encoded, Ok)
    assert encoded.value == _golden_json("ok-envelope.json")

    decoded = await result_codec.deserialize_async(_golden_json("ok-envelope.json"))
    assert isinstance(decoded, Ok)
    assert decoded.value == 42
