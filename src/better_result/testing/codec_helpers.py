"""Assertions for Result codec invariants."""

from __future__ import annotations

from better_result import (
    AsyncResultCodec,
    Ok,
    Result,
    ResultCodec,
    SerializedResult,
    is_ok,
)


def assert_codec_roundtrip[
    OkInput,
    ErrInput,
    OkWire,
    ErrWire,
    OkOutput,
    ErrOutput,
](
    result_codec: ResultCodec[
        OkInput,
        ErrInput,
        OkWire,
        ErrWire,
        OkOutput,
        ErrOutput,
    ],
    source: Result[OkInput, ErrInput],
    *,
    expected_wire: SerializedResult[OkWire, ErrWire],
    expected_result: Result[OkOutput, ErrOutput],
) -> None:
    """Assert encoding and decoding preserve a Result branch and payload."""
    encoded = result_codec.serialize(source)
    assert encoded == Ok(expected_wire)
    assert is_ok(encoded)
    assert result_codec.deserialize(encoded.ok_value) == expected_result


async def assert_async_codec_roundtrip[
    OkInput,
    ErrInput,
    OkWire,
    ErrWire,
    OkOutput,
    ErrOutput,
](
    result_codec: AsyncResultCodec[
        OkInput,
        ErrInput,
        OkWire,
        ErrWire,
        OkOutput,
        ErrOutput,
    ],
    source: Result[OkInput, ErrInput],
    *,
    expected_wire: SerializedResult[OkWire, ErrWire],
    expected_result: Result[OkOutput, ErrOutput],
) -> None:
    """Assert the asynchronous codec's equivalent roundtrip invariant."""
    encoded = await result_codec.serialize(source)
    assert encoded == Ok(expected_wire)
    assert is_ok(encoded)
    assert await result_codec.deserialize(encoded.ok_value) == expected_result
