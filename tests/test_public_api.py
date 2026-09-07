"""Smoke tests for the supported package-level API."""

from __future__ import annotations

import better_result


def test_package_root_exports_core_and_combinator_api() -> None:
    assert better_result.ok(2).map(lambda value: value + 1).value == 3
    assert isinstance(better_result.err("bad"), better_result.Err)
    assert better_result.unwrap_or(better_result.ok(1), 0) == 1
    mapped = better_result.map_result(str)(better_result.ok(2))
    assert isinstance(mapped, better_result.Ok)
    assert mapped.value == "2"


def test_package_root_exports_codec_and_error_api() -> None:
    assert better_result.TaggedError.is_tagged_error(
        better_result.TaggedError(message="bad"),
    )
    configuration = better_result.codec_config(
        serialize_ok=str,
        serialize_err=str,
        deserialize_ok=int,
        deserialize_err=str,
    )
    result_codec = better_result.codec(configuration)
    encoded = result_codec.serialize(better_result.ok(2))
    assert isinstance(encoded, better_result.Ok)
    assert encoded.value == {"status": "ok", "value": "2"}
