"""Compose async boundaries, collect results, and encode the final result."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from better_result import (
    Err,
    Ok,
    Result,
    SchemaFailure,
    async_codec,
    capture_async,
    collect_results_async,
    is_ok,
    partition_results_async,
)


async def fetch_score(raw: str) -> int:
    await asyncio.sleep(0)
    return int(raw)


async def add_prefix(error: str) -> str:
    await asyncio.sleep(0)
    return f"score error: {error}"


async def fetch_score_result(raw: str) -> Result[int, str]:
    parsed = await capture_async(lambda: fetch_score(raw), catch=str)
    return await parsed.map_err_async(add_prefix)


async def serialize_score(score: int) -> dict[str, int]:
    await asyncio.sleep(0)
    return {"score": score}


async def serialize_error(error: str) -> dict[str, str]:
    await asyncio.sleep(0)
    return {"message": error}


async def deserialize_score(value: object) -> int | SchemaFailure:
    await asyncio.sleep(0)
    if not isinstance(value, Mapping):
        return SchemaFailure([{"message": "score must be an integer", "path": []}])
    score = value.get("score")
    if not isinstance(score, int):
        return SchemaFailure([{"message": "score must be an integer", "path": []}])
    return score


async def deserialize_error(value: object) -> str | SchemaFailure:
    await asyncio.sleep(0)
    if not isinstance(value, Mapping):
        return SchemaFailure([{"message": "message must be a string", "path": []}])
    message = value.get("message")
    if not isinstance(message, str):
        return SchemaFailure([{"message": "message must be a string", "path": []}])
    return message


async def main() -> None:
    results = await collect_results_async(
        (
            fetch_score_result("10"),
            fetch_score_result("bad"),
            fetch_score_result("20"),
        )
    )
    expected_error = "score error: invalid literal for int() with base 10: 'bad'"
    assert results == Err((expected_error,))
    print(f"collected: {results}")

    values, errors = await partition_results_async(
        (
            fetch_score_result("10"),
            fetch_score_result("bad"),
            fetch_score_result("20"),
        )
    )
    assert values == [10, 20]
    assert errors == [expected_error]
    print(f"partitioned: values={values}, errors={errors}")

    result_codec = async_codec(
        serialize_ok=serialize_score,
        serialize_err=serialize_error,
        deserialize_ok=deserialize_score,
        deserialize_err=deserialize_error,
    )
    encoded = await result_codec.serialize(Ok(42))
    assert is_ok(encoded)
    assert encoded == Ok({"status": "ok", "value": {"score": 42}})
    print(f"encoded: {encoded}")
    decoded = await result_codec.deserialize(encoded.ok_value)
    assert decoded == Ok(42)
    print(f"decoded: {decoded}")

    # A wire-level error remains a domain Err after decoding.
    decoded_error = await result_codec.deserialize(
        {"status": "error", "error": {"message": "offline"}}
    )
    print(f"decoded error: {decoded_error}")
    assert decoded_error == Err("offline")


if __name__ == "__main__":
    asyncio.run(main())
