"""Static type contracts for collection operation overloads."""

from __future__ import annotations

from typing import Never, assert_type

import pytest

from better_result import (
    Err,
    Ok,
    Result,
    all_results,
    all_results_async,
    capture,
    capture_async,
    collect_results,
    collect_results_async,
    flatten_result,
    partition_results,
    partition_results_async,
    traverse,
    traverse_async,
)


def _int_value() -> int:
    return 1


def _text_value() -> str:
    return "two"


def test_collection_operations_preserve_heterogeneous_tuple_types() -> None:
    one = _int_value()
    two = _text_value()
    bad = _text_value()

    values = all_results((Ok(one), Ok(two)))
    assert_type(values, Result[tuple[int, str], Never])

    partitioned = partition_results((Ok(one), Err(bad)))
    assert_type(partitioned, tuple[list[int], list[str]])


@pytest.mark.asyncio
async def test_async_collection_operations_preserve_tuple_types() -> None:
    async def text_result() -> Ok[str]:
        return Ok("two")

    one = _int_value()
    values = await all_results_async((Ok(one), text_result()))
    assert_type(values, Result[tuple[int, str], Never])

    partitioned = await partition_results_async((Ok(one), text_result()))
    assert_type(partitioned, tuple[list[int | str], list[Never]])


def test_capture_type_inference() -> None:
    def ok_operation() -> int:
        return 42

    success = capture(ok_operation)
    assert_type(success, Result[int, Exception])

    mapped = capture(ok_operation, catch=str)
    assert_type(mapped, Result[int, str])


@pytest.mark.asyncio
async def test_capture_async_type_inference() -> None:
    async def op() -> int:
        return 42

    async def catch(exc: Exception) -> str:
        return str(exc)

    result = await capture_async(op)
    assert_type(result, Result[int, Exception])

    mapped = await capture_async(op, catch=catch)
    assert_type(mapped, Result[int, str])


def test_collect_results_type_inference() -> None:
    collected = collect_results([Ok(1), Err("bad")])
    assert_type(collected, Result[tuple[int, ...], tuple[str, ...]])

    heterogeneous = collect_results((_one_result(), _bad_result(), _two_result()))
    assert_type(
        heterogeneous,
        Result[tuple[int, Never, str], tuple[str]],
    )


def _one_result() -> Result[int, Never]:
    return Ok(1)


def _bad_result() -> Result[Never, str]:
    return Err("bad")


def _two_result() -> Result[str, Never]:
    return Ok("two")


def _nested_result() -> Result[Result[int, str], float]:
    return Ok(Ok(42))


def test_flatten_result_type_inference() -> None:
    flattened = flatten_result(_nested_result())
    assert_type(flattened, Result[int, str | float])


def test_traverse_type_inference() -> None:
    def to_result(value: int) -> Result[str, str]:
        return Ok(str(value))

    result = traverse([1, 2], to_result)
    assert_type(result, Result[tuple[str, ...], str])


@pytest.mark.asyncio
async def test_async_collection_helpers_type_inference() -> None:
    async def to_result(value: int) -> Result[str, ValueError]:
        return Ok(str(value))

    collected = await collect_results_async((to_result(1), to_result(2)))
    assert_type(collected, Result[tuple[str, str], tuple[ValueError]])

    heterogeneous = await collect_results_async(
        (_async_one_result(), _async_bad_result(), _async_two_result())
    )
    assert_type(
        heterogeneous,
        Result[tuple[int, Never, str], tuple[str]],
    )

    traversed = await traverse_async((1, 2), to_result, max_concurrency=1)
    assert_type(traversed, Result[tuple[str, ...], ValueError])


async def _async_one_result() -> Result[int, Never]:
    return Ok(1)


async def _async_bad_result() -> Result[Never, str]:
    return Err("bad")


async def _async_two_result() -> Result[str, Never]:
    return Ok("two")
