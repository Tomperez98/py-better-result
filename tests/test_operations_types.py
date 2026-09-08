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
    partition_results,
    partition_results_async,
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
