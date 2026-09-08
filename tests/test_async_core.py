"""Tests for the narrow asynchronous Result API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import pytest

from better_result import Err, Ok, PanicError, Result


@pytest.mark.asyncio
async def test_and_then_async_composes_and_short_circuits() -> None:
    success = await Ok(2).and_then_async(lambda value: _completed(Ok(str(value))))
    failure = await Err("bad").and_then_async(lambda _: _completed(Ok("unused")))

    assert success == Ok("2")
    assert failure == Err("bad")


@pytest.mark.asyncio
async def test_and_then_async_validates_callbacks_and_preserves_cancellation() -> None:
    async def invalid(_value: int) -> object:
        return "not a Result"

    invalid_result = cast(
        "Callable[[int], Awaitable[Result[object, object]]]",
        invalid,
    )
    with pytest.raises(PanicError, match="must return a Result"):
        await Ok(1).and_then_async(invalid_result)

    async def broken(_value: int) -> object:
        message = "broken"
        raise RuntimeError(message)

    broken_result = cast(
        "Callable[[int], Awaitable[Result[object, object]]]",
        broken,
    )
    with pytest.raises(PanicError, match="callback threw"):
        await Ok(1).and_then_async(broken_result)

    async def cancelled(_value: int) -> object:
        raise asyncio.CancelledError

    cancelled_result = cast(
        "Callable[[int], Awaitable[Result[object, object]]]",
        cancelled,
    )
    with pytest.raises(asyncio.CancelledError):
        await Ok(1).and_then_async(cancelled_result)


async def _completed[T](value: T) -> T:
    await asyncio.sleep(0)
    return value
