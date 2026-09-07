"""Async tests for the Result core."""

from __future__ import annotations

import asyncio
from typing import Never, TypeVar, cast

import pytest

from better_result.core import Err, Ok, PanicError, Result

Value = TypeVar("Value")


async def _completed[T](value: T) -> T:
    """Return a value after yielding control once."""
    await asyncio.sleep(0)
    return value


@pytest.mark.asyncio
async def test_ok_async_combinators_chain_and_return_self() -> None:
    seen: list[object] = []
    success = Ok(2)

    mapped = await success.and_then_async(lambda value: _completed(Ok(value + 1)))
    assert isinstance(mapped, Ok)
    assert mapped.value == 3

    tapped = await success.tap_async(lambda value: _completed(seen.append(value)))
    assert tapped is success
    assert seen == [2]

    recovered = await success.try_recover_async(
        lambda _: _completed(Err("must not run")),
    )
    assert recovered is success


@pytest.mark.asyncio
async def test_err_async_combinators_short_circuit_or_recover() -> None:
    seen: list[object] = []
    failure = Err("missing")

    chained = await failure.and_then_async(lambda _: _completed(Ok("must not run")))
    assert isinstance(chained, Err)
    assert chained.error == "missing"

    tapped = await failure.tap_async(lambda value: _completed(seen.append(value)))
    assert tapped is failure
    assert seen == []

    tapped_error = await failure.tap_error_async(
        lambda value: _completed(seen.append(value)),
    )
    assert tapped_error is failure
    assert seen == ["missing"]

    recovered = await failure.try_recover_async(
        lambda value: _completed(Ok(len(value))),
    )
    assert isinstance(recovered, Ok)
    assert recovered.value == 7


@pytest.mark.asyncio
async def test_async_callback_exceptions_become_panics() -> None:
    async def rejected(_: int) -> Result[int, str]:
        msg = "network unavailable"
        raise ValueError(msg)

    with pytest.raises(PanicError, match="and_then_async callback threw") as raised:
        await Ok(1).and_then_async(rejected)
    assert isinstance(raised.value.cause, ValueError)

    async def rejected_side_effect(_: int) -> None:
        msg = "side effect failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="tap_async callback threw"):
        await Ok(1).tap_async(rejected_side_effect)

    async def rejected_error_side_effect(_: str) -> None:
        msg = "side effect failed"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match="tap_error_async callback threw"):
        await Err("bad").tap_error_async(rejected_error_side_effect)


@pytest.mark.asyncio
async def test_async_callback_cancellation_propagates() -> None:
    async def cancel(_: int) -> Result[int, str]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await Ok(1).and_then_async(cancel)


@pytest.mark.asyncio
async def test_async_callbacks_must_return_results() -> None:
    async def invalid(_: int) -> Result[object, object]:
        return cast("Result[object, object]", object())

    with pytest.raises(
        PanicError, match="and_then_async callback must return a Result"
    ):
        await Ok(1).and_then_async(invalid)


@pytest.mark.asyncio
async def test_async_noop_callbacks_are_not_called() -> None:
    called = False

    async def should_not_run_result(_: object) -> Result[str, Never]:
        nonlocal called
        called = True
        return Ok("unexpected")

    async def should_not_run_side_effect(_: Never) -> None:
        nonlocal called
        called = True

    await Err("bad").and_then_async(should_not_run_result)
    await Ok(1).try_recover_async(should_not_run_result)
    await Err("bad").tap_async(should_not_run_side_effect)
    await Ok(1).tap_error_async(should_not_run_side_effect)

    assert not called
