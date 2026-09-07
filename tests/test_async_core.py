"""Async tests for the Result core."""

from __future__ import annotations

import asyncio
from typing import Never, TypeVar

import pytest

from better_result.core import Err, Ok, Panic, Result, err, ok

Value = TypeVar("Value")


async def _completed(value: Value) -> Value:  # noqa: UP047
    """Return a value after yielding control once."""
    await asyncio.sleep(0)
    return value


@pytest.mark.asyncio
async def test_ok_async_combinators_chain_and_return_self() -> None:
    seen: list[object] = []
    success = ok(2)

    mapped = await success.and_then_async(lambda value: _completed(ok(value + 1)))
    assert isinstance(mapped, Ok)
    assert mapped.value == 3

    tapped = await success.tap_async(lambda value: _completed(seen.append(value)))
    assert tapped is success
    assert seen == [2]

    both = await success.tap_both_async(
        {"ok": lambda value: _completed(seen.append(value)), "err": _completed}
    )
    assert both is success
    assert seen == [2, 2]

    recovered = await success.try_recover_async(
        lambda _: _completed(err("must not run"))
    )
    assert recovered is success


@pytest.mark.asyncio
async def test_err_async_combinators_short_circuit_or_recover() -> None:
    seen: list[object] = []
    failure = err("missing")

    chained = await failure.and_then_async(lambda _: _completed(ok("must not run")))
    assert isinstance(chained, Err)
    assert chained.error == "missing"

    tapped = await failure.tap_async(lambda value: _completed(seen.append(value)))
    assert tapped is failure
    assert seen == []

    tapped_error = await failure.tap_error_async(
        lambda value: _completed(seen.append(value))
    )
    assert tapped_error is failure
    assert seen == ["missing"]

    both = await failure.tap_both_async(
        {"ok": _completed, "err": lambda value: _completed(seen.append(value))}
    )
    assert both is failure
    assert seen == ["missing", "missing"]

    recovered = await failure.try_recover_async(
        lambda value: _completed(ok(len(value)))
    )
    assert isinstance(recovered, Ok)
    assert recovered.value == 7


@pytest.mark.asyncio
async def test_async_callback_exceptions_become_panics() -> None:
    async def rejected(_: int) -> Result[int, str]:
        raise ValueError("network unavailable")

    with pytest.raises(Panic, match="and_then_async callback threw") as raised:
        await ok(1).and_then_async(rejected)
    assert isinstance(raised.value.cause, ValueError)

    async def rejected_side_effect(_: int) -> object:
        raise RuntimeError("side effect failed")

    with pytest.raises(Panic, match="tap_async callback threw"):
        await ok(1).tap_async(rejected_side_effect)

    async def rejected_error_side_effect(_: str) -> object:
        raise RuntimeError("side effect failed")

    with pytest.raises(Panic, match="tap_error_async callback threw"):
        await err("bad").tap_error_async(rejected_error_side_effect)


@pytest.mark.asyncio
async def test_async_noop_callbacks_are_not_called() -> None:
    called = False

    async def should_not_run(_: object) -> Result[str, Never]:
        nonlocal called
        called = True
        return ok("unexpected")

    await err("bad").and_then_async(should_not_run)
    await ok(1).try_recover_async(should_not_run)
    await err("bad").tap_async(should_not_run)
    await ok(1).tap_error_async(should_not_run)

    assert not called
