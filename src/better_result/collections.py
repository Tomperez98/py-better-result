"""Result collection and flattening helpers."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, cast

from better_result.core import Err, Ok, PanicError, Result, _require_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable


def all_results[A, E](results: Iterable[Result[A, E]]) -> Result[list[A], E]:
    """Collect success values in input order or return the first error."""
    values: list[A] = []
    for result in results:
        if isinstance(result, Err):
            return Err(result.error)
        values.append(result.value)
    return Ok(values)


async def _await_results[A, E](
    results: Iterable[Result[A, E] | Awaitable[Result[A, E]]],
) -> list[Result[A, E]]:
    awaitables: list[Awaitable[Result[A, E]]] = []
    for result in results:
        if inspect.isawaitable(result):
            awaitables.append(cast("Awaitable[Result[A, E]]", result))
        else:
            awaitables.append(_completed(result))

    async def await_one(awaitable: Awaitable[Result[A, E]]) -> Result[A, E]:
        return await awaitable

    tasks = [asyncio.create_task(await_one(awaitable)) for awaitable in awaitables]
    try:
        resolved = list(await asyncio.gather(*tasks))
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except PanicError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except Exception as cause:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        msg = "Result.all_async input awaitable rejected"
        raise PanicError(msg, cause) from cause

    return [
        cast(
            "Result[A, E]",
            _require_result(result, "Result.all_async input must be a Result"),
        )
        for result in resolved
    ]


async def _completed[T](value: T) -> T:
    return value


async def all_results_async[A, E](
    results: Iterable[Result[A, E] | Awaitable[Result[A, E]]],
) -> Result[list[A], E]:
    """Await all inputs concurrently and collect success values."""
    return all_results(await _await_results(results))


def partition[A, E](results: Iterable[Result[A, E]]) -> tuple[list[A], list[E]]:
    """Split results into success and error values while preserving order."""
    values: list[A] = []
    errors: list[E] = []
    for result in results:
        if isinstance(result, Ok):
            values.append(result.value)
        else:
            errors.append(result.error)
    return values, errors


async def partition_async[A, E](
    results: Iterable[Result[A, E] | Awaitable[Result[A, E]]],
) -> tuple[list[A], list[E]]:
    """Await all inputs concurrently and partition their payloads."""
    return partition(await _await_results(results))
