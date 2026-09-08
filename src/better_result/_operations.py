"""Standalone Result operations and boundary helpers."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, cast, overload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

from ._core import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class CancellationToken:
    """Cooperative cancellation signal for async operations and retry waits."""

    _event: asyncio.Event = field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise asyncio.CancelledError

    async def wait(self) -> None:
        await self._event.wait()


@dataclass(frozen=True, slots=True)
class TryContext:
    """Context passed to an exception-capturing operation attempt."""

    attempt: int
    cancel_token: CancellationToken | None = None


@dataclass(frozen=True, slots=True)
class RetryPolicy[E]:
    """
    Retry controls for :func:`try_async`.

    ``times`` counts retries after the initial attempt. Delays are in seconds.
    A callable delay receives the mapped error and the failed attempt context.
    """

    times: int
    delay: float | Callable[[E, TryContext], float] = 0
    backoff: str = "constant"
    should_retry: Callable[[E, TryContext], bool] | None = None
    jitter: bool | float = False


@overload
def try_result[T](
    operation: Callable[[TryContext], T],
    *,
    catch: None = None,
    retry: int = 0,
) -> Result[T, Exception]: ...


@overload
def try_result[T, E](
    operation: Callable[[TryContext], T],
    *,
    catch: Callable[[Exception], E],
    retry: int = 0,
) -> Result[T, E]: ...


def try_result[T](
    operation: Callable[[TryContext], T],
    *,
    catch: Callable[[Exception], object] | None = None,
    retry: int = 0,
) -> Result[T, object]:
    """Execute a synchronous operation and return expected exceptions as Err."""
    if retry < 0:
        _raise_retry_policy_error("retry must be non-negative")

    context = TryContext(attempt=1)
    retry_index = 0
    while True:
        try:
            return cast("Result[T, object]", Ok(operation(context)))
        except Exception as cause:
            mapped = catch(cause) if catch is not None else cause
            if retry_index >= retry:
                return cast("Result[T, object]", Err(mapped))
            retry_index += 1
            context = TryContext(attempt=context.attempt + 1)


def _raise_retry_policy_error(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_retry_policy[E](policy: RetryPolicy[E]) -> None:
    if policy.times < 0:
        _raise_retry_policy_error("retry times must be non-negative")
    if policy.backoff not in ("constant", "linear", "exponential"):
        _raise_retry_policy_error(
            "retry backoff must be constant, linear, or exponential"
        )

    if isinstance(policy.jitter, bool):
        jitter_factor = 1.0 if policy.jitter else 0.0
    else:
        jitter_factor = policy.jitter
        if not math.isfinite(jitter_factor) or not 0 <= jitter_factor <= 1:
            _raise_retry_policy_error(
                "retry jitter must be a finite number between 0 and 1"
            )

    if callable(policy.delay):
        if policy.backoff != "constant":
            _raise_retry_policy_error("dynamic retry delay cannot use backoff")
        if jitter_factor:
            _raise_retry_policy_error("dynamic retry delay cannot use jitter")
        return

    delay = cast("float", policy.delay)
    if not math.isfinite(delay) or delay < 0:
        _raise_retry_policy_error("retry delay must be a finite non-negative number")


def _retry_delay[E](
    policy: RetryPolicy[E],
    error: E,
    context: TryContext,
    retry_index: int,
) -> float:
    if callable(policy.delay):
        delay_fn = cast("Callable[[E, TryContext], float]", policy.delay)
        delay = delay_fn(error, context)
    else:
        base_delay = cast("float", policy.delay)
        if policy.backoff == "constant":
            delay = base_delay
        elif policy.backoff == "linear":
            delay = base_delay * (retry_index + 1)
        else:
            delay = base_delay * 2**retry_index

    delay = float(delay)
    if not math.isfinite(delay) or delay < 0:
        _raise_retry_policy_error("retry delay must be a finite non-negative number")

    if callable(policy.delay) or not policy.jitter:
        return delay
    jitter_factor = 1.0 if policy.jitter is True else policy.jitter
    return delay * (1 - jitter_factor + random.SystemRandom().random() * jitter_factor)


async def _wait_for_retry(
    delay: float,
    cancel_token: CancellationToken | None,
) -> bool:
    if cancel_token is None:
        await asyncio.sleep(delay)
        return True
    if cancel_token.is_cancelled:
        return False
    try:
        await asyncio.wait_for(cancel_token.wait(), timeout=delay)
    except TimeoutError:
        return True
    return False


async def _map_async_exception[E](
    cause: Exception,
    catch: Callable[[Exception], E | Awaitable[E]] | None,
) -> E | Exception:
    if catch is None:
        return cause
    mapped = catch(cause)
    if inspect.isawaitable(mapped):
        return await cast("Awaitable[E]", mapped)
    return cast("E", mapped)


@overload
async def try_async[T](
    operation: Callable[[TryContext], Awaitable[T]],
    *,
    catch: None = None,
    retry: RetryPolicy[Exception] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, Exception]: ...


@overload
async def try_async[T, E](
    operation: Callable[[TryContext], Awaitable[T]],
    *,
    catch: Callable[[Exception], E | Awaitable[E]],
    retry: RetryPolicy[E] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, E]: ...


async def try_async[T](
    operation: Callable[[TryContext], Awaitable[T]],
    *,
    catch: Callable[[Exception], object | Awaitable[object]] | None = None,
    retry: RetryPolicy[object] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, object]:
    """Execute an async operation with optional mapping and bounded retries."""
    if retry is not None:
        _validate_retry_policy(retry)

    context = TryContext(attempt=1, cancel_token=cancel_token)
    retry_index = 0
    while True:
        try:
            return cast("Result[T, object]", Ok(await operation(context)))
        except Exception as cause:
            error = await _map_async_exception(cause, catch)
            if retry is None or retry_index >= retry.times:
                return cast("Result[T, object]", Err(error))
            if retry.should_retry is not None and not retry.should_retry(
                error, context
            ):
                return cast("Result[T, object]", Err(error))
            should_continue = await _wait_for_retry(
                _retry_delay(retry, error, context, retry_index),
                cancel_token,
            )
            if not should_continue:
                return cast("Result[T, object]", Err(error))
            retry_index += 1
            context = TryContext(
                attempt=context.attempt + 1,
                cancel_token=cancel_token,
            )


def flatten_result[T, E, F](
    result: Result[Result[T, E], F],
) -> Result[T, E | F]:
    """Flatten a nested Result while preserving either error value."""
    if isinstance(result, Ok):
        return result.ok_value
    return cast("Result[T, E | F]", result)


@overload
def all_results[A, E](results: tuple[Result[A, E]]) -> Result[tuple[A], E]: ...


@overload
def all_results[A, E, B, F](
    results: tuple[Result[A, E], Result[B, F]],
) -> Result[tuple[A, B], E | F]: ...


@overload
def all_results[A, E, B, F, C, G](
    results: tuple[Result[A, E], Result[B, F], Result[C, G]],
) -> Result[tuple[A, B, C], E | F | G]: ...


@overload
def all_results[A, E, B, F, C, G, D, H](
    results: tuple[Result[A, E], Result[B, F], Result[C, G], Result[D, H]],
) -> Result[tuple[A, B, C, D], E | F | G | H]: ...


@overload
def all_results[T, E](results: Iterable[Result[T, E]]) -> Result[tuple[T, ...], E]: ...


def all_results[T, E](results: Iterable[Result[T, E]]) -> Result[tuple[T, ...], E]:
    """Collect all success values, or return the first error in input order."""
    values: list[T] = []
    for result in results:
        if isinstance(result, Err):
            return Err(result.err_value)
        assert isinstance(result, Ok)
        values.append(result.ok_value)
    return Ok(tuple(values))


@overload
def partition_results[A, E](
    results: tuple[Result[A, E]],
) -> tuple[list[A], list[E]]: ...


@overload
def partition_results[A, E, B, F](
    results: tuple[Result[A, E], Result[B, F]],
) -> tuple[list[A | B], list[E | F]]: ...


@overload
def partition_results[A, E, B, F, C, G](
    results: tuple[Result[A, E], Result[B, F], Result[C, G]],
) -> tuple[list[A | B | C], list[E | F | G]]: ...


@overload
def partition_results[A, E, B, F, C, G, D, H](
    results: tuple[Result[A, E], Result[B, F], Result[C, G], Result[D, H]],
) -> tuple[list[A | B | C | D], list[E | F | G | H]]: ...


@overload
def partition_results[T, E](
    results: Iterable[Result[T, E]],
) -> tuple[list[T], list[E]]: ...


def partition_results[T, E](
    results: Iterable[Result[T, E]],
) -> tuple[list[T], list[E]]:
    """Split Results into success and error values while preserving order."""
    values: list[T] = []
    errors: list[E] = []
    for result in results:
        if isinstance(result, Ok):
            values.append(result.ok_value)
        else:
            assert isinstance(result, Err)
            errors.append(result.err_value)
    return values, errors


async def _resolve_result[T, E](
    result: Result[T, E] | Awaitable[Result[T, E]],
) -> Result[T, E]:
    if isinstance(result, Result):
        return cast("Result[T, E]", result)
    return await result


@overload
async def all_results_async[A, E](
    results: tuple[Result[A, E] | Awaitable[Result[A, E]]],
) -> Result[tuple[A], E]: ...


@overload
async def all_results_async[A, E, B, F](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
    ],
) -> Result[tuple[A, B], E | F]: ...


@overload
async def all_results_async[A, E, B, F, C, G](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
    ],
) -> Result[tuple[A, B, C], E | F | G]: ...


@overload
async def all_results_async[A, E, B, F, C, G, D, H](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
        Result[D, H] | Awaitable[Result[D, H]],
    ],
) -> Result[tuple[A, B, C, D], E | F | G | H]: ...


@overload
async def all_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> Result[tuple[T, ...], E]: ...


async def all_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> Result[tuple[T, ...], E]:
    """Await Results concurrently, then collect successes in input order."""
    resolved = cast(
        "list[Result[T, E]]",
        await asyncio.gather(*(_resolve_result(result) for result in results)),
    )
    return all_results(resolved)


@overload
async def partition_results_async[A, E](
    results: tuple[Result[A, E] | Awaitable[Result[A, E]]],
) -> tuple[list[A], list[E]]: ...


@overload
async def partition_results_async[A, E, B, F](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
    ],
) -> tuple[list[A | B], list[E | F]]: ...


@overload
async def partition_results_async[A, E, B, F, C, G](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
    ],
) -> tuple[list[A | B | C], list[E | F | G]]: ...


@overload
async def partition_results_async[A, E, B, F, C, G, D, H](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
        Result[D, H] | Awaitable[Result[D, H]],
    ],
) -> tuple[list[A | B | C | D], list[E | F | G | H]]: ...


@overload
async def partition_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> tuple[list[T], list[E]]: ...


async def partition_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> tuple[list[T], list[E]]:
    """Await Results concurrently, then partition values in input order."""
    resolved = cast(
        "list[Result[T, E]]",
        await asyncio.gather(*(_resolve_result(result) for result in results)),
    )
    return partition_results(resolved)
