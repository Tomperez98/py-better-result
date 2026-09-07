"""
Result-level utilities for synchronous and asynchronous workflows.

The module mirrors the utility portion of the TypeScript ``result.ts`` API.
Python callers compose coroutines with ``await`` directly; generator-based
``yield*`` composition is intentionally not part of this API.
"""

# ``map`` and ``all`` intentionally mirror the Result API, despite shadowing builtins.
# ruff: noqa: A001

from __future__ import annotations

import asyncio
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from .codec import (
    ResultCodec,
    ResultCodecConfig,
    Schema,
    SchemaFailure,
    SerializedErr,
    SerializedOk,
    SerializedResult,
    codec,
    codec_config,
)
from .core import Err, Ok, Panic, Result, _require_result
from .error import UnhandledException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class TryContext:
    """Context passed to each synchronous attempt."""

    attempt: int


@dataclass(frozen=True, slots=True)
class TryAsyncContext:
    """Context passed to each asynchronous attempt and retry decision."""

    attempt: int
    cancel_event: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Number of additional attempts for a synchronous operation."""

    times: int = 0


@dataclass(frozen=True)
class AsyncRetryConfig[E]:
    """
    Retry policy for :func:`try_async`.

    ``delay_ms`` may be a fixed delay or a callback that computes the delay
    from the current error and retry context. Delays are kept in milliseconds
    to match the TypeScript API.
    """

    times: int = 0
    delay_ms: (
        float
        | Callable[
            [E, TryAsyncContext],
            float | Awaitable[float],
        ]
    ) = 0.0
    backoff: Literal["linear", "constant", "exponential"] = "constant"
    should_retry: Callable[[E, TryAsyncContext], bool | Awaitable[bool]] | None = None
    jitter: bool | float = False
    cancel_event: asyncio.Event | None = None


def _retry_times(retry: int | RetryConfig | None) -> int:
    if retry is None:
        return 0
    times = retry if isinstance(retry, int) else retry.times
    if times < 0:
        message = f"retry times must not be negative, got {times}"
        raise ValueError(message)
    return times


@overload
def try_result[A](
    operation: Callable[[TryContext], A],
    catch: None = None,
    retry: int | RetryConfig | None = None,
) -> Result[A, UnhandledException]: ...


@overload
def try_result[A, E](
    operation: Callable[[TryContext], A],
    catch: Callable[[BaseException], E],
    retry: int | RetryConfig | None = None,
) -> Result[A, E]: ...


def try_result[A, E](
    operation: Callable[[TryContext], A],
    catch: Callable[[BaseException], E] | None = None,
    retry: int | RetryConfig | None = None,
) -> Result[A, E | UnhandledException]:
    """
    Run a callback and return its value or a handled failure.

    Callback failures are expected operation failures and become
    ``UnhandledException`` when no ``catch`` callback is supplied. A failing
    ``catch`` callback is a programmer error and becomes ``Panic``.
    """
    times = _retry_times(retry)

    def execute(context: TryContext) -> Result[A, E | UnhandledException]:
        try:
            return Ok[A, E | UnhandledException](operation(context))
        except Panic:
            raise
        except Exception as cause:
            if catch is None:
                return Err[A, E | UnhandledException](UnhandledException(cause))
            try:
                return Err[A, E | UnhandledException](catch(cause))
            except Panic:
                raise
            except Exception as catch_error:
                raise Panic(
                    "Result.try catch handler threw",
                    catch_error,
                ) from catch_error

    result: Result[A, E | UnhandledException] = execute(TryContext(attempt=1))
    for attempt in range(2, times + 2):
        if isinstance(result, Ok):
            break
        result = execute(TryContext(attempt=attempt))
    return result


async def _await_value[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return cast("T", await value)
    return value


async def _sleep_for_retry(
    delay_ms: float,
    cancel_event: asyncio.Event | None,
) -> bool:
    if cancel_event is None:
        await asyncio.sleep(max(0.0, delay_ms / 1000))
        return True
    if cancel_event.is_set():
        return False
    if delay_ms <= 0:
        await asyncio.sleep(0)
        return not cancel_event.is_set()
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay_ms / 1000)
    except TimeoutError:
        return not cancel_event.is_set()
    return False


def _validate_delay(delay_ms: float) -> float:
    try:
        delay = float(delay_ms)
    except (TypeError, ValueError) as cause:
        raise Panic("Result.try_async retry delay must be a number", cause) from cause
    if not math.isfinite(delay) or delay < 0:
        raise Panic("Result.try_async retry delay must be finite and non-negative")
    return delay


def _static_retry_delay(
    delay_ms: float,
    backoff: Literal["linear", "constant", "exponential"],
    retry_attempt: int,
) -> float:
    if backoff == "constant":
        return delay_ms
    if backoff == "linear":
        return delay_ms * (retry_attempt + 1)
    return delay_ms * float(2**retry_attempt)


def _jitter_factor(*, jitter: bool | float) -> float:
    if jitter is True:
        return 1.0
    if jitter is False:
        return 0.0
    if not math.isfinite(jitter) or not 0 <= jitter <= 1:
        raise Panic(
            "Result.try_async retry jitter must be a finite number between 0 and 1",
        )
    return jitter


@overload
async def try_async[A](
    operation: Callable[[TryAsyncContext], Awaitable[A]],
    catch: None = None,
    retry: AsyncRetryConfig[UnhandledException] | None = None,
) -> Result[A, UnhandledException]: ...


@overload
async def try_async[A, E](
    operation: Callable[[TryAsyncContext], Awaitable[A]],
    catch: Callable[[BaseException], E | Awaitable[E]],
    retry: AsyncRetryConfig[E] | None = None,
) -> Result[A, E]: ...


async def try_async[A, E](
    operation: Callable[[TryAsyncContext], Awaitable[A]],
    catch: Callable[[BaseException], E | Awaitable[E]] | None = None,
    retry: AsyncRetryConfig[E] | None = None,
) -> Result[A, E | UnhandledException]:
    """Run an async callback with optional retry, backoff, and cancellation."""
    policy = retry or AsyncRetryConfig[E]()
    if policy.times < 0:
        message = f"retry times must not be negative, got {policy.times}"
        raise ValueError(message)
    if policy.backoff not in {"constant", "linear", "exponential"}:
        msg = f"unsupported retry backoff: {policy.backoff!r}"
        raise ValueError(msg)
    if not callable(policy.delay_ms):
        _validate_delay(policy.delay_ms)
    jitter_factor = _jitter_factor(jitter=policy.jitter)

    async def execute(context: TryAsyncContext) -> Result[A, E | UnhandledException]:
        try:
            return Ok[A, E | UnhandledException](await operation(context))
        except asyncio.CancelledError:
            raise
        except Panic:
            raise
        except Exception as cause:
            if catch is None:
                return Err[A, E | UnhandledException](UnhandledException(cause))
            try:
                return Err[A, E | UnhandledException](await _await_value(catch(cause)))
            except asyncio.CancelledError:
                raise
            except Panic:
                raise
            except Exception as catch_error:
                raise Panic(
                    "Result.try_async catch handler threw",
                    catch_error,
                ) from catch_error

    context = TryAsyncContext(attempt=1, cancel_event=policy.cancel_event)
    result = await execute(context)
    should_retry = policy.should_retry or (lambda _error, _context: True)

    for retry_attempt in range(policy.times):
        if not isinstance(result, Err) or (
            policy.cancel_event is not None and policy.cancel_event.is_set()
        ):
            break
        error = cast("E", result.error)
        retry_predicate = cast(
            "Callable[[E, TryAsyncContext], bool | Awaitable[bool]]",
            should_retry,
        )
        try:
            continue_retry = await _await_value(retry_predicate(error, context))
        except asyncio.CancelledError:
            raise
        except Panic:
            raise
        except Exception as cause:
            raise Panic(
                "Result.try_async should_retry predicate threw",
                cause,
            ) from cause
        if not continue_retry:
            break

        if callable(policy.delay_ms):
            delay_callback = cast(
                "Callable[[E, TryAsyncContext], float | Awaitable[float]]",
                policy.delay_ms,
            )
            try:
                delay_ms = await _await_value(delay_callback(error, context))
            except asyncio.CancelledError:
                raise
            except Panic:
                raise
            except Exception as cause:
                raise Panic(
                    "Result.try_async delay_ms callback threw",
                    cause,
                ) from cause
        else:
            delay_ms = _static_retry_delay(
                policy.delay_ms,
                policy.backoff,
                retry_attempt,
            )
            delay_ms *= 1 - jitter_factor + random.random() * jitter_factor  # noqa: S311
        delay_ms = _validate_delay(delay_ms)
        if not await _sleep_for_retry(delay_ms, policy.cancel_event):
            break
        context = TryAsyncContext(
            attempt=context.attempt + 1,
            cancel_event=policy.cancel_event,
        )
        result = await execute(context)

    return result


try_ = try_result
try_promise = try_async


@overload
def map[A, B, E](result: Result[A, E], fn: Callable[[A], B]) -> Result[B, E]: ...


@overload
def map[A, B, E](fn: Callable[[A], B]) -> Callable[[Result[A, E]], Result[B, E]]: ...


def map[A, B, E](
    result_or_fn: Result[A, E] | Callable[[A], B],
    fn: Callable[[A], B] | None = None,
) -> Result[B, E] | Callable[[Result[A, E]], Result[B, E]]:
    """Map a success value in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("map data-last form requires a callback")
        return lambda result: result.map(result_or_fn)
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("map data-first form requires a Result")
    return cast("Result[B, E]", result_or_fn.map(fn))


@overload
def map_error[A, E, E2](
    result: Result[A, E],
    fn: Callable[[E], E2],
) -> Result[A, E2]: ...


@overload
def map_error[A, E, E2](
    fn: Callable[[E], E2],
) -> Callable[[Result[A, E]], Result[A, E2]]: ...


def map_error[A, E, E2](
    result_or_fn: Result[A, E] | Callable[[E], E2],
    fn: Callable[[E], E2] | None = None,
) -> Result[A, E2] | Callable[[Result[A, E]], Result[A, E2]]:
    """Map an error value in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("map_error data-last form requires a callback")
        return lambda result: result.map_error(result_or_fn)
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("map_error data-first form requires a Result")
    return cast("Result[A, E2]", result_or_fn.map_error(fn))


@overload
def try_recover[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[E], Result[B, E2]],
) -> Result[A | B, E2]: ...


@overload
def try_recover[A, E, B, E2](
    fn: Callable[[E], Result[B, E2]],
) -> Callable[[Result[A, E]], Result[A | B, E2]]: ...


def try_recover[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[E], Result[B, E2]],
    fn: Callable[[E], Result[B, E2]] | None = None,
) -> Result[A | B, E2] | Callable[[Result[A, E]], Result[A | B, E2]]:
    """Recover an error in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("try_recover data-last form requires a callback")
        return lambda result: cast(
            "Result[A | B, E2]",
            result.try_recover(result_or_fn),
        )
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("try_recover data-first form requires a Result")
    return cast("Result[A | B, E2]", result_or_fn.try_recover(fn))


@overload
def try_recover_async[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[E], Awaitable[Result[B, E2]]],
) -> Awaitable[Result[A | B, E2]]: ...


@overload
def try_recover_async[A, E, B, E2](
    fn: Callable[[E], Awaitable[Result[B, E2]]],
) -> Callable[[Result[A, E]], Awaitable[Result[A | B, E2]]]: ...


def try_recover_async[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[E], Awaitable[Result[B, E2]]],
    fn: Callable[[E], Awaitable[Result[B, E2]]] | None = None,
) -> (
    Awaitable[Result[A | B, E2]]
    | Callable[[Result[A, E]], Awaitable[Result[A | B, E2]]]
):
    """Async recovery in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("try_recover_async data-last form requires a callback")

        async def recover_later(result: Result[A, E]) -> Result[A | B, E2]:
            return cast(
                "Result[A | B, E2]",
                await result.try_recover_async(result_or_fn),
            )

        return recover_later
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("try_recover_async data-first form requires a Result")
    return cast("Awaitable[Result[A | B, E2]]", result_or_fn.try_recover_async(fn))


@overload
def and_then[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[A], Result[B, E2]],
) -> Result[B, E | E2]: ...


@overload
def and_then[A, E, B, E2](
    fn: Callable[[A], Result[B, E2]],
) -> Callable[[Result[A, E]], Result[B, E | E2]]: ...


def and_then[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[A], Result[B, E2]],
    fn: Callable[[A], Result[B, E2]] | None = None,
) -> Result[B, E | E2] | Callable[[Result[A, E]], Result[B, E | E2]]:
    """Chain a Result-returning callback in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("and_then data-last form requires a callback")
        return lambda result: result.and_then(result_or_fn)
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("and_then data-first form requires a Result")
    return cast("Result[B, E | E2]", result_or_fn.and_then(fn))


@overload
def and_then_async[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[A], Awaitable[Result[B, E2]]],
) -> Awaitable[Result[B, E | E2]]: ...


@overload
def and_then_async[A, E, B, E2](
    fn: Callable[[A], Awaitable[Result[B, E2]]],
) -> Callable[[Result[A, E]], Awaitable[Result[B, E | E2]]]: ...


def and_then_async[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[A], Awaitable[Result[B, E2]]],
    fn: Callable[[A], Awaitable[Result[B, E2]]] | None = None,
) -> (
    Awaitable[Result[B, E | E2]]
    | Callable[[Result[A, E]], Awaitable[Result[B, E | E2]]]
):
    """Async Result chaining in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("and_then_async data-last form requires a callback")

        async def chain_later(result: Result[A, E]) -> Result[B, E | E2]:
            return await result.and_then_async(result_or_fn)

        return chain_later
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("and_then_async data-first form requires a Result")
    return result_or_fn.and_then_async(fn)


@overload
def match[A, E, T](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., T]],
) -> T: ...


@overload
def match[A, E, T](
    handlers: Mapping[str, Callable[..., T]],
) -> Callable[[Result[A, E]], T]: ...


def match[A, E, T](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., T]],
    handlers: Mapping[str, Callable[..., T]] | None = None,
) -> T | Callable[[Result[A, E]], T]:
    """Match a Result in data-first or data-last form."""
    if handlers is None:
        if isinstance(result_or_handlers, (Ok, Err)):
            raise TypeError("match data-last form requires handlers")
        return lambda result: result.match(result_or_handlers)
    if not isinstance(result_or_handlers, (Ok, Err)):
        raise TypeError("match data-first form requires a Result")
    return result_or_handlers.match(handlers)


@overload
def tap[A, E](result: Result[A, E], fn: Callable[[A], object]) -> Result[A, E]: ...


@overload
def tap[A, E](fn: Callable[[A], object]) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap[A, E](
    result_or_fn: Result[A, E] | Callable[[A], object],
    fn: Callable[[A], object] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run a success side effect in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("tap data-last form requires a callback")
        return lambda result: result.tap(result_or_fn)
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("tap data-first form requires a Result")
    return cast("Result[A, E]", result_or_fn.tap(fn))


@overload
def tap_async[A, E](
    result: Result[A, E],
    fn: Callable[[A], Awaitable[object]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_async[A, E](
    fn: Callable[[A], Awaitable[object]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_async[A, E](
    result_or_fn: Result[A, E] | Callable[[A], Awaitable[object]],
    fn: Callable[[A], Awaitable[object]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run an async success side effect in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("tap_async data-last form requires a callback")

        async def tap_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_async(result_or_fn)

        return tap_later
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("tap_async data-first form requires a Result")
    return result_or_fn.tap_async(fn)


@overload
def tap_error[A, E](
    result: Result[A, E],
    fn: Callable[[E], object],
) -> Result[A, E]: ...


@overload
def tap_error[A, E](
    fn: Callable[[E], object],
) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap_error[A, E](
    result_or_fn: Result[A, E] | Callable[[E], object],
    fn: Callable[[E], object] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run an error side effect in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("tap_error data-last form requires a callback")
        return lambda result: result.tap_error(result_or_fn)
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("tap_error data-first form requires a Result")
    return cast("Result[A, E]", result_or_fn.tap_error(fn))


@overload
def tap_error_async[A, E](
    result: Result[A, E],
    fn: Callable[[E], Awaitable[object]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_error_async[A, E](
    fn: Callable[[E], Awaitable[object]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_error_async[A, E](
    result_or_fn: Result[A, E] | Callable[[E], Awaitable[object]],
    fn: Callable[[E], Awaitable[object]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run an async error side effect in data-first or data-last form."""
    if fn is None:
        if isinstance(result_or_fn, (Ok, Err)):
            raise TypeError("tap_error_async data-last form requires a callback")

        async def tap_error_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_error_async(result_or_fn)

        return tap_error_later
    if not isinstance(result_or_fn, (Ok, Err)):
        raise TypeError("tap_error_async data-first form requires a Result")
    return result_or_fn.tap_error_async(fn)


@overload
def tap_both[A, E](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., object]],
) -> Result[A, E]: ...


@overload
def tap_both[A, E](
    handlers: Mapping[str, Callable[..., object]],
) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap_both[A, E](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., object]],
    handlers: Mapping[str, Callable[..., object]] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run the active side effect in data-first or data-last form."""
    if handlers is None:
        if isinstance(result_or_handlers, (Ok, Err)):
            raise TypeError("tap_both data-last form requires handlers")
        return lambda result: result.tap_both(result_or_handlers)
    if not isinstance(result_or_handlers, (Ok, Err)):
        raise TypeError("tap_both data-first form requires a Result")
    return cast("Result[A, E]", result_or_handlers.tap_both(handlers))


@overload
def tap_both_async[A, E](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., Awaitable[object]]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_both_async[A, E](
    handlers: Mapping[str, Callable[..., Awaitable[object]]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_both_async[A, E](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., Awaitable[object]]],
    handlers: Mapping[str, Callable[..., Awaitable[object]]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run the active async side effect in data-first or data-last form."""
    if handlers is None:
        if isinstance(result_or_handlers, (Ok, Err)):
            raise TypeError("tap_both_async data-last form requires handlers")

        async def tap_both_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_both_async(result_or_handlers)

        return tap_both_later
    if not isinstance(result_or_handlers, (Ok, Err)):
        raise TypeError("tap_both_async data-first form requires a Result")
    return result_or_handlers.tap_both_async(handlers)


_MISSING = object()


def unwrap[A, E](result: Result[A, E], message: str | None = None) -> A:
    """Extract a success value or raise ``Panic`` for an error result."""
    return result.unwrap(message)


@overload
def unwrap_or[A, E, B](result: Result[A, E], fallback: B) -> A | B: ...


@overload
def unwrap_or[A, E, B](fallback: B) -> Callable[[Result[A, E]], A | B]: ...


def unwrap_or[A, E, B](
    result_or_fallback: Result[A, E] | B,
    fallback: B | object = _MISSING,
) -> A | B | Callable[[Result[A, E]], A | B]:
    """Extract a success value or return a fallback in either call form."""
    if fallback is _MISSING:
        if isinstance(result_or_fallback, (Ok, Err)):
            raise TypeError("unwrap_or data-last form requires a fallback")
        return lambda result: result.unwrap_or(result_or_fallback)
    if not isinstance(result_or_fallback, (Ok, Err)):
        raise TypeError("unwrap_or data-first form requires a Result")
    return cast("A | B", result_or_fallback.unwrap_or(fallback))


def all[A, E](results: Iterable[Result[A, E]]) -> Result[list[A], E]:
    """Collect success values in input order or return the first error."""
    values: list[A] = []
    for result in results:
        checked = _require_result(result, "Result.all input must be a Result")
        if isinstance(checked, Err):
            return Err[list[A], E](cast("E", checked.error))
        values.append(cast("A", checked.value))
    return Ok[list[A], E](values)


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
    except Panic:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except Exception as cause:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise Panic("Result.all_async input awaitable rejected", cause) from cause

    return [
        cast(
            "Result[A, E]",
            _require_result(result, "Result.all_async input must be a Result"),
        )
        for result in resolved
    ]


async def _completed[T](value: T) -> T:
    return value


async def all_async[A, E](
    results: Iterable[Result[A, E] | Awaitable[Result[A, E]]],
) -> Result[list[A], E]:
    """Await all inputs concurrently and collect success values."""
    return all(await _await_results(results))


def partition[A, E](results: Iterable[Result[A, E]]) -> tuple[list[A], list[E]]:
    """Split results into success and error values while preserving order."""
    values: list[A] = []
    errors: list[E] = []
    for result in results:
        checked = _require_result(result, "Result.partition input must be a Result")
        if isinstance(checked, Ok):
            values.append(cast("A", checked.value))
        else:
            errors.append(cast("E", checked.error))
    return values, errors


async def partition_async[A, E](
    results: Iterable[Result[A, E] | Awaitable[Result[A, E]]],
) -> tuple[list[A], list[E]]:
    """Await all inputs concurrently and partition their payloads."""
    return partition(await _await_results(results))


@overload
def flatten[A, E, E2](result: Ok[Ok[A, E], E2]) -> Result[A, E | E2]: ...


@overload
def flatten[A, E, E2](result: Ok[Err[A, E], E2]) -> Result[A, E | E2]: ...


@overload
def flatten[A, E, E2](result: Err[Result[A, E], E2]) -> Result[A, E | E2]: ...


@overload
def flatten[A, E, E2](result: Result[Result[A, E], E2]) -> Result[A, E | E2]: ...


def flatten[A, E, E2](result: Result[Result[A, E], E2]) -> Result[A, E | E2]:
    """Flatten a nested Result into one Result."""
    checked = _require_result(result, "Result.flatten input must be a Result")
    if isinstance(checked, Ok):
        nested = checked.value
        nested_result = _require_result(
            nested,
            "Result.flatten nested input must be a Result",
        )
        if isinstance(nested_result, Ok):
            return Ok[A, E | E2](cast("A", nested_result.value))
        return Err[A, E | E2](cast("E", nested_result.error))
    return Err[A, E | E2](cast("E2", checked.error))


__all__ = [
    "AsyncRetryConfig",
    "ResultCodec",
    "ResultCodecConfig",
    "RetryConfig",
    "Schema",
    "SchemaFailure",
    "SerializedErr",
    "SerializedOk",
    "SerializedResult",
    "TryAsyncContext",
    "TryContext",
    "all",
    "all_async",
    "and_then",
    "and_then_async",
    "codec",
    "codec_config",
    "flatten",
    "map",
    "map_error",
    "match",
    "partition",
    "partition_async",
    "tap",
    "tap_async",
    "tap_both",
    "tap_both_async",
    "tap_error",
    "tap_error_async",
    "try_",
    "try_async",
    "try_promise",
    "try_recover",
    "try_recover_async",
    "try_result",
    "unwrap",
    "unwrap_or",
]
