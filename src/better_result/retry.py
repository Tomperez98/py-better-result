"""Synchronous and asynchronous retry workflows."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from .core import Err, Ok, Panic, Result
from .error import UnhandledException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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
            return Ok(operation(context))
        except Panic:
            raise
        except Exception as cause:
            if catch is None:
                return Err(UnhandledException(cause))
            try:
                return Err(catch(cause))
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
            return Ok(await operation(context))
        except asyncio.CancelledError:
            raise
        except Panic:
            raise
        except Exception as cause:
            if catch is None:
                return Err(UnhandledException(cause))
            try:
                handled_error: E = await _await_value(catch(cause))
                return Err(handled_error)
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
