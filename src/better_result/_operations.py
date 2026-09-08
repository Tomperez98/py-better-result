"""Standalone Result operations and boundary helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import inspect
import math
import random
import time
from typing import TYPE_CHECKING, NoReturn, cast, overload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

from better_result._core import Err, Ok, Result, is_err, is_ok


class CancellationToken:
    """Best-effort cancellation signal for async operations and retry waits."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

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
class RetryContext[E]:
    """Context supplied to a retry policy after an error is mapped."""

    error: E
    attempt: int
    cancel_token: CancellationToken | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            _raise_retry_policy_error("retry attempt must be a positive integer")
        if self.attempt < 1:
            _raise_retry_policy_error("retry attempt must be a positive integer")


@dataclass(frozen=True, slots=True)
class StopRetry:
    """A retry policy decision that ends the operation with its current error."""


@dataclass(frozen=True, slots=True)
class RetryAfter:
    """A retry policy decision that waits before starting another attempt."""

    delay: float

    def __post_init__(self) -> None:
        _validate_delay(self.delay)


@dataclass(frozen=True, slots=True)
class ConstantDelay:
    """Use the same delay before every retry."""

    seconds: float

    def __post_init__(self) -> None:
        _validate_delay(self.seconds)

    def delay_for[E](self, context: RetryContext[E]) -> float:
        del context
        return self.seconds


@dataclass(frozen=True, slots=True)
class LinearBackoff:
    """Increase the initial delay by one base interval per retry."""

    initial: float

    def __post_init__(self) -> None:
        _validate_delay(self.initial)

    def delay_for[E](self, context: RetryContext[E]) -> float:
        delay = self.initial * context.attempt
        _validate_delay(delay)
        return float(delay)


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Multiply the initial delay by ``factor`` for each retry."""

    initial: float
    factor: float = 2.0

    def __post_init__(self) -> None:
        _validate_delay(self.initial)
        _validate_backoff_factor(self.factor)

    def delay_for[E](self, context: RetryContext[E]) -> float:
        try:
            delay = self.initial * self.factor ** (context.attempt - 1)
        except OverflowError:
            _raise_retry_policy_error("retry delay overflowed")
        _validate_delay(delay)
        return cast("float", delay)


@dataclass(frozen=True, slots=True)
class DynamicDelay[E]:
    """Compute a delay from the mapped error and retry context."""

    function: Callable[[RetryContext[E]], float]

    def delay_for(self, context: RetryContext[E]) -> float:
        delay = self.function(context)
        _validate_delay(delay)
        return delay


@dataclass(frozen=True, slots=True)
class Jittered[E]:
    """Apply multiplicative jitter to another retry schedule."""

    schedule: RetrySchedule[E]
    factor: float

    def __post_init__(self) -> None:
        _validate_jitter_factor(self.factor)

    def delay_for(self, context: RetryContext[E]) -> float:
        delay = self.schedule.delay_for(context)
        factor = self.factor
        jittered_delay = delay * (1 - factor + random.SystemRandom().random() * factor)
        _validate_delay(jittered_delay)
        return jittered_delay


type RetrySchedule[E] = (
    ConstantDelay | LinearBackoff | ExponentialBackoff | DynamicDelay[E] | Jittered[E]
)
type RetryDecision = StopRetry | RetryAfter


def _raise_retry_policy_error(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_delay(delay: float) -> None:
    if isinstance(delay, bool):
        _raise_retry_policy_error("retry delay must be a finite non-negative number")
    try:
        valid = math.isfinite(delay) and delay >= 0
    except (TypeError, ValueError):
        _raise_retry_policy_error("retry delay must be a finite non-negative number")
    if not valid:
        _raise_retry_policy_error("retry delay must be a finite non-negative number")


def _validate_backoff_factor(factor: float) -> None:
    if isinstance(factor, bool):
        _raise_retry_policy_error(
            "retry backoff factor must be a finite number at least 1"
        )
    try:
        valid = math.isfinite(factor) and factor >= 1
    except (TypeError, ValueError):
        _raise_retry_policy_error(
            "retry backoff factor must be a finite number at least 1"
        )
    if not valid:
        _raise_retry_policy_error(
            "retry backoff factor must be a finite number at least 1"
        )


def _validate_jitter_factor(factor: float) -> None:
    if isinstance(factor, bool):
        _raise_retry_policy_error(
            "retry jitter must be a finite number between 0 and 1"
        )
    try:
        valid = math.isfinite(factor) and 0 <= factor <= 1
    except (TypeError, ValueError):
        _raise_retry_policy_error(
            "retry jitter must be a finite number between 0 and 1"
        )
    if not valid:
        _raise_retry_policy_error(
            "retry jitter must be a finite number between 0 and 1"
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy[E]:
    """Bounded retry decisions shared by synchronous and async operations."""

    times: int
    schedule: RetrySchedule[E]
    should_retry: Callable[[RetryContext[E]], bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.times, bool) or not isinstance(self.times, int):
            _raise_retry_policy_error("retry times must be a non-negative integer")
        if self.times < 0:
            _raise_retry_policy_error("retry times must be a non-negative integer")
        if not isinstance(
            self.schedule,
            (ConstantDelay, LinearBackoff, ExponentialBackoff, DynamicDelay, Jittered),
        ):
            _raise_retry_policy_error("retry schedule must be a supported schedule")
        if self.should_retry is not None and not callable(self.should_retry):
            _raise_retry_policy_error("retry predicate must be callable")

    @classmethod
    def from_schedule(
        cls,
        times: int,
        schedule: RetrySchedule[E],
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        return cls(times, schedule, should_retry)

    @classmethod
    def immediate(
        cls,
        times: int,
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        return cls(times, ConstantDelay(0), should_retry)

    @classmethod
    def constant(
        cls,
        times: int,
        delay: float = 0,
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        return cls(times, ConstantDelay(delay), should_retry)

    @classmethod
    def linear(
        cls,
        times: int,
        initial_delay: float,
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        return cls(times, LinearBackoff(initial_delay), should_retry)

    @classmethod
    def exponential(
        cls,
        times: int,
        initial_delay: float,
        factor: float = 2.0,
        *,
        jitter: float = 0,
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        schedule: RetrySchedule[E] = ExponentialBackoff(initial_delay, factor)
        if jitter:
            schedule = Jittered(schedule, jitter)
        return cls(times, schedule, should_retry)

    @classmethod
    def dynamic(
        cls,
        times: int,
        delay: Callable[[RetryContext[E]], float],
        should_retry: Callable[[RetryContext[E]], bool] | None = None,
    ) -> RetryPolicy[E]:
        return cls(times, DynamicDelay(delay), should_retry)

    def decide(self, context: RetryContext[E]) -> RetryDecision:
        if context.attempt > self.times:
            return StopRetry()
        if self.should_retry is not None and not self.should_retry(context):
            return StopRetry()
        return RetryAfter(self.schedule.delay_for(context))


def _coerce_retry_policy[E](
    retry: int | RetryPolicy[E] | None,
) -> RetryPolicy[E] | None:
    if retry is None:
        return None
    if isinstance(retry, RetryPolicy):
        return retry
    if isinstance(retry, bool) or not isinstance(retry, int):
        _raise_retry_policy_error("retry must be a non-negative integer or policy")
    if retry < 0:
        _raise_retry_policy_error("retry must be non-negative")
    if retry == 0:
        return None
    return cast("RetryPolicy[E]", RetryPolicy.immediate(retry))


@overload
def try_result[T](
    operation: Callable[[TryContext], T],
    *,
    catch: None = None,
    retry: int | RetryPolicy[Exception] | None = 0,
) -> Result[T, Exception]: ...


@overload
def try_result[T, E](
    operation: Callable[[TryContext], T],
    *,
    catch: Callable[[Exception], E],
    retry: int | RetryPolicy[E] | None = 0,
) -> Result[T, E]: ...


def try_result[T](
    operation: Callable[[TryContext], T],
    *,
    catch: Callable[[Exception], object] | None = None,
    retry: int | RetryPolicy[object] | None = 0,
) -> Result[T, object]:
    """Execute a synchronous operation with optional bounded retries."""
    retry_policy = _coerce_retry_policy(retry)
    context = TryContext(attempt=1)
    while True:
        try:
            return cast("Result[T, object]", Ok(operation(context)))
        except Exception as cause:
            mapped = catch(cause) if catch is not None else cause
            if retry_policy is None:
                return cast("Result[T, object]", Err(mapped))
            decision = retry_policy.decide(
                RetryContext(
                    error=mapped,
                    attempt=context.attempt,
                )
            )
            if isinstance(decision, StopRetry):
                return cast("Result[T, object]", Err(mapped))
            assert isinstance(decision, RetryAfter)
            time.sleep(decision.delay)
            context = TryContext(attempt=context.attempt + 1)


async def _wait_for_retry(
    delay: float,
    cancel_token: CancellationToken | None,
) -> None:
    if cancel_token is None:
        await asyncio.sleep(delay)
        return
    with suppress(TimeoutError):
        await asyncio.wait_for(cancel_token.wait(), timeout=delay)
    cancel_token.raise_if_cancelled()


async def _await_with_cancellation[T](
    awaitable: Awaitable[T],
    cancel_token: CancellationToken,
) -> T:
    """Best-effort cancel an in-flight awaitable when the token is cancelled."""
    operation_task = asyncio.ensure_future(awaitable)
    cancellation_task = asyncio.create_task(cancel_token.wait())
    try:
        done, _ = await asyncio.wait(
            (operation_task, cancellation_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task

        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        cancelled_by_token = True
    except asyncio.CancelledError:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise
    finally:
        if not cancellation_task.done():
            cancellation_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancellation_task

    assert cancelled_by_token
    raise asyncio.CancelledError


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
    retry: int | RetryPolicy[Exception] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, Exception]: ...


@overload
async def try_async[T, E](
    operation: Callable[[TryContext], Awaitable[T]],
    *,
    catch: Callable[[Exception], E | Awaitable[E]],
    retry: int | RetryPolicy[E] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, E]: ...


async def try_async[T](
    operation: Callable[[TryContext], Awaitable[T]],
    *,
    catch: Callable[[Exception], object | Awaitable[object]] | None = None,
    retry: int | RetryPolicy[object] | None = None,
    cancel_token: CancellationToken | None = None,
) -> Result[T, object]:
    """Execute an async operation with optional mapping and bounded retries."""
    retry_policy = _coerce_retry_policy(retry)
    context = TryContext(attempt=1, cancel_token=cancel_token)
    while True:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        try:
            awaitable = operation(context)
            if cancel_token is None:
                value = await awaitable
            else:
                value = await _await_with_cancellation(awaitable, cancel_token)
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            return cast("Result[T, object]", Ok(value))
        except Exception as cause:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            if cancel_token is None:
                error = await _map_async_exception(cause, catch)
            else:
                error = await _await_with_cancellation(
                    _map_async_exception(cause, catch),
                    cancel_token,
                )
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            if retry_policy is None:
                return cast("Result[T, object]", Err(error))
            decision = retry_policy.decide(
                RetryContext(
                    error=error,
                    attempt=context.attempt,
                    cancel_token=cancel_token,
                )
            )
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            if isinstance(decision, StopRetry):
                return cast("Result[T, object]", Err(error))
            assert isinstance(decision, RetryAfter)
            await _wait_for_retry(decision.delay, cancel_token)
            context = TryContext(
                attempt=context.attempt + 1,
                cancel_token=cancel_token,
            )


def flatten_result[T, E, F](
    result: Result[Result[T, E], F],
) -> Result[T, E | F]:
    """Flatten a nested Result while preserving either error value."""
    if is_ok(result):
        return result.ok_value
    if is_err(result):
        return result
    message = "expected a concrete Result variant"
    raise TypeError(message)


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
        if is_ok(result):
            values.append(result.ok_value)
            continue
        return Err(result.unwrap_err())
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
        if is_ok(result):
            values.append(result.ok_value)
        else:
            errors.append(result.unwrap_err())
    return values, errors


async def _resolve_results[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> list[Result[T, E]]:
    pending = list(results)
    awaitables: list[Awaitable[Result[T, E]]] = []
    indexes: list[int] = []

    for index, result in enumerate(pending):
        if not isinstance(result, Result):
            indexes.append(index)
            awaitables.append(result)

    if not awaitables:
        # The scan above proves every item is already a Result. Avoid copying
        # the list on this hot path; the object cast only bridges list invariance.
        return cast("list[Result[T, E]]", cast("object", pending))

    if len(awaitables) == len(pending):
        return list(await asyncio.gather(*awaitables))

    resolved = [cast("Result[T, E]", result) for result in pending]
    awaited = await asyncio.gather(*awaitables)
    for index, result in zip(indexes, awaited, strict=True):
        resolved[index] = result
    return resolved


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
    resolved = await _resolve_results(results)
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
    resolved = await _resolve_results(results)
    return partition_results(resolved)


# --- capture / capture_async ---


@overload
def capture[T](
    operation: Callable[[], T],
    *,
    catch: None = None,
) -> Result[T, Exception]: ...


@overload
def capture[T, E](
    operation: Callable[[], T],
    *,
    catch: Callable[[Exception], E],
) -> Result[T, E]: ...


def capture[T](
    operation: Callable[[], T],
    *,
    catch: Callable[[Exception], object] | None = None,
) -> Result[T, object]:
    """
    Execute a zero-argument operation and capture expected exceptions.

    Unlike ``try_result``, this helper does not accept a ``TryContext``
    or a retry policy. Use it at a simple exception boundary where retries
    are not needed.
    """
    try:
        return cast("Result[T, object]", Ok(operation()))
    except Exception as cause:
        mapped = catch(cause) if catch is not None else cause
        return cast("Result[T, object]", Err(mapped))


@overload
async def capture_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    catch: None = None,
) -> Result[T, Exception]: ...


@overload
async def capture_async[T, E](
    operation: Callable[[], Awaitable[T]],
    *,
    catch: Callable[[Exception], Awaitable[E]],
) -> Result[T, E]: ...


@overload
async def capture_async[T, E](
    operation: Callable[[], Awaitable[T]],
    *,
    catch: Callable[[Exception], E],
) -> Result[T, E]: ...


async def capture_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    catch: Callable[[Exception], object | Awaitable[object]] | None = None,
) -> Result[T, object]:
    """
    Execute a zero-argument async operation and capture expected exceptions.

    Unlike ``try_async``, this helper does not accept a ``TryContext``,
    retry policy, or cancellation token. Use it at a simple async exception
    boundary where retries are not needed.
    """
    try:
        value = await operation()
        return cast("Result[T, object]", Ok(value))
    except Exception as cause:
        if catch is None:
            return cast("Result[T, object]", Err(cause))
        mapped = catch(cause)
        if inspect.isawaitable(mapped):
            awaited = await mapped
            mapped = awaited
        return cast("Result[T, object]", Err(mapped))


# --- collect_results / collect_results_async ---


@overload
def collect_results[A, E](
    results: tuple[Result[A, E]],
) -> Result[tuple[A], tuple[E]]: ...


@overload
def collect_results[A, E, B, F](
    results: tuple[Result[A, E], Result[B, F]],
) -> Result[tuple[A, B], tuple[E | F]]: ...


@overload
def collect_results[A, E, B, F, C, G](
    results: tuple[Result[A, E], Result[B, F], Result[C, G]],
) -> Result[tuple[A, B, C], tuple[E | F | G]]: ...


@overload
def collect_results[A, E, B, F, C, G, D, H](
    results: tuple[Result[A, E], Result[B, F], Result[C, G], Result[D, H]],
) -> Result[tuple[A, B, C, D], tuple[E | F | G | H]]: ...


@overload
def collect_results[T, E](
    results: Iterable[Result[T, E]],
) -> Result[tuple[T, ...], tuple[E, ...]]: ...


def collect_results[T, E](
    results: Iterable[Result[T, E]],
) -> Result[tuple[T, ...], tuple[E, ...]]:
    """
    Accumulate every error instead of returning the first one.

    Unlike ``all_results`` which short-circuits on the first error, this
    helper evaluates every supplied Result and returns all errors as a
    tuple. When there are no errors the success values are returned as a
    tuple in input order.
    """
    values: list[T] = []
    errors: list[E] = []
    for result in results:
        if is_ok(result):
            values.append(result.ok_value)
        else:
            errors.append(result.unwrap_err())
    if errors:
        return Err(tuple(errors))
    return Ok(tuple(values))


@overload
async def collect_results_async[A, E](
    results: tuple[Result[A, E] | Awaitable[Result[A, E]]],
) -> Result[tuple[A], tuple[E]]: ...


@overload
async def collect_results_async[A, E, B, F](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
    ],
) -> Result[tuple[A, B], tuple[E | F]]: ...


@overload
async def collect_results_async[A, E, B, F, C, G](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
    ],
) -> Result[tuple[A, B, C], tuple[E | F | G]]: ...


@overload
async def collect_results_async[A, E, B, F, C, G, D, H](
    results: tuple[
        Result[A, E] | Awaitable[Result[A, E]],
        Result[B, F] | Awaitable[Result[B, F]],
        Result[C, G] | Awaitable[Result[C, G]],
        Result[D, H] | Awaitable[Result[D, H]],
    ],
) -> Result[tuple[A, B, C, D], tuple[E | F | G | H]]: ...


@overload
async def collect_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> Result[tuple[T, ...], tuple[E, ...]]: ...


async def collect_results_async[T, E](
    results: Iterable[Result[T, E] | Awaitable[Result[T, E]]],
) -> Result[tuple[T, ...], tuple[E, ...]]:
    """Resolve Results concurrently, then accumulate errors in input order."""
    resolved = await _resolve_results(results)
    return collect_results(resolved)


# --- traverse / traverse_async ---


def _validate_concurrency(value: object) -> None:
    message = "max_concurrency must be a positive integer or None"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)  # noqa: TRY004
    if value <= 0:
        raise ValueError(message)


def traverse[A, T, E](
    values: Iterable[A],
    operation: Callable[[A], Result[T, E]],
) -> Result[tuple[T, ...], E]:
    """
    Apply a Result-returning operation over input values and short-circuit.

    Returns the first error in input order, or a tuple of all success values.
    This is the same short-circuit semantics as ``all_results``.
    """
    values_out: list[T] = []
    for value in values:
        result = operation(value)
        if is_ok(result):
            values_out.append(result.ok_value)
            continue
        return Err(result.unwrap_err())
    return Ok(tuple(values_out))


async def traverse_async[A, T, E](
    values: Iterable[A],
    operation: Callable[[A], Awaitable[Result[T, E]]],
    *,
    max_concurrency: int | None = None,
) -> Result[tuple[T, ...], E]:
    """Apply an async Result-returning operation with optional concurrency limit."""
    if max_concurrency is not None:
        _validate_concurrency(max_concurrency)

    if max_concurrency is None:
        return await all_results_async(operation(value) for value in values)

    iterator = iter(values)
    ordered: list[Result[T, E] | None] = []

    async def worker() -> None:
        while True:
            try:
                value = next(iterator)
            except StopIteration:
                return
            index = len(ordered)
            ordered.append(None)
            ordered[index] = await operation(value)

    await asyncio.gather(*(worker() for _ in range(max_concurrency)))
    # Every worker fills its reserved slot before gather() completes. Avoid a
    # second O(n) list copy while narrowing the sentinel-based working list.
    resolved = cast("list[Result[T, E]]", cast("object", ordered))
    return all_results(resolved)
