"""Standalone Result operations and boundary helpers."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, cast, overload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

from ._core import Err, Ok, Result


class CancellationToken:
    """Cooperative cancellation signal for async operations and retry waits."""

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
    if isinstance(retry, bool) or not isinstance(retry, (int, RetryPolicy)):
        _raise_retry_policy_error("retry must be a non-negative integer or policy")
    if isinstance(retry, int):
        if retry < 0:
            _raise_retry_policy_error("retry must be non-negative")
        return cast("RetryPolicy[E]", RetryPolicy.immediate(retry))
    return retry


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
    if cancel_token.is_cancelled:
        raise asyncio.CancelledError
    try:
        await asyncio.wait_for(cancel_token.wait(), timeout=delay)
    except TimeoutError:
        return
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
            value = await operation(context)
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            return cast("Result[T, object]", Ok(value))
        except Exception as cause:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            error = await _map_async_exception(cause, catch)
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
