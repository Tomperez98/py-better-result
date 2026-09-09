"""Typed synchronous and asynchronous retries for Result-returning operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import random
import time
from typing import TYPE_CHECKING, Protocol, TypeVar

from better_result._core import Err, Ok, Result, _require_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class StopRetry:
    """Stop retrying and return the current error."""


@dataclass(frozen=True, slots=True)
class RetryAfter:
    """Wait for ``delay`` seconds before the next attempt."""

    delay: float

    def __post_init__(self) -> None:
        _validate_delay(value=self.delay)


type RetryDecision = RetryAfter | StopRetry


@dataclass(frozen=True, slots=True)
class RetryContext[E]:
    """Context supplied to retry schedules and predicates."""

    error: E
    attempt: int

    def __post_init__(self) -> None:
        message = "retry attempt must be a positive integer"
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise TypeError(message)
        if self.attempt < 1:
            raise ValueError(message)


class RetrySchedule[E](Protocol):
    """Computes the next retry decision."""

    def decide(self, context: RetryContext[E]) -> RetryDecision: ...


class RetryPredicate[E](Protocol):
    """Decides whether a failed operation should be retried."""

    def should_retry(self, context: RetryContext[E]) -> bool: ...


@dataclass(frozen=True, slots=True)
class AlwaysRetry:
    """The default predicate, which retries every error until the limit."""

    def should_retry[E](self, context: RetryContext[E]) -> bool:
        del context
        return True


@dataclass(frozen=True, slots=True)
class FixedBackoff:
    """A fixed delay before every retry."""

    delay: float

    def __post_init__(self) -> None:
        _validate_delay(value=self.delay)
        object.__setattr__(self, "delay", float(self.delay))

    def decide(self, context: RetryContext[E]) -> RetryDecision:
        del context
        return RetryAfter(self.delay)


@dataclass(frozen=True, slots=True)
class LinearBackoff:
    """A delay that grows by one base interval for each attempt."""

    base_delay: float

    def __post_init__(self) -> None:
        _validate_delay(value=self.base_delay)
        object.__setattr__(self, "base_delay", float(self.base_delay))

    def decide(self, context: RetryContext[E]) -> RetryDecision:
        delay = self.base_delay * context.attempt
        if not math.isfinite(delay):
            return StopRetry()
        return RetryAfter(delay)


class InvalidBackoffMultiplier(ValueError):  # noqa: N818
    """Raised when an exponential backoff multiplier is invalid."""

    def __init__(self) -> None:
        super().__init__("backoff multiplier must be at least 1")


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """A delay that multiplies the base interval for each retry."""

    base_delay: float
    multiplier: int

    def __post_init__(self) -> None:
        _validate_delay(value=self.base_delay)
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, int):
            raise InvalidBackoffMultiplier
        if self.multiplier < 1:
            raise InvalidBackoffMultiplier
        object.__setattr__(self, "base_delay", float(self.base_delay))

    def decide(self, context: RetryContext[E]) -> RetryDecision:
        try:
            delay = self.base_delay * self.multiplier ** (context.attempt - 1)
        except OverflowError:
            return StopRetry()
        if not math.isfinite(delay):
            return StopRetry()
        return RetryAfter(delay)


class InvalidJitterFactor(ValueError):  # noqa: N818
    """Raised when a jitter factor is outside ``0.0..=1.0``."""

    def __init__(self) -> None:
        super().__init__("jitter factor must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class Jittered[S]:
    """Apply multiplicative random jitter to another retry schedule."""

    schedule: S
    factor: float

    def __post_init__(self) -> None:
        if isinstance(self.factor, bool) or not isinstance(self.factor, (int, float)):
            raise InvalidJitterFactor
        if not math.isfinite(self.factor) or not 0.0 <= self.factor <= 1.0:
            raise InvalidJitterFactor
        object.__setattr__(self, "factor", float(self.factor))

    def decide(self, context: RetryContext[E]) -> RetryDecision:
        decision = _schedule_decide(self.schedule, context)
        if not isinstance(decision, RetryAfter):
            return StopRetry()
        random_factor = (
            1.0 - self.factor
        ) + random.SystemRandom().random() * self.factor
        return RetryAfter(decision.delay * random_factor)


@dataclass(frozen=True, slots=True)
class RetryPolicy[S, P = AlwaysRetry]:
    """A bounded retry policy shared by sync and async operations."""

    max_retries: int
    schedule: S
    predicate: P | AlwaysRetry = field(default_factory=AlwaysRetry)

    def __post_init__(self) -> None:
        _validate_max_retries(value=self.max_retries)

    @classmethod
    def immediate(cls, max_retries: int) -> RetryPolicy[FixedBackoff, AlwaysRetry]:
        return RetryPolicy(max_retries, FixedBackoff(0), AlwaysRetry())

    @classmethod
    def fixed(
        cls,
        max_retries: int,
        delay: float,
    ) -> RetryPolicy[FixedBackoff, AlwaysRetry]:
        return RetryPolicy(max_retries, FixedBackoff(delay), AlwaysRetry())

    @classmethod
    def linear(
        cls,
        max_retries: int,
        base_delay: float,
    ) -> RetryPolicy[LinearBackoff, AlwaysRetry]:
        return RetryPolicy(max_retries, LinearBackoff(base_delay), AlwaysRetry())

    @classmethod
    def exponential(
        cls,
        max_retries: int,
        base_delay: float,
        multiplier: int,
    ) -> RetryPolicy[ExponentialBackoff, AlwaysRetry]:
        return RetryPolicy(
            max_retries,
            ExponentialBackoff(base_delay, multiplier),
            AlwaysRetry(),
        )

    def with_predicate[Q](self, predicate: Q) -> RetryPolicy[S, Q]:
        return RetryPolicy(self.max_retries, self.schedule, predicate)

    def with_jitter(self, factor: float) -> RetryPolicy[Jittered[S], P]:
        return RetryPolicy(
            self.max_retries,
            Jittered(self.schedule, factor),
            self.predicate,
        )

    def _decide(self, error: E, attempt: int) -> RetryDecision:
        context = RetryContext(error, attempt)
        if attempt > self.max_retries:
            return StopRetry()
        if not _predicate_should_retry(self.predicate, context):
            return StopRetry()
        return _schedule_decide(self.schedule, context)


def retry[T, E, S, P](
    operation: Callable[[], Result[T, E]],
    policy: RetryPolicy[S, P],
) -> Result[T, E]:
    """Execute a Result-returning operation with bounded retries."""
    attempt = 1
    while True:
        result = _require_result(operation())
        if isinstance(result, Ok):
            return result
        assert isinstance(result, Err)

        decision = policy._decide(result.value, attempt)  # noqa: SLF001
        if isinstance(decision, StopRetry):
            return result
        assert isinstance(decision, RetryAfter)

        time.sleep(decision.delay)
        attempt += 1


async def retry_async[T, E, S, P](
    operation: Callable[[], Awaitable[Result[T, E]]],
    policy: RetryPolicy[S, P],
) -> Result[T, E]:
    """Execute an async Result-returning operation with bounded retries."""
    attempt = 1
    while True:
        result = _require_result(await operation())
        if isinstance(result, Ok):
            return result
        assert isinstance(result, Err)

        decision = policy._decide(result.value, attempt)  # noqa: SLF001
        if isinstance(decision, StopRetry):
            return result
        assert isinstance(decision, RetryAfter)

        await asyncio.sleep(decision.delay)
        attempt += 1


def _schedule_decide[T, S](schedule: S, context: RetryContext[T]) -> RetryDecision:
    decider = getattr(schedule, "decide", None)
    if callable(decider):
        decision = decider(context)
    elif callable(schedule):
        decision = schedule(context)  # ty: ignore[call-top-callable]
    else:
        message = "schedule must implement decide(context)"
        raise TypeError(message)

    match decision:
        case RetryAfter() | StopRetry():
            return decision
        case _:
            message = "schedule must return a RetryDecision"
            raise TypeError(message)


def _predicate_should_retry[T, P](predicate: P, context: RetryContext[T]) -> bool:
    retry_decider = getattr(predicate, "should_retry", None)
    if callable(retry_decider):
        return bool(retry_decider(context))
    if callable(predicate):
        return bool(predicate(context))  # ty: ignore[call-top-callable]
    message = "predicate must implement should_retry(context)"
    raise TypeError(message)


def _validate_max_retries(*, value: bool | int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = "max retries must be a non-negative integer"
        raise ValueError(message)


def _validate_delay(*, value: bool | float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = "retry delay must be a finite non-negative number"
        raise TypeError(message)
    if not math.isfinite(value) or value < 0:
        message = "retry delay must be a finite non-negative number"
        raise ValueError(message)
