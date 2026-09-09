"""Retry synchronous Result-producing operations with typed policies."""

from __future__ import annotations

from enum import Enum, auto

from better_result import (
    Err,
    Ok,
    Result,
    RetryAfter,
    RetryContext,
    RetryPolicy,
    StopRetry,
    retry,
)


class FetchError(Enum):
    TEMPORARY = auto()
    PERMANENT = auto()


def fetch(attempts: list[int]) -> Result[str, FetchError]:
    attempts.append(1)
    match len(attempts):
        case 1 | 2:
            return Err(FetchError.TEMPORARY)
        case 3:
            return Ok("response from the service")
        case _:
            return Err(FetchError.PERMANENT)


def main() -> None:
    immediate = RetryPolicy.immediate(2)
    fixed = RetryPolicy.fixed(2, 0)
    linear = RetryPolicy.linear(2, 0)
    exponential = RetryPolicy.exponential(2, 0, 2)

    attempts: list[int] = []
    result = retry(lambda: fetch(attempts), exponential)
    assert result == Ok("response from the service")
    assert len(attempts) == 3
    print(f"exponential policy: {result}, attempts={len(attempts)}")

    for policy in (fixed, linear, immediate):
        policy_attempts: list[int] = []
        assert _retry_fetch(policy_attempts, policy) == Ok("response from the service")

    policy = RetryPolicy.immediate(10).with_predicate(
        lambda context: context.error is FetchError.TEMPORARY,
    )
    permanent_attempts: list[int] = []
    permanent_result = retry(
        lambda: _permanent(permanent_attempts),
        policy,
    )
    assert permanent_result == Err(FetchError.PERMANENT)
    assert len(permanent_attempts) == 1
    print("predicate stopped permanent error after one attempt")

    observed_attempts: list[int] = []

    def schedule(context: RetryContext[FetchError]) -> RetryAfter | StopRetry:
        observed_attempts.append(context.attempt)
        return RetryAfter(0)

    dynamic_policy = RetryPolicy(2, schedule)
    dynamic_calls: list[int] = []
    dynamic_result = retry(lambda: fetch(dynamic_calls), dynamic_policy)
    assert dynamic_result == Ok("response from the service")
    assert observed_attempts == [1, 2]
    print(f"custom schedule saw attempts {observed_attempts}")


def _retry_fetch[S, P](
    attempts: list[int],
    policy: RetryPolicy[S, P],
) -> Result[str, FetchError]:
    return retry(lambda: fetch(attempts), policy)


def _permanent(attempts: list[int]) -> Err[FetchError]:
    attempts.append(1)
    return Err(FetchError.PERMANENT)


if __name__ == "__main__":
    main()
