"""Retry expected exceptions with bounded, typed retry policies."""

from __future__ import annotations

import asyncio

from better_result import (
    Ok,
    RetryContext,
    RetryPolicy,
    TryContext,
    try_async,
    try_result,
)


def main() -> None:
    attempts = 0

    def request(context: TryContext) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            message = f"temporary failure on attempt {context.attempt}"
            raise TimeoutError(message)
        return "response body"

    policy = RetryPolicy[str].exponential(
        times=3,
        initial_delay=0.01,
        should_retry=lambda context: "temporary" in context.error,
    )
    result = try_result(request, catch=str, retry=policy)

    assert result == Ok("response body")
    assert attempts == 3
    print(result)
    print(f"attempts={attempts}")

    # A dynamic schedule can choose a delay from the mapped error and attempt.
    dynamic_attempts = 0
    dynamic_delays: list[float] = []

    def dynamic_request(context: TryContext) -> str:
        nonlocal dynamic_attempts
        dynamic_attempts += 1
        if dynamic_attempts < 3:
            message = f"temporary failure on attempt {context.attempt}"
            raise TimeoutError(message)
        return "dynamic response"

    def delay_for(context: RetryContext[str]) -> float:
        delay = context.attempt / 10
        dynamic_delays.append(delay)
        return delay

    dynamic_policy = RetryPolicy[str].dynamic(
        times=2,
        delay=delay_for,
        should_retry=lambda context: "temporary" in context.error,
    )
    dynamic_result = try_result(
        dynamic_request,
        catch=str,
        retry=dynamic_policy,
    )

    assert dynamic_result == Ok("dynamic response")
    assert dynamic_attempts == 3
    assert dynamic_delays == [0.1, 0.2]
    print(dynamic_result)
    print(f"dynamic_attempts={dynamic_attempts}")
    print(f"dynamic_delays={dynamic_delays}")

    asyncio.run(async_demo())


async def async_request(context: TryContext) -> str:
    if context.attempt == 1:
        message = "temporary async failure"
        raise TimeoutError(message)
    return "async response"


async def async_demo() -> None:
    result = await try_async(
        async_request,
        catch=str,
        retry=RetryPolicy[str].exponential(
            times=2,
            initial_delay=0,
            jitter=0.25,
            should_retry=lambda context: "temporary" in context.error,
        ),
    )
    assert result == Ok("async response")
    print(result)


if __name__ == "__main__":
    main()
