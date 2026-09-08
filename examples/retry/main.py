"""Retry an expected exception with a bounded, typed retry policy."""

from __future__ import annotations

from better_result import RetryContext, RetryPolicy, TryContext, try_result


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

    print(dynamic_result)
    print(f"dynamic_attempts={dynamic_attempts}")
    print(f"dynamic_delays={dynamic_delays}")


if __name__ == "__main__":
    main()
