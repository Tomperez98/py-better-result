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
        initial_delay=0,
        should_retry=lambda context: "temporary" in context.error,
    )
    result = try_result(request, catch=str, retry=policy)

    print(result)  # noqa: T201
    print(f"attempts={attempts}")  # noqa: T201

    # A dynamic schedule can choose a delay from the mapped error and attempt.
    def delay_for(context: RetryContext[str]) -> float:
        return context.attempt / 10

    dynamic_policy = RetryPolicy[str].dynamic(times=2, delay=delay_for)
    print(dynamic_policy)  # noqa: T201


if __name__ == "__main__":
    main()
