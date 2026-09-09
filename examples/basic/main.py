"""A minimal synchronous retry workflow."""

from __future__ import annotations

from better_result import Err, Ok, Result, RetryPolicy, retry


def main() -> None:
    policy = RetryPolicy.fixed(2, 0)
    attempts = 0

    def operation() -> Result[str, str]:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return Err("temporary failure")
        return Ok("loaded from the service")

    result = retry(operation, policy)
    assert result == Ok("loaded from the service")
    print(f"{result} after {attempts} attempts")


if __name__ == "__main__":
    main()
