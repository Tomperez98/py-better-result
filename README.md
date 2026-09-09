# py-better-result

A small Python port of the Rust `retry-result` crate, with an immutable
`Result[T, E]` implementation shaped after Rust's `std::result::Result`.

```python
from better_result import Err, Ok, RetryPolicy, retry

attempts = 0


def request() -> Ok[str] | Err[str]:
    global attempts
    attempts += 1
    return Err("temporary") if attempts < 2 else Ok("done")


result = retry(request, RetryPolicy.fixed(2, 0))
assert result == Ok("done")
```

## Result

`Ok(value)` and `Err(error)` are immutable result variants. The supported
operations follow Rust's `Result` API:

- inspection: `is_ok`, `is_err`, `is_ok_and`, `is_err_and`, `ok`, `err`
- mapping: `map`, `map_or`, `map_or_else`, `map_err`
- composition: `and_`, `and_then`, `or_`, `or_else`
- observation: `inspect`, `inspect_err`
- extraction: `unwrap`, `expect`, `unwrap_err`, `expect_err`, `unwrap_or`,
  `unwrap_or_else`, `unwrap_or_default`
- nesting: `flatten`, `transpose`

Python uses `and_` and `or_` because `and` and `or` are reserved keywords.
`unwrap`-style failures raise `UnwrapError`; callback exceptions are not
silently converted to `Err` values.

## Retry

Retry operates on `Result`-returning operations. It does not catch Python
exceptions; expected failures must be returned as `Err`, matching the Rust
crate's behavior.

```python
from better_result import Err, Ok, RetryPolicy, retry

policy = RetryPolicy.exponential(3, 0.01, 2)
attempts = 0


def operation() -> Ok[str] | Err[str]:
    global attempts
    attempts += 1
    return Err("temporary") if attempts < 3 else Ok("response")


result = retry(operation, policy)
assert result == Ok("response")
```

Available schedules are `FixedBackoff`, `LinearBackoff`,
`ExponentialBackoff`, and `Jittered`. Custom schedules implement
`decide(RetryContext) -> RetryDecision`; custom predicates implement
`should_retry(RetryContext) -> bool`.

Retry delays are finite, non-negative floating-point seconds, representing
Rust's `Duration`. Invalid exponential multipliers and jitter factors raise
`InvalidBackoffMultiplier` and `InvalidJitterFactor`, respectively.

`retry_async` provides the same behavior for awaitable operations. Normal
`asyncio` task cancellation cancels an in-flight operation or retry delay.

## Examples

The examples mirror the Rust workspace:

```bash
uv run better-result-example basic
uv run better-result-example sync-retry
uv run better-result-example async-retry
uv run better-result-example --all
```

## Development

```bash
uv sync --all-packages
uv run pytest
uv run pytest --cov
uv run ruff check .
uv run ty check
```
