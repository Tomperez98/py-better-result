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

Python uses `and_` and `or_` because `and` and `or` are reserved keywords.
Callback exceptions are not silently converted to `Err` values.

## Capture exceptions

Python functions often signal expected failures by raising exceptions. Use
`capture` at that boundary to turn a one-shot operation into a `Result`:

```python
from better_result import capture

result = capture(lambda: int("not-a-number"), catch=str)
assert result.err() == "invalid literal for int() with base 10: 'not-a-number'"
```

Without `catch`, the original `Exception` is stored in `Err`. The mapper is
where the exception should be translated into the domain error type for the
current layer. Mapper failures are not captured. `capture_async` provides the
same behavior for async operations and also accepts an async mapper. Pass
`exceptions=(TimeoutError, OSError)` to capture only selected failures. Only
`Exception` is captured, so `asyncio.CancelledError` and other `BaseException`
values propagate normally.

Capture is deliberately an explicit boundary rather than behavior baked into
`Result.map` or `retry`: callback and programmer errors should not silently
become domain failures.

## Retry

Retry operates on `Result`-returning operations. Wrap exception-based
operations with `capture` when they need retrying:

```python
from better_result import RetryPolicy, try_result

result = try_result(
    read_config,
    RetryPolicy.immediate(2),
    catch=str,
)
```

`try_result` and `try_async` combine capture and retry for ordinary
exception-based functions. Pass `exceptions=(TimeoutError, OSError)` to
capture only expected failures; uncaught programmer errors still propagate.

The `is_ok` and `is_err` helpers are type guards, so type checkers narrow a
`Result[T, E]` to `Ok[T]` or `Err[E]` inside each branch.

Expected failures should be returned as `Err` after crossing the boundary.

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
uv run better-result-example capture
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
