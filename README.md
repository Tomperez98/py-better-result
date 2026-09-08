# py-better-result

> Credits: [better-result.dev](https://better-result.dev)

A typed `Result[T, E]` for Python: return `Ok(value)` or `Err(error)`, compose workflows without exception-driven control flow, and keep expected failures visible to the type checker.

```python
from dataclasses import dataclass

from better_result import Err, Ok, Result


@dataclass(frozen=True)
class User:
    name: str


def parse_user_id(raw: str) -> Result[int, str]:
    if not raw.isdecimal():
        return Err("invalid user id")
    return Ok(int(raw))


def load_user(user_id: int) -> Result[User, str]:
    if user_id == 42:
        return Ok(User("Ada"))
    return Err("user not found")


result = parse_user_id("42").and_then(load_user).map(lambda user: user.name)

match result:
    case Ok(name):
        print(name)
    case Err(message):
        print(f"error: {message}")
```

```text
Ada
```

## Install

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add py-better-result
```

With pip:

```bash
pip install py-better-result
```

`py-better-result` requires Python 3.12 or newer. The runtime dependency is `typing-extensions`.

## Examples

Runnable, focused examples for composition, validation, async workflows,
cancellation, retries, collections, and codecs are in [`examples/`](examples/README.md).
Start with
[`examples/basic/`](examples/basic/README.md) for the smallest complete workflow.

## Why use a Result?

Use a `Result` when failure is an expected part of an operation—validation, a missing record, a rejected request, or a downstream service error. The error stays in the return type instead of being hidden in a broad `try`/`except` or collapsed into `None`.

- `Ok[T]` contains a successful value.
- `Err[E]` contains an expected error value.
- `Result[T, E]` is the common type for either branch.
- `and_then` and `map` short-circuit on the first `Err`.
- Exceptions raised by callbacks are not silently converted into `Err`; unexpected defects propagate.
- `Ok` and `Err` are frozen, unhashable dataclasses and support structural pattern matching.

## Core API

```python
from better_result import Err, Ok, Result, is_err, is_ok

result: Result[int, str] = Ok(2)

result.map(lambda value: value * 10)  # Ok(20)
result.and_then(lambda value: Ok(str(value)))  # Ok("2")
result.map_err(str.upper)  # unchanged Ok(2)
result.unwrap_or(0)  # 2
result.match(ok=str, err=lambda error: error)  # "2"

failure: Result[int, str] = Err("offline")
failure.map(lambda value: value * 10)  # unchanged Err("offline")
failure.map_err(str.upper)  # Err("OFFLINE")
failure.unwrap_or(0)  # 0
failure.unwrap_or_else(lambda error: len(error))  # 7
```

The branch-specific values are available as `ok_value` and `err_value`. Use `isinstance`, `is_ok`, or `is_err` to narrow a `Result`:

```python
if is_ok(result):
    print(result.ok_value)  # int
elif is_err(result):
    print(result.err_value)  # str
```

### Choosing a combinator

| Operation | Runs when | Returns |
| --- | --- | --- |
| `map(fn)` | the result is `Ok` | a new `Result` with the mapped success value |
| `map_err(fn)` | the result is `Err` | a new `Result` with the mapped error |
| `and_then(fn)` | the result is `Ok` | the `Result` returned by the next operation |
| `or_else(fn)` | the result is `Err` | the `Result` returned by the recovery operation |
| `map_or(default, fn)` | either branch | a plain value |
| `map_or_else(default_fn, fn)` | either branch | a plain value |
| `match(ok=..., err=...)` | exactly one branch | the handler's return value |
| `inspect(fn)` / `inspect_err(fn)` | only the selected branch | the original `Result`, for side effects |

`unwrap()` and `expect(message)` return the success value but raise `UnwrapError` on `Err`. Their counterparts `unwrap_err()` and `expect_err()` select the error branch. Prefer `unwrap_or`, `unwrap_or_else`, or explicit matching when failure is expected.

## Async workflows

The core combinators have async forms: `map_async`, `and_then_async`, `or_else_async`, `inspect_async`, `inspect_err_async`, and `inspect_both_async`.

```python
import asyncio

from better_result import Ok


async def fetch_name(user_id: int) -> Ok[str]:
    return Ok(f"user-{user_id}")


async def main() -> None:
    result = await Ok(2).and_then_async(fetch_name)
    print(result)


asyncio.run(main())
```

```text
Ok(value='user-2')
```

Async callbacks are only awaited for the active branch. A failed `Result` therefore skips downstream success callbacks just like the synchronous API.

## Capture exceptions and retry operations

Use `try_result` or `try_async` at a boundary where an exception is an expected failure mode. Without a mapper, the exception itself becomes the error value; `catch` can convert it into a domain error.

```python
from better_result import Err, TryContext, try_result


def read_port(context: TryContext) -> int:
    return int("not-a-port")


result = try_result(read_port, catch=lambda exc: {"message": str(exc)})
assert result == Err(
    {"message": "invalid literal for int() with base 10: 'not-a-port'"}
)
```

Both `try_result` and `try_async` accept the same bounded `RetryPolicy`. An integer remains shorthand for immediate retries. An integer retry count, or a policy without `should_retry`, retries every caught `Exception`; use a predicate that selects transient failures and retry only idempotent or otherwise safe-to-repeat operations. Prefer the named policy constructors when a delay schedule is needed:

```python
import asyncio

from better_result import RetryPolicy, TryContext, try_async


async def fetch(context: TryContext) -> str:
    if context.attempt < 2:
        raise TimeoutError("temporary timeout")
    return "response body"


async def main() -> None:
    result = await try_async(
        fetch,
        catch=str,
        retry=RetryPolicy[str].exponential(
            times=3,  # retries after the first attempt
            initial_delay=0,
            should_retry=lambda context: "timeout" in context.error.lower(),
        ),
    )
    print(result)


asyncio.run(main())
```

`RetryPolicy.constant`, `.linear`, `.exponential`, and `.dynamic` use typed schedule values internally. When a callback does not provide enough information to infer the mapped error type, specialize the policy explicitly, as above. For direct composition, type the callback and compose the schedule explicitly:

```python
from better_result import DynamicDelay, RetryContext, RetryPolicy


def delay_for_attempt(context: RetryContext[str]) -> float:
    return context.attempt / 10


dynamic_policy = RetryPolicy[str].from_schedule(
    times=2,
    schedule=DynamicDelay(delay_for_attempt),
)
```

```text
Ok(value='response body')
```

`TryContext.attempt` starts at `1`. A policy receives a separate `RetryContext` containing the mapped error, attempt number, and optional cancellation token. Retry schedules derive their zero-based retry position from `attempt`; custom delay and retry predicates receive the context as their only argument. `CancellationToken` is best-effort: `try_async` monitors it, cancels an in-flight operation task, checks it before and after attempts and policy decisions, and interrupts retry waits. Ordinary async operations do not need to check the token themselves. Cancellation raises `asyncio.CancelledError`; it is not returned as a domain `Err`.

Delayed `try_result` retries use blocking `time.sleep()` and do not accept a cancellation token. Use `try_async` when retry waits must be interruptible.

Cancellation still follows Python's async cancellation boundaries. CPU-bound code, blocking calls, or dependencies that suppress `CancelledError` may not stop immediately. Code that needs tighter responsiveness can optionally call `context.cancel_token.raise_if_cancelled()` while processing work. Native task cancellation is likewise propagated.

## Collect or partition Results

```python
from better_result import Err, Ok, all_results, partition_results


all_results([Ok(1), Ok(2)])
# Ok(value=(1, 2))

all_results([Ok(1), Err("database unavailable"), Ok(3)])
# Err(value="database unavailable")

partition_results([Ok(1), Err("bad input"), Ok(2)])
# ([1, 2], ["bad input"])
```

- `all_results` returns every success in a tuple, or the first error in input order.
- `partition_results` returns `(success_values, error_values)` while preserving the relative order of each list.
- `flatten_result` turns `Result[Result[T, E], F]` into `Result[T, E | F]`.
- `all_results_async` and `partition_results_async` accept `Result` values or awaitables, await them concurrently, and preserve input order.

## Encode and decode at boundaries

`codec` and `async_codec` turn a `Result` into a typed envelope suitable for JSON or another wire format. Each schema returns either the converted value or `SchemaFailure` with structured issues.

```python
from better_result import Err, Ok, codec


result_codec = codec(
    serialize_ok=lambda user: {"id": user["id"]},
    serialize_err=lambda error: {"code": error},
    deserialize_ok=lambda value: value["id"],
    deserialize_err=lambda value: value["code"],
)

encoded = result_codec.serialize(Ok({"id": 42}))
assert encoded == Ok({"status": "ok", "value": {"id": 42}})

encoded_error = result_codec.serialize(Err("not_found"))
assert encoded_error == Ok({"status": "error", "error": {"code": "not_found"}})

decoded = result_codec.deserialize({"status": "ok", "value": {"id": 42}})
assert decoded == Ok(42)
```

A decoded wire-level error is returned as `Err` with the decoded error value. Malformed envelopes and schema rejections are returned as `Err(ResultDeserializationError)`, so the deserialization error type is `ErrOutput | ResultDeserializationError`. Serialization schema rejections are returned as `Err(ResultSerializationError)`.

Use `async_codec` when schemas are asynchronous. The `serialize_unsafe` and `deserialize_unsafe` methods unwrap codec failures and raise `UnwrapError`; they are useful only when the boundary failure is already handled elsewhere.

## Public API

The package exports:

- Core types: `Result`, `Ok`, `Err`, `UnwrapError`, `is_ok`, `is_err`
- Async and sync operations: `try_result`, `try_async`, `RetryPolicy`, `TryContext`, `RetryContext`, `CancellationToken`
- Retry ADTs: `RetryAfter`, `StopRetry`, `ConstantDelay`, `LinearBackoff`, `ExponentialBackoff`, `DynamicDelay`, `Jittered`
- Collection operations: `all_results`, `all_results_async`, `partition_results`, `partition_results_async`, `flatten_result`
- Codecs: `codec`, `async_codec`, `ResultCodec`, `AsyncResultCodec`
- Codec types: `SchemaFailure`, `CodecIssue`, `SerializedOk`, `SerializedErr`, `SerializedResult`, `SyncSchema`, `AsyncSchema`, `ResultSerializationError`, `ResultDeserializationError`

## Development

```bash
uv sync --all-packages
uv run pytest
uv run pytest --cov
uv run ruff check .
uv run ty check
```

The test suite includes example-based runtime tests, Hypothesis property-based tests, and static type contracts. Property tests exercise Result laws, collection reference models, retry bounds, backoff invariants, and cancellation behavior with automatically generated inputs.
