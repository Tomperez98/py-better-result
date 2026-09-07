# better-result

`better-result` provides typed `Ok`/`Err` values for Python applications that want expected failures to be returned and programmer errors to fail fast.

## Install

```bash
uv add better-result
```

The package requires Python 3.12 or newer.

## Basic results

```python
from better_result import Err, Ok, Result, err, ok


def parse_port(raw: str) -> Result[int, str]:
    try:
        port = int(raw)
    except ValueError:
        return err("port is not an integer")
    if not 1 <= port <= 65_535:
        return err("port is out of range")
    return ok(port)


result = parse_port("8080")
if isinstance(result, Ok):
    print(result.value)
else:
    print(result.error)
```

`Ok` and `Err` are immutable result containers. Their payloads remain ordinary Python references, so a mutable payload can still be mutated by its owner.

The package follows one failure rule:

- expected outcomes such as invalid input, unavailable services, and rejected business rules are returned as `Err` values;
- broken callback contracts and programmer defects raise `Panic`;
- async cancellation and process-control exceptions are never converted into `Panic`.

## Composition

```python
from better_result import and_then, map_result, unwrap_or

result = (
    parse_port("8080")
    .map(lambda port: port + 1)
    .and_then(lambda port: ok(f"port:{port}"))
)

label = unwrap_or(result, "unavailable")
```

Most utilities support both data-first and data-last forms:

```python
from better_result import map_result, ok

add_prefix = map_result(lambda value: f"port:{value}")
result = add_prefix(ok(8080))
```

`and_then` and `try_recover` callbacks must return a `Result`; returning another value raises `Panic` immediately.

## Tagged errors

Use tagged errors for stable application error vocabularies:

```python
from better_result import TaggedError, err, match_error


class InvalidPort(TaggedError, tag="InvalidPort"):
    def __init__(self, raw: str) -> None:
        super().__init__(message="invalid port", raw=raw)


failure = err(InvalidPort("abc"))
message = match_error(
    failure.error,
    {"InvalidPort": lambda error: f"bad input: {error.raw}"},
)
```

`TaggedError.to_json()` and `Panic.to_json()` return recursively JSON-compatible diagnostic dictionaries, including stack traces. Use `to_safe_json()` for transport payloads when stack traces and other diagnostic details should not leave the process. `to_dict()` and `to_safe_dict()` provide the corresponding Python dictionaries.

## Async workflows

Use the async combinators with coroutines. `asyncio.CancelledError` propagates normally:

```python
from better_result import and_then_async, ok


async def load_user(user_id: int):
    return ok({"id": user_id})


async def load_name(user: dict[str, int]):
    return ok(str(user["id"]))


async def workflow():
    result = await and_then_async(ok(42), load_user)
    return await and_then_async(result, load_name)
```

`all_async()` preserves input order and cancels unfinished sibling operations when an input rejects. `partition_async()` waits for all inputs and returns `(success_values, error_values)`.

## Retries

```python
from better_result import AsyncRetryConfig, try_async


async def operation(context): ...


result = await try_async(
    operation,
    retry=AsyncRetryConfig(times=3, backoff="exponential", delay_ms=100),
)
```

Retry delays must be finite and non-negative. Delay callbacks may return a number or an awaitable number. Cancellation stops the retry operation immediately.

## Codecs

Codecs validate Result payloads at serialization boundaries:

```python
from better_result import codec, codec_config, ok

result_codec = codec(
    codec_config(
        serialize_ok=str,
        serialize_err=str,
        deserialize_ok=int,
        deserialize_err=str,
    ),
)

encoded = result_codec.serialize(ok(42))
assert encoded.value == {"status": "ok", "value": "42"}
```

Synchronous schemas return a `Result` directly. If a schema is asynchronous, use `serialize_async()` or `deserialize_async()`; the synchronous methods return an awaitable in that configuration.

## Development

```bash
uv run pytest -q
uv run pytest --cov
uv run ruff check src tests
uv run ty check
```

Coverage is opt-in and is enforced at 100% when running `uv run pytest --cov`.
