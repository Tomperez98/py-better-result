# better-result

`better-result` provides typed `Ok`/`Err` values for Python applications that want expected failures to be returned and programmer errors to fail fast.

## Install

```bash
uv add better-result
```

The package requires Python 3.12 or newer.

## Basic results

```python
from better_result.core import Err, Ok, Result


def parse_port(raw: str) -> Result[int, str]:
    try:
        port = int(raw)
    except ValueError:
        return Err("port is not an integer")
    if not 1 <= port <= 65_535:
        return Err("port is out of range")
    return Ok(port)


result = parse_port("8080")
if isinstance(result, Ok):
    print(result.value)
else:
    print(result.error)
```

`Ok` and `Err` are immutable result containers. Their concrete types describe only the payload they store: `Ok[T]` contains a success value and `Err[E]` contains an error value. Annotate the surrounding `Result[A, E]` when both branches need a complete type. Their payloads remain ordinary Python references, so a mutable payload can still be mutated by its owner.

The package follows one failure rule:

- expected outcomes such as invalid input, unavailable services, and rejected business rules are returned as `Err` values;
- broken callback contracts and programmer defects raise `Panic`;
- async cancellation and process-control exceptions are never converted into `Panic`.

## Composition

```python
from better_result.core import Ok

result = (
    parse_port("8080")
    .map(lambda port: port + 1)
    .and_then(lambda port: Ok(f"port:{port}"))
)

label = result.unwrap_or("unavailable")
```

Result transformations and observations use instance methods. Branch matching
always receives both explicit callbacks:

```python
message = result.match(
    lambda value: f"port:{value}",
    lambda error: f"unavailable: {error}",
)
```

`and_then` and `try_recover` callbacks must return a `Result`; returning another value raises `Panic` immediately.

## Tagged errors

Use tagged errors for stable application error vocabularies:

```python
from better_result.core import Err
from better_result.error import TaggedError


class InvalidPort(TaggedError, tag="InvalidPort"):
    def __init__(self, raw: str) -> None:
        super().__init__(message="invalid port", raw=raw)


failure = Err(InvalidPort("abc"))
message = failure.error.match(
    {"InvalidPort": lambda error: f"bad input: {error.raw}"},
)
```

`TaggedError.to_json()` and `Panic.to_json()` return recursively JSON-compatible diagnostic dictionaries, including stack traces. Use `to_safe_json()` for transport payloads when stack traces and other diagnostic details should not leave the process. `to_dict()` and `to_safe_dict()` provide the corresponding Python dictionaries.

## Async workflows

Use the async instance methods with coroutines. `asyncio.CancelledError`
propagates normally:

```python
from better_result.core import Ok


async def load_user(user_id: int):
    return Ok({"id": user_id})


async def load_name(user: dict[str, int]):
    return Ok(str(user["id"]))


async def workflow():
    result = await Ok(42).and_then_async(load_user)
    return await result.and_then_async(load_name)
```

`all_results_async()` preserves input order and cancels unfinished sibling operations when an input rejects. `partition_async()` waits for all inputs and returns `(success_values, error_values)`.

## Retries

```python
from better_result.retry import AsyncRetryConfig, try_async


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
from better_result.codec import codec
from better_result.core import Ok

result_codec = codec(
    serialize_ok=str,
    serialize_err=str,
    deserialize_ok=int,
    deserialize_err=str,
)

encoded = result_codec.serialize(Ok(42))
assert encoded.value == {"status": "ok", "value": "42"}
```

`codec()` is for synchronous schemas and returns `Result` values directly. Use `async_codec()` when a schema is asynchronous; its `serialize()` and `deserialize()` methods are awaitable.

## Development

```bash
uv run pytest -q
uv run pytest --cov
uv run ruff check src tests
uv run ty check
```

Coverage is opt-in and is enforced at 100% when running `uv run pytest --cov`.
