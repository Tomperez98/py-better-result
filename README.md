# better-result

`better-result` is a small typed Result library for Python. It has one supported
entrypoint and one canonical workflow:

```python
from better_result import Err, Ok, Result, TaggedError
```

The supported public API is intentionally limited to `Ok`, `Err`, `Result`,
`TaggedError`, and `PanicError`. Modules whose names begin with `_` are
implementation details and are not supported import paths.

## The failure rule

- expected outcomes are returned as `Err` values;
- successful values are returned as `Ok` values;
- broken callback contracts and programmer defects raise `PanicError`;
- async cancellation propagates normally.

## Define concrete errors

Define application errors as concrete subclasses. Their constructors are the
single source of truth for their typed properties:

```python
from better_result import TaggedError


class InvalidPort(TaggedError, tag="InvalidPort"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="invalid port", raw=raw)
```

Tags must be non-empty strings. Dynamic tag factories and string-handler error
dispatch are deliberately not part of the public API.

## Compose results

Use `map` for a pure success transformation, `and_then` for a Result-returning
step, and `map_error` to translate a failure type:

```python
from better_result import Err, Ok, Result


def parse_port(raw: str) -> Result[int, InvalidPort]:
    try:
        port = int(raw)
    except ValueError:
        return Err(InvalidPort(raw))
    if not 1 <= port <= 65_535:
        return Err(InvalidPort(raw))
    return Ok(port)


result = parse_port("8080").map(lambda port: port + 1)
```

Consume a result with the one branch-consumption operation, `match`:

```python
message = result.match(
    lambda port: f"listening on {port}",
    lambda error: error.message,
)
```

Use `unwrap_or` only when a fallback value is the intended behavior:

```python
port = parse_port(raw).unwrap_or(8080)
```

## Async composition

Use `and_then_async` for an asynchronous Result-returning step:

```python
async def load_user(user_id: int) -> Result[dict[str, object], InvalidPort]: ...


result = await parse_port("8080").and_then_async(load_user)
```

## Serialization

`TaggedError.to_safe_json()` is the transport serializer. It returns JSON-safe
metadata without diagnostic stack traces. Diagnostic serializers are kept for
local debugging and are not part of the narrow root API.

## Development

```bash
uv run pytest -q
uv run pytest --cov
uv run ruff check
uv run ty check
```
