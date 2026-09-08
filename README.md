# better-result

> Inspired by the TypeScript [`better-result`](https://better-result.dev/) project.

Make expected failures explicit in Python without exceptions for ordinary control
flow.

```python
from better_result import Err, Ok, Result, TaggedError


class InvalidPort(TaggedError, tag="InvalidPort"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="port must be between 1 and 65535", raw=raw)


def parse_port(raw: str) -> Result[int, InvalidPort]:
    try:
        port = int(raw)
    except ValueError:
        return Err(InvalidPort(raw))
    return Ok(port) if 1 <= port <= 65_535 else Err(InvalidPort(raw))


def format_address(port: int) -> str:
    return f"http://localhost:{port}"


def address_for(raw: str) -> str:
    return (
        parse_port(raw)
        .map(format_address)
        .match(
            str,
            lambda error: f"Invalid input {error.raw!r}: {error.message}",
        )
    )


print(address_for("8080"))
print(address_for("nope"))
```

```text
http://localhost:8080
Invalid input 'nope': port must be between 1 and 65535
```

`better-result` is a small, dependency-free, typed `Result` library for Python
3.12+. A `Result[T, E]` is either `Ok[T]` or `Err[E]`, so a function can return
successful values and expected domain failures in one type that callers must
handle explicitly.

## Install

```bash
pip install better-result
```

Or add it to a `uv` project:

```bash
uv add better-result
```

## The supported API

The package has one supported import surface:

```python
from better_result import Err, Ok, PanicError, Result, TaggedError
```

| Name | Use |
| --- | --- |
| `Ok(value)` | Return a successful value. |
| `Err(error)` | Return an expected failure. |
| `Result[T, E]` | Type alias for `Ok[T] | Err[E]`. |
| `TaggedError` | Base class for typed application errors. |
| `PanicError` | Signal a programmer error or broken callback contract. |

Modules whose names begin with `_` are implementation details and are not
supported import paths.

## Compose results

Use the small set of operations that matches the shape of the computation:

- `map` transforms a successful value.
- `and_then` sequences a function that returns another `Result`.
- `map_error` translates an expected failure.
- `match` consumes both branches and is the canonical way to finish a result.
- `unwrap_or` supplies a fallback when that is genuinely the desired behavior.

```python
from better_result import Err, Ok, Result, TaggedError


class EmptyName(TaggedError, tag="EmptyName"):
    def __init__(self) -> None:
        super().__init__(message="name cannot be empty")


class InvalidAge(TaggedError, tag="InvalidAge"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="age must be a positive integer", raw=raw)


def read_name(name: str) -> Result[str, EmptyName]:
    return Ok(name) if name else Err(EmptyName())


def read_age(raw: str) -> Result[int, InvalidAge]:
    try:
        age = int(raw)
    except ValueError:
        return Err(InvalidAge(raw))
    return Ok(age) if age > 0 else Err(InvalidAge(raw))


profile = read_name("Alice").and_then(
    lambda name: read_age("30").map(
        lambda age: {"name": name, "age": age},
    ),
)

# Error types can be translated at a boundary.
error_message = read_age("unknown").map_error(lambda error: error.message)

# A fallback is explicit and local.
age = read_age("unknown").unwrap_or(0)
```

When the next operation is asynchronous, use `and_then_async`:

```python
async def load_preferences(
    profile: dict[str, object],
) -> Result[dict[str, object], TaggedError]:
    return Ok({**profile, "theme": "dark"})


# Use this inside an async function.
preferences = await profile.and_then_async(load_preferences)
```

`Ok` and `Err` are immutable. A callback that raises, or returns something
other than a `Result` where a `Result` is required, raises `PanicError` instead
of silently turning a programming defect into an expected application error.
Async cancellation is propagated normally.

## Define typed errors

Define concrete error classes with a non-empty string tag. Put the properties a
caller needs on the class and initialize them through `TaggedError`:

```python
from better_result import TaggedError


class UserNotFound(TaggedError, tag="UserNotFound"):
    user_id: int

    def __init__(self, user_id: int) -> None:
        super().__init__(message="user was not found", user_id=user_id)


error = UserNotFound(42)
assert error.name == "UserNotFound"
assert error.user_id == 42
```

The tag is a stable discriminator for runtime inspection and transport; it does
not automatically dispatch handlers. Use concrete error types and `Result.match`
when consuming failures.

`TaggedError.to_safe_json()` returns JSON-compatible transport metadata without
diagnostic stack traces:

```python
import json


payload = UserNotFound(42).to_safe_json()
json.dumps(payload)  # safe to hand to a JSON encoder
```

Expected outcomes should be represented by `Err` values. Programmer defects
and broken library contracts are `PanicError`s, not domain errors.

## Development

Clone the repository and run the checks with `uv`:

```bash
uv sync
uv run pytest -q
uv run pytest --cov
uv run ruff check
uv run ty check
```

The test suite is configured to require 100% branch coverage.

## License

Licensed under the [Apache License 2.0](LICENSE).
