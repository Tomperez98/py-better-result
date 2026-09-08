"""Runnable README example."""

from __future__ import annotations

from better_result import Err, Ok, Result, TaggedError


class InvalidPortError(TaggedError, tag="InvalidPort"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="port must be between 1 and 65535", raw=raw)


def parse_port(raw: str) -> Result[int, InvalidPortError]:
    try:
        port = int(raw)
    except ValueError:
        return Err(InvalidPortError(raw))
    return Ok(port) if 1 <= port <= 65_535 else Err(InvalidPortError(raw))


def format_address(port: int) -> str:
    return f"http://localhost:{port}"


def format_error(error: InvalidPortError) -> str:
    return f"Invalid input {error.raw!r}: {error.message}"


def address_for(raw: str) -> str:
    return parse_port(raw).map(format_address).match(str, format_error)


print(address_for("8080"))  # noqa: T201
print(address_for("nope"))  # noqa: T201
